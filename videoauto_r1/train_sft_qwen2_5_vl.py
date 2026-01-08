# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import datetime
import logging
from typing import Any, Dict, List
import torch
from torch import distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import SFTTrainer

from model.modeling_qwen2_5_vl_patched import Qwen2_5_VLForConditionalGeneration
from model.monkey_patch import apply_qwen2_5_vl_monkey_patch
from model.qwen_vl_utils.vision_process import process_vision_info
from utils.arguments import process_args
from utils.data_rl import LazyGRPODataset

logger = logging.getLogger("train_sft")


def set_model(model_args, model, logger):
    if model_args.tune_mm_vision:
        for p in model.visual.parameters():
            p.requires_grad = True
    else:
        for p in model.visual.parameters():
            p.requires_grad = False
    logger.info(f"tune_mm_vision: {model_args.tune_mm_vision}")

    if model_args.tune_mm_mlp:
        for p in model.visual.merger.parameters():
            p.requires_grad = True
    else:
        for p in model.visual.merger.parameters():
            p.requires_grad = False
    logger.info(f"tune_mm_mlp: {model_args.tune_mm_mlp}")

    if model_args.tune_mm_llm:
        for p in model.language_model.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for p in model.language_model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    logger.info(f"tune_mm_llm: {model_args.tune_mm_llm}")

    # in deepspeed, we need to count p.ds_numel instead of p.numel
    if is_deepspeed_zero3_enabled():

        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()

    else:

        def numel(p):
            return p.numel()

    total_params = sum(numel(p) for p in model.parameters())
    trainable_params = sum(numel(p) for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")


class SFTCollator:
    def __init__(self, processor, loss_type="only_assistant", max_length=16384):
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.loss_type = loss_type
        self.max_length = max_length
        logger.info(f"Model max length: {self.max_length}")

        assert isinstance(
            self.processor, Qwen2_5_VLProcessor
        ), "Only Qwen2.5_VLProcessor is supported."
        assert loss_type in [
            "exclude_visual",
            "only_assistant",
            "format_answer",
        ], f"Loss type {loss_type} is not supported."
        logger.info(f"SFT loss type: {loss_type}")

        if loss_type in ["only_assistant", "format_answer"]:
            self.assistant_prefix_ids = self.tokenizer(
                "<|im_start|>assistant\n", add_special_tokens=False
            ).input_ids
            # should be [151644, 77091, 198]
            logger.info(rf"id of <|im_start|>assistant\n: {self.assistant_prefix_ids}")

        if loss_type == "format_answer":
            self.think_open_ids = self.tokenizer(
                "<think>", add_special_tokens=False
            ).input_ids
            self.think_close_ids = self.tokenizer(
                "</think>", add_special_tokens=False
            ).input_ids
            logger.info(
                f"id of <think>: {self.think_open_ids}, </think>: {self.think_close_ids}"
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples for training."""
        for idx in range(len(examples)):
            examples[idx]["messages"].append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": examples[idx]["response"]}],
                }
            )

        messages = [example["messages"] for example in examples]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False) for msg in messages
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )

        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            **video_kwargs,
        )

        if self.loss_type == "exclude_visual":
            labels = batch["input_ids"].clone()

        elif self.loss_type == "only_assistant":
            labels = torch.full_like(batch["input_ids"], -100)

            # <|im_start|>assistant\n
            pattern = torch.tensor(self.assistant_prefix_ids, device=labels.device)
            wins = batch["input_ids"].unfold(dimension=-1, size=pattern.numel(), step=1)
            match_mask = (wins == pattern).all(dim=-1)

            for b_idx in range(match_mask.size(0)):
                pos = torch.nonzero(match_mask[b_idx], as_tuple=False).squeeze(1)

                if pos.numel() == 0:
                    # not found：the text is truncated before assistant response
                    # ignore the sample in this case
                    logger.warning(
                        f"[SFTCollator] No assistant prefix after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue

                # if exists multiple times，use the last one（usually the last round of assistant response）
                start_idx = pos[-1].item() + pattern.numel()
                labels[b_idx, start_idx:] = batch["input_ids"][b_idx, start_idx:]

        elif self.loss_type == "format_answer":
            labels = torch.full_like(batch["input_ids"], -100)

            # <|im_start|>assistant\n
            pattern = torch.tensor(self.assistant_prefix_ids, device=labels.device)
            wins = batch["input_ids"].unfold(dimension=-1, size=pattern.numel(), step=1)
            match_mask = (wins == pattern).all(dim=-1)

            # <think>...</think>
            p_open = torch.tensor(self.think_open_ids, device=labels.device)
            p_close = torch.tensor(self.think_close_ids, device=labels.device)
            wins_close = batch["input_ids"].unfold(
                dimension=-1, size=p_close.numel(), step=1
            )
            match_close_mask = (wins_close == p_close).all(dim=-1)

            for b_idx in range(match_mask.size(0)):
                pos = torch.nonzero(match_mask[b_idx], as_tuple=False).squeeze(1)

                if pos.numel() == 0:
                    # not found：the text is truncated before assistant response
                    # ignore the sample in this case
                    logger.warning(
                        f"[SFTCollator] No assistant prefix after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue

                think_open_start = pos[-1].item() + pattern.numel()
                think_open_end = think_open_start + p_open.numel()
                labels[b_idx, think_open_start:think_open_end] = batch["input_ids"][
                    b_idx, think_open_start:think_open_end
                ]

                close_pos = torch.nonzero(
                    match_close_mask[b_idx], as_tuple=False
                ).squeeze(1)

                if close_pos.numel() == 0:
                    logger.warning(
                        f"[SFTCollator] No </think> after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue

                think_close_start = close_pos[-1].item()
                labels[b_idx, think_close_start:] = batch["input_ids"][
                    b_idx, think_close_start:
                ]

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported.")

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Handle visual tokens based on processor type
        if isinstance(self.processor, Qwen2_5_VLProcessor):
            visual_tokens = [
                self.tokenizer.convert_tokens_to_ids(self.processor.image_token),
                self.tokenizer.convert_tokens_to_ids(self.processor.video_token),
            ]
        else:
            raise NotImplementedError(f"Unsupported processor: {type(self.processor)}")

        for visual_token_id in visual_tokens:
            labels[labels == visual_token_id] = -100

        batch["labels"] = labels
        return batch


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    # parsing arguments
    model_args, data_args, training_args = process_args(is_grpo=False)
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # set up training args
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # set up loggers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger.info(f"Global Rank: {global_rank}, Local Rank: {local_rank}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")

    # monkey patch the Qwen model before initialization for handling the mixed modality dataset
    if model_args.apply_monkey_patch:
        apply_qwen2_5_vl_monkey_patch(model_args.apply_monkey_patch)
        logger.info(f"Qwen2.5-VL patch applied, mode: {model_args.apply_monkey_patch}")

    # build processor
    processor = AutoProcessor.from_pretrained(model_args.model_path)

    # build model
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
        "attn_implementation": "flash_attention_2",
        "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Model loaded from {model_args.model_path}")

    # set model
    set_model(model_args, model, logger)

    # build dataset
    train_dataset = LazyGRPODataset(
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        video_min_pixels=data_args.video_min_pixels,
        video_max_pixels=data_args.video_max_pixels,
        video_total_pixels=data_args.video_total_pixels,
        max_frames=data_args.max_frames,
        nframes=data_args.nframes,
        fps=data_args.fps,
        image_min_pixels=data_args.image_min_pixels,
        image_max_pixels=data_args.image_max_pixels,
        rl_mode="direct_rl",  # use direct prompt
    )

    # build collator
    collator = SFTCollator(
        processor,
        loss_type="only_assistant",
        max_length=model_args.model_max_length,
    )

    # initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()
    logger.info("Training finished")

    # save model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
