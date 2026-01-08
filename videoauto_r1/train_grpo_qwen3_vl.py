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
import torch
from torch import distributed as dist
from transformers import AutoProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration
from model.monkey_patch import apply_qwen3_vl_monkey_patch
from reward.reward import reward_funcs_registry
from trainer.grpo_vllm_trainer_qwen3_vl import Qwen3_VL_GRPOVLLMTrainer
from utils.arguments import process_args
from utils.data_rl import LazyGRPODataset

logger = logging.getLogger("train_grpo")


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
        for p in model.visual.deepstack_merger_list.parameters():
            p.requires_grad = True
    else:
        for p in model.visual.merger.parameters():
            p.requires_grad = False
        for p in model.visual.deepstack_merger_list.parameters():
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


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    # parsing arguments
    model_args, data_args, training_args = process_args(is_grpo=True)
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
        apply_qwen3_vl_monkey_patch(model_args.apply_monkey_patch)
        logger.info(f"Qwen3-VL patch applied, mode: {model_args.apply_monkey_patch}")

    # build processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_path, padding_side="left"
    )

    # build model
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
        "attn_implementation": "flash_attention_2",
        # "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Model loaded from {model_args.model_path}")

    # build reference model
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Reference model loaded from {model_args.model_path}")

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
        rl_mode=data_args.rl_mode,  # direct_rl, cot_rl, answer_twice_rl
    )

    # reward functions
    reward_funcs = [reward_funcs_registry[func] for func in training_args.reward_funcs]

    # initialize GRPO trainer
    trainer = Qwen3_VL_GRPOVLLMTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=processor,
        args=training_args,
        reward_funcs=reward_funcs,
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
