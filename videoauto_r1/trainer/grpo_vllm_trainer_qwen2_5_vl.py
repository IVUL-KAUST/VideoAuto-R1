# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from collections import defaultdict
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import transformers
from accelerate.utils import gather, set_seed
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, Sampler
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models.utils import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import entropy_from_logits, pad, selective_log_softmax

from model.qwen_vl_utils.vision_process import cached_process_vision_info
from utils.arguments import GRPOConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            (self.num_samples // self.batch_size)
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )


def split_to_chunk(input_dict: dict, num_chunks: int) -> list[dict]:
    """
    Splits a dictionary into `num_chunks` equal parts. If the value is tensor, then split along the first dimension.
    If the value is a list, then split accordingly.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> z = [1, 2, 3, None, 5, 6]
    >>> input_dict = {"x": x, "y": y, "z": z}
    >>> split_to_chunk(input_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]]), "z": [1, 2]},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]]), "z": [3, None]},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]]), "z": [5, 6]},
    ]
    ```
    """
    first_tensor = next(v for v in input_dict.values() if isinstance(v, torch.Tensor))
    chunk_size = first_tensor.shape[0] // num_chunks

    splitted_dict = []
    for i in range(num_chunks):
        tmp_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                tmp_dict[k] = v[i * chunk_size : (i + 1) * chunk_size]
            elif isinstance(v, list):
                tmp_dict[k] = [
                    v[j] for j in range(i * chunk_size, (i + 1) * chunk_size)
                ]
            else:
                raise ValueError(f"Unsupported type {type(v)} for key {k}")
        splitted_dict.append(tmp_dict)
    return splitted_dict


def shuffle_sequence_dict(
    seq_dict: dict[str, Optional[Sequence]],
) -> dict[str, Optional[Sequence]]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Optional[Sequence]) -> Optional[Sequence]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_visual_data(batch):
    """
    Splits and reorganizes the visual data (images and videos) in a batch into per-sample structures aligned
    with the message layout. It separates and groups `pixel_values`, `pixel_values_videos`, `image_grid_thw`,
    `video_grid_thw`, and `second_per_grid_ts` into a list of tensors (could be None if no visual information
    in this sample), so that each sample has its own corresponding visual tensors instead of one large
    concatenated tensor across the batch.
    """
    # if only have text data, skip
    if "pixel_values" not in batch and "pixel_values_videos" not in batch:
        return batch

    # split the image pixel values
    if "image_grid_thw" in batch:
        image_lengths = batch["image_grid_thw"].prod(dim=1).tolist()
        if sum(image_lengths) != batch["pixel_values"].size(0):
            raise ValueError(
                f"Mismatch: sum(image_lengths) = {sum(image_lengths)} != pixel_values.size(0) = {batch["pixel_values"].size(0)}"
            )
        split_pixel_values = list(
            torch.split(batch["pixel_values"], image_lengths, dim=0)
        )  # [total, feature_dim]

    # split the video pixel values
    if "video_grid_thw" in batch:
        video_lengths = batch["video_grid_thw"].prod(dim=1).tolist()
        if sum(video_lengths) != batch["pixel_values_videos"].size(0):
            raise ValueError(
                f"Mismatch: sum(video_lengths) = {sum(video_lengths)} != pixel_values_videos.size(0) = {batch['pixel_values_videos'].size(0)}"
            )
        split_pixel_values_videos = list(
            torch.split(batch["pixel_values_videos"], video_lengths, dim=0)
        )  # [total, feature_dim]

    batch_pixel_values = []
    batch_image_grid_thw = []
    batch_pixel_values_videos = []
    batch_video_grid_thw = []
    batch_second_per_grid_ts = []
    image_idx = 0
    video_idx = 0

    for message in batch["messages"]:
        tmp_pixel_values = []
        tmp_image_grid_thw = []
        tmp_pixel_values_videos = []
        tmp_video_grid_thw = []
        tmp_second_per_grid_ts = []

        for msg in message:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if "image" in ele or "image_url" in ele:
                        tmp_pixel_values.append(split_pixel_values[image_idx])
                        tmp_image_grid_thw.append(batch["image_grid_thw"][image_idx])
                        image_idx += 1

                    if "video" in ele:
                        tmp_pixel_values_videos.append(
                            split_pixel_values_videos[video_idx]
                        )
                        tmp_video_grid_thw.append(batch["video_grid_thw"][video_idx])
                        tmp_second_per_grid_ts.append(
                            batch["second_per_grid_ts"][video_idx]
                        )
                        video_idx += 1

        if len(tmp_pixel_values) > 0:
            tmp_pixel_values = torch.cat(tmp_pixel_values, dim=0)
            tmp_image_grid_thw = torch.stack(tmp_image_grid_thw, dim=0)
        batch_pixel_values.append(tmp_pixel_values)
        batch_image_grid_thw.append(tmp_image_grid_thw)

        if len(tmp_pixel_values_videos) > 0:
            tmp_pixel_values_videos = torch.cat(tmp_pixel_values_videos, dim=0)
            tmp_video_grid_thw = torch.stack(tmp_video_grid_thw, dim=0)
            tmp_second_per_grid_ts = torch.stack(tmp_second_per_grid_ts, dim=0)
        batch_pixel_values_videos.append(tmp_pixel_values_videos)
        batch_video_grid_thw.append(tmp_video_grid_thw)
        batch_second_per_grid_ts.append(tmp_second_per_grid_ts)

    # Note that we might have empty list such as in `pixel_values_videos` if there is no video in the batch.
    # We will pop them out in `unsplit_visual_data`.
    return {
        **batch,
        "pixel_values": batch_pixel_values,
        "image_grid_thw": batch_image_grid_thw,
        "pixel_values_videos": batch_pixel_values_videos,
        "video_grid_thw": batch_video_grid_thw,
        "second_per_grid_ts": batch_second_per_grid_ts,
    }


def unsplit_visual_data(batch):
    """
    Opposite of `split_visual_data`. Merges a list of tensors in `batch["pixel_values"]`and others
    back into a single tensor along the first dimension.
    """
    if "pixel_values" in batch:
        non_empty_pixel_values = [pv for pv in batch["pixel_values"] if len(pv) > 0]
        non_empty_image_grid_thw = [
            thw for thw in batch["image_grid_thw"] if len(thw) > 0
        ]
        assert len(non_empty_pixel_values) == len(non_empty_image_grid_thw)
        if non_empty_pixel_values:
            batch["pixel_values"] = torch.cat(non_empty_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(non_empty_image_grid_thw, dim=0)
        else:
            batch.pop("pixel_values")
            batch.pop("image_grid_thw")

    if "pixel_values_videos" in batch:
        non_empty_pixel_values_videos = [
            pv for pv in batch["pixel_values_videos"] if len(pv) > 0
        ]
        non_empty_video_grid_thw = [
            thw for thw in batch["video_grid_thw"] if len(thw) > 0
        ]
        non_empty_second_per_grid_ts = [
            ts for ts in batch["second_per_grid_ts"] if len(ts) > 0
        ]
        if non_empty_pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(
                non_empty_pixel_values_videos, dim=0
            )
            batch["video_grid_thw"] = torch.cat(non_empty_video_grid_thw, dim=0)
            batch["second_per_grid_ts"] = torch.cat(non_empty_second_per_grid_ts, dim=0)
        else:
            batch.pop("pixel_values_videos")
            batch.pop("video_grid_thw")
            batch.pop("second_per_grid_ts")

    return batch


def get_high_entropy_mask(
    entropies: torch.Tensor, mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

    Args:
        entropies (`torch.Tensor`):
            Tensor of shape (batch_size, seq_len) with per-token entropy values.
        mask (`torch.Tensor`):
            Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
        threshold (`float`):
            Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

    Returns:
        `torch.Tensor`:
            Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold and
            `False` otherwise.
    """
    non_pad_entropies = entropies[mask.bool()].float()
    if non_pad_entropies.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.bool)
    entropy_threshold = torch.quantile(non_pad_entropies, threshold)
    masked_entropies = entropies * mask.float()
    entropy_mask = masked_entropies >= entropy_threshold
    return entropy_mask & mask.bool()  # ensure padding tokens are always masked out


class Qwen2_5_VL_GRPOVLLMTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Should be a [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. Can be either: a single reward function, or list of
            reward functions, where each item can independently be any of the above types.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        # Models
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        assert self.vllm_mode == "colocate", "Only colocate mode is supported on MAST."
        self.vllm_gpu_memory_utilization = (
            args.vllm_gpu_memory_utilization
        )  # only applies to colocation mode
        self.vllm_tensor_parallel_size = (
            args.vllm_tensor_parallel_size
        )  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.importance_sampling_level = args.importance_sampling_level
        self.top_entropy_quantile = args.top_entropy_quantile

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon

        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            processing_class=processing_class,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            # For deepspeed or non-distributed models, the reference model has been created
            self.ref_model = ref_model

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            assert self.vllm_mode == "colocate"
            # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                    f"({self.accelerator.num_processes}) evenly."
                )

            if self.vllm_tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(
                            range(
                                i * self.vllm_tensor_parallel_size,
                                (i + 1) * self.vllm_tensor_parallel_size,
                            )
                        )
                        for i in range(
                            self.accelerator.num_processes
                            // self.vllm_tensor_parallel_size
                        )
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

            if (
                self.max_prompt_length is not None
                and self.max_completion_length is not None
            ):
                max_model_len = self.max_prompt_length + self.max_completion_length
            else:
                max_model_len = None
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_num_seqs=self.args.per_device_train_batch_size
                * self.vllm_tensor_parallel_size
                * self.args.steps_per_generation,
                max_model_len=max_model_len,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                max_num_batched_tokens=4096,
            )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        if version.parse(transformers.__version__) >= version.parse("4.52.0"):
            # from transformers 4.52.0, the `seed_worker` requires the `num_workers` and `rank` arguments
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        else:
            dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ):
        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw

        if video_grid_thw is not None and pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_grid_thw"] = video_grid_thw
            model_inputs["second_per_grid_ts"] = second_per_grid_ts

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        model_inputs["logits_to_keep"] = logits_to_keep + 1

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[
            :, -logits_to_keep:, :
        ]  # (B, logits_to_keep, H)
        return last_hidden_state

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        # used in Qwen2.5VL
        messages=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        # Chunk inputs into smaller batches to reduce memory peak
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        batch = {
            "messages": messages,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
            batch["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos
            batch["video_grid_thw"] = video_grid_thw
            batch["second_per_grid_ts"] = second_per_grid_ts

        splitted_batch = split_visual_data(batch)
        chunked_batch = split_to_chunk(
            splitted_batch,
            num_chunks=input_ids.size(0) // batch_size,
        )
        for chunk in chunked_batch:
            model_inputs = unsplit_visual_data(chunk)
            model_inputs.pop("messages")

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = model_inputs["input_ids"][:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # For non-PEFT models, simply gather (if needed) and update each parameter individually.
        for name, param in self.model.named_parameters():
            name = self._fix_param_name_to_vllm(name)
            with gather_if_zero3([param]):
                assert self.vllm_mode == "colocate"
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        assert self.vllm_mode == "colocate"
        self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = split_visual_data(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_to_chunk(
                    generation_batch,
                    num_chunks=self.args.steps_per_generation,
                )
                self._buffered_inputs = [
                    unsplit_visual_data(batch) for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, completions, solutions, problem_types):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(completions),
            len(self.reward_funcs),
            device=device,
        )
        for i, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    completions=completions,
                    solutions=solutions,
                    problem_types=problem_types,
                )

                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _prepare_vllm_inputs(
        self,
        messages,
        prompts_text,
        image_inputs,
        video_inputs,
        video_kwargs,
    ):
        # Restore the image_inputs, video_inputs, and video_kwargs that were assembled into a batch through process_vision_info
        # back into a list, where each item corresponds to a vLLM input containing prompt, multi_modal_data, and mm_processor_kwargs.
        vllm_inputs = []
        image_idx = 0
        video_idx = 0
        video_fps_list = video_kwargs["fps"]

        for message, prompt in zip(messages, prompts_text):
            tmp_image_inputs = []
            tmp_video_inputs = []
            tmp_video_fps_list = []

            for msg in message:
                if isinstance(msg["content"], list):
                    for ele in msg["content"]:
                        if "image" in ele or "image_url" in ele:
                            tmp_image_inputs.append(image_inputs[image_idx])
                            image_idx += 1
                        elif "video" in ele:
                            tmp_video_inputs.append(video_inputs[video_idx])
                            tmp_video_fps_list.append(video_fps_list[video_idx])
                            video_idx += 1

            tmp_llm_inputs = {"prompt": prompt}

            # if contains multi-modal data
            tmp_mm_data = {}
            if len(tmp_image_inputs) > 0:
                tmp_mm_data["image"] = tmp_image_inputs
            if len(tmp_video_inputs) > 0:
                tmp_mm_data["video"] = tmp_video_inputs
            if tmp_mm_data:
                tmp_llm_inputs["multi_modal_data"] = tmp_mm_data

            # if contains mm_processor_kwargs
            if len(tmp_video_fps_list) > 0:
                tmp_llm_inputs["mm_processor_kwargs"] = {
                    "fps": tmp_video_fps_list,
                }

            vllm_inputs.append(tmp_llm_inputs)

        if image_inputs:
            assert image_idx == len(image_inputs), (
                f"Not all images were processed: "
                f"image_idx {image_idx} != len(image_inputs) {len(image_inputs)}"
            )

        if video_inputs:
            assert video_idx == len(video_inputs), (
                f"Not all videos were processed: "
                f"video_idx {video_idx} != len(video_inputs) {len(video_inputs)}"
            )
        return vllm_inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        solutions = [example["response"] for example in inputs]  # gt solutions
        problem_types = [example["problem_type"] for example in inputs]
        messages = [example["messages"] for example in inputs]

        prompts_text = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        image_inputs, video_inputs, video_kwargs = cached_process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            **video_kwargs,
        )

        # move to device
        if "pixel_values" in prompt_inputs:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(device)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].to(device)
        if "pixel_values_videos" in prompt_inputs:
            prompt_inputs["pixel_values_videos"] = prompt_inputs[
                "pixel_values_videos"
            ].to(device)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].to(device)
            prompt_inputs["second_per_grid_ts"] = prompt_inputs[
                "second_per_grid_ts"
            ].to(device)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"].to(device),
            prompt_inputs["attention_mask"].to(device),
        )

        assert prompt_ids.shape[-1] <= self.max_prompt_length, print(
            f"Prompt length {prompt_ids.shape[-1]} exceeds max_prompt_length {self.max_prompt_length}"
        )

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            assert self.vllm_mode == "colocate"
            if self.guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
            else:
                guided_decoding = None

            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": self.max_completion_length,
                "guided_decoding": guided_decoding,
            }
            if self.args.generation_kwargs is not None:
                generation_kwargs.update(self.args.generation_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            # convert input data to vLLM inputs
            vllm_inputs = self._prepare_vllm_inputs(
                messages,
                prompts_text,
                image_inputs,
                video_inputs,
                video_kwargs,
            )

            if self.vllm_tensor_parallel_size > 1:
                # Gather vLLM inputs from all ranks in the TP group and flatten.
                # Each rank starts with its own vLLM inputs; after gathering, all ranks see the full group set.
                orig_size = len(vllm_inputs)
                gathered_vllm_inputs = [
                    None for _ in range(self.vllm_tensor_parallel_size)
                ]
                torch.distributed.all_gather_object(
                    gathered_vllm_inputs, vllm_inputs, group=self.tp_group
                )
                all_vllm_inputs = [
                    p for sublist in gathered_vllm_inputs for p in sublist
                ]
            else:
                all_vllm_inputs = vllm_inputs

            with profiling_context(self, "vLLM.generate"):
                all_outputs = self.llm.generate(
                    all_vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            completion_ids = [
                output.token_ids
                for outputs in all_outputs
                for output in outputs.outputs
            ]

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(
                    local_rank_in_group * orig_size,
                    (local_rank_in_group + 1) * orig_size,
                )
                completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = (
                    prompt_ids,
                    prompt_mask,
                )
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    # used in Qwen2.5VL
                    messages=messages,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                    video_grid_thw=prompt_inputs.get("video_grid_thw"),
                    second_per_grid_ts=prompt_inputs.get("second_per_grid_ts"),
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        # used in Qwen2.5VL
                        messages=messages,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                        video_grid_thw=prompt_inputs.get("video_grid_thw"),
                        second_per_grid_ts=prompt_inputs.get("second_per_grid_ts"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                # used in Qwen2.5VL
                                messages=messages,
                                pixel_values=prompt_inputs.get("pixel_values"),
                                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                                pixel_values_videos=prompt_inputs.get(
                                    "pixel_values_videos"
                                ),
                                video_grid_thw=prompt_inputs.get("video_grid_thw"),
                                second_per_grid_ts=prompt_inputs.get(
                                    "second_per_grid_ts"
                                ),
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            completions_text,
            solutions,
            problem_types,
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        output = {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_values_videos" in prompt_inputs:
            output["pixel_values_videos"] = prompt_inputs["pixel_values_videos"]
        if "video_grid_thw" in prompt_inputs:
            output["video_grid_thw"] = prompt_inputs["video_grid_thw"]
        if "second_per_grid_ts" in prompt_inputs:
            output["second_per_grid_ts"] = prompt_inputs["second_per_grid_ts"]
        return output

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=self.top_entropy_quantile < 1.0,
            messages=inputs.get("messages"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(
                -1
            ) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        if self.top_entropy_quantile < 1.0:
            mean_entropy = masked_batch_mean(entropies)
            self._metrics[mode]["entropy"].append(
                self.accelerator.gather(mean_entropy).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
