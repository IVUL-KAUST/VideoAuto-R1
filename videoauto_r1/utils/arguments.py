from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import transformers
import yaml
from transformers import TrainingArguments
from transformers.hf_argparser import DataClass
from trl import SFTConfig


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of input model"},
    )
    tune_mm_llm: bool = field(
        default=True,
        metadata={"help": "Whether to tune the llm."},
    )
    tune_mm_mlp: bool = field(
        default=True,
        metadata={"help": "Whether to tune the mlp projector."},
    )
    tune_mm_vision: bool = field(
        default=True,
        metadata={"help": "Whether to tune the vision encoder."},
    )
    model_max_length: int = field(
        default=16384,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    apply_monkey_patch: Optional[
        Literal["enforce_image", "enforce_video", "enforce_image_video"]
    ] = field(
        default=None,
        metadata={
            "help": "Monkey patch mode for Qwen2.5-VL for text/image/video mixed dataset. Options: 'enforce_image', "
            "'enforce_video', 'enforce_image_video', or None (no patch). Default: None"
        },
    )


@dataclass
class DataArguments:
    dataset_info: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataset yaml file."},
    )
    dataset_name: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "A list of dataset names. Must be one of the names in the dataset config."
        },
    )
    image_min_pixels: Optional[int] = field(
        default=4 * 28 * 28,
        metadata={"help": "The minimum number of pixels for an image."},
    )
    image_max_pixels: Optional[int] = field(
        default=16384 * 28 * 28,
        metadata={"help": "The maximum number of pixels for an image."},
    )
    video_min_pixels: Optional[int] = field(
        default=128 * 28 * 28,
        metadata={"help": "The minimum number of pixels for a frame in a video."},
    )
    video_max_pixels: Optional[int] = field(
        default=768 * 28 * 28,
        metadata={"help": "The maximum number of pixels for a frame in a video."},
    )
    video_total_pixels: Optional[int] = field(
        default=115200 * 28 * 28,
        metadata={"help": "The maximum total number of pixels for a video."},
    )
    max_frames: Optional[int] = field(
        default=768,
        metadata={"help": "The maximum number of frames allowed for a video."},
    )
    nframes: Optional[int] = field(
        default=None,
        metadata={
            "help": "The exact number of frames to extract for a video. If None, the frame number is decided by fps and max_frames."
        },
    )
    fps: Optional[float] = field(
        default=2.0,
        metadata={"help": "The frames per second to extract frames for a video."},
    )
    use_confirm_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use think + segment + confirm + answer format in GQA"},
    )
    rl_mode: Optional[Literal["direct_rl", "cot_rl", "answer_twice_rl"]] = field(
        default="cot_rl",
        metadata={
            "help": (
                "RL mode: 'direct_rl' (\\boxed{...}); 'cot_rl' (<think>...</think>\\boxed{...}); "
                "'answer_twice_rl' (\\boxed{...}<think>...</think>\\boxed{...})."
            ),
            "choices": ["direct_rl", "cot_rl", "answer_twice_rl"],
        },
    )
    task_type: Optional[list[str]] = field(
        default_factory=lambda: ["all"],
        metadata={"help": "Different task type for videor1 data ablation"},
    )

    def __post_init__(self):
        # read dataset config
        with open(self.dataset_info, "r", encoding="utf-8") as f:
            self.dataset_config = yaml.safe_load(f)

        # make sure dataset_name is a list
        if isinstance(self.dataset_name, str):
            self.dataset_name = [self.dataset_name]

        # check if dataset_name is in dataset_config
        for name in self.dataset_name:
            if name not in self.dataset_config:
                raise ValueError(f"Dataset {name} not found in dataset config.")


@dataclass
class GRPOConfig(TrainingArguments):
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`."
        },
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    max_completion_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the generated completion."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations to sample."},
    )
    num_generations_compare: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations for think group and nothink group."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={
            "help": "Number of iterations per batch (denoted as μ in the algorithm)."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
        },
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the Liger GRPO loss."},
    )
    generation_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `supress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of completions to print with `rich`. If `None`, all completions are logged."
        },
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `1.0` masks all but the highest entropy token; `0.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )

    # VLLM configuration
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = (
                self.per_device_train_batch_size
                * num_processes
                * self.steps_per_generation
            )
        elif (
            self.generation_batch_size is not None and self.steps_per_generation is None
        ):
            # Just ensure the value is divisible by the global batch size
            if (
                self.generation_batch_size
                % (self.per_device_train_batch_size * num_processes)
                != 0
            ):
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif (
            self.generation_batch_size is None and self.steps_per_generation is not None
        ):
            self.generation_batch_size = (
                self.per_device_train_batch_size
                * num_processes
                * self.steps_per_generation
            )
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )


def process_args(is_grpo=False) -> Tuple[DataClass, ...]:
    if is_grpo:
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, GRPOConfig)
        )
    else:
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, SFTConfig)
        )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args
