timestamp=$(date +%Y%m%d%H%M)
exp=qwen2_5_vl_grpo_auto_text_image_video

video_min_pixels=$((16 * 28 * 28))
video_max_pixels=$((768 * 28 * 28))
video_total_pixels=$((4096 * 28 * 28))
max_frames=256

image_min_pixels=$((4 * 28 * 28))
image_max_pixels=$((768 * 28 * 28))

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 --nnodes=4 \
videoauto_r1/train_grpo_qwen2_5_vl.py \
--model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
--output_dir "experiments/train_qwen_rl_auto/${timestamp}-${exp}/" \
--dataset_info "data/data_config.yaml" \
--dataset_name "DAPO-Math" "VIRL" "ThinkLite-VL-Hard" "ActivityNet-TVG" "Charades-STA" "TimeR1" "NeXT-GQA" "VideoR1" "TVBench" "STI-Bench" "MMR-VBench" \
--video_min_pixels "$video_min_pixels" \
--video_max_pixels "$video_max_pixels" \
--video_total_pixels "$video_total_pixels" \
--max_frames "$max_frames" \
--image_min_pixels "$image_min_pixels" \
--image_max_pixels "$image_max_pixels" \
--log_on_each_node False \
--logging_dir /tmp/videothink/log/ \
--bf16 \
--per_device_train_batch_size 8 \
--learning_rate 1e-6 \
--lr_scheduler_type "constant_with_warmup" \
--max_grad_norm 1.0 \
--num_train_epochs 1 \
--gradient_checkpointing True \
--deepspeed "scripts/train/dsconfig/zero3.json" \
--save_strategy "steps" \
--save_steps 200 \
--logging_steps 10 \
--eval_strategy "no" \
--report_to "tensorboard" \
--optim "adamw_torch_fused" \
--weight_decay 0.01 \
--tune_mm_llm True \
--tune_mm_mlp True \
--tune_mm_vision False \
--max_prompt_length 8192 \
--max_completion_length 2048 \
--num_generations 16 \
--steps_per_generation 8 \
--beta 0.01 \
--reward_funcs "accuracy_boxed1" "accuracy_boxed2" "format_twice_boxed" \
--use_vllm True \
--vllm_mode "colocate" \
--vllm_gpu_memory_utilization 0.4 \
--vllm_tensor_parallel_size 4 \
--reward_weights 0.9 1.1 1 \
--apply_monkey_patch "enforce_image_video" \
--rl_mode "answer_twice_rl" \
--save_total_limit 1
