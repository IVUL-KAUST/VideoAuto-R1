# data configuration
image_min_tokens=128
image_max_tokens=16384
video_min_tokens=16
video_max_tokens=768
video_total_tokens=16384

image_min_pixels=$((image_min_tokens * 28 * 28))
image_max_pixels=$((image_max_tokens * 28 * 28))
video_min_pixels=$((video_min_tokens * 28 * 28))
video_max_pixels=$((video_max_tokens * 28 * 28))
video_total_pixels=$((video_total_tokens * 28 * 28))

# experiment configuration
model_path=IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B
output_path=experiments/videoauto-r1/VideoAuto-R1-Qwen2.5-VL-7B/

# task list
tasks=(videomme_boxed mvbench_boxed longvideobench_val_v_boxed mmvu_val_mc_boxed video_mmmu_boxed charades_boxed activitynet_tvg_boxed nextgqa_boxed)

master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
for task in "${tasks[@]}"; do
    echo "Running task $task"

    if [[ "$task" == "video_mmmu_boxed" || "$task" == "mmvu_val_mc_boxed" ]]; then
    max_frames=64
    elif [[ "$task" == "activitynet_tvg_boxed" ]]; then
    max_frames=512
    else
    max_frames=256
    fi

    accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model qwen2_5_vl_autothink \
    --model_args inference_mode=auto,early_exit_thresh=0.97,pretrained=$model_path,video_min_pixels=$video_min_pixels,video_max_pixels=$video_max_pixels,video_total_pixels=$video_total_pixels,max_frames=$max_frames,image_min_pixels=$image_min_pixels,image_max_pixels=$image_max_pixels \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/min${video_min_tokens}_max${video_max_tokens}_total${video_total_tokens}_maxf${max_frames}/"
    sleep 30
done
