# VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

<p align="left">
<a href="https://arxiv.org/abs/2601.05175" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2601.05175-b31b1b.svg?style=flat" /></a>
<a href='https://ivul-kaust.github.io/projects/videoauto-r1/'>
    <img src='https://img.shields.io/badge/Project%20Page-VideoAuto--R1-green'></a>
<a href="https://huggingface.co/collections/IVUL-KAUST/videoauto-r1" alt="models">
    <img src="https://img.shields.io/badge/Models-HuggingFace-yellow.svg" /></a>
<a href="https://huggingface.co/datasets/IVUL-KAUST/VideoAuto-R1-Data" alt="data">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow.svg" /></a>
<a href="https://github.com/IVUL-KAUST/VideoAuto-R1/blob/main/LICENSE.txt" alt="license">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a href="https://img.shields.io/github/stars/IVUL-KAUST/VideoAuto-R1" alt="stars">
    <img src="https://img.shields.io/github/stars/IVUL-KAUST/VideoAuto-R1" /></a>
</p>

## 📖 Overview
<div align="center">

![Method Overview](./assets/method.png)

</div>

We propose VideoAuto-R1, a video understanding framework that adopts a "reason-when-necessary" strategy. During training, our approach follows a Thinking Once, Answering Twice paradigm: the model first generates an initial answer, then performs reasoning, and finally outputs a reviewed answer. Both answers are supervised via verifiable rewards. During inference, the model uses the confidence score of the initial answer to determine whether to proceed with reasoning.


## 🔥 Updates
- **[2025-01-09]**: Try our online demo at [HuggingFace Spaces](https://huggingface.co/spaces/sming256/VideoAuto-R1_Demo)!
- **[2025-01-08]**: We have released the training code and data for VideoAuto-R1!

## Installation

```bash
git clone git@github.com:IVUL-KAUST/VideoAuto-R1.git
cd VideoAuto-R1

conda create -n videoauto-r1 python=3.12
source activate videoauto-r1

pip install -r requirements.txt

conda install "ffmpeg<8"
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

The code is tested with Python 3.12, PyTorch 2.8, CUDA 12.4 on linux, and may also work on other versions.

## Training

Please download the data from [HuggingFace](https://huggingface.co/datasets/IVUL-KAUST/VideoAuto-R1-Data) and put them under the `data/` folder.

For training, please run the following scripts:
```bash
# for Qwen2.5-VL
bash scripts/train/grpo_autothink/train_qwen2.5vl_grpo_auto_text_image_video.sh

# for Qwen3-VL
bash scripts/train/grpo_autothink/train_qwen3vl_grpo_auto_text_image_video.sh
```
Our models are trained on 32 H100 GPUs. You may need to adjust the batch size and accumulation steps according to your hardware settings.

## Evaluation

We use lmms_eval framework to evaluate our models.

For evaluating the baseline Qwen models, please run the following scripts:
```bash
# for Qwen2.5-VL
bash scripts/eval/benchmark_qwen/eval_qwen2_5_vl_16k.sh

# for Qwen3-VL
bash scripts/eval/benchmark_qwen/eval_qwen3_vl_128k.sh
```

For evaluating our VideoAuto-R1 models, please run the following scripts:
```bash
# for Qwen2.5-VL
bash scripts/eval/grpo_autothink/eval_qwen2_5_vl_auto_16k.sh

# for Qwen3-VL
bash scripts/eval/grpo_autothink/eval_qwen3_vl_auto_128k.sh
```
Our models are evaluated on 8 H100 GPUs. You may need to adjust according to your hardware settings.


Expected Results:
| Benchmarks     | Qwen2.5-VL-7B | VideoAuto-R1-7B | Qwen3-VL-8B | VideoAuto-R1-8B |
| -------------- | ------------- | --------------- | ----------- | --------------- |
| VideoMME       | 66.0          | **67.3**        | 72.5        | 71.7            |
| MVBench        | 67.1          | **71.0**        | 69.4        | **72.0**        |
| LongVideoBench | 60.9          | 60.5            | 67.6        | 67.4            |
| MMVU           | 66.2          | **69.7**        | 69.9        | **71.1**        |
| VideoMMMU      | 54.7          | **58.6**        | 61.0        | **65.0**        |
| MVP            | 36.5          | **39.4**        | 40.5        | **43.0**        |
| Charades-STA   | 52.9          | **60.0**        | 44.6        | **63.7**        |
| ActivityNet-QA | 26.9          | **47.6**        | 36.1        | **51.9**        |
| Next-GQA       | 20.2          | **36.7**        | 37.1        | **44.2**        |

Due to the different environment or library versions, the performance may vary slightly from the reported results in the paper (±0.5%).

## Acknowledgement

This project builds upon the following excellent works: [Qwen-VL](https://github.com/QwenLM/Qwen3-VL), [TRL](https://github.com/huggingface/trl), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), etc. We thank all researchers and developers who contributed to these foundational projects.

## Citation

If you use VideoAuto-R1 in your research, please cite:

```bibtex
@article{liu2026videoautor1,
  title={VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice},
  author={Liu, Shuming and Zhuge, Mingchen and Zhao, Changsheng and Chen, Jun and Wu, Lemeng and Liu, Zechun and Zhu, Chenchen and Cai, Zhipeng and Zhou, Chong and Liu, Haozhe and Chang, Ernie and Suri, Saksham and Xu, Hongyu and Qian, Qi and Wen, Wei and Varadarajan, Balakrishnan and Liu, Zhuang and Xu, Hu and Bordes, Florian and Krishnamoorthi, Raghuraman and Ghanem, Bernard and Chandra, Vikas and Xiong, Yunyang},
  journal={arXiv preprint arXiv:2601.05175},
  year={2026}
}
```

This project is licensed under the Apache License 2.0. See LICENSE file for details.

If you have any questions, please contact: shuming.liu@kaust.edu.sa.