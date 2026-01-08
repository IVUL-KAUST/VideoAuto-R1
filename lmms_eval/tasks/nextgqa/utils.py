import os
import re
import sys
from pathlib import Path

import yaml
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "nextgqa_boxed.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# Pass in video path here
# Can only work correctly with video llm
def gqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "nextgqa_videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    options = "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(doc["candidates"])])

    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"Question: {question}\nOptions:\n{options}\n{post_prompt}"


def gqa_doc_to_answer(doc):
    return {
        "answer": chr(ord("A") + doc["candidates"].index(doc["answer"])),
        "segment": doc["solution"],
    }


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]


def parse_timestamps_from_string(pred_text):
    """
    Parse a single interval from free text.

    Supports the following interval forms (numbers only; no time like 1:23):
      - [t1, t2], [t1 - t2], [t1 — t2], [t1 to t2], [t1 and t2]
      - (t1, t2) and similar with the same separators
      - t1, t2 (without brackets), with the same separators

    Returns:
      - [start, end] as floats if a match is found (start <= end; swapped if needed)
      - None if no valid interval is found
    """
    text = (pred_text or "").strip()

    # 1) Token definitions
    # Non-negative integer or decimal number (e.g., 12, 12.5)
    NUM = r"(?:\d+(?:\.\d+)?)"

    # Separators: comma/hyphen/en dash/em dash/"to"/"and" (case-insensitive)
    SEP = r"(?:,|-|–|—|\bto\b|\band\b)"

    # 2) Unified interval pattern:
    #    Match one of:
    #      [start SEP end]  |  (start SEP end)  |  start SEP end
    #    with boundary guards to avoid sticking to neighboring digits or dots.
    INTERVAL_RE = re.compile(
        rf"""
        (?<![\d.])                                  # Do not start in the middle of a number/decimal
        (?:
            \[\s*(?P<sb>{NUM})\s*{SEP}\s*(?P<eb>{NUM})\s*\]   # [a, b]
          | \(\s*(?P<sp>{NUM})\s*{SEP}\s*(?P<ep>{NUM})\s*\)   # (a, b)
          | (?P<s>{NUM})\s*{SEP}\s*(?P<e>{NUM})               # a, b
        )
        (?![\d.])                                   # Do not end in the middle of a number/decimal
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # 3) Search for the first interval occurrence
    m = INTERVAL_RE.search(text)
    if not m:
        return None

    # 4) Extract the numeric strings from whichever branch matched
    start_str = m.group("sb") or m.group("sp") or m.group("s")
    end_str = m.group("eb") or m.group("ep") or m.group("e")

    # 5) Convert to floats before any comparison (avoid lexicographic string comparison)
    start = float(start_str)
    end = float(end_str)

    # 6) Ensure start <= end by swapping if necessary
    if start > end:
        start, end = end, start

    # 7) Return the interval as floats
    return [start, end]


def compute_iou(gt_timestamps, pred_timestamps, eps: float = 1e-9):
    # Compute IoU for single intervals
    gs, ge = gt_timestamps[0], gt_timestamps[1]
    ps, pe = pred_timestamps[0], pred_timestamps[1]

    if ge < gs:
        gs, ge = ge, gs

    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), return 0
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return 0.0

    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter
    return 0.0 if union <= eps else inter / union


def gqa_process_results_generation(doc, result):
    pred = result[0]
    pred_answer = pred.split("<>")[0].strip()
    pred_segment = pred.split("<>")[-1].strip()
    pred_timestamps = parse_timestamps_from_string(pred_segment)
    if pred_timestamps is None:
        pred_timestamps = [0.0, 0.0]

    pred_answer = extract_characters_regex(pred_answer)
    gt_answer = chr(ord("A") + doc["candidates"].index(doc["answer"]))

    gt_timestamps = doc["solution"]
    gt_timestamps = [[float(gt[0]), float(gt[1])] for gt in gt_timestamps]

    result = {
        "query": f'{doc["video"]}>>>{doc["candidates"]}',
        "gt_answer": gt_answer,
        "gt_segment": gt_timestamps,
        "pred": pred,
        "score": float(gt_answer == pred_answer),
        "iou": max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps]),
    }
    return {
        "accuracy": result,
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def gqa_process_boxed_results_generation(doc, result):
    pred = extract_final_boxed_content(result[0])
    pred_answer = pred.split("<>")[0].strip()
    pred_segment = pred.split("<>")[-1].strip()
    pred_timestamps = parse_timestamps_from_string(pred_segment)
    if pred_timestamps is None:
        pred_timestamps = [0.0, 0.0]

    pred_answer = extract_characters_regex(pred_answer)
    gt_answer = chr(ord("A") + doc["candidates"].index(doc["answer"]))

    gt_timestamps = doc["solution"]
    gt_timestamps = [[float(gt[0]), float(gt[1])] for gt in gt_timestamps]

    result = {
        "query": f'{doc["video"]}>>>{doc["candidates"]}',
        "gt_answer": gt_answer,
        "gt_segment": gt_timestamps,
        "pred": pred,
        "score": float(gt_answer == pred_answer),
        "iou": max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps]),
    }

    return {
        "accuracy": result,
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result["iou"])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))


def temporal_grounding_aggregate_nextgqa_iou_03(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.3)


def temporal_grounding_aggregate_nextgqa_iou_05(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.5)


def temporal_grounding_aggregate_nextgqa_iou_07(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.7)


def temporal_grounding_aggregate_nextgqa_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result["iou"])

    return float(sum(ious) * 100 / len(ious))


def multi_choice_aggregate_nextgqa_accuracy(results, args):
    accuracy = []
    for result in results:
        accuracy.append(result["score"])

    return float(sum(accuracy) * 100 / len(accuracy))
