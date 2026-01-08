import datetime
import json
import os
import re
import sys
from pathlib import Path

import yaml
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
from loguru import logger as eval_logger

# with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "activitynet_tvg.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# Pass in video path here
# Can only work correctly with video llm
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "val2_videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["caption"]

    return f"{pre_prompt}{question}{post_prompt}"


def temporal_grounding_doc_to_answer(doc):
    return doc["timestamp"]


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    # Ensure result is a regular Python float, not float16
    return float(max(min1 - max0, 0) / (max1 - min0))

    # # hacked!
    # return float(max(min1 - max0, 0) / (B[1] - B[0]))


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


def temporal_grounding_process_results_generation(doc, result):
    """
    Parse the predicted text to get (start_timestamp, end_timestamp) in seconds.
    Supports:
      - [mm:ss(.ms), mm:ss(.ms)]
      - Natural language with HH:MM(:SS)(.ms) or explicit seconds

    Notes:
      - Expects a global logger named `eval_logger` available in the caller's scope.
      - Returns (0.0, 0.0) if parsing fails; emits warnings accordingly.
    """
    pred = result[0]
    try:
        start_timestamp, end_timestamp = parse_timestamps_from_string(pred)
    except Exception as e:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}, error: {e}")
        start_timestamp, end_timestamp = 0, 0

    if start_timestamp == 0 and end_timestamp == 0:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}")

    # Convert float16 to regular Python floats to make them JSON serializable
    gt_timestamp = doc["timestamp"]
    if hasattr(gt_timestamp, "tolist"):
        # Handle numpy arrays or tensors
        gt_timestamp = gt_timestamp.tolist()
    elif isinstance(gt_timestamp, (list, tuple)):
        # Handle lists/tuples that might contain float16 values
        gt_timestamp = [float(x) for x in gt_timestamp]
    else:
        # Handle single values
        gt_timestamp = float(gt_timestamp)

    result = {
        "query": f'{doc["video"]}>>>{doc["caption"]}>>>{gt_timestamp}',
        "gt": gt_timestamp,
        "pred": [start_timestamp, end_timestamp],
        "iou": iou(gt_timestamp, [start_timestamp, end_timestamp]),
    }

    return {
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_process_boxed_results_generation(doc, result):
    """
    Parse the predicted text to get (start_timestamp, end_timestamp) in seconds.
    Supports:
      - \boxed{[mm:ss(.ms), mm:ss(.ms)]}
      - [mm:ss(.ms), mm:ss(.ms)]
      - Natural language with HH:MM(:SS)(.ms) or explicit seconds

    Notes:
      - Expects a global logger named `eval_logger` available in the caller's scope.
      - Returns (0.0, 0.0) if parsing fails; emits warnings accordingly.
    """
    pred = extract_final_boxed_content(result[0])
    try:
        start_timestamp, end_timestamp = parse_timestamps_from_string(pred)
    except Exception as e:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}, error: {e}")
        start_timestamp, end_timestamp = 0, 0

    if start_timestamp == 0 and end_timestamp == 0:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}")

    # Convert float16 to regular Python floats to make them JSON serializable
    gt_timestamp = doc["timestamp"]
    if hasattr(gt_timestamp, "tolist"):
        # Handle numpy arrays or tensors
        gt_timestamp = gt_timestamp.tolist()
    elif isinstance(gt_timestamp, (list, tuple)):
        # Handle lists/tuples that might contain float16 values
        gt_timestamp = [float(x) for x in gt_timestamp]
    else:
        # Handle single values
        gt_timestamp = float(gt_timestamp)

    result = {
        "query": f'{doc["video"]}>>>{doc["caption"]}>>>{gt_timestamp}',
        "gt": gt_timestamp,
        "pred": [start_timestamp, end_timestamp],
        "iou": iou(gt_timestamp, [start_timestamp, end_timestamp]),
    }

    return {
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_activitynet_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result["iou"])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))


def temporal_grounding_aggregate_activitynet_iou_03(results, args):
    return temporal_grounding_aggregate_activitynet_iou_threshold(results, args, 0.3)


def temporal_grounding_aggregate_activitynet_iou_05(results, args):
    return temporal_grounding_aggregate_activitynet_iou_threshold(results, args, 0.5)


def temporal_grounding_aggregate_activitynet_iou_07(results, args):
    return temporal_grounding_aggregate_activitynet_iou_threshold(results, args, 0.7)


def temporal_grounding_aggregate_activitynet_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result["iou"])

    return float(sum(ious) * 100 / len(ious))


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
