# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
import re
import sys
from typing import Union

import pandas as pd
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

idx_map = ["a", "b"]


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
cache_dir = os.path.join(base_cache_dir, "minimal_video_pairs")


def mvp_doc_to_visual(doc):
    video_path = doc["video_path"].lstrip("/")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def get_candidates(doc):
    if type(doc["candidates"]) == str:
        cands = ast.literal_eval(doc["candidates"])
    else:
        cands = doc["candidates"]
    return cands


def mvp_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = ""
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    cands = get_candidates(doc)
    option_prompt = f"A. {cands[0]}\nB. {cands[1]}"
    full_text = "Question: " + question + "\nOptions:\n" + option_prompt + post_prompt
    return full_text


# Process result for yes_no
def mvp_process_results(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    answer_pred = extract_pred(pred)
    cand = get_candidates(doc)
    answer_idx = str(cand.index(str(doc["answer"])))

    # Some hand-crafted matching rules
    if answer_pred:
        rating = 1 if answer_pred == answer_idx else 0

    return {
        "pair_accuracy": {
            "video_id": doc["video_id"],
            "video-llm-prediction": pred,
            "prediction_idx": answer_pred,
            "answer_idx": answer_idx,
            "match_success": match_success,
            "rating": rating,
            "candidates": cand,
        },
        "single_accuracy": {
            "video_id": doc["video_id"],
            "video-llm-prediction": pred,
            "prediction_idx": answer_pred,
            "answer_idx": answer_idx,
            "match_success": match_success,
            "rating": rating,
        },
    }


# Process result for yes_no
def mvp_process_boxed_results(doc, result):
    pred = extract_final_boxed_content(result[0])
    rating = 0
    match_success = True
    answer_pred = extract_pred(pred)
    cand = get_candidates(doc)
    answer_idx = str(cand.index(str(doc["answer"])))

    # Some hand-crafted matching rules
    if answer_pred:
        rating = 1 if answer_pred == answer_idx else 0

    return {
        "pair_accuracy": {
            "video_id": doc["video_id"],
            "video-llm-prediction": pred,
            "prediction_idx": answer_pred,
            "answer_idx": answer_idx,
            "match_success": match_success,
            "rating": rating,
            "candidates": cand,
        },
        "single_accuracy": {
            "video_id": doc["video_id"],
            "video-llm-prediction": pred,
            "prediction_idx": answer_pred,
            "answer_idx": answer_idx,
            "match_success": match_success,
            "rating": rating,
        },
    }


def mvp_doc_to_answer(doc):
    return doc["answer"]


def extract_pred(video_llm_output) -> Union[str, bool]:
    video_llm_output = video_llm_output.lower()
    pattern = r"(Answer|Assistant)?:?\s*([AB])\b"
    matches = re.findall(pattern, video_llm_output, re.IGNORECASE)
    if matches:
        actual_answer = matches[1] if len(matches) > 1 else matches[0]
        answer = actual_answer[1].lower()
        pred_idx = idx_map.index(answer)
        return str(pred_idx)
    elif len(video_llm_output) == 1 and video_llm_output.lower() in idx_map:
        answer = video_llm_output.lower()
        pred_idx = idx_map.index(answer)
        return str(pred_idx)
    else:
        return False


def compute_metrics(results):
    """
    Compute single and paired accuracy metrics
    """
    single_correct_count = 0
    pair_correct_count = 0

    # results is a list of dict
    result_by_vid = {}
    for answer_dict in results:
        if answer_dict["rating"] == 1:
            single_correct_count += 1
        video_id = "_".join(answer_dict["video_id"].split("_")[:-1])
        if video_id not in result_by_vid:
            result_by_vid[video_id] = [answer_dict]
        else:
            result_by_vid[video_id].append(answer_dict)

    for _, answer_dict_pair in result_by_vid.items():
        answer_dict_1, answer_dict_2 = answer_dict_pair

        if answer_dict_1["rating"] == 1 and answer_dict_2["rating"] == 1:
            pair_correct_count += 1

    single_accuracy = single_correct_count / len(results)
    pair_accuracy = pair_correct_count / len(result_by_vid)

    return single_accuracy * 100, pair_accuracy * 100


def mvp_single_accuracy(results, args):
    sa, _ = compute_metrics(results)
    return sa


# Generate leaderboard submission files in pair accuracy call


def generate_leaderboard_submission_df(results, subset="mvp"):
    ldb_rows = []
    for row in results:
        ldb_row = {
            "data_name": subset,
            "row_id": row["video_id"],
            "model_answer": row["candidates"][int(row["prediction_idx"])],
        }
        ldb_rows.append(ldb_row)
    return pd.DataFrame(ldb_rows)


def mvp_pair_accuracy(results, args, task="", subset="mvp"):
    _, pa = compute_metrics(results)
    ldb_file = generate_submission_file(f"{subset}_{task}_valid.jsonl", args)
    ldb = generate_leaderboard_submission_df(results, subset=subset)
    ldb["task"] = task
    ldb.to_json(ldb_file, lines=True, orient="records")
    return pa


def mvp_hoi_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="human_object_interactions", subset="mvp")


def mvp_ip_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="intuitive_physics", subset="mvp")


def mvp_roi_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="robot_object_interactions", subset="mvp")


def mvp_tr_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="temporal_reasoning", subset="mvp")


def mvp_mini_hoi_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="human_object_interactions", subset="mvp_mini")


def mvp_mini_ip_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="intuitive_physics", subset="mvp_mini")


def mvp_mini_roi_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="robot_object_interactions", subset="mvp_mini")


def mvp_mini_tr_pair_accuracy(results, args):
    return mvp_pair_accuracy(results, args, task="temporal_reasoning", subset="mvp_mini")
