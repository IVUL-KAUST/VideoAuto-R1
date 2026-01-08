import os
import re
import sys
from pathlib import Path

import yaml
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")

base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mmvu_val_mc.yaml", "r") as f:
    raw_data_val = f.readlines()
    safe_data_val = []
    for i, line in enumerate(raw_data_val):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_val.append(line)
cache_name_val = yaml.safe_load("".join(safe_data_val))["dataset_kwargs"]["cache_dir"]
cache_dir_val = os.path.join(base_cache_dir, cache_name_val)


def mmvu_doc_to_visual_val(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir_val, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


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


def mmvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = "\n".join([f"{k}. {v}" for k, v in doc["choices"].items()])
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = f"Question: {question}\nOptions:\n{options}\n{post_prompt}"
    return full_prompt


def mmvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    data_dict = {
        "question_id": doc["id"],
        "category": doc["video_path"].split("/")[-2],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
    }
    return {f"accuracy": data_dict}


def mmvu_process_boxed_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = extract_final_boxed_content(results[0])
    pred_ans = extract_characters_regex(pred)

    data_dict = {
        "question_id": doc["id"],
        "category": doc["video_path"].split("/")[-2],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
    }
    return {f"accuracy": data_dict}


def mmvu_aggregate_results_val(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    TASK_MAP = {
        "Biology": "Science",
        "Chemistry": "Science",
        "Modern_Physics": "Science",
        "Astronomy": "Science",
        "Geography": "Science",
        "Materials_Science": "Science",
        "Neurobiology": "Science",
        "Electromagnetism": "Science",
        "Thermodynamics": "Science",
        "Mechanics": "Science",
        "Civil_Engineering": "Engineering",
        "Electrical_Engineering": "Engineering",
        "Mechanical_Engineering": "Engineering",
        "Biomedical_Engineering": "Engineering",
        "Electronics_and_Communication": "Engineering",
        "Computer_Science": "Engineering",
        "Clinical_Medicine": "Healthcare",
        "Basic_Medicine": "Healthcare",
        "Preventive_Medicine": "Healthcare",
        "Pharmacy": "Healthcare",
        "Dentistry": "Healthcare",
        "Art": "Humanities_and_Social_Science",
        "Literature": "Humanities_and_Social_Science",
        "History": "Humanities_and_Social_Science",
        "Law": "Humanities_and_Social_Science",
        "Economics": "Humanities_and_Social_Science",
        "Management": "Humanities_and_Social_Science",
    }

    TASK_TYPES = list(set(TASK_MAP.values()))

    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["category"]
        if category in TASK_MAP:
            category = TASK_MAP[category]
            category2score[category]["answered"] += 1
            category2score[category]["correct"] += result["pred_answer"] == result["answer"]
    category_scores = {}

    for category in TASK_TYPES:
        total_correct = category2score[category]["correct"]
        total_answered = category2score[category]["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        category_scores[category] = accuracy

    total_correct = sum(category2score[category]["correct"] for category in TASK_TYPES)
    total_answered = sum(category2score[category]["answered"] for category in TASK_TYPES)
    accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info("=" * 50)
    eval_logger.info(f"Average Accuracy: {accuracy:.2f}%")
    eval_logger.info("Categorical accuracy: ")
    for key, value in category_scores.items():
        eval_logger.info(f"{key} accuracy: {value:.2f}%")
    eval_logger.info("=" * 50)
    return accuracy
