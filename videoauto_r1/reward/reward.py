import logging
import re

from reward.mc_grader import equal_answer
from reward.tg_grader import compute_iou_reward
from math_verify import parse, verify

logger = logging.getLogger(__name__)


_PATTERN_ANSWER_THINK_ANSWER = re.compile(
    r"(?!.*<think>.*<think>)(?!.*</think>.*</think>)"  # only one <think>...</think> pair
    r"\s*\\boxed\{(?P<ans1>(?:[^{}]|\{[^{}]*\})*)\}\s*"  # first boxed
    r"<think>(?=[\s\S]*?\S)[\s\S]*?</think>\s*"  # <think>...</think>, should not be empty
    r"\\boxed\{(?P<ans2>(?:[^{}]|\{[^{}]*\})*)\}\s*$",  # second boxed
    re.DOTALL,
)

_PATTERN_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_content(text):
    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    # Return the last match if any matches were found
    return matches[-1] if matches else text


def extract_twice_boxed_content(text, index=0):
    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    if index == 0:
        text = text.split("<think>")[0]
    elif index == 1:
        text = text.split("</think>")[-1]

    matches = _PATTERN_BOXED.findall(text)
    if len(matches) != 1:
        return ""
    else:
        return matches[0]


def accuracy_boxed_reward(completions, solutions, problem_types, **kwargs):
    rewards = []
    for pred, gt, question_type in zip(completions, solutions, problem_types):
        try:
            pred_ans = extract_boxed_content(pred).strip()
            gt_ans = gt.strip() if isinstance(gt, str) else gt
            if question_type == "exact_match":
                reward = 1.0 if equal_answer(gt_ans, pred_ans) else 0.0
            elif question_type == "math":
                gt_math = parse(gt_ans)
                pred_math = parse(pred_ans)
                reward = 1.0 if verify(gt_math, pred_math) else 0.0
            elif question_type == "iou":
                reward = compute_iou_reward(gt_ans, pred_ans)
            elif question_type == "gqa":
                assert isinstance(gt, dict)
                print(pred_ans)
                gt_answer, gt_segment = gt["answer"].strip(), gt["segment"]
                pred_answer = pred_ans.split("<>")[0].strip()
                pred_segment = pred_ans.split("<>")[-1].strip()
                reward_answer = 1.0 if equal_answer(pred_answer, gt_answer) else 0.0
                reward_segment = compute_iou_reward(gt_segment, pred_segment)
                reward = reward_answer + reward_segment
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)
    return rewards


def format_boxed_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the format: <think>...</think> ... \boxed{...}"""
    pattern = r"<think>.*?</think>.*?\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def format_direct_boxed_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the format: <think>...</think> ... \boxed{...}"""
    pattern = r"\s*\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_twice_boxed_reward1(completions, solutions, problem_types, **kwargs):
    rewards = []
    for pred, gt, question_type in zip(completions, solutions, problem_types):
        try:
            pred_ans = extract_twice_boxed_content(pred, index=0).strip()
            gt_ans = gt.strip() if isinstance(gt, str) else gt
            if question_type == "exact_match":
                reward = 1.0 if equal_answer(gt_ans, pred_ans) else 0.0
            elif question_type == "math":
                gt_math = parse(gt_ans)
                pred_math = parse(pred_ans)
                reward = 1.0 if verify(gt_math, pred_math) else 0.0
            elif question_type == "iou":
                reward = compute_iou_reward(gt_ans, pred_ans)
            elif question_type == "gqa":
                assert isinstance(gt, dict)
                gt_answer, gt_segment = gt["answer"].strip(), gt["segment"]
                pred_answer = pred_ans.split("<>")[0].strip()
                pred_segment = pred_ans.split("<>")[-1].strip()
                reward_answer = 1.0 if equal_answer(pred_answer, gt_answer) else 0.0
                reward_segment = compute_iou_reward(gt_segment, pred_segment)
                reward = reward_answer + reward_segment
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)
    return rewards


def accuracy_twice_boxed_reward2(completions, solutions, problem_types, **kwargs):
    rewards = []
    for pred, gt, question_type in zip(completions, solutions, problem_types):
        try:
            pred_ans = extract_twice_boxed_content(pred, index=1).strip()
            gt_ans = gt.strip() if isinstance(gt, str) else gt
            if question_type == "exact_match":
                reward = 1.0 if equal_answer(gt_ans, pred_ans) else 0.0
            elif question_type == "math":
                gt_math = parse(gt_ans)
                pred_math = parse(pred_ans)
                reward = 1.0 if verify(gt_math, pred_math) else 0.0
            elif question_type == "iou":
                reward = compute_iou_reward(gt_ans, pred_ans)
            elif question_type == "gqa":
                assert isinstance(gt, dict)
                gt_answer, gt_segment = gt["answer"].strip(), gt["segment"]
                pred_answer = pred_ans.split("<>")[0].strip()
                pred_segment = pred_ans.split("<>")[-1].strip()
                reward_answer = 1.0 if equal_answer(pred_answer, gt_answer) else 0.0
                reward_segment = compute_iou_reward(gt_segment, pred_segment)
                reward = reward_answer + reward_segment

            if reward > 0.7:
                pred_ans_1 = extract_twice_boxed_content(pred, index=0).strip()
                if pred_ans_1.lstrip().startswith("Let's analyze"):
                    reward += 0.3 / 1.1
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)
    return rewards


def format_twice_boxed_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the format: \\boxed{...} <think>...</think> \boxed{...}"""
    rewards = []
    for content in completions:
        match = _PATTERN_ANSWER_THINK_ANSWER.match(content)
        if not match:
            print(f"Failed format, Content: {content}")
        rewards.append(1.0 if match else 0.0)
    return rewards


reward_funcs_registry = {
    "accuracy_boxed": accuracy_boxed_reward,
    "format_boxed": format_boxed_reward,
    "format_direct_boxed": format_direct_boxed_reward,
    "accuracy_boxed1": accuracy_twice_boxed_reward1,
    "accuracy_boxed2": accuracy_twice_boxed_reward2,
    "format_twice_boxed": format_twice_boxed_reward,
}
