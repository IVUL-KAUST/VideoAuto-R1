import re

from loguru import logger as eval_logger


def extract_after_think(text):
    """
    Extracts the content after the last </think> tag in the given text.

    Args:
        text (str): The text containing </think> tags

    Returns:
        str: The content after the last </think> tag, or the original text if no </think> tag is found
    """
    # Find the last occurrence of </think>
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        think_content = text[:last_think_end].strip()
        answer_content = text[last_think_end + len("</think>") :].strip()

        think_matches = re.findall(
            r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", think_content
        )
        answer_matches = re.findall(
            r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", answer_content
        )

        if think_matches and not answer_matches:
            eval_logger.info(
                f"\nBoxed not found in answer but found in think, use think content: {text}"
            )
            return think_matches[-1]

        return answer_content
    else:
        eval_logger.info(f"\nCould not find </think>, use full text: {text}")
        return text


def extract_final_boxed_content(text, strict=False):
    """
    Extracts the content of the final \\boxed{} command in the given text.

    Args:
        text (str): The text containing \\boxed{} commands

    Returns:
        str or None: The content of the final \\boxed{} command, or None if no \\boxed{} command is found
    """
    # extract the content after the last </think> tag
    # text = extract_after_think(text)

    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    # Return the last match if any matches were found
    if boxed_matches:
        return boxed_matches[-1]
    else:
        eval_logger.info(f"no boxed found in {text}")
        if strict:
            return ""
        else:
            return text
