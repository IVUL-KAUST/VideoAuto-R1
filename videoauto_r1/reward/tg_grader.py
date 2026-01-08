import re


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
    ps, pe = float(pred_timestamps[0]), float(pred_timestamps[1])
    gs, ge = float(gt_timestamps[0]), float(gt_timestamps[1])
    if ge < gs:
        gs, ge = ge, gs

    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), force reward 0.0
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return 0.0

    # Compute IoU for single intervals
    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter

    return 0.0 if union <= eps else inter / union


def compute_iou_reward(gt_timestamps, pred_text):
    r"""
    Compute IoU between a single GT interval and a single predicted interval.

    Rules:
      - The predicted interval must be parsed by `parse_timestamps_from_string`, i.e.,
        it must be `[start, end]` in seconds.
      - If parsing fails, return 0.0.
      - GT must also contain exactly one interval [gt_start, gt_end].

    Returns:
      IoU in [0.0, 1.0].
    """
    # Parse predicted interval (must be exactly one)
    pred_timestamps = parse_timestamps_from_string(pred_text)
    if pred_timestamps is None:
        print(f"Failed tvg parsing, Content: {pred_text}")
        return 0.0

    if isinstance(gt_timestamps[0], (list, tuple)):
        return max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps])
    else:
        return compute_iou(gt_timestamps, pred_timestamps)


def compute_giou(gt_timestamps, pred_timestamps, eps: float = 1e-9):
    ps, pe = float(pred_timestamps[0]), float(pred_timestamps[1])
    gs, ge = float(gt_timestamps[0]), float(gt_timestamps[1])
    if ge < gs:
        gs, ge = ge, gs

    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), force reward 0.0
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return -1.0

    # Standard IoU pieces
    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter

    # IoU (guard against numerical issues)
    iou = 0.0 if union <= eps else (inter / union)

    # Smallest enclosing interval C
    c_len = max(ge, pe) - min(gs, ps)

    # GIoU = IoU - (|C \ U|)/|C| = IoU - (c_len - union)/c_len
    giou = iou - (c_len - union) / c_len

    # Clamp to valid range [-1, 1] to avoid tiny numerical spillover
    giou = max(-1.0, min(1.0, giou))
    return giou


def compute_giou_reward(gt_timestamps, pred_text, eps: float = 1e-9):
    r"""
    Compute Generalized IoU (GIoU) between a single GT interval and a single predicted interval (1D).

    Policy (mirrors compute_simple_iou_reward):
      - If parsing the predicted interval fails -> return 0.0
      - If the predicted interval has zero length -> return 0.0
      - Otherwise compute 1D GIoU:
            GIoU = IoU - (|C \ U|) / |C|
        where:
            U = union of GT and Pred (length)
            C = length of the smallest enclosing interval covering both GT and Pred
        Range: [-1.0, 1.0], with 1.0 at perfect match, potentially negative when disjoint.

    Returns:
      GIoU in [-1.0, 1.0] (use the commented line at bottom to remap to [0,1] if desired).
    """
    # Parse predicted interval
    pred_timestamps = parse_timestamps_from_string(pred_text)
    if pred_timestamps is None:
        print(f"Failed tvg parsing, Content: {pred_text}")
        return -1.0

    if isinstance(gt_timestamps[0], (list, tuple)):
        return max([compute_giou(gt, pred_timestamps) for gt in gt_timestamps])
    else:
        return compute_giou(gt_timestamps, pred_timestamps)


def compute_diou(gt_timestamps, pred_timestamps, eps: float = 1e-9):
    """
    Compute Distance-IoU (DIoU) for 1D intervals.

    Args:
        gt_timestamps: [gs, ge] for the ground-truth interval.
        pred_timestamps: [ps, pe] for the predicted interval.
        eps: small constant to guard against numerical issues.

    Returns:
        DIoU in [-1.0, 1.0]. (Perfect match -> 1.0; can be negative when disjoint.)
        If the predicted interval is degenerate (length ~ 0), returns 0.0.
    """
    # Normalize endpoints (ensure start <= end)
    ps, pe = float(pred_timestamps[0]), float(pred_timestamps[1])
    gs, ge = float(gt_timestamps[0]), float(gt_timestamps[1])
    if ge < gs:
        gs, ge = ge, gs
    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), return -1.0
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return -1.0

    # IoU components
    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter
    iou = 0.0 if union <= eps else (inter / union)

    # Smallest enclosing interval C and center distance
    c_left, c_right = min(gs, ps), max(ge, pe)
    c_len = c_right - c_left

    c_g = 0.5 * (gs + ge)
    c_p = 0.5 * (ps + pe)
    d = abs(c_g - c_p)

    # DIoU = IoU - (d^2 / |C|^2)
    diou = iou - (d * d) / (c_len * c_len)

    # Clamp to [-1, 1] against tiny numerical spillover
    diou = max(-1.0, min(1.0, diou))
    return diou


def compute_diou_reward(gt_timestamps, pred_text, eps: float = 1e-9):
    """
    Compute DIoU (1D) between a GT interval (or list of GT intervals) and a single predicted interval
    parsed from free text via `parse_timestamps_from_string`.

    Policy:
      - If parsing fails -> return 0.0 (and print a debug line).
      - If GT is a list of intervals -> take the max DIoU over all GT intervals.
      - If predicted interval degenerates -> underlying compute_diou returns 0.0.

    Returns:
      DIoU in [-1.0, 1.0].
    """
    pred_timestamps = parse_timestamps_from_string(pred_text)
    if pred_timestamps is None:
        print(f"Failed tvg parsing, Content: {pred_text}")
        return -1.0

    if isinstance(gt_timestamps[0], (list, tuple)):
        return max(compute_diou(gt, pred_timestamps, eps=eps) for gt in gt_timestamps)
    else:
        return compute_diou(gt_timestamps, pred_timestamps, eps=eps)
