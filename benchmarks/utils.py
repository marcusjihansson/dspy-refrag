import re
import numpy as np


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    if not model_name:
        return "unknown_model"
    # Replace problematic characters with underscores
    sanitized = re.sub(r"[^\w\-.]", "_", model_name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")


def _safe_number(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _dist(a: str | None, b: str | None):
    # Basic similarity metrics: exact match, Jaccard, length-normalized overlap
    if not a or not b:
        return {"exact": False, "jaccard": 0.0, "len_ratio": 0.0}
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens) or 1
    j = inter / union
    len_ratio = min(len(a), len(b)) / (max(len(a), len(b)) or 1)
    exact = a.strip().lower() == b.strip().lower()
    return {"exact": exact, "jaccard": j, "len_ratio": len_ratio}