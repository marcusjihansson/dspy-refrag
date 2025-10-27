import time
import numpy as np
from utils import _safe_number, _dist


def benchmark_model(model, queries, wait_time=0.0):
    """Measure detailed metrics for a list of queries."""
    results = []
    for q in queries:
        start = time.time()
        output = model(q)
        elapsed = time.time() - start

        if wait_time > 0:
            time.sleep(wait_time)

        # Support both dict outputs and objects with attributes
        if isinstance(output, dict):
            answer = output.get("answer")
            token_usage = output.get("token_usage", 0)
            meta = output.get("meta", {})
            # Handle OpenRouter/OpenAI response format
            if isinstance(token_usage, dict):
                tokens = token_usage.get("total_tokens", 0)
            elif isinstance(token_usage, (int, float)):
                tokens = token_usage
            else:
                tokens = 0
        else:
            answer = getattr(output, "answer", None)
            token_usage = getattr(output, "token_usage", 0)
            meta = getattr(output, "meta", {}) if hasattr(output, "meta") else {}
            # Handle OpenRouter/OpenAI response format
            if isinstance(token_usage, dict):
                tokens = token_usage.get("total_tokens", 0)
            elif isinstance(token_usage, (int, float)):
                tokens = token_usage
            else:
                tokens = 0

        results.append(
            {
                "query": q,
                "answer": answer,
                "latency": elapsed,
                "tokens": tokens,
                "meta": meta,
            }
        )

    def _agg(vals):
        arr = np.array([_safe_number(v) for v in vals], dtype=float)
        return {
            "avg": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    agg = {
        "latency": _agg([r["latency"] for r in results]),
        "tokens": _agg([r["tokens"] for r in results]),
        "context_chars": _agg(
            [r.get("meta", {}).get("context_chars", 0) for r in results]
        ),
        "prompt_chars": _agg(
            [r.get("meta", {}).get("prompt_chars", 0) or 0 for r in results]
        ),
        "retrieved": _agg([r.get("meta", {}).get("retrieved", 0) for r in results]),
        "selected": _agg([r.get("meta", {}).get("selected", 0) for r in results]),
    }

    return agg, results


def evaluate_accuracy(results_ref, results_base):
    """Compute multiple similarity metrics across paired results."""
    pairs = []
    for r1, r2 in zip(results_ref, results_base):
        d = _dist(r1.get("answer"), r2.get("answer"))
        pairs.append(d)
    if not pairs:
        return {"exact": 0.0, "jaccard": 0.0, "len_ratio": 0.0}
    exact = sum(1 for p in pairs if p["exact"]) / len(pairs)
    jacc = float(np.mean([p["jaccard"] for p in pairs]))
    lr = float(np.mean([p["len_ratio"] for p in pairs]))
    return {"exact": exact, "jaccard": jacc, "len_ratio": lr}