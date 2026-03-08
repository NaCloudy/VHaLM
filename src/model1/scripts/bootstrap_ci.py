#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import sys
from typing import Dict, Any, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.eval import tokenize_text  # noqa: E402
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

ALLOWED_METRICS = ["METEOR", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
SMOOTH_FN = SmoothingFunction().method4
BLEU_WEIGHTS = {
    "BLEU-1": (1.0, 0.0, 0.0, 0.0),
    "BLEU-2": (0.5, 0.5, 0.0, 0.0),
    "BLEU-3": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
    "BLEU-4": (0.25, 0.25, 0.25, 0.25),
}


def build_sample_metrics(preds, refs):
    per_metric = {metric: [] for metric in ALLOWED_METRICS}

    for pred, ref in zip(preds, refs):
        ref_tokens = tokenize_text(ref)
        pred_tokens = tokenize_text(pred)

        if ref_tokens and pred_tokens:
            meteor_val = float(meteor_score([ref_tokens], pred_tokens))
        else:
            meteor_val = 0.0
        per_metric["METEOR"].append(meteor_val)

        for metric, weights in BLEU_WEIGHTS.items():
            if not pred_tokens or not ref_tokens:
                bleu_val = 0.0
            else:
                try:
                    bleu_val = float(sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=SMOOTH_FN))
                except ZeroDivisionError:
                    bleu_val = 0.0
            per_metric[metric].append(bleu_val)

    # Convert lists to numpy arrays for fast indexing
    return {metric: np.array(values, dtype=np.float64) for metric, values in per_metric.items()}


def aggregate_metrics(sample_metrics, indices) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if len(indices) == 0:
        return {metric: 0.0 for metric in ALLOWED_METRICS}

    idx_array = np.array(indices, dtype=np.int64)
    for metric, values in sample_metrics.items():
        subset = values[idx_array]
        results[metric] = float(subset.mean()) if subset.size else 0.0
    return results


def load_strings(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 支持两种结构：
    # 1) {"items": [{"pred": "...", "ref": "..."}, ...]}
    # 2) ["...", "..."]  仅字符串列表
    if isinstance(data, dict) and "items" in data:
        # 若是联合文件，调用方分别传入 pred/ref 路径即可；此分支保留兼容
        raise ValueError("Expect plain list JSON for preds/refs; got dict with 'items'.")
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
    return [str(x) for x in data]


def bootstrap_ci(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    lo = float(np.quantile(series, alpha / 2))
    hi = float(np.quantile(series, 1 - alpha / 2))
    return {"mean": float(np.mean(series)), "ci95": [lo, hi]}


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI for metrics (BLEU-1..4, METEOR)")
    parser.add_argument("--pred_path", required=True, help="模型预测 JSON（字符串列表）")
    parser.add_argument("--ref_path", required=True, help="参考文本 JSON（字符串列表）")
    parser.add_argument("--B", type=int, default=200, help="bootstrap轮数，默认200")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    preds = load_strings(args.pred_path)
    refs = load_strings(args.ref_path)

    if len(preds) != len(refs):
        raise ValueError(f"Preds/Refs length mismatch: {len(preds)} vs {len(refs)}")

    N = len(preds)

    sample_metrics = build_sample_metrics(preds, refs)
    full_indices = list(range(N))
    full_metrics = aggregate_metrics(sample_metrics, full_indices)

    metrics_names = list(full_metrics.keys())
    metrics_series = {k: [] for k in metrics_names}

    rng = np.random.default_rng(args.seed)

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(range(args.B), desc="Bootstrap", ncols=88, file=sys.stderr)
        use_tqdm = True
    except Exception:
        iterator = range(args.B)
        use_tqdm = False
        step = max(1, args.B // 10)

    for b_idx in iterator:
        idxs = rng.integers(0, N, size=N).tolist()
        m = aggregate_metrics(sample_metrics, idxs)
        for k in metrics_names:
            metrics_series[k].append(float(m[k]))
        if not use_tqdm and ((b_idx + 1) % step == 0 or (b_idx + 1) == args.B):
            print(f"Bootstrap progress: {b_idx + 1}/{args.B}", file=sys.stderr, flush=True)

    if use_tqdm:
        iterator.close()

    ci = {k: bootstrap_ci(v) for k, v in metrics_series.items()}

    out = {
        "full": full_metrics,
        "bootstrap": ci,
        "N": N,
        "B": args.B,
        "seed": args.seed,
        "pred_path": args.pred_path,
        "ref_path": args.ref_path,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
