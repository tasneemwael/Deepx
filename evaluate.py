"""
evaluate.py — Compute official competition metrics on the validation set.

Usage:
    # After running predict_val.py:
    python evaluate.py val_predictions.json DeepX_validation.xlsx

    # Or to evaluate submission.json directly (it will auto-detect the right split):
    python evaluate.py submission.json DeepX_validation.xlsx
"""

import json, ast, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict

ASPECTS    = ["food", "service", "price", "cleanliness",
              "delivery", "ambiance", "app_experience", "general", "none"]
SENTIMENTS = ["positive", "negative", "neutral"]


def predictions_to_vectors(aspects: list, aspect_sentiments: dict) -> set:
    return {(a, aspect_sentiments[a]) for a in aspects}


def compute_pair_f1(predictions: list, ground_truth: pd.DataFrame) -> dict:
    gt_map = {}
    for _, row in ground_truth.iterrows():
        rid   = int(row["review_id"])
        asps  = ast.literal_eval(row["aspects"])  if isinstance(row["aspects"],  str) else row["aspects"]
        sents = ast.literal_eval(row["aspect_sentiments"]) if isinstance(row["aspect_sentiments"], str) else row["aspect_sentiments"]
        gt_map[rid] = predictions_to_vectors(asps, sents)

    pred_map = {}
    for p in predictions:
        rid = int(p["review_id"])
        pred_map[rid] = predictions_to_vectors(p["aspects"], p["aspect_sentiments"])

    total_tp = total_fp = total_fn = 0
    per_aspect_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for rid, gt_pairs in gt_map.items():
        pred_pairs = pred_map.get(rid, set())
        tp = gt_pairs & pred_pairs
        fp = pred_pairs - gt_pairs
        fn = gt_pairs - pred_pairs

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for pair in tp:
            per_aspect_stats[pair[0]]["tp"] += 1
        for pair in fp:
            per_aspect_stats[pair[0]]["fp"] += 1
        for pair in fn:
            per_aspect_stats[pair[0]]["fn"] += 1

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    micro_f1  = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n{'='*50}")
    print(f"  ABSA Evaluation Results")
    print(f"{'='*50}")
    print(f"  Predictions matched : {len(pred_map)}")
    print(f"  Ground truth rows   : {len(gt_map)}")
    print(f"  Micro Precision : {precision:.4f}")
    print(f"  Micro Recall    : {recall:.4f}")
    print(f"  Micro F1        : {micro_f1:.4f}  ← competition metric")
    print(f"\n  Per-aspect breakdown:")
    print(f"  {'Aspect':<18} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*40}")

    for asp in ASPECTS:
        stats = per_aspect_stats[asp]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2*p*r / (p + r + 1e-9)
        print(f"  {asp:<18} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")

    return {"precision": precision, "recall": recall, "micro_f1": micro_f1}


def compute_aspect_only_f1(predictions: list, ground_truth: pd.DataFrame) -> float:
    gt_map, pred_map = {}, {}
    for _, row in ground_truth.iterrows():
        rid  = int(row["review_id"])
        asps = row["aspects"]
        gt_map[rid] = set(ast.literal_eval(asps) if isinstance(asps, str) else asps)
    for p in predictions:
        pred_map[int(p["review_id"])] = set(p["aspects"])

    tp = fp = fn = 0
    for rid, gt_asps in gt_map.items():
        pred_asps = pred_map.get(rid, set())
        tp += len(gt_asps & pred_asps)
        fp += len(pred_asps - gt_asps)
        fn += len(gt_asps - pred_asps)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\n  Aspect-only F1  : {f1:.4f}")
    return f1


if __name__ == "__main__":
    pred_file = sys.argv[1] if len(sys.argv) > 1 else "submission.json"
    val_file  = sys.argv[2] if len(sys.argv) > 2 else "DeepX_validation.xlsx"

    print(f"📂 Loading predictions : {pred_file}")
    print(f"📂 Loading ground truth: {val_file}")

    with open(pred_file, encoding="utf-8") as f:
        predictions = json.load(f)

    df_val = pd.read_excel(val_file)

    # ── Normalize all IDs to int for comparison ──────────────────
    val_ids  = set(int(x) for x in df_val["review_id"].tolist())
    pred_ids = set(int(p["review_id"]) for p in predictions)

    overlap = val_ids & pred_ids
    print(f"\n📊 ID Summary:")
    print(f"   Predictions file has : {len(predictions)} entries  (IDs {min(pred_ids)}–{max(pred_ids)})")
    print(f"   Validation file has  : {len(df_val)} rows      (IDs {min(val_ids)}–{max(val_ids)})")
    print(f"   Overlapping IDs      : {len(overlap)}")

    if not overlap:
        print("\n❌  ERROR: No overlapping review_ids found between prediction file and validation file.")
        print("\n   This usually means your submission.json was generated for the UNLABELED set,")
        print("   not the validation set.")
        print("\n   ✅ Fix: run  →  python predict_val.py")
        print("   Then:          python evaluate.py val_predictions.json DeepX_validation.xlsx")
        sys.exit(1)

    if len(overlap) < len(val_ids):
        missing = len(val_ids) - len(overlap)
        print(f"\n⚠   Warning: {missing} validation IDs are missing from predictions.")
        print("   Missing reviews will be counted as false negatives.\n")

    # Filter predictions to only validation IDs
    pred_val = [p for p in predictions if int(p["review_id"]) in val_ids]

    compute_pair_f1(pred_val, df_val)
    compute_aspect_only_f1(pred_val, df_val)