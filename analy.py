import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("output/buy_point_predictions.csv")

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

rows = []
for dataset in sorted(df["dataset"].dropna().unique()):
    sub = df[df["dataset"] == dataset].copy()
    y = sub["label"].astype(int)
    score = sub["pred_score"]

    for th in thresholds:
        pred = (score >= th).astype(int)
        picked = sub[score >= th]

        rows.append({
            "dataset": dataset,
            "threshold": th,
            "picked_count": int(len(picked)),
            "picked_ratio_pct": round(len(picked) / len(sub) * 100, 2) if len(sub) else 0,
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "avg_future_return_pct": round(picked["future_max_return_pct"].mean(), 2) if len(picked) else None,
            "avg_future_drawdown_pct": round(picked["future_max_drawdown_pct"].mean(), 2) if len(picked) else None,
            "hit_rate_pct": round(picked["label"].mean() * 100, 2) if len(picked) else None,
        })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv("output/high_precision_threshold_analysis.csv", index=False, encoding="utf-8-sig")
print("\n已保存: output/high_precision_threshold_analysis.csv")
