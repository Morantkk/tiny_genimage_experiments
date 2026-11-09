import csv
import json
from pathlib import Path


MODELS = ["resnet18_baseline", "resnet18_frequency", "patch_transformer"]
KEYS = [
    "multiclass_accuracy",
    "multiclass_macro_precision",
    "multiclass_macro_recall",
    "multiclass_macro_f1",
    "multiclass_weighted_f1",
    "binary_accuracy",
    "binary_precision_fake",
    "binary_recall_fake",
    "binary_f1_fake",
    "real_recall",
    "binary_auc_fake",
    "multiclass_ovr_auc",
]


def main():
    output_root = Path("/data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments/outputs")
    rows = []
    for model_name in MODELS:
        metrics_path = output_root / model_name / "best_eval" / "metrics.json"
        if not metrics_path.exists():
            rows.append({"model": model_name, "status": "missing_metrics"})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"model": model_name, "status": "ok"}
        for key in KEYS:
            row[key] = metrics.get(key)
        rows.append(row)

    comparison_csv = output_root / "model_comparison.csv"
    comparison_json = output_root / "model_comparison.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "status", *KEYS])
        writer.writeheader()
        writer.writerows(rows)
    comparison_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        best_multi = max(ok_rows, key=lambda row: row.get("multiclass_macro_f1") or -1)
        best_binary = max(ok_rows, key=lambda row: row.get("binary_f1_fake") or -1)
        analysis = {
            "best_multiclass_macro_f1": best_multi,
            "best_binary_f1_fake": best_binary,
            "notes": [
                "multiclass metrics evaluate real plus generator attribution.",
                "binary metrics map Real to real and all generators to fake.",
                "confusion matrices and classification reports are saved under each model's best_eval directory.",
            ],
        }
        (output_root / "comparison_analysis.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8"
        )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
