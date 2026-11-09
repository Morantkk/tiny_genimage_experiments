import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path("/data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments/outputs")
EXPERIMENTS = [
    "resnet18_baseline",
    "resnet18_traditional_first20_concat",
    "resnet18_traditional_color_concat",
    "resnet18_traditional_texture_concat",
    "resnet18_traditional_frequency_concat",
    "resnet18_traditional_all_concat",
    "resnet18_traditional_all_gated",
    "resnet18_traditional_all_film",
    "resnet18_traditional_all_hand_only",
]
METRICS = [
    "multiclass_accuracy",
    "multiclass_macro_f1",
    "binary_accuracy",
    "binary_f1_fake",
    "binary_recall_fake",
    "binary_auc_fake",
]


def load_best_metrics(exp):
    path = OUTPUT_ROOT / exp / "best_eval" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = []
    for exp in EXPERIMENTS:
        metrics = load_best_metrics(exp)
        if metrics is None:
            print(f"skip missing {exp}", flush=True)
            continue
        row = {"experiment": exp}
        for key in METRICS:
            row[key] = metrics.get(key)
        rows.append(row)

    out_dir = OUTPUT_ROOT / "traditional_feature_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "traditional_fusion_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "traditional_fusion_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if not rows:
        print("no result rows found", flush=True)
        return

    labels = [r["experiment"].replace("resnet18_traditional_", "").replace("resnet18_", "") for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, [r["multiclass_macro_f1"] for r in rows], width, label="Macro-F1")
    ax.bar(x + width / 2, [r["binary_f1_fake"] for r in rows], width, label="Fake-F1")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Traditional Feature Fusion Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "traditional_fusion_metric_bars.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels, [r["multiclass_macro_f1"] for r in rows], marker="o", label="Macro-F1")
    ax.plot(labels, [r["binary_recall_fake"] for r in rows], marker="o", label="Fake Recall")
    ax.plot(labels, [r["binary_auc_fake"] for r in rows], marker="o", label="AUC")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Which handcrafted features are learnable/useful?")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "traditional_fusion_metric_lines.png", dpi=220)
    plt.close(fig)

    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
