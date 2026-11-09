import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path("/data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments/outputs")
FIG_DIR = OUTPUT_ROOT / "figures"
MODELS = {
    "resnet18_baseline": "ResNet18",
    "resnet18_frequency": "ResNet18+Freq",
    "patch_transformer": "Patch-Transformer",
}
CLASS_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_metric_bars():
    metrics = {
        name: load_json(OUTPUT_ROOT / key / "best_eval" / "metrics.json")
        for key, name in MODELS.items()
    }
    metric_keys = [
        ("multiclass_accuracy", "Multi Acc"),
        ("multiclass_macro_f1", "Macro F1"),
        ("binary_accuracy", "Binary Acc"),
        ("binary_f1_fake", "Fake F1"),
        ("binary_recall_fake", "Fake Recall"),
        ("binary_auc_fake", "Binary AUC"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.24
    plt.figure(figsize=(12, 6))
    for idx, (model_name, model_metrics) in enumerate(metrics.items()):
        values = [model_metrics[k] if model_metrics[k] is not None else 0 for k, _ in metric_keys]
        plt.bar(x + (idx - 1) * width, values, width=width, label=model_name)
        for xi, value in zip(x + (idx - 1) * width, values):
            plt.text(xi, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, [label for _, label in metric_keys])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Metric Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_metric_comparison.png", dpi=300)
    plt.close()


def plot_training_curves():
    plt.figure(figsize=(12, 5))
    for key, name in MODELS.items():
        history = load_json(OUTPUT_ROOT / key / "history.json")
        epochs = [row["epoch"] for row in history]
        vals = [row["multiclass_macro_f1"] for row in history]
        plt.plot(epochs, vals, marker="o", linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro-F1 Curve")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "macro_f1_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    for key, name in MODELS.items():
        history = load_json(OUTPUT_ROOT / key / "history.json")
        epochs = [row["epoch"] for row in history]
        vals = [row["binary_f1_fake"] for row in history]
        plt.plot(epochs, vals, marker="o", linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Fake F1")
    plt.title("Validation Binary Fake-F1 Curve")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "binary_fake_f1_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, labels, title, out_path: Path, normalize: bool = True):
    cm = cm.astype(float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sum, 1), where=row_sum != 0)
    fig_w = max(7, len(labels) * 0.85)
    plt.figure(figsize=(fig_w, fig_w * 0.82))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else str(int(value))
            plt.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_all_confusion_matrices():
    for key, name in MODELS.items():
        eval_dir = OUTPUT_ROOT / key / "best_eval"
        multi_cm = np.loadtxt(eval_dir / "confusion_matrix_multiclass.csv", delimiter=",")
        binary_cm = np.loadtxt(eval_dir / "confusion_matrix_binary.csv", delimiter=",")
        plot_confusion_matrix(
            multi_cm,
            CLASS_NAMES,
            f"{name} Multiclass Confusion Matrix",
            FIG_DIR / f"{key}_confusion_multiclass.png",
            normalize=True,
        )
        plot_confusion_matrix(
            binary_cm,
            ["Real", "Fake"],
            f"{name} Binary Confusion Matrix",
            FIG_DIR / f"{key}_confusion_binary.png",
            normalize=True,
        )


def plot_per_class_f1():
    x = np.arange(len(CLASS_NAMES))
    width = 0.24
    plt.figure(figsize=(13, 6))
    for idx, (key, name) in enumerate(MODELS.items()):
        report = load_json(OUTPUT_ROOT / key / "best_eval" / "classification_report.json")
        vals = [report[label]["f1-score"] for label in CLASS_NAMES]
        plt.bar(x + (idx - 1) * width, vals, width=width, label=name)
    plt.xticks(x, CLASS_NAMES, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1-score")
    plt.title("Per-Class F1 Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_class_f1_comparison.png", dpi=300)
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric_bars()
    plot_training_curves()
    plot_all_confusion_matrices()
    plot_per_class_f1()
    print(f"saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
