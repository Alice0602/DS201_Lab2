import argparse
import json
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from data import build_test_loader
from engine import evaluate
from models import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default="VinaFood21")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f"File checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)
    model_name = args.model or ckpt.get("model_name", "lenet_avg")

    print(f"\n--- Testing {model_name} ({img_size}x{img_size}) from {ckpt_path} ---")

    model = build_model(model_name, num_classes=len(classes), img_size=img_size).to(device)
    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(args.data_root, img_size, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | Macro-F1: {f1:.4f} | Prec: {p_macro:.4f} | Rec: {r_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = ckpt_path.parent / "confusion_matrix.png"
    save_confusion_matrix(cm, classes, cm_path, title=f"Confusion Matrix - {model_name}")
    np.save(ckpt_path.parent / "cm.npy", cm)
    np.save(ckpt_path.parent / "classes.npy", np.array(classes))

    metrics = {
        "model": model_name,
        "img_size": img_size,
        "test_loss": loss,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1
    }
    with open(ckpt_path.parent / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
