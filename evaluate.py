import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from data import build_test_loader
from engine import evaluate
from models import build_model


def get_args():
    p = argparse.ArgumentParser(description="Đánh giá mô hình CNN trên tập test VinaFood21")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Đường dẫn file checkpoint (ví dụ: outputs/lenet_avg/best.pt)")
    p.add_argument("--data_root", type=str, default="VinaFood21",
                   help="Đường dẫn tới thư mục chứa dữ liệu VinaFood21")
    p.add_argument("--model", type=str, default=None,
                   help="Tên mô hình (nếu không truyền, tự động đọc từ checkpoint)")
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
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
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
    assert ckpt_path.is_file(), f"Không tìm thấy file checkpoint: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)
    model_name = args.model or ckpt.get("model_name", "lenet_avg")

    print("=" * 70)
    print(f"ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()} TRÊN TẬP TEST")
    print(f"Checkpoint : {ckpt_path} (Epoch {ckpt.get('epoch', '?')})")
    print(f"Kích thước : {img_size}x{img_size} | Số lớp: {len(classes)}")
    print("=" * 70)

    model = build_model(model_name, num_classes=len(classes), img_size=img_size).to(device)
    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(args.data_root, img_size, args.batch_size, args.num_workers)
    assert test_classes == classes, "Thứ tự lớp ở tập test khác tập train/val!"

    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print("\n===== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST =====")
    print(f"Test Loss        : {loss:.4f}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (macro): {p_macro:.4f}")
    print(f"Recall (macro)   : {r_macro:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print("-" * 70)
    print("Báo cáo chi tiết theo 21 lớp món ăn:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = ckpt_path.parent / "confusion_matrix.png"
    save_confusion_matrix(cm, classes, cm_path, title=f"Confusion Matrix - {model_name}")
    print(f"Đã lưu ma trận nhầm lẫn tại: {cm_path}")

    # Lưu metrics ra file json
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
    print(f"Đã lưu số liệu test ra: {ckpt_path.parent / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
