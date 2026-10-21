import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from models import build_model, n_params


def get_args():
    p = argparse.ArgumentParser(description="Huấn luyện các mô hình CNN trên VinaFood21")
    p.add_argument("--model", type=str, required=True,
                   choices=["lenet_avg", "lenet_max", "googlenet", "resnet18", "resnet50_fe", "resnet50_ft"],
                   help="Tên mô hình cần huấn luyện")
    p.add_argument("--data_root", type=str, default="VinaFood21",
                   help="Đường dẫn tới thư mục chứa dữ liệu VinaFood21")
    p.add_argument("--img_size", type=int, default=None,
                   help="Kích thước ảnh (mặc định: 64 cho LeNet/GoogLeNet, 96 cho ResNet18, 224 cho ResNet50)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (mặc định 1e-3, riêng ResNet-50 Fine-tuning là 1e-4)")
    p.add_argument("--no_augment", action="store_true",
                   help="Tắt Data Augmentation trên tập train (dùng cho thí nghiệm so sánh ResNet-18)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None,
                   help="Thư mục lưu outputs (mặc định: outputs/<model>)")
    return p.parse_args()


def save_loss_curve(history, path, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]
    best_idx = int(np.argmin(va))
    best_ep = ep[best_idx]

    plt.figure(figsize=(7, 4))
    plt.plot(ep, tr, "o-", label="Train Loss")
    plt.plot(ep, va, "s-", label="Val Loss")
    plt.axvline(best_ep, ls="--", color="gray", label=f"Val Loss thấp nhất (Epoch {best_ep})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title(f"Đường cong Loss - {model_name}")
    plt.xticks(ep[::2] if len(ep) > 5 else ep)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Xác định kích thước ảnh mặc định nếu không truyền vào
    if args.img_size is None:
        if args.model.startswith("lenet") or args.model == "googlenet":
            args.img_size = 64
        elif args.model == "resnet18":
            args.img_size = 96
        elif args.model.startswith("resnet50"):
            args.img_size = 224

    # Xác định learning rate mặc định nếu không truyền vào
    if args.lr is None:
        if args.model == "resnet50_ft":
            args.lr = 1e-4
        else:
            args.lr = 1e-3

    # Xác định thư mục output mặc định
    if args.out_dir is None:
        aug_tag = "_noaug" if args.no_augment else ""
        args.out_dir = f"outputs/{args.model}{aug_tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    augment = not args.no_augment
    print("=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN: {args.model.upper()}")
    print(f"Thiết bị: {device} | Kích thước ảnh: {args.img_size}x{args.img_size}")
    print(f"Data Augmentation: {augment} | Batch size: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Lưu kết quả tại: {out_dir.resolve()}")
    print("=" * 70)

    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, args.img_size, args.batch_size,
        augment=augment, num_workers=args.num_workers, seed=args.seed
    )
    print(f"Số lớp: {len(classes)} | Số ảnh train: {len(train_dl.dataset)} | Số ảnh val: {len(val_dl.dataset)}")

    model = build_model(args.model, num_classes=len(classes), img_size=args.img_size).to(device)
    total_p = n_params(model)
    train_p = n_params(model, trainable_only=True)
    print(f"Mô hình {args.model}: {total_p:,} tham số (Cần cập nhật: {train_p:,})")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    history = []
    best_f1 = -1.0
    total_time = 0.0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        elapsed = time.time() - t0
        total_time += elapsed

        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
            "time_sec": elapsed
        })

        mark = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "model_name": args.model,
                "img_size": args.img_size,
                "epoch": ep,
                "best_val_f1": best_f1
            }, out_dir / "best.pt")
            mark = " *"

        print(f"Epoch {ep:>2}/{args.epochs} | Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {va_loss:.4f}/{va_acc:.4f} F1 {va_f1:.4f} | {elapsed:.1f}s{mark}")

    # Ghi lại file history.csv
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    # Lưu biểu đồ Loss Curve
    save_loss_curve(history, out_dir / "loss_curve.png", args.model)
    avg_sec = total_time / args.epochs
    print(f"\nHuấn luyện hoàn tất! Thời gian trung bình: {avg_sec:.1f}s/epoch")
    print(f"Best Val Macro-F1: {best_f1:.4f} -> Đã lưu checkpoint tại: {out_dir / 'best.pt'}")
    print(f"Biểu đồ loss đã lưu tại: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
