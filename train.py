import argparse
import csv
import time
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from models import build_model, n_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data_root", type=str, default="VinaFood21")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
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
    plt.axvline(best_ep, ls="--", color="gray", label=f"Best Val (Epoch {best_ep})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.xticks(ep[::2] if len(ep) > 5 else ep)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.img_size is None:
        if args.model.startswith("lenet") or args.model == "googlenet":
            args.img_size = 64
        elif args.model == "resnet18":
            args.img_size = 96
        elif args.model.startswith("resnet50"):
            args.img_size = 224

    if args.lr is None:
        args.lr = 1e-4 if args.model == "resnet50_ft" else 1e-3

    if args.out_dir is None:
        if args.model == "resnet18":
            aug_tag = "_noaug" if args.no_augment else "_aug"
        else:
            aug_tag = "_noaug" if args.no_augment else ""
        args.out_dir = f"outputs/{args.model}{aug_tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    augment = not args.no_augment
    print(f"\n--- Training {args.model} | device={device} | size={args.img_size}x{args.img_size} | lr={args.lr} | epochs={args.epochs} ---")

    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, args.img_size, args.batch_size,
        augment=augment, num_workers=args.num_workers, seed=args.seed
    )
    assert len(classes) == 21, f"Tap du lieu chi tim thay {len(classes)}/21 lop! Kiem tra thu muc {args.data_root}"

    model = build_model(args.model, num_classes=len(classes), img_size=args.img_size).to(device)
    total_p = n_params(model)
    train_p = n_params(model, trainable_only=True)
    print(f"Model params: {total_p:,} (trainable: {train_p:,})")

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

        print(f"Epoch {ep:>2}/{args.epochs} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {va_loss:.4f}/{va_acc:.4f} F1 {va_f1:.4f} | {elapsed:.1f}s{mark}")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    save_loss_curve(history, out_dir / "loss_curve.png", args.model)
    avg_sec = total_time / args.epochs
    print(f"Done! {avg_sec:.1f}s/epoch, Best Val F1: {best_f1:.4f}\n")


if __name__ == "__main__":
    main()
