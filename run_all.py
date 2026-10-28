import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTS = [
    {
        "id": "lenet_avg",
        "name": "LeNet-5 (AvgPool)",
        "model": "lenet_avg",
        "img_size": 64,
        "epochs": 20,
        "lr": 1e-3,
        "no_augment": False,
        "params": "339,421",
        "out_dir": "outputs/lenet_avg"
    },
    {
        "id": "lenet_max",
        "name": "LeNet-5 (MaxPool)",
        "model": "lenet_max",
        "img_size": 64,
        "epochs": 20,
        "lr": 1e-3,
        "no_augment": False,
        "params": "339,421",
        "out_dir": "outputs/lenet_max"
    },
    {
        "id": "googlenet",
        "name": "GoogLeNet (3 Inception)",
        "model": "googlenet",
        "img_size": 64,
        "epochs": 20,
        "lr": 1e-3,
        "no_augment": False,
        "params": "332,557",
        "out_dir": "outputs/googlenet"
    },
    {
        "id": "resnet18_noaug",
        "name": "ResNet-18 (No Aug)*",
        "model": "resnet18",
        "img_size": 96,
        "epochs": 20,
        "lr": 1e-3,
        "no_augment": True,
        "params": "11,187,285",
        "out_dir": "outputs/resnet18_noaug"
    },
    {
        "id": "resnet18_aug",
        "name": "ResNet-18 (Aug)*",
        "model": "resnet18",
        "img_size": 96,
        "epochs": 20,
        "lr": 1e-3,
        "no_augment": False,
        "params": "11,187,285",
        "out_dir": "outputs/resnet18_aug"
    },
    {
        "id": "resnet50_fe",
        "name": "ResNet-50 (Feature Extraction)*",
        "model": "resnet50_fe",
        "img_size": 224,
        "epochs": 15,
        "lr": 1e-3,
        "no_augment": False,
        "params": "23,551,061 (học 43,029)",
        "out_dir": "outputs/resnet50_fe"
    },
    {
        "id": "resnet50_ft",
        "name": "ResNet-50 (Fine-Tuning)*",
        "model": "resnet50_ft",
        "img_size": 224,
        "epochs": 10,
        "lr": 1e-4,
        "no_augment": False,
        "params": "23,551,061",
        "out_dir": "outputs/resnet50_ft"
    }
]


def plot_resnet18_comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_noaug = Path("outputs/resnet18_noaug/history.csv")
    p_aug   = Path("outputs/resnet18_aug/history.csv")
    if not (p_noaug.is_file() and p_aug.is_file()):
        return

    df_noaug = pd.read_csv(p_noaug)
    df_aug   = pd.read_csv(p_aug)

    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(df_noaug["epoch"], df_noaug["train_loss"], "o--", label="Train Loss (No Aug)")
    plt.plot(df_noaug["epoch"], df_noaug["val_loss"], "s-", label="Val Loss (No Aug)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-18 (Khong Augment)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_aug["epoch"], df_aug["train_loss"], "o--", label="Train Loss (Aug)", color="green")
    plt.plot(df_aug["epoch"], df_aug["val_loss"], "s-", label="Val Loss (Aug)", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-18 (Co Augment)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    out_cmp = Path("outputs/resnet18_overfitting_comparison.png")
    plt.savefig(out_cmp, dpi=120, bbox_inches="tight")
    plt.close()


def plot_all_loss_curves(out_path="outputs/loss_all_models.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [
        ("lenet_avg", "LeNet-5 (AvgPool)"),
        ("lenet_max", "LeNet-5 (MaxPool)"),
        ("googlenet", "GoogLeNet (3 Inception)"),
        ("resnet18_noaug", "ResNet-18 (Khong Augment)*"),
        ("resnet18_aug", "ResNet-18 (Co Augment)*"),
        ("resnet50_fe", "ResNet-50 (Feature Extraction)*"),
        ("resnet50_ft", "ResNet-50 (Fine-Tuning)*"),
    ]

    has_any = any(Path(f"outputs/{m}/history.csv").is_file() for m, _ in models)
    if not has_any:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes_flat = axes.flatten()

    for i, (m_id, title) in enumerate(models):
        ax = axes_flat[i]
        p = Path(f"outputs/{m_id}/history.csv")
        if p.is_file():
            df = pd.read_csv(p)
            best_va = df["val_loss"].min()
            best_ep = df.loc[df["val_loss"].idxmin(), "epoch"]
            ax.plot(df["epoch"], df["train_loss"], "o--", label="Train Loss", color="#1f77b4", linewidth=1.5, markersize=4)
            ax.plot(df["epoch"], df["val_loss"], "s-", label="Val Loss", color="#d62728", linewidth=1.5, markersize=4)
            ax.axvline(best_ep, ls=":", color="gray", alpha=0.7, label=f"Best ({best_va:.3f})")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Loss", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        else:
            ax.axis("off")

    ax8 = axes_flat[7]
    p_noaug = Path("outputs/resnet18_noaug/history.csv")
    p_aug = Path("outputs/resnet18_aug/history.csv")
    if p_noaug.is_file() and p_aug.is_file():
        df_noaug = pd.read_csv(p_noaug)
        df_aug = pd.read_csv(p_aug)
        ax8.plot(df_noaug["epoch"], df_noaug["val_loss"], "s-", label="Val Loss (Khong Aug)", color="#d62728", linewidth=1.5, markersize=4)
        ax8.plot(df_aug["epoch"], df_aug["val_loss"], "^-", label="Val Loss (Co Aug)", color="#2ca02c", linewidth=1.5, markersize=4)
        ax8.set_title("Doi chieu Overfitting (Bai 3*)", fontsize=10, fontweight="bold")
        ax8.set_xlabel("Epoch", fontsize=9)
        ax8.set_ylabel("Val Loss", fontsize=9)
        ax8.grid(True, alpha=0.25)
        ax8.legend(fontsize=8, loc="best")
    else:
        ax8.axis("off")

    plt.suptitle("Đồ thị Loss (Train vs Validation) của tất cả 7 mô hình", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_row(models=None, out_path="outputs/confusion_matrices_row.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if models is None:
        models = [
            ("lenet_avg", "LeNet-5 (Bài 1)"),
            ("googlenet", "GoogLeNet (Bài 2)"),
            ("resnet18_aug", "ResNet-18 (Bài 3*)"),
            ("resnet50_ft", "ResNet-50 FT (Bài 4*)"),
        ]

    available = []
    for m_id, title in models:
        cm_f = Path(f"outputs/{m_id}/cm.npy")
        img_f = Path(f"outputs/{m_id}/confusion_matrix.png")
        if cm_f.is_file() or img_f.is_file():
            available.append((m_id, title, cm_f, img_f))

    if not available:
        return

    has_all_npy = all(cm_f.is_file() for _, _, cm_f, _ in available)
    n = len(available)

    if has_all_npy:
        import seaborn as sns
        classes = None
        for m_id, _, _, _ in available:
            cls_f = Path(f"outputs/{m_id}/classes.npy")
            if cls_f.is_file():
                classes = list(np.load(cls_f))
                break

        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, (m_id, title, cm_f, _) in enumerate(available):
            cm = np.load(cm_f)
            xt = classes if classes is not None else False
            yt = classes if (classes is not None and i == 0) else False
            sns.heatmap(
                cm, annot=False, cmap="Blues",
                cbar=(i == n - 1),
                ax=axes[i],
                xticklabels=xt,
                yticklabels=yt
            )
            axes[i].set_title(title, fontsize=11, fontweight="bold")
            if classes is not None:
                axes[i].tick_params(axis="x", labelsize=6.5, rotation=90)
                if i == 0:
                    axes[i].tick_params(axis="y", labelsize=6.5, rotation=0)
                    axes[i].set_ylabel("True label", fontsize=10)
                else:
                    axes[i].set_ylabel("")
            axes[i].set_xlabel("Predicted label", fontsize=10)

        plt.subplots_adjust(wspace=0.12)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for i, (_, _, _, img_f) in enumerate(available):
            if img_f.is_file():
                axes[i].imshow(plt.imread(img_f))
            axes[i].axis("off")
        plt.subplots_adjust(wspace=0.03)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_all_confusion_matrices(out_path="outputs/confusion_matrices_all.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [
        ("lenet_avg", "LeNet-5 (AvgPool)"),
        ("lenet_max", "LeNet-5 (MaxPool)"),
        ("googlenet", "GoogLeNet (3 Inception)"),
        ("resnet18_noaug", "ResNet-18 (Khong Aug)*"),
        ("resnet18_aug", "ResNet-18 (Co Aug)*"),
        ("resnet50_fe", "ResNet-50 (FE)*"),
        ("resnet50_ft", "ResNet-50 (FT)*"),
    ]

    available = []
    for m_id, title in models:
        cm_f = Path(f"outputs/{m_id}/cm.npy")
        img_f = Path(f"outputs/{m_id}/confusion_matrix.png")
        if cm_f.is_file() or img_f.is_file():
            available.append((m_id, title, cm_f, img_f))

    if not available:
        return

    has_all_npy = all(cm_f.is_file() for _, _, cm_f, _ in available)
    if has_all_npy:
        import seaborn as sns
        classes = None
        for m_id, _, _, _ in available:
            cls_f = Path(f"outputs/{m_id}/classes.npy")
            if cls_f.is_file():
                classes = list(np.load(cls_f))
                break

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes_flat = axes.flatten()
        for i, (m_id, title, cm_f, _) in enumerate(available):
            ax = axes_flat[i]
            cm = np.load(cm_f)
            is_left = (i % 4 == 0)
            xt = classes if classes is not None else False
            yt = classes if (classes is not None and is_left) else False
            sns.heatmap(
                cm, annot=False, cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=xt,
                yticklabels=yt
            )
            ax.set_title(title, fontsize=10, fontweight="bold")
            if classes is not None:
                ax.tick_params(axis="x", labelsize=5.5, rotation=90)
                if is_left:
                    ax.tick_params(axis="y", labelsize=5.5, rotation=0)
                    ax.set_ylabel("True label", fontsize=9)
                else:
                    ax.set_ylabel("")
            ax.set_xlabel("Predicted label", fontsize=9)

        for j in range(len(available), 8):
            axes_flat[j].axis("off")

        plt.suptitle("Confusion Matrix của tất cả 7 mô hình", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        n = len(available)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, (_, _, _, img_f) in enumerate(available):
            if img_f.is_file():
                axes_flat[i].imshow(plt.imread(img_f))
            axes_flat[i].axis("off")
        for j in range(len(available), len(axes_flat)):
            axes_flat[j].axis("off")
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="VinaFood21")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    from data import download_vinafood21
    download_vinafood21(args.data_root)

    summary_rows = []

    for exp in EXPERIMENTS:
        if args.only and exp["model"] != args.only and exp["id"] != args.only:
            continue

        print(f"\n>>> Running: {exp['name']}")

        cmd_train = [
            sys.executable, "train.py",
            "--model", exp["model"],
            "--data_root", args.data_root,
            "--img_size", str(exp["img_size"]),
            "--batch_size", str(args.batch_size),
            "--epochs", str(exp["epochs"]),
            "--lr", str(exp["lr"]),
            "--out_dir", exp["out_dir"]
        ]
        if exp["no_augment"]:
            cmd_train.append("--no_augment")

        subprocess.run(cmd_train, check=True)

        cmd_eval = [
            sys.executable, "evaluate.py",
            "--ckpt", f"{exp['out_dir']}/best.pt",
            "--data_root", args.data_root,
            "--batch_size", str(args.batch_size)
        ]
        subprocess.run(cmd_eval, check=True)

        history_path = Path(exp["out_dir"]) / "history.csv"
        metrics_path = Path(exp["out_dir"]) / "test_metrics.json"

        sec_per_ep = "?"
        if history_path.is_file():
            df_h = pd.read_csv(history_path)
            if "time_sec" in df_h.columns:
                sec_per_ep = f"{df_h['time_sec'].mean():.1f}s"

        acc, f1 = "?", "?"
        if metrics_path.is_file():
            with open(metrics_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
                acc = f"{m_data.get('accuracy', 0):.4f}"
                f1  = f"{m_data.get('f1_macro', 0):.4f}"

        summary_rows.append({
            "Mô hình": exp["name"],
            "Resize": f"{exp['img_size']}x{exp['img_size']}",
            "Tham số": exp.get("params", "?"),
            "Giây/epoch": sec_per_ep,
            "Accuracy": acc,
            "macro-F1": f1
        })

    plot_resnet18_comparison()
    plot_all_loss_curves()
    plot_confusion_matrices_row()
    plot_all_confusion_matrices()

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "summary_table.csv"
        out_md  = out_dir / "summary_table.md"
        df_summary.to_csv(out_csv, index=False, encoding="utf-8")
        try:
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(df_summary.to_markdown(index=False))
        except Exception:
            with open(out_md, "w", encoding="utf-8") as f:
                f.write("| " + " | ".join(df_summary.columns) + " |\n")
                f.write("| " + " | ".join([":---:"] * len(df_summary.columns)) + " |\n")
                for _, row in df_summary.iterrows():
                    f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")

        print("\n--- Bang tong hop ket qua ---")
        print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
