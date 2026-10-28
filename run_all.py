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
