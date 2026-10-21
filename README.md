# DS201 · Lab 2: Mạng Nơ-ron Tích chập (CNN) trên VinaFood21

Dự án triển khai và thực nghiệm 4 mô hình CNN theo cấu trúc module hoá tiêu chuẩn của PyTorch:
- **Bài 1:** LeNet-5 (So sánh giữa `AvgPool2d` và `MaxPool2d`).
- **Bài 2:** GoogLeNet (Kiến trúc đa nhánh với các Inception Blocks và Global Average Pooling).
- **Bài 3\*:** ResNet-18 (Dựng từ đầu với các Residual BasicBlock, so sánh hiện tượng Overfitting khi Có vs Không có Data Augmentation).
- **Bài 4\*:** Transfer Learning với Pretrained ResNet-50 từ HuggingFace (So sánh Feature Extraction và Fine-Tuning).

---

## 1. Cấu trúc thư mục

```
cnn_project/
├── models/
│   ├── __init__.py       # Factory build_model, đếm tham số n_params
│   ├── lenet.py          # LeNet-5 (hỗ trợ avgpool & maxpool)
│   ├── googlenet.py      # MiniGoogLeNet với InceptionBlock
│   ├── resnet18.py       # ResNet-18 [2, 2, 2, 2] from scratch
│   └── resnet50.py       # Pretrained ResNet-50 từ HuggingFace
├── data.py               # Dataset VinaFood21, Augmentation, DataLoader
├── engine.py             # train_one_epoch, evaluate
├── train.py              # Huấn luyện mô hình qua dòng lệnh (CLI)
├── evaluate.py           # Đánh giá checkpoint trên test set
├── run_all.py            # Script chạy tự động toàn bộ 4 bài thực hành
├── run_colab.ipynb       # Notebook chạy trực tiếp trên Google Colab / Kaggle
├── requirements.txt      # Danh sách thư viện phụ thuộc
└── outputs/              # Chứa các file kết quả (.pt, .png, .csv)
```

---

## 2. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

---

## 3. Chạy trên Google Colab / Kaggle (Khuyến nghị GPU T4)

### Cách A: Chạy qua Notebook Colab (Giao diện trực quan)
1. Mở [run_colab.ipynb](run_colab.ipynb) trên Google Colab.
2. Chọn **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. Bấm **Run all** để tự động tải dữ liệu, huấn luyện, đánh giá và hiển thị toàn bộ đồ thị.

### Cách B: Chạy qua Terminal / dòng lệnh

#### Tự động chạy tất cả trong 1 lệnh duy nhất:
```bash
python run_all.py --data_root VinaFood21
```

#### Hoặc chạy từng bài riêng biệt:

* **Bài 1: LeNet-5**
```bash
# LeNet-5 với AvgPool
python train.py --model lenet_avg --epochs 20
python evaluate.py --ckpt outputs/lenet_avg/best.pt

# LeNet-5 với MaxPool
python train.py --model lenet_max --epochs 20
python evaluate.py --ckpt outputs/lenet_max/best.pt
```

* **Bài 2: GoogLeNet (Inception Blocks)**
```bash
python train.py --model googlenet --epochs 20
python evaluate.py --ckpt outputs/googlenet/best.pt
```

* **Bài 3\*: ResNet-18 (From Scratch)**
```bash
# Lần 1: Không có Augmentation (Bị Overfitting)
python train.py --model resnet18 --no_augment --epochs 20
python evaluate.py --ckpt outputs/resnet18_noaug/best.pt

# Lần 2: Có Augmentation
python train.py --model resnet18 --epochs 20
python evaluate.py --ckpt outputs/resnet18_aug/best.pt
```

* **Bài 4\*: Pretrained ResNet-50 (HuggingFace)**
```bash
# Chế độ 1: Feature Extraction (đóng băng backbone)
python train.py --model resnet50_fe --epochs 15 --lr 1e-3
python evaluate.py --ckpt outputs/resnet50_fe/best.pt

# Chế độ 2: Fine-Tuning (huấn luyện toàn bộ mạng)
python train.py --model resnet50_ft --epochs 10 --lr 1e-4
python evaluate.py --ckpt outputs/resnet50_ft/best.pt
```

---

## 4. Kết quả đầu ra trong thư mục `outputs/`

Mỗi mô hình sau khi chạy xong sẽ tự động tạo thư mục con trong `outputs/` chứa:
- `best.pt`: Checkpoint trọng số mô hình có `val macro-F1` cao nhất.
- `history.csv`: Bảng ghi nhận `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1` qua từng epoch.
- `loss_curve.png`: Biểu đồ đường cong Loss (Train vs Val) chỉ ra epoch tối ưu.
- `confusion_matrix.png`: Ma trận nhầm lẫn kích thước $21 \times 21$ trên tập test.
- `test_metrics.json`: Các chỉ số độ đo Accuracy, Precision-macro, Recall-macro, F1-macro.

Bảng tổng hợp đối chiếu toàn diện được lưu tại:
- `outputs/summary_table.csv`
- `outputs/summary_table.md`
- `outputs/resnet18_overfitting_comparison.png`

---

## 5. Hướng dẫn Push lên GitHub và Nộp bài

1. **Commit và Push toàn bộ mã nguồn cùng thư mục kết quả `outputs/` lên GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Hoàn thành Lab 2: Mạng CNN trên VinaFood21"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
2. **Tạo file nộp bài:**
   - Tạo file text đặt tên theo MSSV: ví dụ `21120000.txt`.
   - Dán link GitHub repository của bạn vào file `21120000.txt`.
   - Nộp file text này lên hệ thống khóa học.
