# DS201 - Lab 2: Mạng nơ-ron tích chập (CNN) trên VinaFood21

Thực hành tuần 2: Huấn luyện và đánh giá 4 mô hình CNN trên bộ dữ liệu VinaFood21:
- **Bài 1:** LeNet-5 (so sánh AvgPool và MaxPool)
- **Bài 2:** GoogLeNet (Inception blocks + GAP)
- **Bài 3\*:** ResNet-18 (from scratch, so sánh Không Augment vs Có Augment)
- **Bài 4\*:** ResNet-50 Pretrained (HuggingFace, so sánh Feature Extraction vs Fine-Tuning)

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Chạy trên Google Colab

Mở file `run_colab.ipynb` trên Google Colab (chọn GPU T4) và chạy lần lượt các ô lệnh hoặc chạy nhanh qua script:

```bash
python run_all.py --data_root VinaFood21
```

## Cấu trúc thư mục

```
cnn_project/
├── models/             # Định nghĩa các mô hình CNN
├── data.py             # Dataloader và tiền xử lý ảnh
├── engine.py           # Vòng lặp train và evaluate
├── train.py            # Huấn luyện mô hình
├── evaluate.py         # Đánh giá trên tập test
├── run_all.py          # Chạy tất cả thí nghiệm
├── run_colab.ipynb     # Notebook chạy trên Colab
└── requirements.txt
```
