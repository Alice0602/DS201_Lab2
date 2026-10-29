| Mô hình | Resize | Tham số | Giây/epoch | Accuracy | macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LeNet-5 (AvgPool) | 64x64 | 339,421 | 123.3s | 0.3099 | 0.2835 |
| LeNet-5 (MaxPool) | 64x64 | 339,421 | 122.8s | 0.3129 | 0.2917 |
| GoogLeNet (3 Inception) | 64x64 | 332,557 | 127.3s | 0.4300 | 0.4053 |
| ResNet-18 (No Aug)* | 96x96 | 11,187,285 | 119.8s | 0.4126 | 0.4089 |
| ResNet-18 (Aug)* | 96x96 | 11,187,285 | 128.6s | 0.4782 | 0.4599 |
| ResNet-50 (Feature Extraction)* | 224x224 | 23,551,061 (học 43,029) | 161.7s | 0.6136 | 0.6153 |
| ResNet-50 (Fine-Tuning)* | 224x224 | 23,551,061 | 179.9s | 0.8411 | 0.8389 |
