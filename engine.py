import torch
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Huấn luyện mô hình qua một epoch (bật mode train: Dropout & BatchNorm cập nhật batch stats).
    """
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return running / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Đánh giá mô hình trên tập validation hoặc test (tắt Dropout & BatchNorm dùng running stats).
    Trả về loss, accuracy, macro-F1, y_true, y_pred.
    """
    model.eval()
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true += yb.cpu().tolist()
        y_pred += pred.cpu().tolist()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return running / total, correct / total, macro_f1, y_true, y_pred
