"""
训练 LSTM 序列模型，用前 SEQ_LEN 分的特征预测下一分 player1 是否赢。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

from src.config import SEQ_LEN
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_sequences_for_lstm,
    split_match_ids,
    mask_by_matches,
)
from src.datasets import make_seq_dataloader
from src.models import LSTMPointPredictor


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 1,
    device: str = "cpu",
) -> LSTMPointPredictor:
    train_loader = make_seq_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_seq_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = LSTMPointPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    def run_loader(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_logits, all_y = [], []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

        n = len(loader.dataset)
        avg_loss = total_loss / n
        all_logits = np.concatenate(all_logits)
        all_y = np.concatenate(all_y)

        probs = 1.0 / (1.0 + np.exp(-all_logits))
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(all_y, preds)
        f1 = f1_score(all_y, preds)
        try:
            auc = roc_auc_score(all_y, probs)
        except ValueError:
            auc = float("nan")

        return avg_loss, acc, f1, auc

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_auc = run_loader(train_loader, train=True)
        val_loss, val_acc, val_f1, val_auc = run_loader(val_loader, train=False)

        print(
            f"[LSTM] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
    os.makedirs("figures", exist_ok=True)

    epochs_axis = range(1, epochs + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, history["train_loss"], label="Train Loss")
    plt.plot(epochs_axis, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/lstm_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, history["val_acc"], label="Val Accuracy")
    plt.plot(epochs_axis, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("LSTM Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/lstm_metrics_curve.png", dpi=200)
    plt.close()

    return model


def eval_lstm(
    model: LSTMPointPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu",
):
    test_loader = make_seq_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits, all_y = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_y.append(yb.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_y = np.concatenate(all_y)

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_y, preds)
    f1 = f1_score(all_y, preds)
    try:
        auc = roc_auc_score(all_y, probs)
    except ValueError:
        auc = float("nan")

    print(f"[LSTM] Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    return acc, f1, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读数据 & 预处理
    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 2. 构造 LSTM 序列
    X_seq, y_seq, match_ids_seq = build_sequences_for_lstm(matches_df, seq_len=SEQ_LEN)

    # 3. 按比赛划分 train/val/test
    train_matches, val_matches, test_matches = split_match_ids(match_ids_seq)

    train_mask = mask_by_matches(match_ids_seq, train_matches)
    val_mask = mask_by_matches(match_ids_seq, val_matches)
    test_mask = mask_by_matches(match_ids_seq, test_matches)

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]

    print("LSTM train/val/test:", len(y_train), len(y_val), len(y_test))

    # 4. 训练
    input_dim = X_train.shape[2]
    model = train_lstm(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=input_dim,
        epochs=15,
        batch_size=64,
        hidden_dim=128,
        device=device,
    )

    # 5. 测试评估
    eval_lstm(model, X_test, y_test, device=device)

    # 6. 保存模型
    torch.save(model.state_dict(), "lstm_point_predictor.pth")
    print("Saved LSTM model to lstm_point_predictor.pth")


if __name__ == "__main__":
    main()
