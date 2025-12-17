"""
train_multistep_lstm.py

使用 LSTMSeqPredictor 进行多步预测：
用前 seq_len 分的特征，预测接下来 horizon 分的结果（P1 是否获胜）。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import SEQ_LEN
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_multistep_sequences,
    split_match_ids,
    mask_by_matches,
)
from src.datasets import make_seq_dataloader
from src.models import LSTMSeqPredictor


HORIZON = 5  # 多步预测长度，可自行调整


def train_multistep_lstm(
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
) -> LSTMSeqPredictor:
    train_loader = make_seq_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_seq_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = LSTMSeqPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3,
        horizon=HORIZON,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def run_loader(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_logits, all_y = [], []

        for xb, yb in loader:
            xb = xb.to(device)                # (batch, seq_len, feat_dim)
            yb = yb.to(device)                # (batch, horizon)

            logits = model(xb)                # (batch, horizon)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

        n = len(loader.dataset)
        avg_loss = total_loss / n

        all_logits = np.concatenate(all_logits, axis=0)  # (N, horizon)
        all_y = np.concatenate(all_y, axis=0)            # (N, horizon)

        # 拉平后一起评估
        logits_flat = all_logits.reshape(-1)
        y_flat = all_y.reshape(-1)

        probs = 1.0 / (1.0 + np.exp(-logits_flat))
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_flat, preds)
        f1 = f1_score(y_flat, preds)
        try:
            auc = roc_auc_score(y_flat, probs)
        except ValueError:
            auc = float("nan")

        return avg_loss, acc, f1, auc

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_auc = run_loader(train_loader, train=True)
        val_loss, val_acc, val_f1, val_auc = run_loader(val_loader, train=False)

        print(
            f"[MultiLSTM] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
        )

    return model


def eval_multistep_lstm(
    model: LSTMSeqPredictor,
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

    all_logits = np.concatenate(all_logits, axis=0)  # (N, horizon)
    all_y = np.concatenate(all_y, axis=0)            # (N, horizon)

    logits_flat = all_logits.reshape(-1)
    y_flat = all_y.reshape(-1)

    probs = 1.0 / (1.0 + np.exp(-logits_flat))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_flat, preds)
    f1 = f1_score(y_flat, preds)
    try:
        auc = roc_auc_score(y_flat, probs)
    except ValueError:
        auc = float("nan")

    print(f"[MultiLSTM] Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    return acc, f1, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 构造多步序列
    X_seq, y_seq, match_ids_seq = build_multistep_sequences(
        matches_df,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
    )

    # 按 match 划分
    train_matches, val_matches, test_matches = split_match_ids(match_ids_seq)
    train_mask = mask_by_matches(match_ids_seq, train_matches)
    val_mask = mask_by_matches(match_ids_seq, val_matches)
    test_mask = mask_by_matches(match_ids_seq, test_matches)

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]

    print("Multi-step LSTM train/val/test:", len(y_train), len(y_val), len(y_test))

    input_dim = X_train.shape[2]

    model = train_multistep_lstm(
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

    eval_multistep_lstm(model, X_test, y_test, device=device)

    torch.save(model.state_dict(), f"lstm_multistep_h{HORIZON}.pth")
    print(f"Saved multi-step LSTM model to lstm_multistep_h{HORIZON}.pth")


if __name__ == "__main__":
    main()
