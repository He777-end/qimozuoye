"""
train_transformer.py

使用 TransformerPointPredictor 做单步预测：
用前 seq_len 分的特征，预测下一分 player1 是否获胜。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import SEQ_LEN
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_sequences_for_lstm,
    split_match_ids,
    mask_by_matches,
)
from src.datasets import make_seq_dataloader
from src.models import TransformerPointPredictor


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    device: str = "cpu",
) -> TransformerPointPredictor:
    train_loader = make_seq_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_seq_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = TransformerPointPredictor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=256,
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            f"[Transformer] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
        )

    return model


def eval_transformer(
    model: TransformerPointPredictor,
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

    print(f"[Transformer] Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    return acc, f1, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 单步序列和 LSTM 一样复用
    X_seq, y_seq, match_ids_seq = build_sequences_for_lstm(matches_df, seq_len=SEQ_LEN)

    train_matches, val_matches, test_matches = split_match_ids(match_ids_seq)
    train_mask = mask_by_matches(match_ids_seq, train_matches)
    val_mask = mask_by_matches(match_ids_seq, val_matches)
    test_mask = mask_by_matches(match_ids_seq, test_matches)

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]

    print("Transformer train/val/test:", len(y_train), len(y_val), len(y_test))

    input_dim = X_train.shape[2]

    model = train_transformer(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=input_dim,
        epochs=15,
        batch_size=64,
        d_model=128,
        nhead=8,
        num_layers=2,
        device=device,
    )

    eval_transformer(model, X_test, y_test, device=device)

    torch.save(model.state_dict(), "transformer_point_predictor.pth")
    print("Saved Transformer model to transformer_point_predictor.pth")


if __name__ == "__main__":
    main()
