"""
train_reversal.py

使用已经训练好的 LSTM 模型，构造“势头反转”数据集，
再训练一个小 MLP 来预测：下一时刻是否发生动量反转。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import RANDOM_STATE
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_flat_features_for_baseline,
)
from src.models import BaselineMLP, LSTMPointPredictor
from src.momentum import build_reversal_dataset


def train_reversal_model(
    model: BaselineMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(y_train)
    n_val = len(y_val)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        idx = np.arange(n_train)
        np.random.shuffle(idx)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = torch.from_numpy(X_train[batch_idx]).to(device)
            yb = torch.from_numpy(y_train[batch_idx]).to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        train_loss = total_loss / n_train

        # ---- val ----
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_val).to(device)
            yb = torch.from_numpy(y_val).to(device)
            logits = model(xb).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
            auc = float("nan")

        print(
            f"[Reversal] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={acc:.4f}, val_auc={auc:.4f}"
        )


def eval_reversal_model(
    model: BaselineMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        logits = model(xb).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")

    print(f"[Reversal] Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    return acc, f1, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读数据 & 预处理
    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 2. 根据展平特征算出 input_dim（要和训练 LSTM 时一致）
    feat_df, _ = build_flat_features_for_baseline(matches_df)
    input_dim = feat_df.shape[1]
    print("LSTM input_dim (flattened features):", input_dim)

    # 3. 构建 LSTM 模型并加载权重（确保你已经跑过 train_lstm.py）
    lstm_model = LSTMPointPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
    ).to(device)

    lstm_state = torch.load("lstm_point_predictor.pth", map_location=device)
    lstm_model.load_state_dict(lstm_state)
    print("Loaded LSTM weights from lstm_point_predictor.pth")

    # 4. 构造 “势头反转” 数据集
    X_rev_all, y_rev_all = build_reversal_dataset(
        lstm_model,
        matches_df,
        feat_df,
        device=device,
    )

    # 5. 拆分 train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_rev_all,
        y_rev_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=(y_rev_all > 0).astype(int),  # 有可能正例比较少
    )

    print("Reversal train/test size:", len(y_train), len(y_test))

    # 6. 再从 train 中划一小部分做 val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=(y_train > 0).astype(int),
    )

    # 7. 初始化并训练 MLP 反转预测模型
    rev_model = BaselineMLP(input_dim=X_train.shape[1], hidden_dim=64).to(device)

    train_reversal_model(
        rev_model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        batch_size=512,
        lr=1e-3,
        device=device,
    )

    # 8. 在 test set 上评估
    eval_reversal_model(rev_model, X_test, y_test, device=device)

    # 9. 保存模型
    torch.save(rev_model.state_dict(), "reversal_mlp.pth")
    print("Saved reversal model to reversal_mlp.pth")


if __name__ == "__main__":
    main()
