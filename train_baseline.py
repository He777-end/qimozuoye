"""
训练基线模型（MLP）来预测：每一分 player1 是否赢。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from src.config import RANDOM_STATE
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_flat_features_for_baseline,
    split_match_ids,
    mask_by_matches,
)
from src.models import BaselineMLP


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BaselineMLP:
    model = BaselineMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    def run_epoch(X, y, train: bool):
        model.train(train)
        n = len(y)
        indices = np.arange(n)
        if train:
            np.random.shuffle(indices)
        total_loss = 0.0
        all_logits, all_y = [], []

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = torch.from_numpy(X[batch_idx]).to(device)
            yb = torch.from_numpy(y[batch_idx]).to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

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
        train_loss, train_acc, train_f1, train_auc = run_epoch(X_train, y_train, train=True)
        val_loss, val_acc, val_f1, val_auc = run_epoch(X_val, y_val, train=False)

        print(
            f"[Baseline] Epoch {epoch:02d} | "
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
    plt.title("Baseline MLP Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/baseline_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, history["val_acc"], label="Val Accuracy")
    plt.plot(epochs_axis, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Baseline MLP Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/baseline_metrics_curve.png", dpi=200)
    plt.close()

    return model


def eval_baseline(
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

    print(f"[Baseline] Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    return acc, f1, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读数据 & 预处理
    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 2. 展平特征
    feat_df, y_all = build_flat_features_for_baseline(matches_df)
    X_all = feat_df.values.astype(np.float32)
    match_ids_all = matches_df["match_id"].values

    # 3. 按比赛划分 train/val/test
    train_matches, val_matches, test_matches = split_match_ids(match_ids_all)

    train_mask = mask_by_matches(match_ids_all, train_matches)
    val_mask = mask_by_matches(match_ids_all, val_matches)
    test_mask = mask_by_matches(match_ids_all, test_matches)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    print("Baseline train/val/test:", len(y_train), len(y_val), len(y_test))

    # 4. 训练
    model = train_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=X_train.shape[1],
        epochs=10,
        device=device,
    )

    # 5. 测试评估
    eval_baseline(model, X_test, y_test, device=device)

    # 6. 保存模型
    torch.save(model.state_dict(), "baseline_mlp.pth")
    print("Saved baseline model to baseline_mlp.pth")


if __name__ == "__main__":
    main()
