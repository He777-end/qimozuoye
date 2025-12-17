# app.py

import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import SEQ_LEN
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    get_match_sequence,
    build_flat_features_for_baseline,
)
from src.models import LSTMPointPredictor, TransformerPointPredictor


@st.cache_data
def load_data():
    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)
    return matches_df


@st.cache_resource
def load_model(model_type: str, input_dim: int, device: str):
    if model_type == "LSTM":
        model = LSTMPointPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=1,
            dropout=0.3,
        ).to(device)
        weight_path = "lstm_point_predictor.pth"
    else:
        model = TransformerPointPredictor(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.3,
        ).to(device)
        weight_path = "transformer_point_predictor.pth"

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    st.title("Wimbledon Momentum Dashboard")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"Using device: **{device}**")

    matches_df = load_data()
    feat_df, _ = build_flat_features_for_baseline(matches_df)
    input_dim = feat_df.shape[1]

    # Sidebar: 模型选择
    model_type = st.sidebar.selectbox("Model Type", ["LSTM", "Transformer"])
    model = load_model(model_type, input_dim, device)

    # Sidebar: 比赛选择
    match_ids = matches_df["match_id"].unique().tolist()
    default_match = match_ids[0] if match_ids else None
    selected_match = st.sidebar.selectbox("Match ID", match_ids, index=0)

    st.write(f"### 模型: {model_type}, 比赛: `{selected_match}`")

    # 构造该比赛的序列
    X_m, y_m, meta_m = get_match_sequence(matches_df, selected_match, seq_len=SEQ_LEN)
    if X_m is None:
        st.warning("该比赛有效分数太少，无法构造序列。换一场试试。")
        return

    with torch.no_grad():
        xb = torch.from_numpy(X_m).float().to(device)
        logits = model(xb).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

    meta_m = meta_m.copy()
    meta_m["p_p1_win"] = probs
    meta_m["momentum_index"] = meta_m["p_p1_win"] - 0.5

    # 简单准确率统计
    if "y_point_p1_win" in meta_m.columns:
        y_true = meta_m["y_point_p1_win"].values
    else:
        # 如果 get_match_sequence 没带出来，可以从原 df 对齐
        y_true = matches_df.loc[meta_m.index, "y_point_p1_win"].values

    y_pred = (probs >= 0.5).astype(int)
    acc = (y_pred == y_true).mean()

    st.write(f"**这场比赛的逐分预测准确率:** {acc:.3f}")

    # 画动量曲线
    if "elapsed_seconds" in meta_m.columns and meta_m["elapsed_seconds"].notna().any():
        x = meta_m["elapsed_seconds"].values
        x_label = "Elapsed seconds"
    else:
        x = np.arange(len(meta_m))
        x_label = "Point index (from SEQ_LEN)"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, probs, marker="o", linewidth=1.5, label="P(P1 wins next point)")
    ax.axhline(0.5, linestyle="--", color="gray", label="0.5 threshold")

    # 标出 P1 实际赢的点
    win_mask = y_true == 1
    ax.scatter(x[win_mask], probs[win_mask], marker="^", label="P1 actually won", alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel("P(P1 wins next point)")
    ax.set_title(f"Momentum curve ({model_type}) - Match {selected_match}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.write("#### 原始数据预览（这场比赛后半部分）")
    st.dataframe(meta_m.tail(20))


if __name__ == "__main__":
    main()
