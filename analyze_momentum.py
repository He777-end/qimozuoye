"""
analyze_momentum.py

载入 LSTM 模型，对指定比赛画出“动量曲线”：
P(player1 wins next point) 随时间或随 point_no 的变化。
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.config import SEQ_LEN
from src.data_utils import (
    load_raw_data,
    preprocess_matches,
    build_flat_features_for_baseline,
)
from src.models import LSTMPointPredictor
from src.momentum import compute_match_momentum


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze momentum of a match using LSTM.")
    parser.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="match_id to analyze (如果不指定，会自动选择一场).",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Optional: 保存图片的文件名，例如 'momentum.png'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读数据 & 预处理
    matches_df, dict_df = load_raw_data()
    matches_df = preprocess_matches(matches_df)

    # 2. 计算 LSTM 输入维度 & 初始化模型
    feat_df, _ = build_flat_features_for_baseline(matches_df)
    input_dim = feat_df.shape[1]
    print("LSTM input_dim:", input_dim)

    lstm_model = LSTMPointPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
    ).to(device)

    lstm_state = torch.load("lstm_point_predictor.pth", map_location=device)
    lstm_model.load_state_dict(lstm_state)
    print("Loaded LSTM weights from lstm_point_predictor.pth")

    # 3. 确定要分析的比赛 match_id
    if args.match_id is not None:
        match_id = args.match_id
        if match_id not in matches_df["match_id"].unique():
            raise ValueError(f"Match id {match_id} not found in data.")
    else:
        # 默认选第一场
        match_id = matches_df["match_id"].iloc[0]
    print("Analyzing match:", match_id)

    # 4. 计算该比赛的动量 DataFrame
    momentum_df = compute_match_momentum(
        lstm_model,
        matches_df,
        match_id=match_id,
        device=device,
        seq_len=SEQ_LEN,
    )

    # 5. 画图：上面是概率曲线，下面是真实结果（0/1）序列
    if "elapsed_seconds" in momentum_df.columns and momentum_df["elapsed_seconds"].notna().any():
        x = momentum_df["elapsed_seconds"].values
        x_label = "Elapsed seconds"
    else:
        x = np.arange(len(momentum_df))
        x_label = "Point index (from SEQ_LEN)"

    y_prob = momentum_df["p_p1_win"].values
    # 确保把真实标签也带进来（前面 compute_match_momentum 返回的 meta_df 里如果没有，可以手动 merge 一下）
    y_true = momentum_df["y_point_p1_win"].values if "y_point_p1_win" in momentum_df.columns else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # (1) 概率曲线
    axes[0].plot(x, y_prob, marker="o", linewidth=1.5, label="P(P1 wins next point)")
    axes[0].axhline(0.5, linestyle="--", label="0.5 threshold")
    axes[0].set_ylabel("Predicted prob")
    axes[0].set_title(f"Momentum curve for match {match_id}")
    axes[0].legend()
    axes[0].grid(True)

    # (2) 真实 0/1 序列
    if y_true is not None:
        # 用 step 的方式画 0/1 序列更直观
        axes[1].step(x, y_true, where="post", label="Actual P1 win (0/1)")
        axes[1].set_ylim(-0.2, 1.2)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["P2 wins", "P1 wins"])
        axes[1].set_ylabel("Actual")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No y_point_p1_win column in momentum_df",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    axes[1].set_xlabel(x_label)
    plt.tight_layout()

    if args.save_fig:
        plt.savefig(args.save_fig, dpi=200)
        print("Saved figure to:", args.save_fig)
    else:
        plt.show()



if __name__ == "__main__":
    main()
