import numpy as np
import pandas as pd
import torch

from .data_utils import (
    build_flat_features_for_baseline,
    get_match_sequence,
)
from .config import SEQ_LEN


def compute_match_momentum(
    model: torch.nn.Module,
    df: pd.DataFrame,
    match_id: str,
    device: str = "cpu",
    seq_len: int = SEQ_LEN,
) -> pd.DataFrame:
    """
    对某场比赛计算 LSTM 预测的下一分获胜概率，作为动量指标。
    返回一个 DataFrame，至少包含：
      - match_id, set_no, game_no, point_no（如果原始数据有）
      - p_p1_win: 预测 player1 赢下一分的概率
      - momentum_index: p_p1_win - 0.5
      - y_point_p1_win: 真实标签（0/1，若能从 df 中恢复）
    """
    model.eval()

    # --- 1. 确保整体 df 中有 y_point_p1_win（以防外面忘了调用 preprocess_matches） ---
    df = df.copy()
    if "y_point_p1_win" not in df.columns:
        if "point_victor" in df.columns:
            df["y_point_p1_win"] = (df["point_victor"] == 1).astype(int)
        else:
            # 实在没有 point_victor 就没法恢复真实标签，只能后面不画 0/1 序列
            print("[compute_match_momentum] WARNING: no 'y_point_p1_win' or 'point_victor' in df.")
    
    # --- 2. 用 get_match_sequence 构建该场比赛的序列 + 元信息 ---
    X_seq, y_seq, meta_df = get_match_sequence(df, match_id, seq_len=seq_len)
    if X_seq is None:
        raise ValueError(f"Match {match_id} 样本太少，无法构造序列（需要 > seq_len 个点）")

    # --- 3. 用模型算出概率 ---
    with torch.no_grad():
        xb = torch.from_numpy(X_seq).to(device)
        logits = model(xb).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    meta_df = meta_df.copy()
    meta_df["p_p1_win"] = probs
    meta_df["momentum_index"] = meta_df["p_p1_win"] - 0.5

    # --- 4. 确保 meta_df 里也有 y_point_p1_win，用于画真实 0/1 序列 ---
    if "y_point_p1_win" not in meta_df.columns and "y_point_p1_win" in df.columns:
        # 用关键列对齐再 merge，一般都有这四个
        key_cols = [c for c in ["match_id", "set_no", "game_no", "point_no"] if c in meta_df.columns]
        if key_cols:
            meta_df = meta_df.merge(
                df[key_cols + ["y_point_p1_win"]],
                on=key_cols,
                how="left",
                suffixes=("", "_y"),
            )
        else:
            # 如果没有这些列，只能简单按行对齐（不推荐，但至少不报错）
            meta_df["y_point_p1_win"] = y_seq

    return meta_df


def build_reversal_dataset(
    model: torch.nn.Module,
    df,
    feat_df_full,
    device: str = "cpu",
    seq_len: int = SEQ_LEN,
):
    """
    构造“势头反转”监督数据：
    - 使用和训练 LSTM 时完全一致的 one-hot 特征列 feat_df_full
    - 逐场比赛计算动量 m_t = p_t - 0.5
    - sign(m_t) != sign(m_{t-1}) 视为 t 时刻发生反转
    - 用 [当前分的展平特征 + 当前 m_t] 预测下一分是否发生反转

    参数：
    - model: 已训练好的 LSTMPointPredictor
    - df: 完整的 matches_df（预处理后）
    - feat_df_full: 与 df 行完全对齐的 one-hot 特征 DataFrame（全局 get_dummies）
    - device: "cpu" 或 "cuda"
    - seq_len: LSTM 输入长度

    返回：
    - X_rev: (N, feat_dim + 1)  最后一个分的特征 + 当前动量 m_t
    - y_rev: (N,)               下一时刻是否反转 (0/1)
    """

    model.eval()

    # 确保行索引对齐
    df = df.reset_index(drop=True)
    feat_df_full = feat_df_full.reset_index(drop=True)

    feat_mat_full = feat_df_full.values.astype(np.float32)

    X_list = []
    y_list = []

    # 按 match_id 分组，但只用它的行号，到全局 feat_mat_full 里取特征
    grouped = df.groupby("match_id").indices

    for match_id, sub_idx in grouped.items():
        sub_idx = np.array(sub_idx)

        # 按 (set_no, game_no, point_no) 排序，保证时间顺序
        order = np.lexsort(
            (
                df.loc[sub_idx, "point_no"].values,
                df.loc[sub_idx, "game_no"].values,
                df.loc[sub_idx, "set_no"].values,
            )
        )
        idx_sorted = sub_idx[order]

        feats = feat_mat_full[idx_sorted]  # (L, feat_dim) —— 这里 feat_dim 一定是 48
        y_point = df.loc[idx_sorted, "y_point_p1_win"].values.astype(np.float32)

        # 至少要有 seq_len + 1 个点，才能：
        #  - 用前 seq_len 构造第一个窗口
        #  - 第一个窗口预测得到 m_seq_len，对应的 label 是 m_{seq_len+1}
        if len(idx_sorted) <= seq_len + 1:
            continue

        # 构造 LSTM 输入序列，用于计算 m_t (动量)
        X_seq = []
        for t in range(seq_len, len(idx_sorted)):
            X_seq.append(feats[t - seq_len : t])
        X_seq = np.stack(X_seq, axis=0)  # (T', seq_len, feat_dim)

        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(device)
            logits = model(xb).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))   # (T',)

        m = probs - 0.5              # 动量指数
        sign_m = np.sign(m)
        reversal = np.zeros_like(m, dtype=np.float32)
        reversal[1:] = (sign_m[1:] != sign_m[:-1]).astype(np.float32)

        # 构建监督样本：用时刻 t 的特征 + m_t 预测 t+1 是否反转
        # 注意 X_seq[t] 对应的是 df 中 index = seq_len + t 行之前的 seq_len 个点
        # 我们把“上一分”的展平特征取自 feats[index_in_df]
        for t in range(len(m) - 1):
            index_in_df = seq_len + t  # 在 idx_sorted 里的位置
            last_feat = feats[index_in_df]  # (feat_dim,)
            x_vec = np.concatenate([last_feat, [m[t]]])  # 拼上当前动量
            y_val = reversal[t + 1]                      # 下一时刻是否反转

            X_list.append(x_vec)
            y_list.append(y_val)

    if not X_list:
        raise RuntimeError("没有构造出任何势头反转样本，请检查 seq_len 与数据量。")

    X_rev = np.stack(X_list, axis=0)
    y_rev = np.array(y_list, dtype=np.float32)
    print("Reversal 数据集:", X_rev.shape)
    return X_rev, y_rev
