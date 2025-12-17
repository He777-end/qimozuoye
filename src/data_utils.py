import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    MATCH_CSV,
    DICT_CSV,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    RANDOM_STATE,
    SEQ_LEN,
)


# ---------- 基础工具 ----------

def time_to_seconds(t: str) -> float:
    """把 HH:MM:SS 转成秒（有异常返回 NaN）"""
    if pd.isna(t):
        return np.nan
    parts = str(t).split(":")
    if len(parts) != 3:
        return np.nan
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------- 数据加载 & 预处理 ----------

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取原始比赛数据和字典"""
    matches_df = pd.read_csv(MATCH_CSV)
    dict_df = pd.read_csv(DICT_CSV)
    return matches_df, dict_df


# 评分映射：网球 0/15/30/40/AD -> 等级编码 0..4
_SCORE_MAP = {
    "0": 0,
    0: 0,
    "15": 1,
    15: 1,
    "30": 2,
    30: 2,
    "40": 3,
    40: 3,
    "AD": 4,
    "Ad": 4,
    "A": 4,
}

def _convert_score_column(series: pd.Series) -> pd.Series:
    """
    将 p1_score / p2_score 这一类网球比分列从 '0','15','30','40','AD'
    映射为 0..4 的整数，便于统计和建模。
    """
    s = series.copy()
    # 先统一成字符串，strip 一下
    s = s.astype(str).str.strip()
    s = s.map(_SCORE_MAP).astype("float32")
    return s


def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始数据做基础清洗：
    - 处理时间
    - 将目标转成 0/1
    - 缺失值填充
    - 将比分型列（p1_score/p2_score）转换为数值
    """

    df = df.copy()

    # -------- 1) 时间 -> 秒 --------
    if "elapsed_time" in df.columns:
        from .data_utils import time_to_seconds  # 避免循环导入报错时你可以挪到文件顶部

        df["elapsed_seconds"] = df["elapsed_time"].apply(time_to_seconds)
    else:
        df["elapsed_seconds"] = np.nan

    # -------- 2) 将比分列转换为数值 --------
    # 注意：config.NUMERIC_COLS 里目前包含 p1_score/p2_score，我们在这里先把它变成数字
    for score_col in ["p1_score", "p2_score"]:
        if score_col in df.columns:
            df[score_col] = _convert_score_column(df[score_col])

    # -------- 3) 目标：player1 是否赢这一分 --------
    # TARGET_COL = "point_victor": 1 -> p1 赢，2 -> p2 赢
    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL {TARGET_COL} 不在数据中，请检查列名。")

    df["y_point_p1_win"] = (df[TARGET_COL] == 1).astype("float32")

    # -------- 4) 数值列缺失填中位数 --------
    numeric_cols = NUMERIC_COLS + ["elapsed_seconds"]
    for col in numeric_cols:
        if col in df.columns:
            # 有些列在 CSV 中可能是 object，这里强制转成 float
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # -------- 5) 类别列缺失填 Unknown --------
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")

    return df


# ---------- 特征工程：one-hot 展平特征（基线 & LSTM 都会用到） ----------

def build_flat_features_for_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    对类别列做 one-hot，返回：
    - feat_df: 展平后的特征 DataFrame
    - y_all:   目标 y (player1 是否赢这一分)
    """
    use_cols = NUMERIC_COLS + ["elapsed_seconds"] + CATEGORICAL_COLS
    feat_df = df[use_cols].copy()

    # one-hot 编码
    feat_df = pd.get_dummies(feat_df, columns=CATEGORICAL_COLS, drop_first=False)

    y_all = df["y_point_p1_win"].values.astype(np.float32)
    return feat_df, y_all


# ---------- LSTM 序列构造 ----------

def build_sequences_for_lstm(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据比赛顺序构造 LSTM 序列：
    返回：
    - X_seq: (N, seq_len, feat_dim)
    - y_seq: (N,)
    - match_ids_seq: (N,) 每个样本属于哪一场比赛
    """
    feat_df, _ = build_flat_features_for_baseline(df)
    df = df.reset_index(drop=True)
    feat_mat = feat_df.values.astype(np.float32)
    y_point = df["y_point_p1_win"].values.astype(np.float32)
    match_ids = df["match_id"].values

    X_list, y_list, match_list = [], [], []

    # groupby 得到每场比赛的索引
    grouped = df.groupby("match_id").indices

    for match_id, indices in grouped.items():
        idx = np.array(indices)

        # 按 set_no, game_no, point_no 排序
        order = np.lexsort(
            (
                df.loc[idx, "point_no"].values,
                df.loc[idx, "game_no"].values,
                df.loc[idx, "set_no"].values,
            )
        )
        idx = idx[order]

        match_feats = feat_mat[idx]
        match_y = y_point[idx]

        if len(idx) <= seq_len:
            continue

        # 滑窗：用前 seq_len 个点预测第 t 个点
        for t in range(seq_len, len(idx)):
            X_list.append(match_feats[t - seq_len : t])
            y_list.append(match_y[t])
            match_list.append(match_id)

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.array(y_list, dtype=np.float32)
    match_ids_seq = np.array(match_list)

    print("LSTM 序列形状:", X_seq.shape)
    return X_seq, y_seq, match_ids_seq


# ---------- 按比赛划分 train/val/test ----------

def split_match_ids(
    match_ids_seq: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定每个样本的 match_id，先取并集得到所有比赛，
    然后随机拆分为 train / val / test 三部分（按照比赛维度）.
    """
    unique_matches = np.unique(match_ids_seq)
    temp_ratio = val_ratio + test_ratio
    train_ratio = 1.0 - temp_ratio

    train_matches, temp_matches = train_test_split(
        unique_matches,
        test_size=temp_ratio,
        random_state=random_state,
        shuffle=True,
    )

    # 在剩下的 temp 中，按比例切成 val / test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_matches, test_matches = train_test_split(
        temp_matches,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    return train_matches, val_matches, test_matches


def mask_by_matches(match_ids: np.ndarray, keep_matches: np.ndarray) -> np.ndarray:
    keep_set = set(keep_matches.tolist())
    return np.array([m in keep_set for m in match_ids])


# ---------- 动量 & 势头反转用到的辅助 ----------

def get_match_sequence(
    df: pd.DataFrame,
    match_id: str,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | tuple[None, None, None]:
    """
    返回某一场比赛的 LSTM 序列输入和标签：
    - X_seq: (N, seq_len, feat_dim)
    - y_seq: (N,)
    - meta_df: 对应的元信息（从第 seq_len 个点开始）
    """
    sub_df = df[df["match_id"] == match_id].copy()
    if sub_df.empty:
        return None, None, None

    sub_df = sub_df.sort_values(by=["set_no", "game_no", "point_no"])
    feat_df, _ = build_flat_features_for_baseline(sub_df)
    feat_mat = feat_df.values.astype(np.float32)
    y_point = sub_df["y_point_p1_win"].values.astype(np.float32)

    if len(sub_df) <= seq_len:
        return None, None, None

    X_list, y_list, idx_list = [], [], []
    for t in range(seq_len, len(sub_df)):
        X_list.append(feat_mat[t - seq_len : t])
        y_list.append(y_point[t])
        idx_list.append(t)

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.array(y_list, dtype=np.float32)
    meta_df = sub_df.iloc[idx_list].reset_index(drop=True)
    return X_seq, y_seq, meta_df


# src/data_utils.py 末尾追加

def build_multistep_sequences(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造多步预测用的序列数据：
    - X: (N, seq_len, feat_dim)
    - y: (N, horizon)  表示未来 horizon 个点是否由 P1 赢
    - match_ids_seq: (N,)  每个样本对应的 match_id

    对于每个 match:
      使用 [t-seq_len, ..., t-1] 的特征去预测 [t, t+1, ..., t+horizon-1] 的结果
    """
    feat_df, _ = build_flat_features_for_baseline(df)
    df = df.reset_index(drop=True)

    feat_mat = feat_df.values.astype(np.float32)
    y_point = df["y_point_p1_win"].values.astype(np.float32)
    match_ids = df["match_id"].values

    X_list, y_list, match_list = [], [], []

    grouped = df.groupby("match_id").indices

    for match_id, indices in grouped.items():
        idx = np.array(indices)

        order = np.lexsort(
            (
                df.loc[idx, "point_no"].values,
                df.loc[idx, "game_no"].values,
                df.loc[idx, "set_no"].values,
            )
        )
        idx = idx[order]

        match_feats = feat_mat[idx]
        match_y = y_point[idx]

        # 需要至少 seq_len + horizon 个点
        if len(idx) <= seq_len + horizon:
            continue

        # t 是预测起点，y 为 [t ... t+horizon-1]
        for t in range(seq_len, len(idx) - horizon + 1):
            X_list.append(match_feats[t - seq_len : t])
            y_list.append(match_y[t : t + horizon])
            match_list.append(match_id)

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.stack(y_list, axis=0)   # (N, horizon)
    match_ids_seq = np.array(match_list)

    print("Multi-step 序列形状:", X_seq.shape, ", y 形状:", y_seq.shape)
    return X_seq, y_seq, match_ids_seq
