from pathlib import Path

# 数据路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

MATCH_CSV = DATA_DIR / "2024_Wimbledon_featured_matches.csv"
DICT_CSV  = DATA_DIR / "2024_data_dictionary.csv"

# 随机种子
RANDOM_STATE = 42

# LSTM 序列长度
SEQ_LEN = 20

# 数值特征列（请根据实际 csv 再稍微核对一下列名）
NUMERIC_COLS = [
    "set_no", "game_no", "point_no",
    "p1_sets", "p2_sets",
    "p1_games", "p2_games",
    "p1_score", "p2_score",
    "p1_points_won", "p2_points_won",
    "p1_ace", "p2_ace",
    "p1_double_fault", "p2_double_fault",
    "p1_unf_err", "p2_unf_err",
    "p1_net_pt", "p2_net_pt",
    "p1_net_pt_won", "p2_net_pt_won",
    "p1_break_pt", "p2_break_pt",
    "p1_break_pt_won", "p2_break_pt_won",
    "p1_break_pt_missed", "p2_break_pt_missed",
    "p1_distance_run", "p2_distance_run",
    "rally_count", "speed_mph",
]

# 类别特征列
CATEGORICAL_COLS = [
    "server",        # 谁发球
    "serve_no",      # 一发/二发
    "serve_width",
    "serve_depth",
    "return_depth",
]

# 目标列
TARGET_COL = "point_victor"  # 1 表 player1 赢, 2 表 player2
