from pathlib import Path


# 项目根目录：shared/ → analysis/ → repo root（parents[2]）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MC2 原始数据位置（仓库根目录下的 MC2/）。
MC2_ROOT = PROJECT_ROOT / "MC2"
BASE_GRAPH_PATH = MC2_ROOT / "mc2_challenge_graph.json"
BUNDLE_DIR = MC2_ROOT / "bundles"

# 挖掘结果输出根目录（按问题分子目录）。
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
OUTPUT_DIR_Q1 = OUTPUT_DIR / "q1"
OUTPUT_DIR_Q2 = OUTPUT_DIR / "q2"
OUTPUT_DIR_Q3 = OUTPUT_DIR / "q3"
OUTPUT_DIR_Q4 = OUTPUT_DIR / "q4"

# 主图贸易记录的有效日期范围会从数据中计算，这里只作为异常检测的兜底边界。
DEFAULT_DATE_MIN = "2028-01-01"
DEFAULT_DATE_MAX = "2034-12-30"

# 海产品相关 HS 编码前缀。数据中 hscode 有时是整数、有时是字符串，所以统一转字符串比较。
FISH_HSCODE_PREFIXES = (
    "301",
    "302",
    "303",
    "304",
    "305",
    "306",
    "307",
    "308",
    "1604",
    "1605",
)

# 自监督链接预测：用 2028-2033 学习图结构，用 2034 的真实边做验证。
LINK_PREDICTION_TRAIN_END = "2033-12-31"
LINK_PREDICTION_VALID_START = "2034-01-01"
LINK_PREDICTION_VALID_END = "2034-12-31"
LINK_PREDICTION_SAMPLE_SIZE = 6000
RANDOM_SEED = 42
