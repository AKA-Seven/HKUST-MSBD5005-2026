import json
from pathlib import Path

from config import BASE_GRAPH_PATH, BUNDLE_DIR


def load_json(path: Path) -> dict:
    """读取 JSON 文件。MC2 文件较大，调用方应避免重复读取主图。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_base_graph(path: Path = BASE_GRAPH_PATH) -> dict:
    """读取 MC2 主知识图谱。"""
    return load_json(path)


def load_bundles(bundle_dir: Path = BUNDLE_DIR) -> dict[str, dict]:
    """读取 12 组预测链接，返回 {bundle_name: graph}。"""
    bundles = {}
    for path in sorted(bundle_dir.glob("*.json")):
        bundles[path.stem] = load_json(path)
    return bundles
