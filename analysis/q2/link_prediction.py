import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    LINK_PREDICTION_SAMPLE_SIZE,
    LINK_PREDICTION_TRAIN_END,
    LINK_PREDICTION_VALID_END,
    LINK_PREDICTION_VALID_START,
    RANDOM_SEED,
)


def _add_neighbor(neighbors: dict[str, set[str]], source: str, target: str) -> None:
    """按无向图建邻居表，因为贸易关系的相似性主要看共同上下游。"""
    if not source or not target or source == target:
        return
    neighbors[source].add(target)
    neighbors[target].add(source)


def build_temporal_link_index(base_graph: dict) -> dict:
    """按时间切分主图，构造训练期图结构和 2034 验证边。"""
    neighbors = defaultdict(set)
    train_pair_counts = Counter()
    validation_pairs = set()
    all_pairs = set()
    nodes = set()

    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        date = link.get("arrivaldate")
        if not source or not target or source == target or not date:
            continue

        pair = tuple(sorted((source, target)))
        all_pairs.add(pair)
        nodes.update(pair)

        if date <= LINK_PREDICTION_TRAIN_END:
            _add_neighbor(neighbors, source, target)
            train_pair_counts[pair] += 1
        elif LINK_PREDICTION_VALID_START <= date <= LINK_PREDICTION_VALID_END:
            validation_pairs.add(pair)

    return {
        "neighbors": dict(neighbors),
        "train_pair_counts": train_pair_counts,
        "validation_pairs": validation_pairs,
        "all_pairs": all_pairs,
        "nodes": sorted(nodes),
    }


def _common_neighbors(source: str, target: str, neighbors: dict[str, set[str]]) -> set[str]:
    """用较小的邻居集合求交，降低高连接公司带来的计算成本。"""
    left = neighbors.get(source, set())
    right = neighbors.get(target, set())
    if len(left) > len(right):
        left, right = right, left
    return left & right


def pair_features(source: str, target: str, index: dict) -> list[float]:
    """经典链接预测特征：共同邻居、Jaccard、Adamic-Adar、资源分配等。"""
    neighbors = index["neighbors"]
    pair = tuple(sorted((source, target)))
    source_degree = len(neighbors.get(source, set()))
    target_degree = len(neighbors.get(target, set()))
    common = _common_neighbors(source, target, neighbors)
    union_size = source_degree + target_degree - len(common)

    adamic_adar = 0.0
    resource_allocation = 0.0
    for node in common:
        degree = len(neighbors.get(node, set()))
        if degree > 1:
            adamic_adar += 1 / math.log(degree)
            resource_allocation += 1 / degree

    return [
        len(common),
        len(common) / union_size if union_size else 0.0,
        adamic_adar,
        resource_allocation,
        source_degree * target_degree,
        min(source_degree, target_degree),
        max(source_degree, target_degree),
        index["train_pair_counts"].get(pair, 0),
    ]


def _sample_negatives(
    n: int,
    existing_pairs: set,
    nodes: list[str],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """从图中随机采样 n 个不存在的公司对作为负样本。"""
    negatives: set[tuple[str, str]] = set()
    max_attempts = max(1000, n * 30)
    for _ in range(max_attempts):
        if len(negatives) >= n:
            break
        source, target = rng.sample(nodes, 2)
        pair = tuple(sorted((source, target)))
        if pair not in existing_pairs:
            negatives.add(pair)
    return list(negatives)


def _sample_training_pairs(
    index: dict,
) -> tuple[list[tuple[str, str]], list[int], list[tuple[str, str]], list[int]]:
    """把 2034 真实边做正样本、随机未见对做负样本，按 80/20 拆分训练集和留出验证集。

    返回 (train_pairs, train_labels, val_pairs, val_labels)，
    验证集与训练集完全不重叠，避免 AUC 在训练数据上估计的乐观偏差。
    """
    rng = random.Random(RANDOM_SEED)
    positives = list(index["validation_pairs"])
    rng.shuffle(positives)
    positives = positives[:LINK_PREDICTION_SAMPLE_SIZE]

    split = max(1, int(len(positives) * 0.8))
    pos_train, pos_val = positives[:split], positives[split:]

    existing_pairs = index["all_pairs"]
    nodes = index["nodes"]
    neg_train = _sample_negatives(len(pos_train), existing_pairs, nodes, rng)
    neg_val = _sample_negatives(len(pos_val), existing_pairs, nodes, rng)

    train_pairs = pos_train + neg_train
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    val_pairs = pos_val + neg_val
    val_labels = [1] * len(pos_val) + [0] * len(neg_val)
    return train_pairs, train_labels, val_pairs, val_labels


def train_link_prediction_model(base_graph: dict) -> dict:
    """自监督训练链接预测模型，并返回模型、索引和留出验证 AUC。"""
    index = build_temporal_link_index(base_graph)
    train_pairs, train_labels, val_pairs, val_labels = _sample_training_pairs(index)

    if len(set(train_labels)) < 2 or len(set(val_labels)) < 2:
        return {
            "model": None,
            "index": index,
            "validation_auc": None,
            "training_pairs": len(train_pairs),
        }

    X_train = np.array([pair_features(s, t, index) for s, t in train_pairs], dtype=float)
    y_train = np.array(train_labels, dtype=int)
    X_val = np.array([pair_features(s, t, index) for s, t in val_pairs], dtype=float)
    y_val = np.array(val_labels, dtype=int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    model.fit(X_train, y_train)

    # 在与训练集不重叠的留出集上计算 AUC，避免循环验证的乐观偏差。
    val_probs = model.predict_proba(X_val)[:, 1]
    validation_auc = roc_auc_score(y_val, val_probs)

    return {
        "model": model,
        "index": index,
        "validation_auc": round(float(validation_auc), 4),
        "training_pairs": len(train_pairs),
    }


def score_bundle_links(bundle_name: str, bundle_graph: dict, model_info: dict) -> dict:
    """用自监督模型为某个预测集打分，分数越高越像 2034 真实会出现的边。"""
    model = model_info["model"]
    index = model_info["index"]
    links = bundle_graph.get("links", [])

    if not links or model is None:
        return {
            "bundle": bundle_name,
            "ml_link_probability": 0.0,
            "ml_validation_auc": model_info["validation_auc"],
            "ml_training_pairs": model_info["training_pairs"],
        }

    features = np.array(
        [pair_features(link.get("source"), link.get("target"), index) for link in links],
        dtype=float,
    )
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "bundle": bundle_name,
        "ml_link_probability": round(float(probabilities.mean()), 4),
        "ml_link_probability_p90": round(float(np.quantile(probabilities, 0.9)), 4),
        "ml_validation_auc": model_info["validation_auc"],
        "ml_training_pairs": model_info["training_pairs"],
    }


def score_all_bundle_links(base_graph: dict, bundles: dict[str, dict]) -> list[dict]:
    """训练一次自监督模型，然后批量评分 12 组预测链接。"""
    model_info = train_link_prediction_model(base_graph)
    return [
        score_bundle_links(bundle_name, bundle_graph, model_info)
        for bundle_name, bundle_graph in sorted(bundles.items())
    ]
