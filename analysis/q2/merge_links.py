import sys
from pathlib import Path

_here   = str(Path(__file__).resolve().parent)
_shared = str(Path(__file__).resolve().parents[1] / "shared")
for _p in (_here, _shared):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_index import normalize_hscode
from score_bundles import reliable_bundle_names


def collect_reliable_links(bundles: dict[str, dict], bundle_scores: list[dict], base_index: dict) -> list[dict]:
    """收集可靠预测集中的链接，并去掉与主图完全重复的边。"""
    reliable_names = reliable_bundle_names(bundle_scores)
    base_exact_edges = base_index["base_exact_edges"]
    collected = []
    seen_new_edges = set()

    for bundle_name in sorted(reliable_names):
        for link in bundles[bundle_name].get("links", []):
            edge_key = (
                link.get("source"),
                link.get("target"),
                link.get("arrivaldate"),
                normalize_hscode(link.get("hscode")),
            )

            # 完全重复的边不能帮助补全图谱；预测集之间重复也只保留一次。
            if edge_key in base_exact_edges or edge_key in seen_new_edges:
                continue

            new_link = dict(link)
            new_link["generated_by"] = new_link.get("generated_by", bundle_name)
            collected.append(new_link)
            seen_new_edges.add(edge_key)

    return collected
