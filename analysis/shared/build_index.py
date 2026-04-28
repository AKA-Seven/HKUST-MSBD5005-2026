from collections import Counter


def normalize_hscode(value) -> str:
    """统一 HS 编码格式，避免整数和字符串比较失败。"""
    if value is None:
        return ""
    return str(value).strip()


def month_key(date_text: str | None) -> str:
    """把 YYYY-MM-DD 转成 YYYY-MM；缺失日期返回空字符串。"""
    if not date_text:
        return ""
    return date_text[:7]


def build_base_index(base_graph: dict) -> dict:
    """为主图建立轻量索引，供预测链接评估使用。"""
    nodes = base_graph.get("nodes", [])
    links = base_graph.get("links", [])

    base_nodes = {node["id"] for node in nodes if "id" in node}
    base_pairs = set()
    base_exact_edges = set()
    base_hscodes = set()
    pair_counts = Counter()
    dates = []
    # company -> [first_date, last_date]，用于时间一致性评估
    company_date_ranges: dict[str, list[str]] = {}

    for link in links:
        source = link.get("source")
        target = link.get("target")
        date = link.get("arrivaldate")
        hscode = normalize_hscode(link.get("hscode"))

        if source and target:
            pair = (source, target)
            base_pairs.add(pair)
            pair_counts[pair] += 1
            base_exact_edges.add((source, target, date, hscode))

        if hscode:
            base_hscodes.add(hscode)
        if date:
            dates.append(date)
            for node in (source, target):
                if not node:
                    continue
                if node not in company_date_ranges:
                    company_date_ranges[node] = [date, date]
                else:
                    if date < company_date_ranges[node][0]:
                        company_date_ranges[node][0] = date
                    if date > company_date_ranges[node][1]:
                        company_date_ranges[node][1] = date

    return {
        "base_nodes": base_nodes,
        "base_pairs": base_pairs,
        "base_exact_edges": base_exact_edges,
        "base_hscodes": base_hscodes,
        "pair_counts": pair_counts,
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "company_date_ranges": company_date_ranges,
    }
