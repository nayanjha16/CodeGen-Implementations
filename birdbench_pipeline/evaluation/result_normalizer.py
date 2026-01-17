from typing import List, Dict, Any


def normalize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize query results for fair comparison.
    - Sort rows
    - Sort keys inside rows
    """
    normalized = []

    for row in results:
        normalized.append(dict(sorted(row.items())))

    # Sort rows for deterministic comparison
    normalized.sort(key=lambda x: tuple(x.values()))
    return normalized
