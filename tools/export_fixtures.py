from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any


X_TRAIN: list[list[float]] = [
    [0.0, 0.0],
    [0.2, 0.1],
    [4.8, 5.0],
    [5.1, 4.9],
    [9.9, 10.2],
    [10.1, 9.8],
]
Y_TRAIN: list[int] = [0, 0, 1, 1, 2, 2]
X_QUERY: list[list[float]] = [
    [0.1, 0.0],
    [4.9, 5.2],
    [10.0, 10.0],
    [6.0, 6.0],
]
EPS = 1e-12
K = 3
METRIC = "euclidean"


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def safe_divide(numerator: float, denominator: float, default: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or abs(denominator) <= sys.float_info.epsilon
    ):
        return default
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else default


def fit_range_normaliser(x: list[list[float]], eps: float) -> tuple[list[float], list[float], float]:
    if not x:
        return ([], [], max(0.0, eps))

    n_cols = len(x[0])
    mins = [0.0] * n_cols
    maxs = [0.0] * n_cols

    for col_idx in range(n_cols):
        col_values = [row[col_idx] for row in x if math.isfinite(row[col_idx])]
        if col_values:
            mins[col_idx] = min(col_values)
            maxs[col_idx] = max(col_values)

    return (mins, maxs, max(0.0, eps))


def transform_range_normaliser(
    x: list[list[float]], mins: list[float], maxs: list[float], eps: float
) -> list[list[float]]:
    transformed: list[list[float]] = []
    for row in x:
        out_row: list[float] = []
        for col_idx, value in enumerate(row):
            numerator = value - mins[col_idx] if math.isfinite(value) else 0.0
            denominator = maxs[col_idx] - mins[col_idx] + eps
            scaled = safe_divide(numerator, denominator, 0.0)
            out_row.append(clamp01(scaled))
        transformed.append(out_row)
    return transformed


def euclidean_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("distance feature mismatch")
    squared_sum = 0.0
    for left_value, right_value in zip(left, right):
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise ValueError("non-finite value encountered")
        delta = left_value - right_value
        squared_sum += delta * delta
    return math.sqrt(squared_sum)


def query_knn(
    x_train: list[list[float]], x_query: list[list[float]], k: int
) -> tuple[list[list[int]], list[list[float]]]:
    effective_k = min(k, len(x_train))
    if effective_k == 0:
        return (
            [[ ] for _ in x_query],
            [[ ] for _ in x_query],
        )

    all_indices: list[list[int]] = []
    all_distances: list[list[float]] = []
    for query_row in x_query:
        row_distances = [
            (train_idx, euclidean_distance(query_row, train_row))
            for train_idx, train_row in enumerate(x_train)
        ]
        row_distances.sort(key=lambda item: (item[1], item[0]))

        chosen = row_distances[:effective_k]
        all_indices.append([idx for idx, _ in chosen])
        all_distances.append([distance for _, distance in chosen])

    return (all_indices, all_distances)


def safe_normalize_rows(matrix: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        n_cols = len(row)
        if n_cols == 0:
            normalized.append([])
            continue

        sanitized = [value if math.isfinite(value) and value > 0.0 else 0.0 for value in row]
        row_sum = sum(sanitized)
        if row_sum > 0.0 and math.isfinite(row_sum):
            normalized.append([value / row_sum for value in sanitized])
        else:
            normalized.append([1.0 / n_cols for _ in range(n_cols)])

    return normalized


def nn_scores(
    knn_indices: list[list[int]],
    knn_distances: list[list[float]],
    y_train: list[int],
    n_classes: int,
) -> list[list[float]]:
    scores: list[list[float]] = []
    for row_indices, row_distances in zip(knn_indices, knn_distances):
        out = [0.0 for _ in range(n_classes)]
        total_weight = 0.0
        for train_idx, distance in zip(row_indices, row_distances):
            weight = 1.0 if distance <= 0.0 else 1.0 / (1.0 + distance)
            out[y_train[train_idx]] += weight
            total_weight += weight

        if total_weight > 0.0 and math.isfinite(total_weight):
            out = [value / total_weight for value in out]
        scores.append(out)

    return scores


def frnn_scores(
    knn_indices: list[list[int]],
    knn_distances: list[list[float]],
    y_train: list[int],
    n_classes: int,
) -> list[list[float]]:
    raw_scores: list[list[float]] = []
    for row_indices, row_distances in zip(knn_indices, knn_distances):
        out = [0.0 for _ in range(n_classes)]
        for class_idx in range(n_classes):
            upper_approx = 0.0
            lower_approx = 1.0
            for train_idx, distance in zip(row_indices, row_distances):
                similarity = clamp01(1.0 / (1.0 + distance)) if distance >= 0.0 and math.isfinite(distance) else 0.0
                if y_train[train_idx] == class_idx:
                    upper_approx = max(upper_approx, similarity)
                else:
                    lower_approx = min(lower_approx, clamp01(1.0 - clamp01(similarity)))

            out[class_idx] = clamp01(0.5 * (lower_approx + upper_approx))
        raw_scores.append(out)

    return safe_normalize_rows(raw_scores)


def try_import_python_implementation(repo_root: Path) -> str:
    sys.path.insert(0, str(repo_root))
    try:
        importlib.import_module("frlearn")
        return "frlearn"
    except Exception:
        return "embedded_reference"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    backend = try_import_python_implementation(repo_root)

    mins, maxs, eps = fit_range_normaliser(X_TRAIN, EPS)
    x_train_norm = transform_range_normaliser(X_TRAIN, mins, maxs, eps)
    x_query_norm = transform_range_normaliser(X_QUERY, mins, maxs, eps)

    knn_indices, knn_distances = query_knn(x_train_norm, x_query_norm, K)
    n_classes = (max(Y_TRAIN) + 1) if Y_TRAIN else 0
    nn = nn_scores(knn_indices, knn_distances, Y_TRAIN, n_classes)
    frnn = frnn_scores(knn_indices, knn_distances, Y_TRAIN, n_classes)

    fixture_dir = repo_root / "rust" / "fixtures" / "python_reference"
    write_json(
        fixture_dir / "x_norm.json",
        {
            "backend": backend,
            "eps": eps,
            "x_train": X_TRAIN,
            "x_query": X_QUERY,
            "x_train_norm": x_train_norm,
            "x_query_norm": x_query_norm,
        },
    )
    write_json(
        fixture_dir / "knn.json",
        {
            "backend": backend,
            "metric": METRIC,
            "k": K,
            "indices": knn_indices,
            "distances": knn_distances,
        },
    )
    write_json(
        fixture_dir / "scores.json",
        {
            "backend": backend,
            "metric": METRIC,
            "k": K,
            "y_train": Y_TRAIN,
            "nn_scores": nn,
            "frnn_scores": frnn,
        },
    )

    print(f"Wrote fixtures to {fixture_dir}")
    print(f"Python implementation source: {backend}")


if __name__ == "__main__":
    main()
