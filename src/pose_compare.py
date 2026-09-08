"""Align two pose sequences in time (they won't be frame-synchronized -- two
takes of the same routine are never the same length or tempo) and score how
closely they match, frame by aligned frame.

Alignment uses Dynamic Time Warping (DTW): the standard technique for
comparing two time series that represent the same underlying signal at
different speeds/offsets -- exactly the case for two dance takes. A naive
same-index frame comparison would compare frame 50 of a slightly slower take
to the wrong frame of a slightly faster one and produce meaningless numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose_estimation import PoseFrame


def _frame_distance(a: PoseFrame, b: PoseFrame) -> float:
    """Mean per-joint Euclidean distance, weighted by how confidently each
    joint was detected in *both* frames -- an occluded ankle in one frame
    shouldn't be allowed to dominate the score."""
    if not a.detected or not b.detected:
        return np.nan
    weights = a.visibility * b.visibility
    if weights.sum() < 1e-6:
        return np.nan
    dists = np.linalg.norm(a.points - b.points, axis=1)
    return float(np.average(dists, weights=weights))


@dataclass
class AlignedPair:
    ref_idx: int
    cand_idx: int
    distance: float  # np.nan if either frame had no confident pose detection


def dtw_align(reference: list[PoseFrame], candidate: list[PoseFrame], band_frac: float = 0.35) -> list[AlignedPair]:
    """Classic DTW over the two frame sequences, cost = per-frame pose
    distance. `band_frac` caps how far the alignment can stray from the
    diagonal (as a fraction of sequence length) -- without it, DTW can
    degenerate into pathological alignments (e.g. matching one whole
    sequence to a single frame of the other) when two takes are very
    different lengths.
    """
    n, m = len(reference), len(candidate)
    band = max(1, int(band_frac * max(n, m)))

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    frame_cost = np.zeros((n, m))
    for i in range(n):
        for j in range(max(0, i - band), min(m, i + band + 1)):
            frame_cost[i, j] = _frame_distance(reference[i], candidate[j])

    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            fc = frame_cost[i - 1, j - 1]
            fc = 0.0 if np.isnan(fc) else fc  # missing detections: don't penalize, just pass through
            cost[i, j] = fc + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack from (n, m) to (0, 0) to recover the alignment path.
    path: list[AlignedPair] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append(AlignedPair(i - 1, j - 1, frame_cost[i - 1, j - 1]))
        step = min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
        if step == cost[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif step == cost[i - 1, j]:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


@dataclass
class ComparisonSummary:
    aligned_pairs: list[AlignedPair]
    mean_distance: float
    match_score_0_100: float
    worst_pairs: list[AlignedPair]  # top-5 highest-deviation aligned frames


def summarize(aligned: list[AlignedPair], top_k: int = 5) -> ComparisonSummary:
    valid = [p for p in aligned if not np.isnan(p.distance)]
    mean_dist = float(np.mean([p.distance for p in valid])) if valid else float("nan")
    # Empirically, normalized per-joint distance of ~0 is a perfect match and
    # ~1.5+ (roughly a torso-length of average joint deviation) reads as
    # "clearly different pose" -- scaled onto 0-100 for a human-readable score.
    match_score = max(0.0, 100.0 * (1.0 - mean_dist / 1.5)) if valid else 0.0
    worst = sorted(valid, key=lambda p: p.distance, reverse=True)[:top_k]
    return ComparisonSummary(aligned, mean_dist, match_score, worst)
