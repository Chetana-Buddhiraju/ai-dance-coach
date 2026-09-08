"""Compare two dance takes: extract pose sequences, align them with DTW,
score the match, and render a side-by-side skeleton comparison GIF.

Usage:
    python -m src.compare_videos --reference ref.mp4 --candidate take.mp4 --out outputs/comparison.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pose_estimation import extract_pose_sequence
from src.pose_compare import dtw_align, summarize
from src.visualize import build_comparison_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to the reference/sample take")
    parser.add_argument("--candidate", required=True, help="Path to the take being evaluated")
    parser.add_argument("--out", default="outputs/comparison.gif")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Extracting pose sequence from reference: {args.reference}")
    ref_seq = extract_pose_sequence(args.reference, sample_fps=args.sample_fps)
    print(f"  {len(ref_seq)} sampled frames, {sum(f.detected for f in ref_seq)} with a detected pose")

    print(f"Extracting pose sequence from candidate: {args.candidate}")
    cand_seq = extract_pose_sequence(args.candidate, sample_fps=args.sample_fps)
    print(f"  {len(cand_seq)} sampled frames, {sum(f.detected for f in cand_seq)} with a detected pose")

    print("Aligning sequences with DTW...")
    aligned = dtw_align(ref_seq, cand_seq)
    summary = summarize(aligned)

    print(f"Mean pose distance (post-alignment): {summary.mean_distance:.4f}")
    print(f"Match score: {summary.match_score_0_100:.1f} / 100")
    print("Frame ranges of largest deviation:")
    for pair in summary.worst_pairs:
        print(f"  ref@{ref_seq[pair.ref_idx].timestamp_s:.1f}s <-> "
              f"candidate@{cand_seq[pair.cand_idx].timestamp_s:.1f}s  (distance={pair.distance:.3f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_comparison_gif(ref_seq, cand_seq, aligned, str(out_path))
    print(f"Saved comparison GIF to {out_path}")

    report_path = out_path.with_suffix(".json")
    report = {
        "reference": Path(args.reference).name,
        "candidate": Path(args.candidate).name,
        "mean_distance": summary.mean_distance,
        "match_score_0_100": summary.match_score_0_100,
        "worst_frame_pairs": [
            {"ref_timestamp_s": ref_seq[p.ref_idx].timestamp_s,
             "candidate_timestamp_s": cand_seq[p.cand_idx].timestamp_s,
             "distance": p.distance}
            for p in summary.worst_pairs
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
