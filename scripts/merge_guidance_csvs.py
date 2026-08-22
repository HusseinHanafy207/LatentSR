"""Merge guidance shard CSVs (disjoint val_index) into one output dir."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from latentsr.metrics.guidance_eval import (
    accumulate_scores_from_rows,
    load_guidance_checkpoint,
)
from latentsr.metrics.image_metrics import (
    summarize_values,
    write_per_image_csv,
    write_summary_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge guidance per_image shards.")
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_index: dict[int, dict] = {}
    traj: list[dict] = []
    for shard in args.shard:
        rows, traj_rows, _done = load_guidance_checkpoint(
            shard, start_index=args.start_index, num_images=args.num_images
        )
        for row in rows:
            by_index[int(row["val_index"])] = row
        traj.extend(traj_rows)
    merged = [by_index[i] for i in sorted(by_index)]
    if len(merged) != args.num_images:
        print(f"warning: merged n={len(merged)}, expected {args.num_images}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_image_csv(args.output_dir / "per_image.csv", merged)
    if traj:
        traj.sort(key=lambda r: (int(r["val_index"]), int(float(r["t"]))))
        write_per_image_csv(args.output_dir / "trajectory.csv", traj)
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    accumulate_scores_from_rows(merged, scores)
    summary = {
        method: {metric: summarize_values(vals) for metric, vals in metric_map.items()}
        for method, metric_map in scores.items()
    }
    extra = {}
    if merged:
        extra = {
            "condition": merged[0].get("condition"),
            "lambda_g": merged[0].get("lambda_g"),
            "merged_shards": [str(p) for p in args.shard],
        }
    write_summary_files(args.output_dir, summary, len(merged), extra=extra)
    print(f"Wrote {args.output_dir}  n={len(merged)}")


if __name__ == "__main__":
    main()
