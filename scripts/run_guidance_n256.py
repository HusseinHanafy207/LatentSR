"""N=256 confirmation: same protocol as n=64, start_index=0, resume on.

Do not look at running means. Pre-registered pass/fail is in
``CONFIRMATION_N256`` (printed at start).

One GPU, four jobs in order: baseline, late50, late200, late800.
Re-run the same command after a disconnect; ``--resume`` skips completed
``val_index``.

When all four CSVs have n=256::

    python scripts/run_guidance_n256.py --job compare --output-root ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

JOBS = {
    "baseline": {
        "condition": "baseline",
        "lambda_g": 0.0,
        "output": "eval_guidance_n256_baseline",
    },
    "late50": {
        "condition": "late",
        "lambda_g": 50.0,
        "output": "eval_guidance_n256_late_l50",
    },
    "late200": {
        "condition": "late",
        "lambda_g": 200.0,
        "output": "eval_guidance_n256_late_l200",
    },
    "late800": {
        "condition": "late",
        "lambda_g": 800.0,
        "output": "eval_guidance_n256_late_l800",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N=256 late-window confirmation.")
    parser.add_argument("--job", required=True, choices=(*JOBS, "compare"))
    parser.add_argument("--sr-checkpoint", type=Path, default=None)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _compare_all(output_root: Path) -> None:
    from latentsr.metrics.guidance_eval import compare_guidance_conditions

    base = output_root / JOBS["baseline"]["output"] / "per_image.csv"
    if not base.exists():
        raise SystemExit(f"missing baseline CSV: {base}")
    for job_name in ("late50", "late200", "late800"):
        spec = JOBS[job_name]
        cand = output_root / spec["output"] / "per_image.csv"
        if not cand.exists():
            raise SystemExit(f"missing {job_name} CSV: {cand}")
        out = output_root / f"{spec['output']}_vs_baseline"
        compare_guidance_conditions(
            base,
            cand,
            output_dir=out,
            baseline_name="baseline",
            candidate_name=job_name,
        )
        print(f"Paired comparison: {out}")


def main() -> None:
    args = parse_args()
    if args.job == "compare":
        _compare_all(args.output_root)
        return
    if args.sr_checkpoint is None or args.vae_checkpoint is None:
        raise SystemExit("--sr-checkpoint and --vae-checkpoint are required")
    job = JOBS[args.job]
    out = args.output_root / job["output"]
    argv = [
        "evaluate_guidance.py",
        "--sr-checkpoint",
        str(args.sr_checkpoint),
        "--vae-checkpoint",
        str(args.vae_checkpoint),
        "--config",
        str(args.config),
        "--condition",
        job["condition"],
        "--output-dir",
        str(out),
        "--seed",
        str(args.seed),
        "--num-images",
        str(args.num_images),
        "--start-index",
        str(args.start_index),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--resume",
        "--no-download" if not args.download else "--download",
    ]
    if job["condition"] != "baseline":
        argv.extend(["--lambda-g", str(job["lambda_g"])])
    if args.data_dir is not None:
        argv.extend(["--data-dir", str(args.data_dir)])
    sys.argv = argv
    eval_path = Path(__file__).resolve().parent / "evaluate_guidance.py"
    import runpy

    runpy.run_path(str(eval_path), run_name="__main__")


if __name__ == "__main__":
    main()
