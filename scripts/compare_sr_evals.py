"""Paired stats + transfer scatters for two LatentSR per_image.csv files.

Re-evaluate both DDPMs first (same 64 val images, --seed 42, include soft-decode),
then:

  python scripts/compare_sr_evals.py \\
    --baseline outputs/eval_sr_vae1_paired/per_image.csv \\
    --candidate outputs/eval_sr_q2_paired/per_image.csv \\
    --output-dir outputs/eval_sr_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from latentsr.metrics.paired_stats import (
    compare_per_image,
    format_comparison_table,
    load_per_image_csv,
    write_comparison_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired permutation tests + bootstrap CIs + Q2 transfer scatters "
            "from two per_image.csv files."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="VAE-1 LatentSR per_image.csv",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="VAE-SR LatentSR per_image.csv",
    )
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_sr_compare"),
    )
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_per_image_csv(args.baseline)
    candidate = load_per_image_csv(args.candidate)
    result = compare_per_image(
        baseline,
        candidate,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    paths = write_comparison_outputs(result, args.output_dir)
    print(format_comparison_table(result))
    print()
    print(f"Wrote paired analysis under: {args.output_dir}")
    print(f"  {paths['table']}")
    print(f"  {paths['json']}")
    print(f"  {paths['csv']}")
    for plot in paths["plots"]:
        print(f"  {plot}")
    print()
    print(
        "Primary: PSNR / LPIPS permutation p-value and bootstrap 95% CI. "
        "The main scatter is delta_psnr_soft_vs_sr.png. "
        "Decide on a λ sweep from effect size + CI + that relationship, "
        "not from a dB cutoff."
    )


if __name__ == "__main__":
    main()
