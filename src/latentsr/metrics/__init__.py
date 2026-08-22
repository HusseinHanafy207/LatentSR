"""Evaluation metrics (PSNR, SSIM, LPIPS, structure, VAE bottleneck)."""

from latentsr.metrics.evaluate_sr import evaluate_sr
from latentsr.metrics.evaluate_vae import evaluate_vae, format_bottleneck_note
from latentsr.metrics.guidance_eval import (
    PRE_REGISTERED,
    REFERENCE_BANNER,
    run_guidance_condition,
)
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    dataset_filename,
    edge_mae,
    format_metric_table,
    frequency_band_error,
    psnr,
    ssim,
    summarize_values,
    write_per_image_csv,
    write_summary_files,
)
from latentsr.metrics.paired_stats import (
    bootstrap_mean_ci,
    compare_per_image,
    sign_flip_permutation_pvalue,
    spearman_rho,
)
from latentsr.metrics.timestep_diagnostic import run_timestep_diagnostic
from latentsr.metrics.z0_recon_diagnostic import run_z0_recon_diagnostic

__all__ = [
    "LPIPSMetric",
    "batch_metrics",
    "bootstrap_mean_ci",
    "compare_per_image",
    "dataset_filename",
    "edge_mae",
    "evaluate_sr",
    "evaluate_vae",
    "format_bottleneck_note",
    "format_metric_table",
    "frequency_band_error",
    "PRE_REGISTERED",
    "REFERENCE_BANNER",
    "psnr",
    "run_guidance_condition",
    "run_timestep_diagnostic",
    "run_z0_recon_diagnostic",
    "sign_flip_permutation_pvalue",
    "spearman_rho",
    "ssim",
    "summarize_values",
    "write_per_image_csv",
    "write_summary_files",
]
