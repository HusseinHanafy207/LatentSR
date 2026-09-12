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
from latentsr.metrics.collapse_geometry import (
    format_collapse_geometry_table,
    run_collapse_geometry,
)
from latentsr.metrics.representation_geometry import (
    format_geometry_table,
    run_representation_geometry,
)
from latentsr.metrics.condition_utility import (
    compute_step_condition_utility,
    evaluate_condition_utility_forward,
    evaluate_condition_utility_reverse,
    generate_condition_utility_report,
    plot_condition_utility,
    save_condition_utility_results,
    shuffle_condition,
)
from latentsr.metrics.ddim_diagnostic import (
    format_ddpm_vs_ddim_table,
    generate_ddpm_vs_ddim_report,
    plot_ddpm_vs_ddim_curves,
    run_ddpm_vs_ddim_comparison,
    save_ddpm_vs_ddim_grid,
    save_ddpm_vs_ddim_results,
)
from latentsr.metrics.timestep_diagnostic import run_timestep_diagnostic
from latentsr.metrics.z0_recon_diagnostic import run_z0_recon_diagnostic

__all__ = [
    "LPIPSMetric",
    "batch_metrics",
    "bootstrap_mean_ci",
    "compare_per_image",
    "compute_step_condition_utility",
    "dataset_filename",
    "edge_mae",
    "evaluate_condition_utility_forward",
    "evaluate_condition_utility_reverse",
    "evaluate_sr",
    "evaluate_vae",
    "format_bottleneck_note",
    "format_collapse_geometry_table",
    "format_ddpm_vs_ddim_table",
    "format_geometry_table",
    "format_metric_table",
    "frequency_band_error",
    "generate_condition_utility_report",
    "generate_ddpm_vs_ddim_report",
    "PRE_REGISTERED",
    "REFERENCE_BANNER",
    "plot_condition_utility",
    "plot_ddpm_vs_ddim_curves",
    "psnr",
    "run_collapse_geometry",
    "run_ddpm_vs_ddim_comparison",
    "run_guidance_condition",
    "run_representation_geometry",
    "run_timestep_diagnostic",
    "run_z0_recon_diagnostic",
    "save_condition_utility_results",
    "save_ddpm_vs_ddim_grid",
    "save_ddpm_vs_ddim_results",
    "shuffle_condition",
    "sign_flip_permutation_pvalue",
    "spearman_rho",
    "ssim",
    "summarize_values",
    "write_per_image_csv",
    "write_summary_files",
]
