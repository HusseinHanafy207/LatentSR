"""Standalone centerpiece panels (not a combined figure).

Reads the local eval CSVs (no re-sampling) and writes

    paper/figures/fig_a_representation.{png,pdf}
    paper/figures/fig_b_reverse_chain.{png,pdf}
    paper/figures/fig_c_late_dose.{png,pdf}
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SOFT_PSNR = (26.15, 28.48)
SOFT_LPIPS = (0.273, 0.119)
SR_PSNR = (26.26, 26.48)
SR_LPIPS = (0.0683, 0.0685)

LAMBDA_LABELS = ["0\nunguided", "50", "200", "800"]
GUIDE_PSNR = np.array([26.48, 27.02, 27.60, 28.07])
GUIDE_LPIPS = np.array([0.0685, 0.0670, 0.0693, 0.0833])

VAE1 = "#4c78a8"
VAESR = "#f58518"
GUIDED = "#2ca02c"
SOFT = "#6b6f76"
LPIPS_C = "#c44e52"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _mean_by_t(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    acc: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        acc[int(float(row["t"]))].append(float(row[key]))
    t = np.array(sorted(acc, reverse=True), dtype=float)
    y = np.array([float(np.mean(acc[int(tt)])) for tt in t])
    return t, y


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _dumbbell(
    ax: plt.Axes,
    y: float,
    x0: float,
    x1: float,
    *,
    delta: str,
    dy: float = -0.18,
) -> None:
    ax.plot([x0, x1], [y, y], color="#d0d3d6", lw=2.6, zorder=1, solid_capstyle="round")
    ax.scatter([x0], [y], s=64, color=VAE1, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter([x1], [y], s=64, color=VAESR, zorder=3, edgecolors="white", linewidths=0.6)
    ax.text(
        0.5 * (x0 + x1),
        y + dy,
        delta,
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )


def _vae_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=VAE1,
            markeredgecolor="white",
            markersize=8,
            label="VAE-1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=VAESR,
            markeredgecolor="white",
            markersize=8,
            label="VAE-SR",
        ),
    ]


def _save(fig: plt.Figure, stem: str) -> None:
    name = stem if stem.startswith("fig_") else f"fig_{stem}"
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def figure_a() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6), constrained_layout=True)
    ax_p, ax_l = axes

    ax_p.set_title("A   Representation — PSNR", loc="left", fontweight="bold")
    _dumbbell(ax_p, 1.0, SOFT_PSNR[0], SOFT_PSNR[1], delta="+2.33 dB")
    _dumbbell(ax_p, 0.0, SR_PSNR[0], SR_PSNR[1], delta="+0.22 dB")
    ax_p.text(
        SOFT_PSNR[0],
        1.18,
        f"{SOFT_PSNR[0]:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VAE1,
    )
    ax_p.text(
        SOFT_PSNR[1],
        1.18,
        f"{SOFT_PSNR[1]:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VAESR,
    )
    ax_p.text(
        SR_PSNR[1] + 0.18,
        0.0,
        f"{SR_PSNR[0]:.2f}  →  {SR_PSNR[1]:.2f}",
        ha="left",
        va="center",
        fontsize=9,
        color="#333333",
    )
    ax_p.annotate(
        "",
        xy=(SOFT_PSNR[1], 0.48),
        xytext=(SR_PSNR[1], 0.48),
        arrowprops=dict(arrowstyle="<->", color="#8a8f94", lw=1.0),
    )
    ax_p.text(
        0.5 * (SOFT_PSNR[1] + SR_PSNR[1]),
        0.55,
        "transfer gap  2.00 dB",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    ax_p.set_yticks([1.0, 0.0])
    ax_p.set_yticklabels(["Soft decode", "LatentSR"])
    ax_p.set_xlim(24.7, 29.3)
    ax_p.set_ylim(-0.55, 1.55)
    ax_p.set_xlabel("PSNR (dB)")
    ax_p.spines["right"].set_visible(False)
    ax_p.spines["top"].set_visible(False)
    ax_p.legend(handles=_vae_handles(), loc="lower right", frameon=False)

    ax_l.set_title("LPIPS  (lower better, right = better)", loc="left")
    _dumbbell(ax_l, 1.0, SOFT_LPIPS[0], SOFT_LPIPS[1], delta="−0.154")
    _dumbbell(ax_l, 0.0, SR_LPIPS[0], SR_LPIPS[1], delta="null")
    ax_l.text(
        SOFT_LPIPS[0],
        1.16,
        f"{SOFT_LPIPS[0]:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VAE1,
    )
    ax_l.text(
        SOFT_LPIPS[1],
        1.16,
        f"{SOFT_LPIPS[1]:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VAESR,
    )
    ax_l.text(
        SR_LPIPS[1],
        0.16,
        f"{SR_LPIPS[1]:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VAE1,
    )
    ax_l.set_yticks([1.0, 0.0])
    ax_l.set_yticklabels(["Soft decode", "LatentSR"])
    ax_l.set_xlim(0.40, 0.02)
    ax_l.set_ylim(-0.55, 1.55)
    ax_l.set_xlabel("LPIPS")
    ax_l.spines["right"].set_visible(False)
    ax_l.spines["top"].set_visible(False)

    _save(fig, "a_representation")


def figure_b() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.set_title(
        r"B   Reverse chain  —  $\cos(\hat{z}_0, z_{lr})$",
        loc="left",
        fontweight="bold",
    )
    diag = _read_csv(ROOT / "outputs" / "eval_timestep_diagnostic" / "timestep_means.csv")
    t_all = np.array([int(r["t"]) for r in diag], dtype=float)
    cos1 = np.array([float(r["cosine_z0_z_lr_vae1_mean"]) for r in diag])
    coss = np.array([float(r["cosine_z0_z_lr_vaesr_mean"]) for r in diag])
    keep = t_all <= 800
    t_all, cos1, coss = t_all[keep], cos1[keep], coss[keep]

    t_g, cos_g = _mean_by_t(
        _read_csv(ROOT / "outputs" / "eval_guidance_late" / "trajectory.csv"),
        "cosine_z0_z_lr",
    )

    ax.axvspan(0, 500, color=GUIDED, alpha=0.08, zorder=0)
    ax.plot(t_all, cos1, color=VAE1, lw=1.8, label="VAE-1")
    ax.plot(t_all, coss, color=VAESR, lw=1.8, label="VAE-SR")
    ax.plot(
        t_g,
        cos_g,
        color=GUIDED,
        lw=2.0,
        marker="o",
        markersize=3.6,
        label=r"VAE-SR + $\lambda_g{=}200$ (late)",
    )
    ax.axvline(500, color=GUIDED, ls=":", lw=0.9, alpha=0.85)
    ax.set_xlim(800, 0)
    ax.set_ylim(0.58, 1.02)
    ax.set_xlabel(r"$t$  (800 $\rightarrow$ 0)")
    ax.set_ylabel("cosine")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    i_peak = int(np.argmax(coss))
    ax.annotate(
        f"copy  {coss[i_peak]:.2f}",
        xy=(t_all[i_peak], coss[i_peak]),
        xytext=(580, 0.91),
        fontsize=9,
        color=VAESR,
        arrowprops=dict(arrowstyle="-", color=VAESR, lw=0.7),
        ha="center",
    )
    cos_sr0 = float(coss[np.isclose(t_all, 0)][0])
    cos_g0 = float(cos_g[np.isclose(t_g, 0)][0])
    ax.annotate(
        f"t=0   {cos_sr0:.2f} → {cos_g0:.2f}",
        xy=(0.0, cos_g0),
        xytext=(170, 0.70),
        fontsize=9,
        color=GUIDED,
        arrowprops=dict(arrowstyle="-", color=GUIDED, lw=0.7),
    )
    ax.text(250, 0.595, "guidance on", color=GUIDED, fontsize=9, ha="center")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "b_reverse_chain")


def figure_c() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.set_title("C   Late-window dose", loc="left", fontweight="bold")
    x = np.arange(4)
    ax.plot(x, GUIDE_PSNR, color=VAE1, marker="o", lw=2.0, markersize=7, zorder=3)
    ax.axhline(SOFT_PSNR[1], color=SOFT, ls="--", lw=1.0)
    ax.axhline(SR_PSNR[1], color="#b0b4b8", ls=":", lw=1.0)
    ax.set_ylabel("PSNR (dB)", color=VAE1)
    ax.tick_params(axis="y", colors=VAE1)
    ax.set_ylim(26.15, 28.85)
    ax.set_xlim(-0.25, 3.25)
    ax.set_xticks(x)
    ax.set_xticklabels(LAMBDA_LABELS)
    ax.set_xlabel(r"$\lambda_g$  (late, $t\leq 500$)")
    ax.spines["top"].set_visible(False)

    ax.text(0.08, SOFT_PSNR[1] + 0.06, "soft decode  28.48", color=SOFT, fontsize=9, va="bottom")
    ax.scatter(
        [2],
        [GUIDE_PSNR[2]],
        s=110,
        facecolors="none",
        edgecolors=VAE1,
        linewidths=1.4,
        zorder=4,
    )

    ax2 = ax.twinx()
    ax2.plot(x, GUIDE_LPIPS, color=LPIPS_C, marker="s", lw=2.0, markersize=6, zorder=3)
    ax2.axhline(SOFT_LPIPS[1], color=LPIPS_C, ls="--", lw=0.9, alpha=0.5)
    ax2.set_ylabel("LPIPS", color=LPIPS_C)
    ax2.tick_params(axis="y", colors=LPIPS_C)
    ax2.set_ylim(0.058, 0.128)
    ax2.spines["top"].set_visible(False)

    ax.annotate(
        "operating\npoint",
        xy=(2, GUIDE_PSNR[2]),
        xytext=(0.85, 27.85),
        fontsize=9,
        color=VAE1,
        arrowprops=dict(arrowstyle="-", color=VAE1, lw=0.7),
    )
    ax2.annotate(
        "over-guidance",
        xy=(3, GUIDE_LPIPS[3]),
        xytext=(1.45, 0.100),
        fontsize=9,
        color=LPIPS_C,
        arrowprops=dict(arrowstyle="-", color=LPIPS_C, lw=0.7),
    )

    handles = [
        Line2D([0], [0], color=VAE1, marker="o", lw=2.0, label="PSNR"),
        Line2D([0], [0], color=LPIPS_C, marker="s", lw=2.0, label="LPIPS"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left")
    _save(fig, "c_late_dose")


def main() -> None:
    _style()
    figure_a()
    figure_b()
    figure_c()


if __name__ == "__main__":
    main()
