#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a 3-panel infographic for LinkedIn (SPY CNN-LSTM / LSTM research).
Output: experiments/preliminary_analyses/results/linkedin_research_infographic.png

NOTE: Preserved as a preliminary marketing asset. Not part of the revised
paper. Figures for the revised paper are produced separately (not in this repo).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Paths
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "linkedin_research_infographic.png"

# --- Brand-ish colors (accessible contrast)
BG = "#F7F8FA"
TEXT = "#1B2430"
MUTED = "#5C6B7A"
ACCENT_A = "#2563EB"  # blue — baseline / good
ACCENT_B = "#DC2626"  # red — poor early CNN-LSTM
ACCENT_C = "#059669"  # green — regime insight bar emphasis
GRID = "#E2E8F0"


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.06, right=0.97, top=0.82, bottom=0.14)

    # ----- Panel A: early comparison (Oct 2019 window)
    ax1 = fig.add_subplot(gs[0, 0])
    labels_a = ["Simple LSTM", "CNN-LSTM\n(initial setup)"]
    values_a = [0.96, 6.64]
    colors_a = [ACCENT_A, ACCENT_B]
    bars1 = ax1.bar(labels_a, values_a, color=colors_a, width=0.55, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Close MAE ($)")
    ax1.set_title("A. Why apples-to-apples matters", fontsize=13, pad=12, color=TEXT)
    ax1.set_ylim(0, 8.5)
    ax1.axhline(0, color=GRID, linewidth=1)
    for b, v in zip(bars1, values_a):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.2,
            f"${v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )
    ax1.text(
        0.5,
        -0.22,
        "SPY · next-day Close · Oct 14–28, 2019\n(misaligned protocol exaggerates the gap)",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
        linespacing=1.35,
    )

    # ----- Panel B: tuned frozen model OHLC
    ax2 = fig.add_subplot(gs[0, 1])
    feats = ["Open", "High", "Low", "Close"]
    maes = [0.92, 1.01, 1.18, 1.04]
    x = range(len(feats))
    bars2 = ax2.bar(x, maes, color=ACCENT_A, width=0.6, edgecolor="white", linewidth=1.2)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(feats)
    ax2.set_ylabel("MAE ($)")
    ax2.set_title("B. Tuned CNN–LSTM (frozen baseline)", fontsize=13, pad=12, color=TEXT)
    ax2.set_ylim(0, 1.45)
    for b, v in zip(bars2, maes):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f"${v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT,
        )
    ax2.text(
        0.5,
        -0.22,
        "60-day OHLCV windows · parallel CNN+LSTM, fused head\nAdamW · lr 3e-4 · wd 0.002 · 64 hidden · 32 filters",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
        linespacing=1.35,
    )

    # ----- Panel C: regime split
    ax3 = fig.add_subplot(gs[0, 2])
    labels_c = ["Pre-2016\n(proxied “pre-AI”)", "Post-2016\n(proxied “post-AI”)"]
    values_c = [2.31, 8.91]
    colors_c = [ACCENT_A, ACCENT_C]
    bars3 = ax3.bar(labels_c, values_c, color=colors_c, width=0.55, edgecolor="white", linewidth=1.2)
    ax3.set_ylabel("Close MAE ($)")
    ax3.set_title("C. Same model, different era", fontsize=13, pad=12, color=TEXT)
    ax3.set_ylim(0, 11)
    for b, v in zip(bars3, values_c):
        ax3.text(
            b.get_x() + b.get_width() / 2,
            v + 0.25,
            f"${v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )
    ax3.annotate(
        "~3.9× higher error\npost split",
        xy=(1, 8.91),
        xytext=(0.35, 9.6),
        fontsize=10,
        fontweight="bold",
        color=TEXT,
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2),
    )
    ax3.text(
        0.5,
        -0.22,
        "Illustrative regime test on SPY daily data\n(evaluation design dominates single scores)",
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
        linespacing=1.35,
    )

    fig.suptitle(
        "SPY next-day forecasting: lessons from a CNN–LSTM research stack",
        fontsize=18,
        fontweight="bold",
        y=0.94,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.885,
        "PyTorch · daily OHLCV · hyperparameter & regime experiments · not investment advice",
        ha="center",
        fontsize=10,
        color=MUTED,
    )

    leg_elements = [
        mpatches.Patch(facecolor=ACCENT_A, edgecolor="white", label="Stronger / baseline"),
        mpatches.Patch(facecolor=ACCENT_B, edgecolor="white", label="Misleading early compare"),
        mpatches.Patch(facecolor=ACCENT_C, edgecolor="white", label="Regime stress"),
    ]
    fig.legend(
        handles=leg_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
