"""
Visualization utilities for the WC passing network project.

All functions return (fig, ax) or fig so callers can save or display them.
mplsoccer is optional; falls back to plain matplotlib if unavailable.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx

try:
    from mplsoccer import Pitch
    _HAS_MPLSOCCER = True
except ImportError:
    _HAS_MPLSOCCER = False

try:
    from network_builder import PassNetwork
except ModuleNotFoundError:
    from src.network_builder import PassNetwork

# ── colour palette ──────────────────────────────────────────────────────────
QF_COLOR   = "#1f77b4"   # blue  – quarterfinalists
ELIM_COLOR = "#d62728"   # red   – eliminated teams
NEUTRAL    = "#7f7f7f"


# ---------------------------------------------------------------------------
# 1. Pass Network on Pitch
# ---------------------------------------------------------------------------

def plot_pass_network(
    network: PassNetwork,
    title: Optional[str] = None,
    min_edge_weight: int = 3,
    node_scale: float = 300,
    edge_scale: float = 2.0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Draw a directed passing network overlaid on a pitch.

    Node size  ∝ total passes made.
    Edge width ∝ pass count between the pair.
    """
    G = network.graph
    nt = network.node_table
    et = network.edge_table

    if _HAS_MPLSOCCER:
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#22312b",
                      line_color="#c7d5cc", linewidth=1)
        fig, ax = pitch.draw(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#22312b")
        ax.set_facecolor("#22312b")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.invert_yaxis()
        _draw_pitch(ax)

    # Node positions
    pos = {}
    for _, row in nt.iterrows():
        if pd.notna(row.get("avg_x")) and pd.notna(row.get("avg_y")):
            pos[row["player"]] = (row["avg_x"], row["avg_y"])

    if not pos:
        ax.text(60, 40, "No location data", color="white",
                ha="center", va="center", fontsize=14)
        return fig

    # Nodes
    out_degree = dict(G.out_degree(weight="weight"))
    for player, (x, y) in pos.items():
        size = out_degree.get(player, 1)
        ax.scatter(x, y, s=size * node_scale / max(out_degree.values(), default=1),
                   c="white", edgecolors="#333333", linewidths=0.5, zorder=5)
        ax.text(x, y - 3, player.split()[-1], color="white",
                fontsize=6, ha="center", va="top", zorder=6)

    # Edges
    for _, erow in et.iterrows():
        if erow["weight"] < min_edge_weight:
            continue
        if erow["passer"] not in pos or erow["recipient"] not in pos:
            continue
        x1, y1 = pos[erow["passer"]]
        x2, y2 = pos[erow["recipient"]]
        alpha = min(0.9, 0.2 + erow["weight"] / et["weight"].max())
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="white",
                lw=erow["weight"] * edge_scale / et["weight"].max(),
                alpha=alpha,
            ),
            zorder=4,
        )

    ax.set_title(title or f"{network.team} — Match {network.match_id}",
                 color="white", fontsize=13, pad=10)
    fig.patch.set_facecolor("#22312b")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def _draw_pitch(ax: plt.Axes) -> None:
    """Minimal pitch lines when mplsoccer is unavailable."""
    for x in [0, 60, 120]:
        ax.axvline(x, color="#c7d5cc", lw=1)
    for y in [0, 80]:
        ax.axhline(y, color="#c7d5cc", lw=1)
    # Penalty areas
    ax.add_patch(plt.Rectangle((0, 18), 18, 44, fill=False,
                                edgecolor="#c7d5cc", lw=1))
    ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False,
                                edgecolor="#c7d5cc", lw=1))


# ---------------------------------------------------------------------------
# 2. Feature distributions by advancement group
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "reached_quarterfinal",
    ncols: int = 3,
    figsize: Optional[Tuple] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box + strip plots for each feature, split by group."""
    import matplotlib.ticker as ticker

    features = [f for f in features if f in df.columns]
    nrows = int(np.ceil(len(features) / ncols))
    if figsize is None:
        figsize = (ncols * 4.5, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    label_map = {1: "QF+", 0: "Eliminated"}
    color_map  = {1: QF_COLOR, 0: ELIM_COLOR}

    for i, feat in enumerate(features):
        ax = axes[i]
        for group_val in [0, 1]:
            vals = df[df[group_col] == group_val][feat].dropna()
            x    = group_val
            bp   = ax.boxplot(
                vals, positions=[x], widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor=color_map[group_val], alpha=0.6),
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(color=color_map[group_val]),
                capprops=dict(color=color_map[group_val]),
                flierprops=dict(marker="o", color=color_map[group_val],
                                alpha=0.3, markersize=4),
            )
            # Strip jitter
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(x + jitter, vals, color=color_map[group_val],
                       alpha=0.5, s=20, zorder=3)

        feat_label = feat.replace("_mean", "").replace("_", " ").title()
        ax.set_title(feat_label, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Eliminated", "QF+"], fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=color_map[g], alpha=0.7, label=label_map[g])
               for g in [1, 0]]
    fig.legend(handles=patches, loc="lower right", fontsize=9)

    fig.suptitle("Passing Network Features by Tournament Advancement",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 3. Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: pd.DataFrame,
    top_n: int = 10,
    title: str = "Feature Importances (Random Forest)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color=QF_COLOR, alpha=0.8)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. PCA cluster scatter
# ---------------------------------------------------------------------------

def plot_cluster_scatter(
    df: pd.DataFrame,
    group_col: str = "reached_quarterfinal",
    label_teams: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """2-D PCA scatter coloured by cluster and shaped by advancement."""
    if "pca1" not in df.columns:
        raise ValueError("Run cluster_teams() before plotting.")

    fig, ax = plt.subplots(figsize=(9, 6))
    clusters = sorted(df["cluster"].dropna().unique())
    cmap     = plt.get_cmap("tab10")
    markers  = {1: "D", 0: "o"}
    sizes    = {1: 100, 0: 60}

    for cluster in clusters:
        sub = df[df["cluster"] == cluster]
        for adv in [1, 0]:
            s = sub[sub[group_col] == adv]
            ax.scatter(
                s["pca1"], s["pca2"],
                c=[cmap(int(cluster))], marker=markers[adv],
                s=sizes[adv], edgecolors="black", linewidths=0.5,
                alpha=0.85, label=f"Cluster {cluster} — {'QF+' if adv else 'Elim'}",
                zorder=3,
            )
            if label_teams and "team" in s.columns:
                for _, row in s.iterrows():
                    ax.text(row["pca1"] + 0.05, row["pca2"] + 0.05,
                            row["team"], fontsize=6, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Team Passing Network Profiles (PCA + K-Means)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 5. Heatmap of feature correlations
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    df: pd.DataFrame,
    features: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    features = [f for f in features if f in df.columns]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(len(features) * 0.8 + 2, len(features) * 0.8 + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    labels = [f.replace("_mean", "").replace("_", "\n") for f in features]
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Matrix", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 6. Team radar / spider chart
# ---------------------------------------------------------------------------

def plot_team_radar(
    df: pd.DataFrame,
    team_names: List[str],
    features: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Radar chart comparing selected teams across normalised features."""
    features = [f for f in features if f in df.columns]
    n = len(features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")

    # Normalise per feature to [0, 1]
    norm_df = df[features].copy()
    for col in features:
        mn, mx = norm_df[col].min(), norm_df[col].max()
        norm_df[col] = (norm_df[col] - mn) / (mx - mn + 1e-9)

    for i, team in enumerate(team_names):
        row = df[df["team"] == team]
        if row.empty:
            continue
        vals = norm_df.loc[row.index[0], features].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap(i), lw=2, label=team)
        ax.fill(angles, vals, color=cmap(i), alpha=0.1)

    feature_labels = [f.replace("_mean", "").replace("_", "\n") for f in features]
    feature_labels += feature_labels[:1]
    ax.set_xticks(angles)
    ax.set_xticklabels(feature_labels, fontsize=8)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6)
    ax.set_title("Team Passing Network Radar", fontsize=13, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
