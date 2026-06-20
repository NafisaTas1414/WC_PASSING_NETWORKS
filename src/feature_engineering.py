"""
Network feature engineering for the WC passing network project.

Two levels of features are produced:
  1. match_level  – one row per (match_id, team)
  2. team_level   – one row per team, aggregated across all their matches
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    from network_builder import PassNetwork
except ModuleNotFoundError:
    from src.network_builder import PassNetwork


# ---------------------------------------------------------------------------
# Match-level feature extraction
# ---------------------------------------------------------------------------

def _safe(fn, default=np.nan):
    try:
        return fn()
    except Exception:
        return default


def _degree_centralization(G: nx.DiGraph) -> float:
    """
    Freeman degree centralization (out-degree, weighted).

    Measures how much the network is dominated by a single hub.
    Ranges 0 (perfectly equal) to 1 (star topology).
    """
    n = G.number_of_nodes()
    if n < 2:
        return np.nan
    degrees = [d for _, d in G.out_degree(weight="weight")]
    if not degrees:
        return np.nan
    max_deg = max(degrees)
    total_diff = sum(max_deg - d for d in degrees)
    # Maximum possible sum of differences for a directed star with n nodes
    max_possible = (n - 1) * (n - 1)
    return total_diff / max_possible if max_possible > 0 else np.nan


def _pitch_zone(x: float) -> str:
    """Classify a pitch x-coordinate into defensive / mid / final third."""
    if x < 40:
        return "defensive"
    if x < 80:
        return "middle"
    return "final"


def extract_match_features(
    network: PassNetwork,
    passes: pd.DataFrame,
) -> dict:
    """
    Compute all network and pass-quality features for one (match_id, team).

    Parameters
    ----------
    network : PassNetwork for this team-match
    passes  : full pass table (will be filtered internally)
    """
    G = network.graph
    match_id = network.match_id
    team = network.team

    # Subset of passes for this team-match
    mask = (passes["match_id"] == match_id) & (passes["team"] == team)
    tp = passes[mask].dropna(subset=["player", "pass_recipient"])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    total_passes = int(tp.shape[0])

    row: dict = {
        "match_id": match_id,
        "team": team,
        # -- volume --
        "total_completed_passes": total_passes,
        "unique_passing_pairs": n_edges,
        "number_of_players_in_network": n_nodes,
        # -- density --
        "network_density": _safe(lambda: nx.density(G)),
        # -- weighted degree --
        "average_weighted_degree": _safe(
            lambda: np.mean([d for _, d in G.degree(weight="weight")])
            if n_nodes > 0 else np.nan
        ),
        "average_edge_weight": _safe(
            lambda: np.mean([d["weight"] for *_, d in G.edges(data=True)])
            if n_edges > 0 else np.nan
        ),
        # -- centralization --
        "degree_centralization": _degree_centralization(G),
        # -- top player reliance --
        "top_player_pass_share": _safe(
            lambda: max(d for _, d in G.out_degree(weight="weight")) / total_passes
            if total_passes > 0 and n_nodes > 0 else np.nan
        ),
        # -- clustering (undirected view) --
        "clustering_coefficient": _safe(
            lambda: nx.average_clustering(G.to_undirected(), weight="weight")
        ),
    }

    # -- betweenness centrality (mean) ------------------------------------
    if n_nodes >= 3:
        bc = _safe(
            lambda: list(nx.betweenness_centrality(G, weight="weight").values())
        )
        row["mean_betweenness_centrality"] = float(np.mean(bc)) if bc else np.nan
        row["max_betweenness_centrality"]  = float(np.max(bc))  if bc else np.nan
    else:
        row["mean_betweenness_centrality"] = np.nan
        row["max_betweenness_centrality"]  = np.nan

    # -- progressive passes -----------------------------------------------
    if "is_progressive" in tp.columns:
        n_prog = int(tp["is_progressive"].sum())
        row["progressive_pass_count"] = n_prog
        row["progressive_pass_share"] = n_prog / total_passes if total_passes else np.nan
    else:
        row["progressive_pass_count"] = np.nan
        row["progressive_pass_share"] = np.nan

    # -- final third entries ----------------------------------------------
    if "is_final_third" in tp.columns:
        n_ft = int(tp["is_final_third"].sum())
        row["final_third_entries"] = n_ft
        row["final_third_entry_share"] = n_ft / total_passes if total_passes else np.nan
    else:
        row["final_third_entries"] = np.nan
        row["final_third_entry_share"] = np.nan

    # -- lateral balance (left / central / right thirds of the pitch) -----
    if "y" in tp.columns and tp["y"].notna().any():
        y_vals = tp["y"].dropna()
        # Pitch y: 0 = left touchline, 80 = right touchline (StatsBomb)
        left   = (y_vals < 26.67).sum()
        centre = ((y_vals >= 26.67) & (y_vals < 53.33)).sum()
        right  = (y_vals >= 53.33).sum()
        total_y = left + centre + right
        row["left_channel_share"]   = left   / total_y if total_y else np.nan
        row["centre_channel_share"] = centre / total_y if total_y else np.nan
        row["right_channel_share"]  = right  / total_y if total_y else np.nan
        # Imbalance: 0 = perfectly symmetric, 1 = all one side
        row["lateral_imbalance"] = abs(left - right) / total_y if total_y else np.nan
    else:
        for k in ["left_channel_share", "centre_channel_share",
                  "right_channel_share", "lateral_imbalance"]:
            row[k] = np.nan

    # -- goalkeeper involvement -------------------------------------------
    GK_KEYWORDS = {"goalkeeper", "GK", "Goalkeeper"}
    gk_players = set()
    if "pass_recipient" in tp.columns:
        # Heuristic: position info not in pass table, so we proxy by checking
        # if player name matches any node tagged as GK in the node_table
        nt = network.node_table
        if "position" in nt.columns:
            gk_mask = nt["position"].isin(GK_KEYWORDS) | nt["position"].str.contains(
                "Goalkeeper", na=False
            )
            gk_players = set(nt.loc[gk_mask, "player"])
    if gk_players:
        gk_sent     = tp["player"].isin(gk_players).sum()
        gk_received = tp["pass_recipient"].isin(gk_players).sum()
        row["gk_pass_share"] = (gk_sent + gk_received) / (2 * total_passes) if total_passes else np.nan
    else:
        row["gk_pass_share"] = np.nan

    return row


def build_match_feature_table(
    networks: Dict[Tuple[int, str], PassNetwork],
    passes: pd.DataFrame,
    team_labels: pd.DataFrame,
    stage_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the match-level feature table for all networks.

    Merges advancement labels onto the resulting DataFrame.
    """
    rows = []
    for (match_id, team), net in networks.items():
        row = extract_match_features(net, passes)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Merge advancement labels
    df = df.merge(team_labels, on="team", how="left")

    # Merge stage info if available
    if stage_map is not None:
        df = df.merge(stage_map[["match_id", "stage_label", "home_team",
                                  "away_team", "home_score", "away_score"]],
                      on="match_id", how="left")
        # Match result from the team's perspective
        df["match_result"] = df.apply(_assign_result, axis=1)

    return df.reset_index(drop=True)


def _assign_result(row: pd.Series) -> Optional[str]:
    """Derive W/D/L for a team based on home/away scores."""
    try:
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        is_home = row["team"] == row["home_team"]
        team_score   = home_score if is_home else away_score
        opponent_score = away_score if is_home else home_score
        if team_score > opponent_score:
            return "W"
        if team_score == opponent_score:
            return "D"
        return "L"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Team-level aggregation
# ---------------------------------------------------------------------------

AGG_FEATURES = [
    "total_completed_passes",
    "network_density",
    "average_weighted_degree",
    "degree_centralization",
    "top_player_pass_share",
    "clustering_coefficient",
    "mean_betweenness_centrality",
    "max_betweenness_centrality",
    "progressive_pass_share",
    "final_third_entry_share",
    "lateral_imbalance",
    "centre_channel_share",
]


def build_team_feature_table(
    match_features: pd.DataFrame,
    team_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate match-level features to the team level.

    Produces mean, std, and a consistency (1 - CV) metric for each feature
    across all of a team's matches.
    """
    numeric_cols = [c for c in AGG_FEATURES if c in match_features.columns]

    team_df = (
        match_features
        .groupby("team", as_index=True)[numeric_cols]
        .agg(["mean", "std"])
    )
    team_df.columns = [f"{col}_{stat}" for col, stat in team_df.columns]
    team_df = team_df.reset_index()

    # Consistency (1 - coefficient of variation) for selected features
    for col in ["network_density", "degree_centralization", "top_player_pass_share"]:
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        if mean_col in team_df.columns and std_col in team_df.columns:
            cv = team_df[std_col] / team_df[mean_col].replace(0, np.nan)
            team_df[f"{col}_consistency"] = 1 - cv.clip(0, 1)

    played = match_features.groupby("team")["match_id"].count().rename("matches_played").reset_index()
    team_df = team_df.merge(played, on="team", how="left")

    team_df = team_df.merge(team_labels, on="team", how="left")

    return team_df.reset_index(drop=True)
