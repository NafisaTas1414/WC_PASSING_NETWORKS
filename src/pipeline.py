"""
Multi-tournament passing-network pipeline.
Reusable functions for loading, processing, and aggregating World Cup data
across multiple StatsBomb seasons without rewriting logic in every notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsbombpy import sb
except ImportError:
    raise ImportError("statsbombpy required: pip install statsbombpy")


# ── Stage label normalisation ───────────────────────────────────────────────

STAGE_MAP = {
    "Group Stage":       "group",
    "Round of 16":       "R16",
    "Quarter-finals":    "QF",
    "Semi-finals":       "SF",
    "Third-place Play-off": "3rd",
    "3rd Place Final":   "3rd",
    "Final":             "Final",
}

STAGE_RANK = {
    "group": 1, "R16": 2, "QF": 3, "SF": 4, "3rd": 5, "Final": 6,
}

STAGE_LABEL = {
    1: "Group Stage", 2: "Round of 16", 3: "Quarter-final",
    4: "Semi-final",  5: "3rd Place",   6: "Final",
}


def normalize_stage(s) -> str:
    """Convert any StatsBomb stage value (string or dict) to a short code."""
    if isinstance(s, dict):
        s = s.get("name", "")
    return STAGE_MAP.get(str(s), str(s))


# ── Low-level helpers ───────────────────────────────────────────────────────

def _coord(v, i):
    """Extract coordinate at index i from list/tuple/numpy array safely."""
    try:
        return float(v[i])
    except Exception:
        return None


# ── Network metric computation ──────────────────────────────────────────────

def compute_network_metrics(ev: pd.DataFrame, team: str) -> Optional[dict]:
    """
    Compute passing-network metrics for one team in one match.
    Returns None if there are no completed passes for the team.
    """
    tp = ev[
        (ev["type"] == "Pass") &
        (ev["pass_outcome"].isna()) &
        (ev["team"] == team)
    ].copy()

    tp["x"] = tp["location"].apply(lambda v: _coord(v, 0))

    has_end = "pass_end_location" in tp.columns
    if has_end:
        tp["ex"]    = tp["pass_end_location"].apply(lambda v: _coord(v, 0))
        # Final-third entry: pass started outside final third, ended inside it
        tp["is_ft"] = (tp["x"] < 80) & (tp["ex"] >= 80)
        # Progressive pass: ball moves forward at least 10 yards
        tp["is_prog"] = (tp["ex"] - tp["x"]) >= 10
    else:
        tp["is_ft"]   = False
        tp["is_prog"] = False

    completed = tp.dropna(subset=["player", "pass_recipient"])
    if len(completed) == 0:
        return None

    sent_ct     = completed.groupby("player").size()
    recv_ct     = completed.groupby("pass_recipient").size()
    all_players = set(tp.dropna(subset=["x"]).groupby("player")["x"].count().index)

    inv    = {p: int(sent_ct.get(p, 0)) + int(recv_ct.get(p, 0)) for p in all_players}
    ranked = sorted(inv.items(), key=lambda kv: -kv[1])

    edges = []
    for (passer, recip), grp in completed.groupby(["player", "pass_recipient"]):
        edges.append({
            "s": passer, "t": recip, "w": len(grp),
            "ft": int(grp["is_ft"].sum()) if has_end else 0,
        })

    n_nodes  = len(all_players)
    n_edges  = len(edges)
    total    = len(completed)
    max_poss = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density  = n_edges / max_poss if max_poss else 0

    most_tot  = ranked[0][1] if ranked else 0
    total_inv = sum(inv.values())
    top_rel   = most_tot / total_inv if total_inv else 0

    top_e    = max(edges, key=lambda e: e["w"]) if edges else None
    total_ft = sum(e["ft"] for e in edges) if has_end else np.nan

    # Critical pass flags — StatsBomb pre-tags these, just count True values
    def _flag_count(col):
        return int(tp[col].eq(True).sum()) if col in tp.columns else 0

    return {
        "completed_passes":     total,
        "unique_passing_pairs": n_edges,
        "num_players":          n_nodes,
        "network_density":      round(density, 4),
        "top_player_reliance":  round(top_rel, 4),
        "top_connection_count": top_e["w"] if top_e else 0,
        "final_third_entries":  int(total_ft) if not np.isnan(total_ft) else np.nan,
        "progressive_passes":   int(tp["is_prog"].sum()),
        "key_passes":           _flag_count("pass_shot_assist"),
        "goal_assists":         _flag_count("pass_goal_assist"),
        "through_balls":        _flag_count("pass_through_ball"),
        "crosses":              _flag_count("pass_cross"),
    }


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate-based columns to a match-feature DataFrame.
    Safe against divide-by-zero — returns NaN instead of inf.
    Modifies df in place and returns it.
    """
    cp = df["completed_passes"]
    up = df["unique_passing_pairs"]
    ft = df["final_third_entries"]

    df["final_third_entry_share"] = np.where(
        cp > 0, ft / cp, np.nan).round(4)
    df["final_third_entries_per_100_passes"] = np.where(
        cp > 0, ft / cp * 100, np.nan).round(2)
    df["connections_per_100_passes"] = np.where(
        cp > 0, up / cp * 100, np.nan).round(2)
    df["passes_per_connection"] = np.where(
        up > 0, cp / up, np.nan).round(2)
    return df


# ── Data loading ────────────────────────────────────────────────────────────

def load_tournament_matches(
    competition_id: int,
    season_id: int,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and normalise matches for any StatsBomb competition/season.
    Caches to CSV on first run.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        matches = pd.read_csv(cache_path)
    else:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            matches.to_csv(cache_path, index=False)

    # Flatten home_team / away_team if StatsBomb returned dicts
    for col in ("home_team", "away_team"):
        if col in matches.columns:
            matches[col] = matches[col].apply(
                lambda v: (v.get("home_team_name") or v.get("away_team_name"))
                if isinstance(v, dict) else v
            )

    # Normalise stage labels
    stage_col = next(
        (c for c in ("competition_stage", "stage") if c in matches.columns), None
    )
    if stage_col:
        matches["stage_label"] = matches[stage_col].apply(
            lambda s: normalize_stage(s.get("name", "") if isinstance(s, dict) else s)
        )

    return matches


def load_events(match_id: int, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load events for one match, writing parquet cache on first fetch."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        path = cache_dir / f"events_{match_id}.parquet"
        if path.exists():
            return pd.read_parquet(path)
    ev = sb.events(match_id=match_id)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(cache_dir / f"events_{match_id}.parquet", index=False)
    return ev


# ── Tournament-level processing ─────────────────────────────────────────────

def build_match_features(
    matches: pd.DataFrame,
    events_cache_dir: Path,
    tournament_year: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one row per (team, match) for all matches in a tournament.
    Prints progress because downloading 64 matches can take a few minutes.
    """
    rows = []
    total = len(matches)
    for i, (_, mrow) in enumerate(matches.iterrows(), 1):
        mid        = int(mrow["match_id"])
        home_team  = str(mrow["home_team"])
        away_team  = str(mrow["away_team"])
        home_score = int(mrow.get("home_score", 0))
        away_score = int(mrow.get("away_score", 0))
        stage      = str(mrow.get("stage_label", ""))
        date       = str(mrow.get("match_date", ""))

        if verbose:
            cached = (Path(events_cache_dir) / f"events_{mid}.parquet").exists()
            tag    = "cache" if cached else "download"
            print(f"  [{i:02}/{total}] {home_team} vs {away_team}  [{stage}]  ({tag})",
                  flush=True)

        try:
            ev = load_events(mid, cache_dir=events_cache_dir)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        for team, opp, gf, ga in [
            (home_team, away_team, home_score, away_score),
            (away_team, home_team, away_score, home_score),
        ]:
            metrics = compute_network_metrics(ev, team)
            if metrics is None:
                continue
            result = "W" if gf > ga else ("L" if gf < ga else "D")
            rows.append({
                "tournament_year": tournament_year,
                "match_id": mid,
                "match_name": f"{home_team} vs {away_team}",
                "date": date,
                "stage": stage,
                "team": team,
                "opponent": opp,
                "goals_for": gf,
                "goals_against": ga,
                "match_result": result,
                **metrics,
            })

    df = pd.DataFrame(rows)
    return add_normalized_features(df)


# ── Team-level aggregation ───────────────────────────────────────────────────

def get_qf_teams(matches: pd.DataFrame) -> frozenset:
    """Return the set of teams that played in QF matches (derived from data)."""
    qf = matches[matches["stage_label"] == "QF"]
    return frozenset(qf["home_team"].tolist() + qf["away_team"].tolist())


def get_stage_reached(team: str, matches: pd.DataFrame) -> str:
    """Return the furthest stage a team reached, as a human-readable label."""
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ]
    max_rank = max(
        (STAGE_RANK.get(str(s), 0) for s in team_matches["stage_label"]),
        default=0,
    )
    return STAGE_LABEL.get(max_rank, "Unknown")


def aggregate_group_stage_profiles(
    match_features: pd.DataFrame,
    matches: pd.DataFrame,
    tournament_year: int,
) -> pd.DataFrame:
    """
    Aggregate group-stage match features to one row per team.

    Input:  match_features DataFrame (one row per team-match, all stages)
    Output: one row per team with avg_* and std_* columns + labels
    """
    gs = match_features[
        match_features["stage"].str.contains("group", case=False, na=False)
    ].copy()

    avg_cols = [
        "completed_passes", "unique_passing_pairs", "network_density",
        "top_player_reliance", "final_third_entries",
        "final_third_entries_per_100_passes", "connections_per_100_passes",
        "passes_per_connection", "final_third_entry_share",
    ]
    avg_cols = [c for c in avg_cols if c in gs.columns]

    std_cols = [
        "completed_passes", "network_density",
        "passes_per_connection", "final_third_entries_per_100_passes",
    ]
    std_cols = [c for c in std_cols if c in gs.columns]

    mean_df  = gs.groupby("team")[avg_cols].mean().round(3)
    mean_df.columns = [f"avg_{c}" for c in mean_df.columns]

    std_df   = gs.groupby("team")[std_cols].std().round(3)
    std_df.columns   = [f"std_{c}" for c in std_df.columns]

    count_df = gs.groupby("team").size().rename("num_group_matches")

    profiles = mean_df.join(std_df).join(count_df).reset_index()

    qf_teams = get_qf_teams(matches)
    profiles["tournament_year"] = tournament_year
    profiles["reached_qf"]      = profiles["team"].apply(
        lambda t: 1 if t in qf_teams else 0
    )
    profiles["stage_reached"]   = profiles["team"].apply(
        lambda t: get_stage_reached(t, matches)
    )

    lead = ["tournament_year", "team", "num_group_matches", "stage_reached", "reached_qf"]
    rest = [c for c in profiles.columns if c not in lead]
    return profiles[lead + rest].reset_index(drop=True)
