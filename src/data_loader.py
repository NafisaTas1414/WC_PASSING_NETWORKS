"""
Data loading and cleaning for 2022 FIFA World Cup passing network analysis.

Data source: StatsBomb Open Data
  Competition ID : 43  (FIFA World Cup)
  Season ID      : 106 (2022)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from statsbombpy import sb
except ImportError as exc:
    raise ImportError("statsbombpy is required: pip install statsbombpy") from exc

logger = logging.getLogger(__name__)

COMPETITION_ID = 43
SEASON_ID = 106

# Teams that reached the 2022 World Cup quarterfinals or beyond
QUARTERFINALISTS = frozenset([
    "Argentina", "Croatia", "France", "Morocco",
    "Netherlands", "Brazil", "England", "Portugal",
])

# StatsBomb pass_type values that indicate set-piece / non-open-play delivery
SET_PIECE_TYPES = frozenset([
    "Corner", "Free Kick", "Throw-in", "Kick Off", "Goal Kick",
])


def load_matches(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Return a DataFrame of all WC 2022 matches with advancement labels."""
    if cache_path and cache_path.exists():
        logger.info("Loading matches from cache: %s", cache_path)
        matches = pd.read_csv(cache_path)
    else:
        logger.info("Fetching match list from StatsBomb Open Data …")
        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            matches.to_csv(cache_path, index=False)
            logger.info("Cached matches to %s", cache_path)

    # Flatten nested dicts that statsbombpy returns for some columns
    for col in ["home_team", "away_team"]:
        if col in matches.columns and matches[col].dtype == object:
            try:
                matches[col] = matches[col].apply(
                    lambda v: v["home_team_name"] if isinstance(v, dict) and "home_team_name" in v
                    else v["away_team_name"] if isinstance(v, dict) and "away_team_name" in v
                    else v
                )
            except Exception:
                pass

    # Advancement labels
    all_teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    team_labels = pd.DataFrame({
        "team": all_teams,
        "reached_quarterfinal": [int(t in QUARTERFINALISTS) for t in all_teams],
    })

    # Stage mapping from match_week / stage column (StatsBomb uses 'stage')
    stage_order = {
        "Group Stage": "group",
        "Round of 16": "R16",
        "Quarter-finals": "QF",
        "Semi-finals": "SF",
        "Third-place Play-off": "3rd",
        "Final": "Final",
    }
    stage_col = next((c for c in ("stage", "competition_stage") if c in matches.columns), None)
    if stage_col:
        matches["stage_label"] = matches[stage_col].map(
            lambda s: stage_order.get(s, str(s)) if isinstance(s, str)
            else stage_order.get(s.get("name", ""), str(s)) if isinstance(s, dict)
            else str(s)
        )

    return matches, team_labels


def load_events(
    match_id: int,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Return raw event data for a single match."""
    if cache_dir:
        cache_file = cache_dir / f"events_{match_id}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

    events = sb.events(match_id=match_id)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache_file, index=False)

    return events


def extract_passes(
    events: pd.DataFrame,
    open_play_only: bool = False,
) -> pd.DataFrame:
    """
    Filter events to completed passes and return a tidy DataFrame.

    Parameters
    ----------
    events       : raw event DataFrame from statsbombpy
    open_play_only : if True, exclude corner/free-kick/throw-in/kick-off passes
    """
    passes = events[events["type"] == "Pass"].copy()

    # Completed passes: StatsBomb records outcome as NaN when the pass succeeded
    passes = passes[passes["pass_outcome"].isna()].copy()

    if open_play_only:
        mask_set = passes["pass_type"].isin(SET_PIECE_TYPES)
        passes = passes[~mask_set].copy()

    def _coord(v, idx):
        """Extract coordinate at idx from a list, tuple, or numpy-array (parquet round-trip)."""
        try:
            return float(v[idx])
        except (TypeError, IndexError, KeyError):
            return None

    # Unpack location columns
    if "location" in passes.columns:
        loc = passes["location"].dropna()
        passes.loc[loc.index, "x"] = loc.apply(lambda v: _coord(v, 0))
        passes.loc[loc.index, "y"] = loc.apply(lambda v: _coord(v, 1))

    if "pass_end_location" in passes.columns:
        eloc = passes["pass_end_location"].dropna()
        passes.loc[eloc.index, "end_x"] = eloc.apply(lambda v: _coord(v, 0))
        passes.loc[eloc.index, "end_y"] = eloc.apply(lambda v: _coord(v, 1))

    # Progressive pass: end_x at least 10 m further toward opponent goal
    # (pitch is 120 x 80 in StatsBomb coordinates, left-to-right)
    if "x" in passes.columns and "end_x" in passes.columns:
        passes["is_progressive"] = (passes["end_x"] - passes["x"]) >= 10

    # Final-third entry: pass ending in x >= 80
    if "end_x" in passes.columns:
        passes["is_final_third"] = passes["end_x"] >= 80

    required = ["id", "match_id", "team", "player", "pass_recipient",
                "minute", "period", "x", "y", "end_x", "end_y",
                "is_progressive", "is_final_third"]
    available = [c for c in required if c in passes.columns]

    return passes[available].reset_index(drop=True)


def load_lineups(match_id: int) -> pd.DataFrame:
    """Return lineup DataFrame for a match with team and position info."""
    lineups_raw = sb.lineups(match_id=match_id)
    rows = []
    for team_name, lineup_df in lineups_raw.items():
        lineup_df = lineup_df.copy()
        lineup_df["team"] = team_name
        rows.append(lineup_df)
    lineups = pd.concat(rows, ignore_index=True)

    # Flatten position column if nested
    if "positions" in lineups.columns:
        lineups["position"] = lineups["positions"].apply(
            lambda p: p[0]["position"] if isinstance(p, list) and p else None
        )

    return lineups


def build_pass_table(
    matches: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    open_play_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and clean passes for all WC 2022 matches.

    Returns a single tidy DataFrame of completed passes across all matches.
    """
    all_passes = []
    for _, row in matches.iterrows():
        match_id = int(row["match_id"])
        if verbose:
            logger.info("Loading match %s …", match_id)
        try:
            events = load_events(match_id, cache_dir=cache_dir)
            passes = extract_passes(events, open_play_only=open_play_only)
            passes["match_id"] = match_id
            # Carry forward match metadata
            for meta_col in ["home_team", "away_team", "match_date",
                              "home_score", "away_score", "stage_label"]:
                if meta_col in matches.columns:
                    passes[meta_col] = row.get(meta_col)
            all_passes.append(passes)
        except Exception as exc:
            logger.warning("Failed to load match %s: %s", match_id, exc)

    return pd.concat(all_passes, ignore_index=True)
