"""
Pass Direction Analysis
=======================
For every completed pass in every match, classify it as:
  - Forward  : ball moves toward the opponent's goal (x increases)
  - Lateral  : ball moves mostly sideways
  - Backward : ball moves away from the opponent's goal (x decreases)

StatsBomb pitch:  x = 0 (own goal) → 120 (opponent's goal)
                  y = 0 (left) → 80 (right)

Classification uses pass_angle (radians, StatsBomb convention):
  0          = perfectly right (toward positive x, i.e. forward for home team)
  π or -π   = perfectly left (backward)
  ±π/2      = perfectly up/down (lateral)

We classify by x-displacement of the pass:
  dx = end_x - start_x
  dy = end_y - start_y

Direction thresholds (configurable at bottom):
  Forward  : dx >  FWD_THRESHOLD   (net forward gain)
  Backward : dx < -FWD_THRESHOLD   (net backward loss)
  Lateral  : |dx| <= FWD_THRESHOLD (mostly sideways)

Output: data/processed/pass_direction.csv
  match_id, year, team, opponent, result,
  total_completed, pct_forward, pct_lateral, pct_backward,
  won (bool), possession_pct, dominant_team (from existing features)
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────
RAW_DIR      = "data/raw"
PROCESSED    = "data/processed"
OUT_FILE     = os.path.join(PROCESSED, "pass_direction.csv")

# A pass is "forward" if x-gain > this many metres (StatsBomb pitch = 120m long)
# 2.0 means the ball must travel at least 2m toward the opponent goal to count forward
FWD_THRESHOLD = 2.0

# ── Helpers ───────────────────────────────────────────────
def classify_direction(dx: float, threshold: float = FWD_THRESHOLD) -> str:
    if dx > threshold:
        return "forward"
    elif dx < -threshold:
        return "backward"
    else:
        return "lateral"


def load_match_events(match_id: int, year: int) -> pd.DataFrame | None:
    if year == 2022:
        path = os.path.join(RAW_DIR, "events", f"events_{match_id}.parquet")
    else:
        path = os.path.join(RAW_DIR, str(year), "events", f"events_{match_id}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def direction_stats_for_team(passes: pd.DataFrame) -> dict:
    """Given a completed-pass dataframe for one team, return direction percentages."""
    total = len(passes)
    if total == 0:
        return {"total_completed": 0, "pct_forward": None,
                "pct_lateral": None, "pct_backward": None}

    fwd  = (passes["direction"] == "forward").sum()
    lat  = (passes["direction"] == "lateral").sum()
    bwd  = (passes["direction"] == "backward").sum()

    return {
        "total_completed": total,
        "pct_forward":  round(fwd  / total * 100, 1),
        "pct_lateral":  round(lat  / total * 100, 1),
        "pct_backward": round(bwd  / total * 100, 1),
        "n_forward": int(fwd),
        "n_lateral": int(lat),
        "n_backward": int(bwd),
    }


def determine_result(team: str, home_team: str, away_team: str,
                     home_score: int, away_score: int) -> str:
    """Return 'win', 'loss', or 'draw' from the perspective of `team`."""
    is_home = (team == home_team)
    gs = home_score if is_home else away_score
    gc = away_score if is_home else home_score
    if gs > gc:   return "win"
    elif gs < gc: return "loss"
    else:         return "draw"


# ── Main ──────────────────────────────────────────────────
def run():
    records = []

    # Load both years' match metadata
    matches_22 = pd.read_csv(os.path.join(RAW_DIR, "matches.csv"))
    matches_18 = pd.read_csv(os.path.join(RAW_DIR, "2018", "matches.csv"))
    matches_22["year"] = 2022
    matches_18["year"] = 2018
    all_matches = pd.concat([matches_22, matches_18], ignore_index=True)

    total = len(all_matches)
    print(f"Processing {total} matches …")

    for i, row in all_matches.iterrows():
        mid   = int(row["match_id"])
        year  = int(row["year"])
        home  = row["home_team"]
        away  = row["away_team"]
        hs    = int(row["home_score"])
        as_   = int(row["away_score"])

        events = load_match_events(mid, year)
        if events is None:
            print(f"  [{i+1}/{total}] SKIP {mid} (no event file)")
            continue

        # Completed passes only (StatsBomb: null pass_outcome = success)
        passes = events[
            (events["type"] == "Pass") &
            (events["pass_outcome"].isna()) &
            (events["location"].notna()) &
            (events["pass_end_location"].notna())
        ].copy()

        if passes.empty:
            print(f"  [{i+1}/{total}] SKIP {mid} (no completed passes)")
            continue

        # Extract x coordinates
        passes["start_x"] = passes["location"].apply(lambda l: l[0])
        passes["end_x"]   = passes["pass_end_location"].apply(lambda l: l[0])
        passes["dx"]      = passes["end_x"] - passes["start_x"]
        passes["direction"] = passes["dx"].apply(classify_direction)

        # Split by team — StatsBomb flips coordinates so both teams attack right (x→120)
        # No need to flip; each team's events already have their attacking direction as x→120
        for team in [home, away]:
            opponent = away if team == home else home
            team_passes = passes[passes["team"] == team]
            stats = direction_stats_for_team(team_passes)
            result = determine_result(team, home, away, hs, as_)

            records.append({
                "match_id":       mid,
                "year":           year,
                "team":           team,
                "opponent":       opponent,
                "home_score":     hs,
                "away_score":     as_,
                "result":         result,
                **stats,
            })

        print(f"  [{i+1}/{total}] {mid} {home} vs {away} ✓")

    df = pd.DataFrame(records)
    os.makedirs(PROCESSED, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(df)} team-match rows → {OUT_FILE}")
    return df


# ── Quick summary after running ───────────────────────────
def summarise(df: pd.DataFrame):
    print("\n── Direction breakdown by result ──────────────────────")
    summary = (
        df.groupby("result")[["pct_forward", "pct_lateral", "pct_backward"]]
        .mean()
        .round(1)
    )
    print(summary)

    print("\n── Top 10 most forward-passing teams (by match) ────────")
    top_fwd = (
        df.sort_values("pct_forward", ascending=False)
        [["team", "opponent", "year", "result", "pct_forward", "pct_lateral", "pct_backward"]]
        .head(10)
        .to_string(index=False)
    )
    print(top_fwd)

    print("\n── Wins with high backward% (circulators who won) ─────")
    circ_wins = df[(df["result"] == "win") & (df["pct_backward"] > 20)].sort_values(
        "pct_backward", ascending=False
    )[["team", "opponent", "year", "pct_forward", "pct_lateral", "pct_backward"]].head(10)
    print(circ_wins.to_string(index=False))

    print("\n── Average direction split overall ─────────────────────")
    print(df[["pct_forward", "pct_lateral", "pct_backward"]].mean().round(1))


if __name__ == "__main__":
    import sys
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        df = pd.read_csv(OUT_FILE)
        summarise(df)
    else:
        df = run()
        summarise(df)

