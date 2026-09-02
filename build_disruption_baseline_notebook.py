"""One-off builder script for notebooks/08_player_disruption_baseline.ipynb.

Run once to generate the notebook, then execute it with nbconvert. Not part
of the analysis pipeline itself — safe to delete after the notebook exists.
"""
import json
import uuid


def cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


cells = []

cells.append(cell("markdown", """
# 08 — Passing-Network Vulnerability: Part 1, Baseline Data

### Building the clean player-level and team-match-level dataset for a future disruption analysis

*2018 & 2022 FIFA World Cup · StatsBomb event data*

**Where this fits:** this is the first of a planned 5-part analysis on passing-network
vulnerability — how much a team's structure depends on individual players, tested by
simulating player removal. The full plan is:

1. **Baseline network + player role data** (this notebook)
2. Single-player removal simulation → per-player vulnerability score
3. Targeted vs. random disruption (Monte Carlo)
4. Two-player combination search
5. Case studies, before/after visuals, team robustness ranking

**This notebook does Part 1 only.** No player removal, no simulation, no optimization,
no outcome prediction — just building and validating a trustworthy baseline dataset that
Parts 2-5 will consume. It lives in its own module (`src/network_vulnerability.py`) and
its own notebook so it doesn't touch any existing analysis in this repo.

**Node/edge definition** is unchanged from the rest of the project: node = player,
directed edge = completed pass, weight = pass count. Progressive-pass logic is reused
from `data_loader.extract_passes` (`is_progressive`), not redefined.
"""))

cells.append(cell("code", """
import sys
sys.path.insert(0, '../src')

from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import network_vulnerability as nv

PROC_DIR = Path('../data/processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)

player_path = PROC_DIR / 'player_network_baseline.csv'
team_path = PROC_DIR / 'team_network_baseline.csv'

if player_path.exists() and team_path.exists():
    player_network_baseline = pd.read_csv(player_path)
    team_network_baseline = pd.read_csv(team_path)
else:
    player_network_baseline, team_network_baseline = nv.build_baseline_tables()
    player_network_baseline.to_csv(player_path, index=False)
    team_network_baseline.to_csv(team_path, index=False)

print('player_network_baseline:', player_network_baseline.shape)
print('team_network_baseline  :', team_network_baseline.shape)
"""))

cells.append(cell("markdown", """
## Validation checks

Before this dataset feeds into any disruption simulation, confirm it's internally
consistent: passes are conserved between the raw event data and the graph, recipients
aren't silently missing, and player names don't collide across teams.
"""))

cells.append(cell("code", """
nv.run_validation_checks(player_network_baseline, team_network_baseline)
"""))

cells.append(cell("markdown", """
## Output — `player_network_baseline`

One row per match_id + team + player.
"""))

cells.append(cell("code", """
print(f"Shape: {player_network_baseline.shape}")
player_network_baseline.head(10)
"""))

cells.append(cell("markdown", """
## Output — `team_network_baseline`

One row per match_id + team.
"""))

cells.append(cell("code", """
print(f"Shape: {team_network_baseline.shape}")
team_network_baseline.head(10)
"""))

cells.append(cell("markdown", """
## Summary statistics
"""))

cells.append(cell("code", """
players_per_team_match = player_network_baseline.groupby(['match_id', 'team']).size()
print("Players per team-match:")
print(players_per_team_match.describe())
print()
print("Completed passes per team-match:")
print(team_network_baseline['total_completed_passes'].describe())
"""))

cells.append(cell("code", """
print("Passes sent per player (distribution):")
print(player_network_baseline['passes_sent'].describe())
print()
print("Betweenness centrality per player (distribution):")
print(player_network_baseline['betweenness_centrality'].describe())
print()
print("Unique passing partners per player (distribution):")
print(player_network_baseline['unique_passing_partners'].describe())
"""))

cells.append(cell("markdown", """
## Next step

Part 1 output is cached to `data/processed/player_network_baseline.csv` and
`data/processed/team_network_baseline.csv`. Part 2 (single-player removal simulation)
will load these directly rather than rebuilding networks from raw events each time.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/08_player_disruption_baseline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("wrote notebooks/08_player_disruption_baseline.ipynb with", len(cells), "cells")
