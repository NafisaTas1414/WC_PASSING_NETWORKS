"""One-off builder script for notebooks/12_single_player_removal.ipynb.

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
# 12 — Passing-Network Vulnerability: Part 2, Single-Player Removal Simulation

*2018 & 2022 FIFA World Cup · StatsBomb event data*

**Central question:** for each team-match passing network, if one player is removed,
how much does the network structurally deteriorate relative to its original state?

**Scope:** this notebook does Part 2 only — single-player removal. It does not do
Monte Carlo random-removal comparison, two-player combination search, team robustness
rankings, or outcome analysis (Parts 3-5). It builds directly on the validated Part 1
baseline (`notebooks/11_player_disruption_baseline.ipynb`).

**What a damage score does *not* mean:** "if this player were absent, the team would
lose X% of its passing ability." **What it does mean:** "X% of the network's *observed*
structural capacity is directly attributable to this player's position and connections."
This is a structural disruption simulation on the graph as recorded — it does not model
how teammates would adapt if a player actually left the pitch.
"""))

cells.append(cell("code", """
import sys
sys.path.insert(0, '../src')

from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import network_vulnerability as nv

PROC_DIR = Path('../data/processed')

player_network_baseline = pd.read_csv(PROC_DIR / 'player_network_baseline.csv')

removal_path = PROC_DIR / 'single_player_removal.csv'
if removal_path.exists():
    single_player_removal = pd.read_csv(removal_path)
else:
    single_player_removal = nv.simulate_single_player_removal(player_network_baseline)
    single_player_removal.to_csv(removal_path, index=False)

print(single_player_removal.shape)
"""))

cells.append(cell("markdown", """
## Validation checks (Task 6)
"""))

cells.append(cell("code", """
nv.run_removal_validation_checks(single_player_removal)
"""))

cells.append(cell("markdown", """
## Task 7 — sanity output

### 1-2. Shape and first 10 rows
"""))

cells.append(cell("code", """
print(f"Shape: {single_player_removal.shape}")
display_cols = [
    'match_id', 'team', 'removed_player', 'total_pass_involvement',
    'low_involvement_flag', 'density_damage', 'edge_damage',
    'largest_component_damage', 'efficiency_damage', 'progressive_capacity_damage',
]
single_player_removal[display_cols].head(10)
"""))

cells.append(cell("markdown", """
### 3. Top 3 players by efficiency damage and progressive-capacity damage, for 5 example team-matches

Deliberately not filtering out low-involvement players here — the point of this check is
to see whether the two damage metrics agree with each other and with raw involvement, not
to draw conclusions yet.
"""))

cells.append(cell("code", """
example_keys = (
    single_player_removal[['match_id', 'team']]
    .drop_duplicates()
    .sample(5, random_state=7)
    .itertuples(index=False)
)

for match_id, team in example_keys:
    sub = single_player_removal[
        (single_player_removal['match_id'] == match_id) & (single_player_removal['team'] == team)
    ]
    print(f"=== match {match_id} — {team} ===")
    print("Top 3 by efficiency damage:")
    print(sub.sort_values('efficiency_damage', ascending=False)
          [['removed_player', 'total_pass_involvement', 'efficiency_damage']].head(3)
          .to_string(index=False))
    print("Top 3 by progressive-capacity damage:")
    print(sub.sort_values('progressive_capacity_damage', ascending=False)
          [['removed_player', 'total_pass_involvement', 'progressive_capacity_damage']].head(3)
          .to_string(index=False))
    print()
"""))

cells.append(cell("markdown", """
### 4. Does removal damage track ordinary involvement/centrality, or add information?

The question isn't whether highly-involved players are "the most vulnerable" — it's
whether the simulation tells us something involvement or centrality alone wouldn't.
"""))

cells.append(cell("code", """
part1_betweenness = player_network_baseline[['match_id', 'team', 'player', 'betweenness_centrality']].rename(
    columns={'player': 'removed_player'}
)
merged = single_player_removal.merge(part1_betweenness, on=['match_id', 'team', 'removed_player'], how='left')

corr_involvement_eff = merged['total_pass_involvement'].corr(merged['efficiency_damage'])
corr_involvement_prog = merged['total_pass_involvement'].corr(merged['progressive_capacity_damage'])
corr_betweenness_eff = merged['betweenness_centrality'].corr(merged['efficiency_damage'])

print(f"total_pass_involvement vs efficiency_damage        : r = {corr_involvement_eff:.3f}")
print(f"total_pass_involvement vs progressive_capacity_damage: r = {corr_involvement_prog:.3f}")
print(f"betweenness_centrality (Part 1) vs efficiency_damage : r = {corr_betweenness_eff:.3f}")
"""))

cells.append(cell("markdown", """
## Next step

Part 2 output is cached to `data/processed/single_player_removal.csv`. Part 3 (targeted
vs. random disruption via Monte Carlo) will load this directly.
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

with open("notebooks/12_single_player_removal.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("wrote notebooks/12_single_player_removal.ipynb with", len(cells), "cells")
