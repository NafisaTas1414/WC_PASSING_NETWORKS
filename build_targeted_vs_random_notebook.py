"""One-off builder script for notebooks/10_targeted_vs_random_disruption.ipynb.

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
# 10 — Passing-Network Vulnerability: Part 3, Targeted vs. Eligible-Pool Disruption

*2018 & 2022 FIFA World Cup · StatsBomb event data*

**Central question:** for each team-match, does removing the most structurally damaging
player cause substantially more damage than removing a typical (eligible) player from the
same observed network? This distinguishes **distributed/robust** networks (many removals
cause similar damage) from **concentrated/vulnerable** ones (one player's removal is
unusually damaging).

**Scope:** Part 3 only. No two-player removal, no tournament rankings, no outcome
analysis, no visualizations. Builds on the validated Part 1 baseline and Part 2
single-player removal simulation.

**Primary metric:** `efficiency_damage`. `density_damage` is excluded (frequently rises
after removal purely from the shrinking node-count denominator) and
`largest_component_damage` is excluded (near-zero in 98.9% of Part 2 removals).
`progressive_capacity_damage` is kept as a secondary, football-specific metric. No
composite score yet.

**Methodology note:** the "typical player" baseline is the *exact* mean/std over the
fully-observed eligible pool for each team-match — Part 2 already computed damage for
every eligible player, so this is a population statistic, not a Monte Carlo estimate.
Sampling would only add noise to a quantity already known exactly.

**A note on what NOT to headline:** the targeted player is *defined* as the eligible-pool
maximum, so "targeted damage exceeds the pool mean" is mechanically guaranteed — it is not
a finding. The real question is whether that maximum belongs to an ordinary hub (highest
pass volume, highest betweenness) or not. That's what the vulnerability-concentration
check (Section 3 below) actually tests.

**Interpretation:** this measures how unusually dependent the *observed* network is on
its most structurally important participant, relative to other meaningful participants —
not a behavioral counterfactual about what would happen if an opponent man-marked them.
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
single_player_removal = pd.read_csv(PROC_DIR / 'single_player_removal.csv')
"""))

cells.append(cell("markdown", """
## 1. Eligibility rule for the baseline pool

Before applying any rule, check how many players each candidate threshold would exclude.
"""))

cells.append(cell("code", """
nv.report_eligibility_thresholds(single_player_removal)
"""))

cells.append(cell("markdown", """
**Chosen rule:** `total_pass_involvement >= 5` OR `share of team-match involvement >= 2%`.
This excludes 126 / 3,758 players (3.4%) — mostly late substitutes with only a touch or
two — while keeping every meaningful contributor. The same eligible pool is used for both
identifying the targeted player and the baseline statistics, as required.
"""))

cells.append(cell("code", """
robustness_path = PROC_DIR / 'team_match_robustness.csv'
if robustness_path.exists():
    team_match_robustness = pd.read_csv(robustness_path)
else:
    team_match_robustness = nv.build_team_match_robustness(single_player_removal, player_network_baseline)
    team_match_robustness.to_csv(robustness_path, index=False)

print(team_match_robustness.shape)
team_match_robustness.head(10)
"""))

cells.append(cell("markdown", """
## 2. Validation checks
"""))

cells.append(cell("code", """
nv.run_robustness_validation_checks(team_match_robustness, single_player_removal, player_network_baseline)
"""))

cells.append(cell("markdown", """
## 3. Is vulnerability actually concentrated?

**The headline finding is Q3-Q5 below, not the targeted-vs-mean comparison** (which is
mechanical by construction — see the methodology note above).
"""))

cells.append(cell("code", """
n = len(team_match_robustness)

# Q3: how often is the targeted player also #1 by raw pass involvement?
top1_by_involvement = (team_match_robustness['targeted_pass_involvement_rank'] == 1).sum()
print(f"Q3. Targeted player is also #1 by pass involvement: "
      f"{top1_by_involvement} / {n} ({top1_by_involvement/n:.1%})")

# Q4: how often is the targeted player NOT in the top 3 by pass involvement?
not_top3 = (team_match_robustness['targeted_pass_involvement_rank'] > 3).sum()
print(f"Q4. Targeted player is NOT in the top 3 by pass involvement: "
      f"{not_top3} / {n} ({not_top3/n:.1%})")
"""))

cells.append(cell("code", """
# Q5: how often is the targeted player also #1 by betweenness centrality (within their team-match)?
bt_rank = player_network_baseline.copy()
bt_rank['_bt_rank'] = bt_rank.groupby(['match_id', 'team'])['betweenness_centrality'].rank(
    ascending=False, method='min'
)
merged = team_match_robustness.merge(
    bt_rank[['match_id', 'team', 'player', '_bt_rank']],
    left_on=['match_id', 'team', 'targeted_player'],
    right_on=['match_id', 'team', 'player'],
    how='left',
)
top1_by_betweenness = (merged['_bt_rank'] == 1).sum()
print(f"Q5. Targeted player is also #1 by betweenness centrality: "
      f"{top1_by_betweenness} / {n} ({top1_by_betweenness/n:.1%})")
print()
print("=> Structural vulnerability is frequently NOT the player with the ball the most,")
print("   and frequently not even the highest-betweenness player by the standard metric.")
"""))

cells.append(cell("markdown", """
### Absolute / standardized concentration measures

`top1_top2_gap` and `targeted_standardized_gap` = (targeted − eligible mean) / eligible std
are the measures to use going forward — not `targeted_to_random_ratio`, which is unstable
whenever the eligible-pool mean is near zero (105 / 256 team-matches here).
"""))

cells.append(cell("code", """
summary_cols = ['targeted_efficiency_damage', 'eligible_efficiency_damage_mean',
                'eligible_efficiency_damage_std', 'targeted_excess_efficiency_damage',
                'targeted_standardized_gap', 'top1_top2_gap']
team_match_robustness[summary_cols].describe().round(4)
"""))

cells.append(cell("code", """
gap_median = team_match_robustness['top1_top2_gap'].median()
gap_p75 = team_match_robustness['top1_top2_gap'].quantile(0.75)
z_median = team_match_robustness['targeted_standardized_gap'].median()
z_p75 = team_match_robustness['targeted_standardized_gap'].quantile(0.75)

print(f"Median top1-top2 gap: {gap_median:.4f}  |  75th percentile: {gap_p75:.4f}")
print(f"Median standardized gap (z-score): {z_median:.2f}  |  75th percentile: {z_p75:.2f}")
print()
print("Most team-matches show a fairly flat top of the distribution (small median gap),")
print("but the 75th-percentile and max show real cases of one player standing out sharply —")
print("consistent with concentrated vulnerability being the exception, not the rule.")
"""))

cells.append(cell("markdown", """
## 4. Illustrative examples (not case studies, not causal claims)

### A. 5 team-matches with the largest standardized gap (most reliable "stands out" measure)
"""))

cells.append(cell("code", """
example_cols = [
    'team', 'year', 'targeted_player', 'targeted_efficiency_damage',
    'eligible_efficiency_damage_mean', 'targeted_standardized_gap', 'top1_top2_gap',
    'targeted_pass_involvement_rank',
]

team_match_robustness.sort_values('targeted_standardized_gap', ascending=False)[example_cols].head(5)
"""))

cells.append(cell("markdown", """
### B. 5 team-matches with the largest top1-top2 gap
"""))

cells.append(cell("code", """
team_match_robustness.sort_values('top1_top2_gap', ascending=False)[example_cols].head(5)
"""))

cells.append(cell("markdown", """
## Next step

Part 3 output is cached to `data/processed/team_match_robustness.csv`. Part 4 (two-player
combination search) will use the same eligibility pool and `efficiency_damage` metric to
test whether joint removal reveals interaction effects beyond the individually most
damaging players.
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

with open("notebooks/10_targeted_vs_random_disruption.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("wrote notebooks/10_targeted_vs_random_disruption.ipynb with", len(cells), "cells")
