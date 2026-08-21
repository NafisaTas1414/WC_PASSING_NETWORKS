# World Cup Passing Networks

Data science project using StatsBomb event data from the 2018 and 2022 FIFA World Cups to study how passing-network structure relates to tournament advancement, match outcomes, and individual-player structural importance.

**Core research question:** Do teams that advance further exhibit more connected, balanced, and progressive passing networks than teams eliminated earlier — and how much of a team's passing structure depends on any single player?

---

## Data

**Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) — Competition ID 43 (FIFA World Cup), Season IDs for 2018 and 2022

- 128 matches total (64 per tournament), all teams, all stages
- Event-level data: passer, recipient, x/y start/end location, outcome, timestamp, defensive events, shots
- License: non-commercial research use

**Quarterfinalists, 2022 (target = 1):** Argentina, Croatia, France, Morocco, Netherlands, Brazil, England, Portugal

---

## Project structure

```
wc_passing_networks/
├── data/
│   ├── raw/                # StatsBomb event parquets (auto-downloaded, cached), 2018 + 2022
│   └── processed/          # Cleaned pass tables, feature tables, vulnerability-analysis outputs
├── notebooks/               # 01–15, run in order — see below
├── src/
│   ├── data_loader.py           # StatsBomb loading, pass cleaning, labeling
│   ├── network_builder.py       # NetworkX directed graph construction
│   ├── network_vulnerability.py # Player/pair removal simulation (Parts 1–5)
│   ├── network_export.py        # Dashboard JSON export helpers
│   ├── insights_builder.py      # Dominance/defensive insights for the dashboard
│   ├── pipeline.py              # Multi-tournament data pipeline
│   ├── visualization.py         # Pitch plots, distributions, network diagrams
│   └── pass_direction_analysis.py # Exploratory — pass-direction classifier, not yet wired into a notebook
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── web/                # Interactive dashboard (index.html, self-contained)
└── build_*.py               # One-off notebook/dashboard generator scripts (repo root)
```

---

## Setup

```bash
pip install statsbombpy networkx pandas numpy scikit-learn scipy shap matplotlib mplsoccer jupyter nbconvert
```

Event data downloads automatically from StatsBomb on first run and is cached to `data/raw/` as parquet files. Subsequent runs are fast.

Notebooks can be run interactively in Jupyter, or regenerated + executed from the repo root via the matching `build_*.py` script, e.g.:
```bash
python3 build_case_studies_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/15_tactical_case_studies.ipynb
```

---

## Notebooks

Run in order — each stage caches its output to `data/processed/`, so later notebooks load cached CSVs instead of rebuilding from raw events.

**Pipeline**
| # | Notebook | Produces |
|---|---|---|
| 01 | `data_exploration` | `pass_summary.csv` — initial StatsBomb data exploration |
| 02 | `network_feature_engineering` | `pass_network_edges.csv` / `pass_network_nodes.csv` — directed weighted passing networks per team-match |
| 03 | `team_match_features` | `team_match_network_features.csv` — team-match feature table (128 rows, 2022) |
| 04 | `normalized_network_analysis` | `team_match_network_features_normalized.csv` — rate-based/normalized features, feeds nearly everything downstream |
| 06 | `multi_tournament_pipeline` | `2018_match_features.csv` + group-stage team profiles — extends the pipeline to 2018, combined 2018+2022 dataset |

**Association & predictive analysis**
| # | Notebook | Finding |
|---|---|---|
| 05 | `statistical_association` | Mann-Whitney U across all stages (project's core association finding) + a group-stage-only check that the signal isn't a knockout-round artifact |
| 07 | `historical_qf_modeling` | Cross-tournament classifier (train on 2018, test on unseen 2022) — Random Forest ROC-AUC 0.84 |
| 08 | `passing_dominance_defensive_resistance` | Composite dominance score; the passing-dominant team wins only 41% of matches |
| 09 | `possession_zone_distribution` | Pitch-zone distribution barely predicts match outcome |
| 10 | `passes_per_shot` | Possession efficiency (passes/shot) *does* predict outcome — contrast with 09 |

**Network vulnerability — does structural importance follow raw involvement, or hide from it? (Parts 1–5)**
| # | Notebook | Finding |
|---|---|---|
| 11 | `player_disruption_baseline` | Baseline player/team network tables for every team-match |
| 12 | `single_player_removal` | Simulates removing each player; measures structural damage |
| 13 | `targeted_vs_random_disruption` | The most damaging player is the team's #1 passer only 27% of the time, #1 by betweenness only 22.7% |
| 14 | `two_player_removal` | Tests every eligible player pair — the true optimal pair differs from naively combining the two most damaging individuals in ~33% of team-matches |
| 15 | `tactical_case_studies` | Three contrasting case studies (hidden linchpin, pair synergy, fully redundant network) + a summary framework classifying all 256 team-matches |

---

## Interactive dashboard

`outputs/web/index.html` is a self-contained interactive dashboard (open directly, or serve locally with `python -m http.server` from `outputs/web/`). Select a match to explore its passing networks, dominance meter, and defensive stats. Each team panel also shows its **most structurally critical player** (from notebook 15) with a one-click "Remove Critical Player" simulation, rendered on a fixed visual scale so the structural change is directly comparable to the original network.

---

## Network definition

| Element | Definition |
|---|---|
| Node | Player |
| Directed edge | Completed pass from passer → recipient |
| Edge weight | Count of completed passes between the pair |
| Node position | Average x/y of pass start coordinates |

Networks are built at the **team-match** level (one graph per team per match).

---

## Key findings

- **Association:** quarterfinalists complete significantly more passes, more progressive passes, and show higher network density than eliminated teams (Mann-Whitney U, p<0.05 on 7 of 11 tested features).
- **Prediction:** a Random Forest trained on 2018 network features and tested on unseen 2022 data reaches 0.84 ROC-AUC — genuine out-of-sample generalization, not just cross-validation on one dataset.
- **Dominance ≠ winning:** the passing-dominant team wins only 41% of matches; possession efficiency (passes per shot) predicts outcomes better than raw zone distribution.
- **Structural vulnerability is often hidden:** the single player whose removal does the most damage to a team's passing network is frequently *not* the top passer or the highest-betweenness player by standard centrality — and some of the most damaging *pairs* of players are individually unremarkable.

---

## Framing

This is an **association and structural-simulation study, not a prediction system** for real-world outcomes. The classifiers and network-disruption simulations describe patterns in the *observed* data — they don't claim a team would actually lose X% of its passing ability if a player were literally removed, and small-sample results (e.g. n=32 teams) are treated as descriptive, not confirmatory. See the limitations sections in individual notebooks (particularly 13–15) for details specific to each analysis.
