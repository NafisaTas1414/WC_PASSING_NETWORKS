# World Cup Passing Networks

Exploratory data science project using 2022 FIFA World Cup event data to analyze whether passing network structure is associated with how far teams advance in the tournament.

**Research question:** Do teams that reached the quarterfinals exhibit more connected, balanced, and progressive passing networks compared to teams eliminated earlier?

---

## Data

**Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) — Competition ID 43, Season ID 106 (FIFA World Cup 2022)

- 64 matches, all 32 teams
- Event-level pass data: passer, recipient, x/y start/end location, outcome, timestamp
- License: non-commercial research use

**Quarterfinalists (target = 1):** Argentina, Croatia, France, Morocco, Netherlands, Brazil, England, Portugal

---

## Project structure

```
wc_passing_networks/
├── data/
│   ├── raw/                # StatsBomb event JSONs (auto-downloaded)
│   └── processed/          # Cleaned pass table, feature tables
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_network_feature_engineering.ipynb
│   └── 03_modeling_and_results.ipynb
├── src/
│   ├── data_loader.py        # StatsBomb loading, pass cleaning, labeling
│   ├── network_builder.py    # NetworkX directed graph construction
│   ├── feature_engineering.py # Graph metrics + pass quality features
│   ├── modeling.py           # Statistical tests, classifiers, clustering
│   └── visualization.py      # Pitch plots, distributions, radar charts
└── outputs/
    ├── figures/
    └── tables/
```

---

## Setup

```bash
pip install statsbombpy networkx pandas numpy scikit-learn scipy matplotlib mplsoccer
```

---

## Running the project

Run the three notebooks in order:

```
01_data_exploration.ipynb      → understand the data, verify fields
02_network_feature_engineering → build networks, compute ~18 features
03_modeling_and_results        → statistical tests, classifiers, profiles
```

Event data is downloaded automatically from StatsBomb on first run and cached to `data/raw/events/` as parquet files. Subsequent runs are fast.

---

## Network definition

| Element | Definition |
|---|---|
| Node | Player |
| Directed edge | Completed pass from passer → recipient |
| Edge weight | Count of completed passes between the pair |
| Node size (viz) | Total passes made |
| Node position | Average x/y of pass start coordinates |

Networks are built at the **team-match** level (one graph per team per match). Features are then aggregated to the team level across all their matches.

---

## Features engineered

**Network topology**
- `network_density` — fraction of possible directed edges present
- `degree_centralization` — how dominated the network is by one hub (Freeman's C)
- `top_player_pass_share` — fraction of all passes made by the busiest passer
- `clustering_coefficient` — prevalence of passing triangles
- `mean_betweenness_centrality` — average reliance on individual bridges
- `average_weighted_degree` — average passing involvement per player

**Pass quality**
- `progressive_pass_share` — fraction advancing ≥10 m toward opponent goal
- `final_third_entry_share` — fraction landing in the attacking third (x ≥ 80)
- `centre_channel_share` — fraction played through the central corridor
- `lateral_imbalance` — asymmetry between left/right side usage

**Team-level aggregates**
Each match-level feature is averaged across matches, with standard deviation and consistency (1 − CV) for key metrics.

---

## Modeling approach

| Layer | Method | Note |
|---|---|---|
| Statistical test | Mann-Whitney U | Non-parametric; effect size = rank-biserial r |
| Classification | Logistic Regression + Random Forest | LOO-CV (correct for n=32) |
| Feature importance | RF mean decrease in impurity | Identifies key network metrics |
| Team profiles | K-Means (k=3) + PCA | Reveals tactical archetypes |

**Target variable:** `reached_quarterfinal` (binary, 8 of 32 teams = 25%)

---

## Key design decisions

**Open-play only filter:** Set `OPEN_PLAY_ONLY = True` in Notebook 02 (default). Corners, free kicks, throw-ins, and kickoffs are excluded to isolate structured tactical passing patterns.

**Minimum edge weight:** Edges require ≥ 2 passes (configurable via `min_passes` in `build_all_networks`). Single passes between substitutes are excluded.

**Leave-One-Out CV:** With n=32 teams, LOO-CV is the statistically correct evaluation strategy. 5-fold CV is also available (`cv_strategy='stratified5'`).

---

## Framing

This is an **association study**, not a prediction system. The goal is to describe what kinds of passing structures characterize teams that went far in WC 2022, not to claim we can predict future tournaments. The long-term vision is to apply the same pipeline to WC 2026 event data once StatsBomb (or another provider) releases it.
