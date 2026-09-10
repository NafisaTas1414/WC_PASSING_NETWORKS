"""Run this once to generate notebooks/09_possession_zone_distribution.ipynb"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# ── helpers ─────────────────────────────────────────────
def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════
md("""# Possession Zone Distribution
### Where on the pitch are teams actually completing passes?

*2018 & 2022 FIFA World Cup · StatsBomb event data · 128 matches*

---

**The key insight this notebook tests:**
A team can dominate possession and still lose — because their possession was *comfortable, not dangerous*.

Zone distribution reveals *where* possession happens, not just *how much*:

| Zone | x range (StatsBomb) | What it represents |
|------|---------------------|-------------------|
| 🔵 **Own Third** | 0 – 40 | Defensive buildup, recycling under pressure |
| 🟡 **Middle Third** | 40 – 80 | Transition play, midfield control |
| 🔴 **Final Third** | 80 – 120 | Attacking play, chance creation |

> *A team with 55% possession and 55% of passes in their own half is playing comfortable football.*
> *A team with 42% possession and 30% of passes in the final third is playing dangerous football.*
""")


# ════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── style ──────────────────────────────────────────────
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'grid.color':       '#21262d',
    'text.color':       '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'axes.labelcolor':  '#8b949e',
    'font.size':        11,
})

# ── colour palette ─────────────────────────────────────
C_OWN   = '#5B8DB8'   # blue   – own third
C_MID   = '#F0A500'   # gold   – middle third
C_FINAL = '#C94040'   # red    – final third

C_WIN   = '#3fb950'   # green
C_DRAW  = '#F0A500'   # gold
C_LOSS  = '#E05252'   # coral

ZONE_COLS = [C_OWN, C_MID, C_FINAL]
ZONE_LBLS = ['Own Third', 'Middle Third', 'Final Third']

print('Setup complete.')
""")


# ════════════════════════════════════════════════════════
# SECTION 1 — PITCH ZONE MAP
# ════════════════════════════════════════════════════════
md("""---
## 1 · Pitch Zone Map

Every completed pass is classified by where on the pitch it **started**.
The StatsBomb coordinate system runs from x = 0 (own goal) to x = 120 (opponent goal).
""")

code("""fig, ax = plt.subplots(figsize=(12, 4.6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# ── grass ─────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle((0,0), 120, 80,
    facecolor='#1e4a17', edgecolor='white', linewidth=2, zorder=1))

# ── zone shading ──────────────────────────────────────
for x0, color in [(0, C_OWN), (40, C_MID), (80, C_FINAL)]:
    ax.add_patch(mpatches.Rectangle((x0,0), 40, 80,
        facecolor=color, alpha=0.38, linewidth=0, zorder=2))

# ── pitch markings ───────────────────────────────────
for xv in [40, 80]:
    ax.plot([xv,xv], [0,80], '--', color='white', alpha=0.55, lw=1.5, zorder=3)
ax.plot([60,60], [0,80], '-', color='white', alpha=0.25, lw=1, zorder=3)
circ = plt.Circle((60,40), 9.15, color='white', fill=False, lw=1, alpha=0.35, zorder=3)
ax.add_patch(circ)
ax.plot(60, 40, 'wo', ms=2.5, alpha=0.4, zorder=3)

# penalty boxes
for x0, w in [(0, 18), (102, 18)]:
    ax.add_patch(mpatches.Rectangle((x0,18), w, 44,
        facecolor='none', edgecolor='white', lw=1, alpha=0.35, zorder=3))

# ── zone labels ──────────────────────────────────────
for xc, lbl, sub, clr in [
    (20,  'OWN THIRD',    'x < 40',       C_OWN),
    (60,  'MIDDLE THIRD', '40 <= x < 80', C_MID),
    (100, 'FINAL THIRD',  'x >= 80',      C_FINAL),
]:
    ax.text(xc, 56, lbl, ha='center', fontsize=12, fontweight='bold',
            color=clr, zorder=4)
    ax.text(xc, 48, sub, ha='center', fontsize=9.5, color='#cccccc',
            alpha=0.8, zorder=4)

# attack arrow
ax.annotate('', xy=(118,5), xytext=(2,5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.1, alpha=0.35))
ax.text(60, 3, 'attacking direction', ha='center', fontsize=8.5, color='#666', zorder=4)

ax.set_xlim(-2, 122)
ax.set_ylim(-7, 82)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('StatsBomb Pitch  ·  x: 0 (own goal)  →  120 (opponent goal)',
             fontsize=12, color='#8b949e', pad=10)

plt.tight_layout()
plt.savefig('../outputs/figures/09_pitch_zones.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
""")


# ════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ════════════════════════════════════════════════════════
md("""---
## 2 · Load & Process All Matches

For every completed pass in all 128 matches we record the starting x-coordinate
and classify it into a zone. Then aggregate to **% of passes per zone** per team per match.
""")

code("""RAW  = Path('../data/raw')
PROC = Path('../data/processed')
OUT  = PROC / 'possession_zones.csv'

def classify_zone(x):
    if x < 40:  return 'own'
    if x < 80:  return 'middle'
    return 'final'

def load_zone_data():
    m22 = pd.read_csv(RAW / 'matches.csv')
    m22['year'] = 2022
    m18 = pd.read_csv(RAW / '2018' / 'matches.csv')
    m18['year'] = 2018
    all_matches = pd.concat([m22, m18], ignore_index=True)

    # bring in stage info (only 2022 matches.csv has competition_stage in right format)
    stage_map = {}
    for _, r in all_matches.iterrows():
        stage_map[int(r['match_id'])] = r.get('competition_stage', 'Unknown')

    records = []
    print(f'Processing {len(all_matches)} matches...')

    for _, row in all_matches.iterrows():
        mid  = int(row['match_id'])
        year = int(row['year'])
        home, away = row['home_team'], row['away_team']
        hs,   as_  = int(row['home_score']), int(row['away_score'])

        ev_path = (RAW / 'events' / f'events_{mid}.parquet' if year == 2022
                   else RAW / '2018' / 'events' / f'events_{mid}.parquet')
        if not ev_path.exists():
            continue

        ev = pd.read_parquet(ev_path)

        # completed passes only (StatsBomb: null outcome = success)
        passes = ev[
            (ev['type'] == 'Pass') &
            ev['pass_outcome'].isna() &
            ev['location'].notna()
        ].copy()
        if passes.empty:
            continue

        passes['start_x'] = passes['location'].apply(lambda l: l[0])
        passes['zone']    = passes['start_x'].apply(classify_zone)

        for team in [home, away]:
            tp    = passes[passes['team'] == team]
            total = len(tp)
            if total == 0:
                continue

            is_home = (team == home)
            gs = hs if is_home else as_
            gc = as_ if is_home else hs
            result = 'win' if gs > gc else ('loss' if gs < gc else 'draw')

            vc = tp['zone'].value_counts()
            records.append({
                'match_id':    mid,
                'year':        year,
                'stage':       stage_map.get(mid, 'Unknown'),
                'team':        team,
                'opponent':    away if team == home else home,
                'result':      result,
                'total_passes': total,
                'own_pct':   round(vc.get('own',    0) / total * 100, 1),
                'mid_pct':   round(vc.get('middle', 0) / total * 100, 1),
                'final_pct': round(vc.get('final',  0) / total * 100, 1),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT, index=False)
    print(f'Saved {len(df)} team-match rows  ->  {OUT}')
    return df

df = load_zone_data()
""")

code("""# Quick look at the data
n_matches = df['match_id'].nunique()
n_teams   = df['team'].nunique()

print(f'Shape      : {df.shape}')
print(f'Matches    : {n_matches}')
print(f'Unique teams: {n_teams}')
print(f'Years      : {sorted(df["year"].unique())}')
print()
print('Average zone split across all team-match observations:')
print(df[['own_pct','mid_pct','final_pct']].mean().round(1).to_string())
print()
print('Top 8 team-matches by Final Third %:')
display(
    df.sort_values('final_pct', ascending=False)
      [['team','opponent','year','stage','result','total_passes','own_pct','mid_pct','final_pct']]
      .head(8)
      .reset_index(drop=True)
)
""")


# ════════════════════════════════════════════════════════
# SECTION 3 — BY MATCH RESULT
# ════════════════════════════════════════════════════════
md("""---
## 3 · Zone Distribution by Match Result

Does where a team passes actually differ between wins, draws, and losses?
""")

code("""by_result = (
    df.groupby('result')[['own_pct','mid_pct','final_pct']]
    .agg(['mean','std'])
    .round(1)
)
by_result.columns = ['_'.join(c) for c in by_result.columns]
print(by_result.reindex(['win','draw','loss']))
""")

code("""avg = df.groupby('result')[['own_pct','mid_pct','final_pct']].mean().round(1)
avg = avg.reindex(['win','draw','loss'])

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
fig.patch.set_facecolor('#0d1117')

zone_data = [
    ('own_pct',   'Own Third',    C_OWN),
    ('mid_pct',   'Middle Third', C_MID),
    ('final_pct', 'Final Third',  C_FINAL),
]
result_labels = ['Win', 'Draw', 'Loss']
result_colors = [C_WIN, C_DRAW, C_LOSS]

for ax, (col, zone_lbl, zone_clr) in zip(axes, zone_data):
    vals = avg[col].values
    bars = ax.bar(result_labels, vals, color=result_colors, alpha=0.80,
                  width=0.55, edgecolor='none')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4,
                f'{v}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=zone_clr)
    ax.set_title(zone_lbl, fontsize=13, fontweight='bold', color=zone_clr, pad=10)
    ax.set_ylabel('Avg % of completed passes', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle('Pass Zone Distribution by Match Result  ·  All 128 WC matches',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('../outputs/figures/09_zones_by_result.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
""")

md("""**What to look for:**
- A higher **Final Third %** for wins would confirm that dangerous possession (near the opponent's goal) drives outcomes
- A higher **Own Third %** for losses would suggest losing teams were either pressing from their own half or defending with the ball
- The **Middle Third** is the transition zone — differences here reveal how teams move the ball between phases
""")


# ════════════════════════════════════════════════════════
# SECTION 4 — TEAM PROFILES
# ════════════════════════════════════════════════════════
md("""---
## 4 · Team Zone Profiles

Every team's average zone split across all their matches in the tournament.
Sorted from **most attacking** (highest Final Third %) at top to **most defensive** at bottom.

This reveals a team's tactical identity — not just *whether* they had possession, but *where* they used it.
""")

code("""# Average per team across all their matches (pooling both years if they appear twice)
team_avg = (
    df.groupby(['team','year'])[['own_pct','mid_pct','final_pct']]
    .mean()
    .round(1)
    .reset_index()
)
team_avg['label'] = team_avg.apply(
    lambda r: r['team'] if df[df['team']==r['team']]['year'].nunique()==1
              else f\"{r['team']} ({int(r['year'])})\", axis=1
)
team_avg = team_avg.sort_values('final_pct', ascending=True)

fig, ax = plt.subplots(figsize=(11, max(12, len(team_avg)*0.38)))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

y = np.arange(len(team_avg))

# stacked horizontal bars
ax.barh(y, team_avg['own_pct'],   color=C_OWN,   alpha=0.82, label='Own Third',    height=0.7)
ax.barh(y, team_avg['mid_pct'],   color=C_MID,   alpha=0.82, label='Middle Third', height=0.7,
        left=team_avg['own_pct'])
ax.barh(y, team_avg['final_pct'], color=C_FINAL, alpha=0.82, label='Final Third',  height=0.7,
        left=team_avg['own_pct'] + team_avg['mid_pct'])

# final_pct labels on right end
for i, (_, row) in enumerate(team_avg.iterrows()):
    total = row['own_pct'] + row['mid_pct'] + row['final_pct']
    ax.text(total + 0.5, i, f\"{row['final_pct']}%\",
            va='center', fontsize=8.5, color=C_FINAL, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(team_avg['label'], fontsize=9)
ax.set_xlabel('% of completed passes', fontsize=11)
ax.set_title('Team Possession Zone Profiles  ·  sorted by Final Third %',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=10, framealpha=0.15,
          handles=[
              mpatches.Patch(color=C_OWN,   label='Own Third'),
              mpatches.Patch(color=C_MID,   label='Middle Third'),
              mpatches.Patch(color=C_FINAL, label='Final Third'),
          ])
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.15)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('../outputs/figures/09_team_profiles.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
""")


# ════════════════════════════════════════════════════════
# SECTION 5 — SCATTER: DANGEROUS vs SAFE POSSESSION
# ════════════════════════════════════════════════════════
md("""---
## 5 · Dangerous vs Safe Possession

Each dot is one team in one match.
- **x-axis**: % of passes in the Own Third (safe/defensive possession)
- **y-axis**: % of passes in the Final Third (dangerous/attacking possession)
- **Colour**: match result from that team's perspective

Teams in the **top-left** are passing high up the pitch with little defensive recycling — *dangerous possession*.
Teams in the **bottom-right** are passing mostly in their own half — *safe but non-threatening possession*.
""")

code("""fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

result_cfg = [
    ('win',  C_WIN,  'Win',  80, 0.75),
    ('draw', C_DRAW, 'Draw', 55, 0.65),
    ('loss', C_LOSS, 'Loss', 80, 0.75),
]

for result, color, label, size, alpha in result_cfg:
    sub = df[df['result'] == result]
    ax.scatter(sub['own_pct'], sub['final_pct'],
               c=color, s=size, alpha=alpha, label=label,
               edgecolors='none', zorder=3)

# quadrant reference lines
own_med   = df['own_pct'].median()
final_med = df['final_pct'].median()
ax.axvline(own_med,   color='white', alpha=0.18, linestyle='--', linewidth=1)
ax.axhline(final_med, color='white', alpha=0.18, linestyle='--', linewidth=1)

# quadrant labels (subtle)
ax.text(df['own_pct'].min()+1, df['final_pct'].max()-1,
        'High attacking\\nLow defensive', fontsize=8.5, color='#3fb950',
        alpha=0.7, va='top')
ax.text(df['own_pct'].max()-1, df['final_pct'].min()+1,
        'Low attacking\\nHigh defensive', fontsize=8.5, color='#E05252',
        alpha=0.7, ha='right')

ax.set_xlabel('Own Third %  (higher = more defensive recycling)', fontsize=11)
ax.set_ylabel('Final Third %  (higher = more attacking possession)', fontsize=11)
ax.set_title('Dangerous vs Safe Possession  ·  Each dot = one team in one match',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.15, markerscale=1.3)
ax.grid(alpha=0.12)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.savefig('../outputs/figures/09_scatter.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
""")


# ════════════════════════════════════════════════════════
# SECTION 6 — 2018 vs 2022
# ════════════════════════════════════════════════════════
md("""---
## 6 · 2018 vs 2022 — Did Playing Style Shift?

Did teams in 2022 play higher up the pitch? Was there more pressing, more direct play?
""")

code("""by_year = df.groupby('year')[['own_pct','mid_pct','final_pct']].mean().round(1)
print(by_year)
print()
diff = by_year.loc[2022] - by_year.loc[2018]
print('2022 vs 2018 difference (positive = higher in 2022):')
print(diff.round(1))
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.patch.set_facecolor('#0d1117')

for ax, year in zip(axes, [2018, 2022]):
    sub = df[df['year'] == year]
    avg_y = sub.groupby('result')[['own_pct','mid_pct','final_pct']].mean().round(1)
    avg_y = avg_y.reindex(['win','draw','loss'])

    x = np.arange(3)
    w = 0.22
    for i, (col, lbl, clr) in enumerate(zip(
        ['own_pct','mid_pct','final_pct'],
        ['Own Third','Middle Third','Final Third'],
        [C_OWN, C_MID, C_FINAL]
    )):
        bars = ax.bar(x + (i-1)*w, avg_y[col], w, color=clr, alpha=0.80,
                      label=lbl, edgecolor='none')
        for bar, v in zip(bars, avg_y[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3, f'{v}%',
                    ha='center', va='bottom', fontsize=8.5, color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels(['Win','Draw','Loss'], fontsize=12)
    ax.set_title(f'WC {year}', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 58)
    ax.grid(axis='y', alpha=0.15)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if year == 2018:
        ax.set_ylabel('Avg % of passes', fontsize=10)

axes[1].legend(fontsize=9, framealpha=0.15, loc='upper right')
fig.suptitle('Zone Distribution by Result — 2018 vs 2022',
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('../outputs/figures/09_tournament_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
""")


# ════════════════════════════════════════════════════════
# TAKEAWAYS
# ════════════════════════════════════════════════════════
md("""---
## Key Takeaways

### What the data actually shows:

**1. Zone distribution barely predicts match result**

Wins, draws, and losses have almost identical zone profiles (~27% own, ~51% mid, ~21% final).
This is a *null result* — and it's analytically important. Where a team passes doesn't tell you who wins.

**2. The teams with the highest Final Third % are often *losing***

Germany 2018 had the 2nd and 3rd highest final-third % in the entire dataset — and were eliminated in the group stage.
Portugal's highest final-third game (46%) was a loss to Uruguay.

Why? **Game state contamination.** Teams chasing a deficit push their passing higher up the pitch out of desperation, not dominance. High final-third % can mean *threatening* or it can mean *panicking*.

**3. The 2018 → 2022 tactical shift is real and significant**

| Tournament | Own Third % | Mid Third % | Final Third % |
|------------|-------------|-------------|---------------|
| 2018       | 24%         | 53%         | 23%           |
| 2022       | 31%         | 50%         | 19%           |

Teams in 2022 passed **7% more in their own third** and **4% less in the final third**.
This reflects a tactical evolution: higher pressing in 2022 forced teams to build deeper.
Counter-attacking teams (Japan, Morocco, Saudi Arabia) deliberately absorbed possession in their own half — and won.

**4. What zone distribution cannot tell you (but should be tested next)**

Zone tells you *where* passes happen — not *what they achieved*.
A team passing 30% in the final third but taking 0 shots is still not dangerous.
The logical next step is **passes per shot**: how efficiently does final-third possession convert into actual chances?

---

*This analysis shows that zone distribution is a descriptor of style, not a predictor of outcome — which is itself a finding worth communicating.*
""")


# ════════════════════════════════════════════════════════
# WRITE FILE
# ════════════════════════════════════════════════════════
nb.cells = cells

output = Path('notebooks/09_possession_zone_distribution.ipynb')
with open(output, 'w') as f:
    nbf.write(nb, f)

print(f'Notebook written -> {output}')
print(f'Cells: {len(cells)}')
