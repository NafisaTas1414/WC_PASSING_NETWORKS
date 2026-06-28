"""Build match-level dominance and defensive resistance insights for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

EVENTS_22 = Path('data/raw/events')
EVENTS_18 = Path('data/raw/2018/events')
DATA_DIR = Path('data/processed')

DOMINANCE_METRICS = [
    'completed_passes',
    'network_density',
    'progressive_passes',
    'final_third_entries',
    'key_passes',
]

DISPLAY_METRICS = [
    ('completed_passes', 'Passes'),
    ('progressive_passes', 'Progressive'),
    ('final_third_entries', 'Final third'),
    ('key_passes', 'Key passes'),
    ('network_density', 'Density'),
]

DEFENSIVE_POSITION_KEYWORDS = [
    'Back', 'Wing Back', 'Defensive Midfield', 'Goalkeeper',
]

DEFENSIVE_METRICS = [
    'clearances', 'blocks', 'interceptions', 'tackles',
    'pressures', 'ball_recoveries',
]

DEFENSIVE_LABELS = {
    'clearances': 'Clearances',
    'blocks': 'Blocks',
    'interceptions': 'Interceptions',
    'tackles': 'Tackles',
    'pressures': 'Pressures',
    'ball_recoveries': 'Recoveries',
}


def _coord(v, i):
    try:
        return float(v[i])
    except Exception:
        return None


def short_name(n: str, overrides: Optional[dict] = None) -> str:
    overrides = overrides or {}
    if n in overrides:
        return overrides[n]
    p = n.strip().split()
    if len(p) <= 2:
        return p[-1]
    if len(p) == 3:
        return p[2]
    pre = {'de', 'di', 'van', 'von', 'del', 'da', 'dos', 'al', 'bin', 'el', 'af'}
    c = p[2]
    if c.lower() in pre and len(p) > 3:
        return c + ' ' + p[3]
    return c


def load_team_matches() -> pd.DataFrame:
    df_2022 = pd.read_csv(DATA_DIR / 'team_match_network_features_normalized.csv')
    df_2022['tournament_year'] = 2022
    df_2018 = pd.read_csv(DATA_DIR / '2018_match_features.csv')
    return pd.concat([df_2022, df_2018], ignore_index=True)


def load_events(match_id: int, year: int) -> Optional[pd.DataFrame]:
    path = (EVENTS_22 if year == 2022 else EVENTS_18) / f'events_{match_id}.parquet'
    return pd.read_parquet(path) if path.exists() else None


def is_defensive_position(position_str) -> bool:
    if pd.isna(position_str):
        return False
    return any(kw.lower() in str(position_str).lower() for kw in DEFENSIVE_POSITION_KEYWORDS)


def get_defensive_stats(events: pd.DataFrame, team: str) -> pd.DataFrame:
    team_ev = events[events['team'] == team].copy()
    if len(team_ev) == 0:
        return pd.DataFrame()

    if 'position' in team_ev.columns:
        player_positions = (
            team_ev[team_ev['position'].notna()]
            .groupby('player')['position']
            .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
        )
    else:
        player_positions = pd.Series(dtype=str)

    rows = []
    for player in team_ev['player'].dropna().unique():
        pev = team_ev[team_ev['player'] == player]
        position = player_positions.get(player, None)
        if not is_defensive_position(position):
            continue

        row = {'player': player, 'position': str(position)}
        row['clearances'] = int((pev['type'] == 'Clearance').sum())
        row['blocks'] = int((pev['type'] == 'Block').sum())
        row['interceptions'] = int((pev['type'] == 'Interception').sum())
        row['ball_recoveries'] = int((pev['type'] == 'Ball Recovery').sum())
        if 'duel_type' in pev.columns:
            row['tackles'] = int(((pev['type'] == 'Duel') & (pev['duel_type'] == 'Tackle')).sum())
        else:
            row['tackles'] = 0
        row['pressures'] = int((pev['type'] == 'Pressure').sum())
        rows.append(row)

    return pd.DataFrame(rows)


def _team_metric_row(row: pd.Series) -> dict:
    out = {}
    for m in DOMINANCE_METRICS:
        val = row.get(m)
        if pd.isna(val):
            val = 0
        out[m] = round(float(val), 4) if m == 'network_density' else int(val)
    return out


def _def_summary(df: pd.DataFrame, overrides: dict) -> dict:
    if df is None or len(df) == 0:
        return {'totals': {m: 0 for m in DEFENSIVE_METRICS}, 'leaders': []}

    df = df.copy()
    df['def_score'] = (
        df['clearances'] + df['blocks'] + df['interceptions']
        + df['tackles'] + df['pressures'] * 0.25
    )
    totals = {m: int(df[m].sum()) for m in DEFENSIVE_METRICS if m in df.columns}
    leaders = []
    for _, r in df.nlargest(3, 'def_score').iterrows():
        leaders.append({
            'player': short_name(str(r['player']), overrides),
            'player_full': str(r['player']),
            'position': str(r.get('position', '')),
            'clearances': int(r['clearances']),
            'blocks': int(r['blocks']),
            'interceptions': int(r['interceptions']),
            'tackles': int(r['tackles']),
            'pressures': int(r['pressures']),
            'ball_recoveries': int(r['ball_recoveries']),
        })
    return {'totals': totals, 'leaders': leaders}


def compute_global_dominance(team_matches: pd.DataFrame) -> dict[int, dict]:
    """Return match_id -> {score (team_a perspective), team_a, team_b, team_a_result}."""
    rows = []
    for (year, match_id), group in team_matches.groupby(['tournament_year', 'match_id']):
        if len(group) != 2:
            continue
        group = group.sort_values('completed_passes', ascending=False).reset_index(drop=True)
        ta, tb = group.iloc[0], group.iloc[1]
        row = {'match_id': int(match_id), 'year': int(year)}
        for m in DOMINANCE_METRICS:
            row[f'{m}_diff'] = float(ta[m]) - float(tb[m])
        row['team_a'] = str(ta['team'])
        row['team_b'] = str(tb['team'])
        row['team_a_result'] = str(ta['match_result'])
        row['goals_a'] = int(ta['goals_for'])
        row['goals_b'] = int(tb['goals_for'])
        rows.append(row)

    frame = pd.DataFrame(rows)
    diff_cols = [f'{m}_diff' for m in DOMINANCE_METRICS if f'{m}_diff' in frame.columns]
    vals = frame[diff_cols].fillna(0).to_numpy(dtype=float)
    vmin = vals.min(axis=0)
    vmax = vals.max(axis=0)
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
    scaled = (vals - vmin) / span
    frame['dominance_score'] = scaled.mean(axis=1)

    return {
        int(r['match_id']): {
            'score': round(float(r['dominance_score']), 3),
            'team_a': r['team_a'],
            'team_b': r['team_b'],
            'team_a_result': r['team_a_result'],
            'goals_a': int(r['goals_a']),
            'goals_b': int(r['goals_b']),
            'year': int(r['year']),
        }
        for _, r in frame.iterrows()
    }


def _winner_side(home: str, away: str, hs: int, as_: int) -> Optional[str]:
    if hs > as_:
        return 'home'
    if as_ > hs:
        return 'away'
    return None


def _result_for_team(team: str, home: str, away: str, hs: int, as_: int) -> str:
    if hs == as_:
        return 'D'
    if team == home:
        return 'W' if hs > as_ else 'L'
    return 'W' if as_ > hs else 'L'


def build_match_insight(
    match: dict,
    year: int,
    team_matches: pd.DataFrame,
    dominance_global: dict,
    overrides: dict,
    events_cache: dict,
) -> dict:
    mid = int(match['match_id'])
    home = str(match['home_team'])
    away = str(match['away_team'])
    hs = int(match['home_score'])
    as_ = int(match['away_score'])

    home_row = team_matches[(team_matches['match_id'] == mid) & (team_matches['team'] == home)]
    away_row = team_matches[(team_matches['match_id'] == mid) & (team_matches['team'] == away)]
    if len(home_row) == 0 or len(away_row) == 0:
        return {}

    home_stats = _team_metric_row(home_row.iloc[0])
    away_stats = _team_metric_row(away_row.iloc[0])

    dom = dominance_global.get(mid, {})
    score = dom.get('score', 0.5)
    team_a = dom.get('team_a', home)

    if team_a == home:
        home_pct = round(score * 100)
        away_pct = 100 - home_pct
    else:
        away_pct = round(score * 100)
        home_pct = 100 - away_pct

    # Pass-volume leader (notebook team_a) — used for failed-dominance outcome
    if home_stats['completed_passes'] >= away_stats['completed_passes']:
        pass_leader, pass_leader_side = home, 'home'
    else:
        pass_leader, pass_leader_side = away, 'away'

    pass_leader_pct = home_pct if pass_leader_side == 'home' else away_pct

    # Composite leader (who wins on blended passing metrics)
    if home_pct >= away_pct:
        composite_leader, composite_side = home, 'home'
    else:
        composite_leader, composite_side = away, 'away'

    dom_result = _result_for_team(pass_leader, home, away, hs, as_)
    dominant_won = dom_result == 'W'
    failed_dominance = dom_result == 'L'
    close_match = 45 <= pass_leader_pct <= 55

    winner_side = _winner_side(home, away, hs, as_)
    winner = home if winner_side == 'home' else away if winner_side == 'away' else None
    low_poss_winner = winner is not None and winner != pass_leader

    if mid not in events_cache:
        ev = load_events(mid, year)
        events_cache[mid] = ev
    ev = events_cache[mid]

    home_def = _def_summary(get_defensive_stats(ev, home) if ev is not None else pd.DataFrame(), overrides)
    away_def = _def_summary(get_defensive_stats(ev, away) if ev is not None else pd.DataFrame(), overrides)

    return {
        'dominance_pct': pass_leader_pct,
        'home_pct': home_pct,
        'away_pct': away_pct,
        'pass_leader': pass_leader,
        'pass_leader_side': pass_leader_side,
        'dominant_team': pass_leader,
        'dominant_side': pass_leader_side,
        'composite_leader': composite_leader,
        'composite_side': composite_side,
        'dominant_won': dominant_won,
        'failed_dominance': failed_dominance,
        'close_match': close_match,
        'winner': winner,
        'winner_side': winner_side,
        'low_poss_winner': low_poss_winner,
        'goal_diff': hs - as_,
        'metrics': {'home': home_stats, 'away': away_stats},
        'defensive': {'home': home_def, 'away': away_def},
        'scatter': {'x': home_pct, 'y': hs - as_},
    }


def build_tournament_insights(matches: list, year: int, team_matches: pd.DataFrame,
                              dominance_global: dict, overrides: dict) -> dict:
    events_cache: dict = {}
    insights = {}
    for m in matches:
        ins = build_match_insight(m, year, team_matches, dominance_global, overrides, events_cache)
        if ins:
            insights[str(m['match_id'])] = ins
    return insights


def build_context(all_insights: dict, dominance_global: dict, team_matches: pd.DataFrame,
                  overrides: dict) -> dict:
    failed = []
    dominant_wins = 0
    total = len(all_insights)

    for mid_str, ins in all_insights.items():
        if ins.get('dominant_won'):
            dominant_wins += 1
        if ins.get('failed_dominance'):
            mid = int(mid_str)
            dom = dominance_global[mid]
            failed.append({
                'match_id': mid,
                'year': dom['year'],
                'dominance_pct': ins['dominance_pct'],
                'dominant_team': ins['dominant_team'],
                'winner': ins['winner'],
            })

    failed.sort(key=lambda x: -x['dominance_pct'])

    def _avg_rows(rows):
        if not rows:
            return {m: 0 for m in DEFENSIVE_METRICS}
        frame = pd.DataFrame(rows)
        return {m: round(float(frame[m].mean()), 2) for m in DEFENSIVE_METRICS if m in frame.columns}

    low_poss_won_defs = []
    low_poss_lost_defs = []
    events_cache: dict = {}
    for (year, match_id), group in team_matches.groupby(['tournament_year', 'match_id']):
        if len(group) != 2:
            continue
        group = group.sort_values('completed_passes', ascending=False).reset_index(drop=True)
        low_team = str(group.iloc[1]['team'])
        low_result = str(group.iloc[1]['match_result'])
        ev = events_cache.get(match_id)
        if ev is None:
            ev = load_events(int(match_id), int(year))
            events_cache[match_id] = ev
        if ev is None:
            continue
        ddf = get_defensive_stats(ev, low_team)
        if len(ddf) == 0:
            continue
        avg = {m: float(ddf[m].mean()) for m in DEFENSIVE_METRICS if m in ddf.columns}
        if low_result == 'W':
            low_poss_won_defs.append(avg)
        elif low_result == 'L':
            low_poss_lost_defs.append(avg)

    return {
        'total_matches': total,
        'dominant_win_rate': round(dominant_wins / total, 3) if total else 0,
        'failed_dominance_count': len(failed),
        'benchmarks': {
            'low_poss_winners': _avg_rows(low_poss_won_defs),
            'low_poss_losers': _avg_rows(low_poss_lost_defs),
        },
        'top_upsets': failed[:10],
        'defensive_labels': DEFENSIVE_LABELS,
        'display_metrics': [[k, v] for k, v in DISPLAY_METRICS],
    }


def build_all_insights(matches_2022: list, matches_2018: list, overrides: dict) -> tuple:
    team_matches = load_team_matches()
    dominance_global = compute_global_dominance(team_matches)

    ins_2022 = build_tournament_insights(matches_2022, 2022, team_matches, dominance_global, overrides)
    ins_2018 = build_tournament_insights(matches_2018, 2018, team_matches, dominance_global, overrides)
    all_ins = {**ins_2022, **ins_2018}
    context = build_context(all_ins, dominance_global, team_matches, overrides)
    return ins_2022, ins_2018, context
