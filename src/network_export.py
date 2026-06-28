"""Shared helpers for building passing-network JSON payloads."""

import numpy as np
import pandas as pd
from pathlib import Path

EVENTS_18 = Path('data/raw/2018/events')
MATCHES_18 = Path('data/raw/2018/matches.csv')

STAGE_MAP = {
    'Group Stage': 'group',
    'Round of 16': 'R16',
    'Quarter-finals': 'QF',
    'Semi-finals': 'SF',
    'Third-place Play-off': '3rd',
    '3rd Place Final': '3rd',
    'Final': 'Final',
}
STAGE_RANK = {'group': 1, 'R16': 2, 'QF': 3, 'SF': 4, '3rd': 5, 'Final': 6}

OVERRIDES = {
    'Nahuel Molina Lucero': 'Molina',
    'Ángel Fabián Di María Hernández': 'Di María',
    'Lionel Andrés Messi Cuccittini': 'Messi',
    'Rodrigo Javier De Paul': 'De Paul',
    'Diogo José Teixeira da Silva': 'Diogo Jota',
    'Nuno Miguel Soares Pereira Troco': 'Nuno Mendes',
    'João Pedro Cavaco Cancelo': 'Cancelo',
    'Raphaël Adelino José Guerreiro': 'Guerreiro',
    'Rúben Santos Gato Alves Dias': 'Rúben Dias',
    'João Mário Naval Costa Eduardo': 'João Mário',
    'André Filipe Tavares Gomes': 'André Gomes',
    'William Silva de Carvalho': 'W. Carvalho',
    'Bernardo Mota Veiga de Carvalho e Silva': 'Bernardo Silva',
    'Bruno Miguel Borges Fernandes': 'Bruno Fernandes',
    'Cristiano Ronaldo dos Santos Aveiro': 'Ronaldo',
    'Goncalo Bernardo Inácio': 'Gonçalo Inácio',
    'Virgil van Dijk': 'Van Dijk',
    'Kevin De Bruyne': 'De Bruyne',
    'Memphis Depay': 'Depay',
    'Frenkie de Jong': 'F. de Jong',
    'Daley Blind': 'Blind',
    'Matthijs de Ligt': 'De Ligt',
    'Denzel Justus Morris Dumfries': 'Dumfries',
    'Steven Julio Berghuis': 'Berghuis',
}
PRE = {'de', 'di', 'van', 'von', 'del', 'da', 'dos', 'al', 'bin', 'el', 'af'}


def load_2018_matches() -> pd.DataFrame:
    matches = pd.read_csv(MATCHES_18)
    stage_col = next(
        (c for c in ('competition_stage', 'stage') if c in matches.columns), None
    )
    if stage_col:
        matches['stage_label'] = matches[stage_col].map(STAGE_MAP).fillna(matches[stage_col])
    return matches


def short_name(n):
    if n in OVERRIDES:
        return OVERRIDES[n]
    p = n.strip().split()
    if len(p) <= 2:
        return p[-1]
    if len(p) == 3:
        return p[2]
    c = p[2]
    if c.lower() in PRE and len(p) > 3:
        return c + ' ' + p[3]
    return c


def _coord(v, i):
    try:
        return float(v[i])
    except Exception:
        return None


def build_team_network(ev, team):
    tp = ev[
        (ev['type'] == 'Pass') &
        (ev['pass_outcome'].isna()) &
        (ev['team'] == team)
    ].copy()

    if len(tp) == 0:
        return None

    tp['x'] = tp['location'].apply(lambda v: _coord(v, 0))
    tp['y'] = tp['location'].apply(lambda v: _coord(v, 1))

    has_end = 'pass_end_location' in tp.columns
    if has_end:
        tp['ex'] = tp['pass_end_location'].apply(lambda v: _coord(v, 0))
        tp['is_prog'] = (tp['ex'] - tp['x']) >= 10
        tp['is_ft'] = (tp['x'] < 80) & (tp['ex'] >= 80)
    else:
        tp['is_prog'] = False
        tp['is_ft'] = False

    tp['is_h1'] = tp['period'] == 1 if 'period' in tp.columns else True
    tp['is_op'] = tp['pass_type'].isna() if 'pass_type' in tp.columns else True

    node_data = {}
    for _, row in tp.iterrows():
        player = row.get('player')
        recip = row.get('pass_recipient')
        if pd.isna(player):
            continue
        if player not in node_data:
            node_data[player] = {'xs': [], 'ys': [], 'sent': 0, 'recv': 0}
        node_data[player]['sent'] += 1
        x, y = row.get('x'), row.get('y')
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            node_data[player]['xs'].append(x)
        if y is not None and not (isinstance(y, float) and np.isnan(y)):
            node_data[player]['ys'].append(y)
        if not pd.isna(recip):
            if recip not in node_data:
                node_data[recip] = {'xs': [], 'ys': [], 'sent': 0, 'recv': 0}
            node_data[recip]['recv'] += 1

    inv = {p: d['sent'] + d['recv'] for p, d in node_data.items()}
    ranked = sorted(inv.keys(), key=lambda p: -inv[p])

    nodes = [
        {
            'id': p,
            'x': round(float(np.mean(node_data[p]['xs'])), 1) if node_data[p]['xs'] else 60.0,
            'y': round(float(np.mean(node_data[p]['ys'])), 1) if node_data[p]['ys'] else 40.0,
            'sent': node_data[p]['sent'],
            'recv': node_data[p]['recv'],
            'rank': i + 1,
        }
        for i, p in enumerate(ranked)
    ]

    edge_data = {}
    for _, row in tp.iterrows():
        passer = row.get('player')
        recip = row.get('pass_recipient')
        if pd.isna(passer) or pd.isna(recip):
            continue
        key = (str(passer), str(recip))
        if key not in edge_data:
            edge_data[key] = {'w': 0, 'h1': 0, 'h2': 0, 'prog': 0, 'ft': 0, 'op': 0}
        edge_data[key]['w'] += 1
        if bool(row.get('is_h1', True)):
            edge_data[key]['h1'] += 1
        else:
            edge_data[key]['h2'] += 1
        if bool(row.get('is_prog', False)):
            edge_data[key]['prog'] += 1
        if bool(row.get('is_ft', False)):
            edge_data[key]['ft'] += 1
        if bool(row.get('is_op', True)):
            edge_data[key]['op'] += 1

    edges = [{'s': s, 't': t, **v} for (s, t), v in edge_data.items()]

    n = len(nodes)
    max_poss = n * (n - 1) if n > 1 else 1
    density = len(edges) / max_poss if max_poss else 0

    top_player = ranked[0] if ranked else ''
    top_total = inv.get(top_player, 0)
    total_sum = sum(inv.values())
    top_rel = top_total / total_sum if total_sum else 0
    top_edge = max(edges, key=lambda e: e['w']) if edges else None
    total_prog = int(sum(e['prog'] for e in edges))
    total_ft = int(sum(e['ft'] for e in edges))

    return {
        'metrics': {
            'completed_passes': len(tp),
            'unique_pairs': len(edges),
            'num_players': n,
            'density': round(density, 4),
            'most_involved': top_player,
            'most_involved_total': top_total,
            'top_reliance': round(top_rel, 4),
            'top_pair': f'{short_name(top_edge["s"])} → {short_name(top_edge["t"])}' if top_edge else '',
            'top_pair_full': f'{top_edge["s"]} → {top_edge["t"]}' if top_edge else '',
            'top_pair_count': top_edge['w'] if top_edge else 0,
            'total_prog': total_prog,
            'total_ft': total_ft,
        },
        'nodes': nodes,
        'edges': edges,
    }


def build_data_2018() -> dict:
    matches18 = load_2018_matches().sort_values('match_date').reset_index(drop=True)
    match_list, networks = [], {}
    for _, mrow in matches18.iterrows():
        mid = int(mrow['match_id'])
        home, away = str(mrow['home_team']), str(mrow['away_team'])
        match_list.append({
            'match_id': mid,
            'home_team': home,
            'away_team': away,
            'home_score': int(mrow.get('home_score', 0)),
            'away_score': int(mrow.get('away_score', 0)),
            'stage': str(mrow.get('stage_label', 'group')),
            'stage_ord': STAGE_RANK.get(str(mrow.get('stage_label', 'group')), 1),
            'date': str(mrow.get('match_date', '')),
        })
        path = EVENTS_18 / f'events_{mid}.parquet'
        if not path.exists():
            continue
        ev = pd.read_parquet(path)
        hnet = build_team_network(ev, home)
        anet = build_team_network(ev, away)
        if hnet:
            networks[f'{mid}_{home}'] = hnet
        if anet:
            networks[f'{mid}_{away}'] = anet
    return {'matches': match_list, 'networks': networks}
