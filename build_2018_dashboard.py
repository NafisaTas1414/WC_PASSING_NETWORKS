#!/usr/bin/env python3
"""
Generate DATA_2018 network JSON and patch index.html to support
a WC 2022 / WC 2018 tournament toggle.
"""

import sys, json, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

EVENTS_18 = Path('data/raw/2018/events')
MATCHES_18 = Path('data/raw/2018/matches.csv')
HTML_PATH  = Path('outputs/web/index.html')

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


def load_2018_matches() -> pd.DataFrame:
    matches = pd.read_csv(MATCHES_18)
    stage_col = next(
        (c for c in ('competition_stage', 'stage') if c in matches.columns), None
    )
    if stage_col:
        matches['stage_label'] = matches[stage_col].map(STAGE_MAP).fillna(matches[stage_col])
    return matches

# ── Short name (mirrors the JS shortName function already in the HTML) ──────
OVERRIDES = {
    'Nahuel Molina Lucero':                  'Molina',
    'Ángel Fabián Di María Hernández':       'Di María',
    'Lionel Andrés Messi Cuccittini':        'Messi',
    'Rodrigo Javier De Paul':                'De Paul',
    'Diogo José Teixeira da Silva':          'Diogo Jota',
    'Nuno Miguel Soares Pereira Troco':      'Nuno Mendes',
    'João Pedro Cavaco Cancelo':             'Cancelo',
    'Raphaël Adelino José Guerreiro':        'Guerreiro',
    'Rúben Santos Gato Alves Dias':          'Rúben Dias',
    'João Mário Naval Costa Eduardo':        'João Mário',
    'André Filipe Tavares Gomes':            'André Gomes',
    'William Silva de Carvalho':             'W. Carvalho',
    'Bernardo Mota Veiga de Carvalho e Silva': 'Bernardo Silva',
    'Bruno Miguel Borges Fernandes':         'Bruno Fernandes',
    'Cristiano Ronaldo dos Santos Aveiro':   'Ronaldo',
    'Goncalo Bernardo Inácio':               'Gonçalo Inácio',
    'Virgil van Dijk':   'Van Dijk',
    'Kevin De Bruyne':   'De Bruyne',
    'Memphis Depay':     'Depay',
    'Frenkie de Jong':   'F. de Jong',
    'Daley Blind':       'Blind',
    'Matthijs de Ligt':  'De Ligt',
    'Denzel Justus Morris Dumfries': 'Dumfries',
    'Steven Julio Berghuis': 'Berghuis',
}
PRE = {'de','di','van','von','del','da','dos','al','bin','el','af'}

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

# ── Build passing network for one team in one match ──────────────────────────
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
        tp['ex']      = tp['pass_end_location'].apply(lambda v: _coord(v, 0))
        tp['is_prog'] = (tp['ex'] - tp['x']) >= 10
        tp['is_ft']   = (tp['x'] < 80) & (tp['ex'] >= 80)
    else:
        tp['is_prog'] = False
        tp['is_ft']   = False

    tp['is_h1'] = tp['period'] == 1 if 'period' in tp.columns else True
    tp['is_op'] = tp['pass_type'].isna() if 'pass_type' in tp.columns else True

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_data = {}
    for _, row in tp.iterrows():
        player = row.get('player')
        recip  = row.get('pass_recipient')
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

    inv    = {p: d['sent'] + d['recv'] for p, d in node_data.items()}
    ranked = sorted(inv.keys(), key=lambda p: -inv[p])

    nodes = [
        {
            'id':   p,
            'x':    round(float(np.mean(node_data[p]['xs'])), 1) if node_data[p]['xs'] else 60.0,
            'y':    round(float(np.mean(node_data[p]['ys'])), 1) if node_data[p]['ys'] else 40.0,
            'sent': node_data[p]['sent'],
            'recv': node_data[p]['recv'],
            'rank': i + 1,
        }
        for i, p in enumerate(ranked)
    ]

    # ── Edges ────────────────────────────────────────────────────────────────
    edge_data = {}
    for _, row in tp.iterrows():
        passer = row.get('player')
        recip  = row.get('pass_recipient')
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

    # ── Metrics ──────────────────────────────────────────────────────────────
    n        = len(nodes)
    max_poss = n * (n - 1) if n > 1 else 1
    density  = len(edges) / max_poss if max_poss else 0

    top_player = ranked[0] if ranked else ''
    top_total  = inv.get(top_player, 0)
    total_sum  = sum(inv.values())
    top_rel    = top_total / total_sum if total_sum else 0
    top_edge   = max(edges, key=lambda e: e['w']) if edges else None
    total_prog = int(sum(e['prog'] for e in edges))
    total_ft   = int(sum(e['ft'] for e in edges))

    return {
        'metrics': {
            'completed_passes':    len(tp),
            'unique_pairs':        len(edges),
            'num_players':         n,
            'density':             round(density, 4),
            'most_involved':       top_player,
            'most_involved_total': top_total,
            'top_reliance':        round(top_rel, 4),
            'top_pair':      f'{short_name(top_edge["s"])} → {short_name(top_edge["t"])}' if top_edge else '',
            'top_pair_full': f'{top_edge["s"]} → {top_edge["t"]}' if top_edge else '',
            'top_pair_count': top_edge['w'] if top_edge else 0,
            'total_prog':    total_prog,
            'total_ft':      total_ft,
        },
        'nodes': nodes,
        'edges': edges,
    }

if __name__ == '__main__':
    # ── Generate DATA_2018 ───────────────────────────────────────────────────
    print('Loading 2018 match list...')
    matches18 = load_2018_matches()
    matches18 = matches18.sort_values('match_date').reset_index(drop=True)
    print(f'  {len(matches18)} matches found')

    match_list = []
    networks   = {}

    for i, (_, mrow) in enumerate(matches18.iterrows(), 1):
    mid       = int(mrow['match_id'])
    home      = str(mrow['home_team'])
    away      = str(mrow['away_team'])
    hs        = int(mrow.get('home_score', 0))
    as_       = int(mrow.get('away_score', 0))
    stage     = str(mrow.get('stage_label', 'group'))
    date      = str(mrow.get('match_date', ''))
    stage_ord = STAGE_RANK.get(stage, 1)

    match_list.append({
        'match_id':   mid,
        'home_team':  home,
        'away_team':  away,
        'home_score': hs,
        'away_score': as_,
        'stage':      stage,
        'stage_ord':  stage_ord,
        'date':       date,
    })

    path = EVENTS_18 / f'events_{mid}.parquet'
    if not path.exists():
        print(f'  [{i:02}/64] MISSING parquet: {mid}')
        continue

    ev   = pd.read_parquet(path)
    hnet = build_team_network(ev, home)
    anet = build_team_network(ev, away)
    if hnet:
        networks[f'{mid}_{home}'] = hnet
    if anet:
        networks[f'{mid}_{away}'] = anet

    print(f'  [{i:02}/64] {home} vs {away} [{stage}]')

    data_2018      = {'matches': match_list, 'networks': networks}
    data_2018_json = json.dumps(data_2018, separators=(',', ':'), ensure_ascii=False)
    print(f'\nDATA_2018: {len(match_list)} matches, {len(networks)} team networks')
    print(f'JSON size: {len(data_2018_json):,} bytes ({len(data_2018_json)/1024:.0f} KB)')

    # ── Patch index.html ─────────────────────────────────────────────────────────
    print('\nPatching HTML...')
    with open(HTML_PATH, encoding='utf-8') as f:
    html = f.read()

    original_size = len(html)

    # 1. Wrap sidebar-building code in buildSidebar() FIRST (while DATA.matches is still original)
    OLD_SIDEBAR = (
    "const listEl=document.getElementById('match-list');\n"
    "STAGES.forEach(st=>{\n"
    "  const sm=DATA.matches.filter(m=>m.stage===st.key);\n"
    "  if(!sm.length)return;\n"
    "  const lbl=document.createElement('div');\n"
    "  lbl.className='stage-label';lbl.textContent=st.label;\n"
    "  listEl.appendChild(lbl);\n"
    "  sm.forEach(m=>{\n"
    "    const it=document.createElement('div');\n"
    "    it.className='match-item';\n"
    "    it.innerHTML=`<div class=\"match-teams\">${m.home_team} vs ${m.away_team}</div>`+\n"
    "      `<div class=\"match-score\">${m.home_score}–${m.away_score} · ${m.date}</div>`;\n"
    "    it.addEventListener('click',()=>selectMatch(m,it));\n"
    "    listEl.appendChild(it);\n"
    "  });\n"
    "});"
    )

    NEW_SIDEBAR = (
    "function buildSidebar(){\n"
    "  const listEl=document.getElementById('match-list');\n"
    "  listEl.innerHTML='';\n"
    "  STAGES.forEach(st=>{\n"
    "    const sm=CUR.matches.filter(m=>m.stage===st.key);\n"
    "    if(!sm.length)return;\n"
    "    const lbl=document.createElement('div');\n"
    "    lbl.className='stage-label';lbl.textContent=st.label;\n"
    "    listEl.appendChild(lbl);\n"
    "    sm.forEach(m=>{\n"
    "      const it=document.createElement('div');\n"
    "      it.className='match-item';\n"
    "      it.innerHTML=`<div class=\"match-teams\">${m.home_team} vs ${m.away_team}</div>`+\n"
    "        `<div class=\"match-score\">${m.home_score}–${m.away_score} · ${m.date}</div>`;\n"
    "      it.addEventListener('click',()=>selectMatch(m,it));\n"
    "      listEl.appendChild(it);\n"
    "    });\n"
    "  });\n"
    "}\n"
    "buildSidebar();"
    )

    if 'function buildSidebar()' in html:
    print('  [SKIP] buildSidebar() already present')
    elif OLD_SIDEBAR in html:
    html = html.replace(OLD_SIDEBAR, NEW_SIDEBAR, 1)
    print('  [OK] Wrapped sidebar in buildSidebar()')
    else:
    print('  [WARN] Could not find sidebar block — checking variant...')
    # Try with the score en-dash as literal character
    alt = OLD_SIDEBAR.replace('–', '–').replace('·', '·')
    if alt in html:
        html = html.replace(alt, NEW_SIDEBAR, 1)
        print('  [OK] Wrapped sidebar (alt encoding)')
    else:
        print('  [FAIL] Sidebar block not found — manual check needed')

    # 2. Rename const DATA → const DATA_2022 (skip if already patched)
    if 'const DATA_2022' in html:
    print('  [SKIP] DATA_2022 already present')
    else:
    html = html.replace('const DATA = {', 'const DATA_2022 = {', 1)
    print('  [OK] Renamed DATA → DATA_2022')

    # 3. Insert DATA_2018 + CUR variable after DATA_2022 (JS section only).
    #    The generic '\n\n/* ── Sidebar' marker also appears in <style> and
    #    would corrupt the page if matched there first.
    INSERT_AFTER = '\n\n/* ── Sidebar ───────────────────────────────────── */\nconst STAGES'
    if 'const DATA_2018' in html:
    print('  [SKIP] DATA_2018 already present — refreshing payload only')
    import re
    html = re.sub(
        r'const DATA_2018 = \{.*?\};\n',
        f'const DATA_2018 = {data_2018_json};\n',
        html,
        count=1,
        flags=re.DOTALL,
    )
    elif INSERT_AFTER not in html:
    raise RuntimeError(
        'Could not find JS sidebar marker after DATA_2022 — '
        'index.html structure may have changed'
    )
    else:
    INJECTION = (
        f'\nconst DATA_2018 = {data_2018_json};\n'
        'let CUR = DATA_2022;\n\n'
        '/* ── Sidebar ───────────────────────────────────── */\nconst STAGES'
    )
    html = html.replace(INSERT_AFTER, INJECTION, 1)
    print('  [OK] Injected DATA_2018 and CUR variable')

    # 4. Replace remaining DATA.networks references (sidebar already handled)
    if 'DATA.networks' in html:
    html = html.replace('DATA.networks', 'CUR.networks')
    print('  [OK] Updated DATA.networks → CUR.networks')
    else:
    print('  [SKIP] DATA.networks already updated')

    # 5. Fix STAGES array: '3rd Place Final' key should be '3rd' to match pipeline output
    if "{key:'3rd',label:'3rd Place Final'}" in html:
    print('  [SKIP] STAGES 3rd key already fixed')
    else:
    html = html.replace(
        "{key:'3rd Place Final',label:'3rd Place Final'}",
        "{key:'3rd',label:'3rd Place Final'}"
    )
    print('  [OK] Fixed STAGES 3rd key')

    # 6. Change sidebar-title
    if 'id="btn22"' in html:
    print('  [SKIP] Tournament toggle already in sidebar')
    else:
    html = html.replace(
        '<div id="sidebar-title">⚽ WC 2022 Matches</div>',
        '<div id="sidebar-title">⚽ WC Matches'
        '<div id="tt">'
        '<button class="ttb active" id="btn22" onclick="switchTournament(2022)">2022</button>'
        '<button class="ttb" id="btn18" onclick="switchTournament(2018)">2018</button>'
        '</div></div>'
    )
    print('  [OK] Updated sidebar-title with toggle buttons')

    # 7. Add id to h1 so JS can update it
    if 'id="main-title"' in html:
    print('  [SKIP] main-title id already present')
    else:
    html = html.replace(
        '<h1>FIFA World Cup 2022 — Passing Networks</h1>',
        '<h1 id="main-title">FIFA World Cup 2022 — Passing Networks</h1>'
    )
    print('  [OK] Added id to h1')

    # 8. Add CSS for tournament toggle buttons (insert before </style>)
    TOGGLE_CSS = (
    '\n#tt{display:flex;gap:4px;margin-top:7px;}'
    '\n.ttb{flex:1;padding:5px 0;background:#21262d;border:1px solid #30363d;'
    'color:#8b949e;border-radius:5px;cursor:pointer;font-size:11px;font-weight:700;'
    'letter-spacing:.3px;transition:background .15s,color .15s;}'
    '\n.ttb.active{background:#f0a500;color:#0d1117;border-color:#f0a500;}'
    '\n.ttb:hover:not(.active){background:#30363d;color:#e6edf3;}\n'
    )
    if '#tt{display:flex' in html:
    print('  [SKIP] Toggle button CSS already present')
    else:
    html = html.replace('</style>', TOGGLE_CSS + '</style>', 1)
    print('  [OK] Added toggle button CSS')

    # 9. Add switchTournament() before </script>
    if 'function switchTournament(' in html:
    print('  [SKIP] switchTournament() already present')
    SWITCH_FN = None
    else:
    SWITCH_FN = (
    '\nfunction switchTournament(yr){'
    '\n  CUR=yr===2022?DATA_2022:DATA_2018;'
    '\n  curMatch=null;cur=null;'
    '\n  document.getElementById("btn22").classList.toggle("active",yr===2022);'
    '\n  document.getElementById("btn18").classList.toggle("active",yr===2018);'
    '\n  document.getElementById("main-title").textContent='
    '"FIFA World Cup "+yr+" — Passing Networks";'
    '\n  document.getElementById("desc").textContent='
    '"Select a match from the sidebar to explore passing networks";'
    '\n  document.getElementById("ph").style.display="";'
    '\n  document.getElementById("ph-home").style.display="none";'
    '\n  document.getElementById("ph-away").style.display="none";'
    '\n  document.getElementById("cmp").style.display="none";'
    '\n  buildSidebar();'
    '\n}\n'
    )
    html = html.replace('</script>', SWITCH_FN + '</script>', 1)
    print('  [OK] Added switchTournament() function')

    # 10. Update page title
    html = html.replace(
    '<title>WC 2022 — Passing Networks</title>',
    '<title>WC Passing Networks</title>'
    )

    # ── Write updated HTML ───────────────────────────────────────────────────────
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

    new_size = len(html)
    print(f'\nDone.')
    print(f'  Original: {original_size/1024/1024:.2f} MB')
    print(f'  Updated:  {new_size/1024/1024:.2f} MB')
    print(f'  Added:    {(new_size-original_size)/1024:.0f} KB')
