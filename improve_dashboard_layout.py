#!/usr/bin/env python3
"""
Dashboard layout improvements:
  1. Fix the "Central player" tooltip, which described betweenness-style
     behavior ("a link between teammates") for a field that's actually just
     raw pass volume. Rename the label to "Top passer" so it stops reading
     as a synonym for "Most Critical Player".
  2. Add a tooltip to "Most Critical Player" explaining what it means.
  3. Move the always-visible critical-player panel out of the team-network
     view (where it was competing for space with the diagram) into its own
     "Vulnerability" tab in the existing drawer, alongside Metrics/Defense/
     Explore — consistent with how Defense already handles two-column
     home/away content in that drawer.

Patches outputs/web/index.html directly (the served/authoritative file) and
best-effort syncs the same JS changes to outputs/web/app.js. Safe to re-run.
"""
import re
from pathlib import Path

HTML_PATH = Path('outputs/web/index.html')
APP_JS_PATH = Path('outputs/web/app.js')


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------

OLD_INS_HOME = '<div class="ins" id="ins-home"></div>\n      <div class="vuln-panel" id="vuln-home"></div>'
NEW_INS_HOME = '<div class="ins" id="ins-home"></div>'
OLD_INS_AWAY = '<div class="ins" id="ins-away"></div>\n      <div class="vuln-panel" id="vuln-away"></div>'
NEW_INS_AWAY = '<div class="ins" id="ins-away"></div>'

OLD_TAB_BUTTONS = '<button class="dt" data-t="explore">Explore</button>\n        <button id="drawer-close">▼</button>'
NEW_TAB_BUTTONS = (
    '<button class="dt" data-t="explore">Explore</button>\n'
    '        <button class="dt" data-t="vulnerability">Vulnerability</button>\n'
    '        <button id="drawer-close">▼</button>'
)

OLD_DH_LABEL = '<span id="dh-label">Metrics · Defense · Explore</span>'
NEW_DH_LABEL = '<span id="dh-label">Metrics · Defense · Vulnerability · Explore</span>'


# ---------------------------------------------------------------------------
# CSS fragments
# ---------------------------------------------------------------------------

OLD_VULN_PANEL_CSS = (
    '.vuln-panel{position:absolute;left:0;right:0;bottom:0;padding:10px 12px;'
    'background:#0d1117;border-top:1px solid #21262d;display:none;'
    'overflow-y:auto;max-height:60%;z-index:6}'
)
NEW_VULN_COLS_CSS = (
    '.vuln-cols{display:grid;grid-template-columns:1fr 1fr;gap:24px;height:100%}\n'
    '.vuln-col{display:flex;flex-direction:column}'
)


# ---------------------------------------------------------------------------
# JS fragments — terminology fixes
# ---------------------------------------------------------------------------

OLD_STATS_CENTRAL = (
    '`<div class="si" title="Most central player — appeared most often as a link between teammates">'
    '<div class="sv hi">${sn2}</div><div class="sl">Central player</div></div>`+'
)
NEW_STATS_CENTRAL = (
    '`<div class="si" title="Player with the most total passes (sent + received) — '
    'pure volume, not the same as structural importance"><div class="sv hi">${sn2}</div>'
    '<div class="sl">Top passer</div></div>`+'
)

OLD_INS_CENTRAL = (
    "el.innerHTML=`<span title=\"Most central player — appeared most often as a link between teammates\">"
    "Most central: <b>${mi}</b></span>${pair}`;"
)
NEW_INS_CENTRAL = (
    "el.innerHTML=`<span title=\"Player with the most total passes (sent + received)\">"
    "Top passer: <b>${mi}</b></span>${pair}`;"
)


# ---------------------------------------------------------------------------
# JS fragments — restructure the vulnerability UI into a drawer tab
# ---------------------------------------------------------------------------

OLD_RENDER_VULN_PANEL = '''function renderVulnPanel(side,matchId,team){
  const el=document.getElementById('vuln-'+side);
  if(!el) return;
  const info=getVulnInfo(matchId,team);
  if(!info){ el.innerHTML=''; el.style.display='none'; return; }
  el.style.display='block';
  const removed=vulnRemoved[side];
  const notObvious=info.pass_involvement_rank>1||(info.betweenness_rank&&info.betweenness_rank>1);
  let html=`<div class="vuln-head"><span class="vuln-icon">\\ud83c\\udfaf</span> Most Critical Player: <b>${sn(info.critical_player)}</b></div>`;
  if(notObvious){
    const btRank=info.betweenness_rank?`#${info.betweenness_rank}`:'n/a';
    html+=`<div class="vuln-rank-contrast">Structural-damage rank <b>#1</b> \\u00b7 Pass involvement <b>#${info.pass_involvement_rank}</b> \\u00b7 Betweenness <b>${btRank}</b></div>`;
  }
  html+=`<button class="vuln-btn ${removed?'restore':'remove'}" id="vuln-btn-${side}">${removed?'Restore Original Network':'Remove Critical Player'}</button>`;
  if(removed){
    html+=`<table class="vuln-table"><tr><th></th><th>Original</th><th>Without</th><th>Change</th></tr>`+
      `<tr><td>Network efficiency</td><td>${info.original_efficiency.toFixed(3)}</td><td>${info.removed_efficiency.toFixed(3)}</td><td class="vd">-${(info.efficiency_damage*100).toFixed(1)}%</td></tr>`+
      `<tr><td>Passing connections</td><td>${info.original_connections}</td><td>${info.removed_connections}</td><td class="vd">-${(info.edge_damage*100).toFixed(1)}%</td></tr>`+
      `<tr><td>Progressive-passing capacity</td><td>${info.original_progressive}</td><td>${info.removed_progressive}</td><td class="vd">-${(info.progressive_capacity_damage*100).toFixed(1)}%</td></tr>`+
      `</table>`+
      `<div class="vuln-note">This simulation removes the player's observed passing connections to show how dependent the recorded network was on their structural role. It does not predict how the team would tactically reorganize without them.</div>`;
  }
  el.innerHTML=html;
  const btn=document.getElementById('vuln-btn-'+side);
  if(btn) btn.addEventListener('click',()=>toggleVulnRemoval(side,matchId,team));
}'''

NEW_RENDER_VULNERABILITY_TAB = '''function renderVulnerabilityTab(m,ins){
  let html='<div class="vuln-cols">';
  ['home','away'].forEach(side=>{
    const team=side==='home'?m.home_team:m.away_team;
    const info=getVulnInfo(m.match_id,team);
    html+=`<div class="vuln-col"><div class="def-col-title">${team}</div>`;
    if(!info){
      html+='<div class="def-empty">No vulnerability data available</div>';
    } else {
      const removed=vulnRemoved[side];
      const notObvious=info.pass_involvement_rank>1||(info.betweenness_rank&&info.betweenness_rank>1);
      html+=`<div class="vuln-head" title="Player whose removal causes the largest drop in network efficiency — not necessarily the player who passes the most"><span class="vuln-icon">\\ud83c\\udfaf</span> Most Critical Player: <b>${sn(info.critical_player)}</b></div>`;
      if(notObvious){
        const btRank=info.betweenness_rank?`#${info.betweenness_rank}`:'n/a';
        html+=`<div class="vuln-rank-contrast">Structural-damage rank <b>#1</b> \\u00b7 Pass involvement <b>#${info.pass_involvement_rank}</b> \\u00b7 Betweenness <b>${btRank}</b></div>`;
      }
      html+=`<button class="vuln-btn ${removed?'restore':'remove'}" id="vuln-btn-${side}">${removed?'Restore Original Network':'Remove Critical Player'}</button>`;
      if(removed){
        html+=`<table class="vuln-table"><tr><th></th><th>Original</th><th>Without</th><th>Change</th></tr>`+
          `<tr><td>Network efficiency</td><td>${info.original_efficiency.toFixed(3)}</td><td>${info.removed_efficiency.toFixed(3)}</td><td class="vd">-${(info.efficiency_damage*100).toFixed(1)}%</td></tr>`+
          `<tr><td>Passing connections</td><td>${info.original_connections}</td><td>${info.removed_connections}</td><td class="vd">-${(info.edge_damage*100).toFixed(1)}%</td></tr>`+
          `<tr><td>Progressive-passing capacity</td><td>${info.original_progressive}</td><td>${info.removed_progressive}</td><td class="vd">-${(info.progressive_capacity_damage*100).toFixed(1)}%</td></tr>`+
          `</table>`+
          `<div class="vuln-note">This simulation removes the player's observed passing connections to show how dependent the recorded network was on their structural role. It does not predict how the team would tactically reorganize without them.</div>`;
      }
    }
    html+='</div>';
  });
  html+='</div>';
  setTimeout(()=>{
    ['home','away'].forEach(side=>{
      const btn=document.getElementById('vuln-btn-'+side);
      if(btn){
        const team=side==='home'?m.home_team:m.away_team;
        btn.addEventListener('click',()=>toggleVulnRemoval(side,m.match_id,team));
      }
    });
  },0);
  return html;
}'''

OLD_TOGGLE_VULN = '''function toggleVulnRemoval(side,matchId,team){
  vulnRemoved[side]=!vulnRemoved[side];
  const info=getVulnInfo(matchId,team);
  renderNet(side==='home'?'sv-home':'sv-away',matchId,team,vulnRemoved[side]&&info?info.critical_player:null);
  renderVulnPanel(side,matchId,team);
}'''

NEW_TOGGLE_VULN = '''function toggleVulnRemoval(side,matchId,team){
  vulnRemoved[side]=!vulnRemoved[side];
  const info=getVulnInfo(matchId,team);
  renderNet(side==='home'?'sv-home':'sv-away',matchId,team,vulnRemoved[side]&&info?info.critical_player:null);
  if(curMatch) renderDrawer(curMatch);
}'''

OLD_RENDER_BOTH_TAIL = '''  renderNet('sv-home',m.match_id,m.home_team,vulnRemoved.home&&hVulnInfo?hVulnInfo.critical_player:null);
  renderNet('sv-away',m.match_id,m.away_team,vulnRemoved.away&&aVulnInfo?aVulnInfo.critical_player:null);
  renderVulnPanel('home',m.match_id,m.home_team);
  renderVulnPanel('away',m.match_id,m.away_team);
}'''

NEW_RENDER_BOTH_TAIL = '''  renderNet('sv-home',m.match_id,m.home_team,vulnRemoved.home&&hVulnInfo?hVulnInfo.critical_player:null);
  renderNet('sv-away',m.match_id,m.away_team,vulnRemoved.away&&aVulnInfo?aVulnInfo.critical_player:null);
}'''

OLD_RENDER_DRAWER = '''function renderDrawer(m){
  const body=document.getElementById('drawer-body');
  const ins=getIns(m.match_id);
  if(!ins){ body.innerHTML=''; return; }
  if(drawerTab==='metrics') body.innerHTML=renderMetricsTab(m,ins);
  else if(drawerTab==='defense') body.innerHTML=renderDefenseTab(m,ins);
  else body.innerHTML=renderExploreTab(m,ins);
}'''

NEW_RENDER_DRAWER = '''function renderDrawer(m){
  const body=document.getElementById('drawer-body');
  const ins=getIns(m.match_id);
  if(!ins){ body.innerHTML=''; return; }
  if(drawerTab==='metrics') body.innerHTML=renderMetricsTab(m,ins);
  else if(drawerTab==='defense') body.innerHTML=renderDefenseTab(m,ins);
  else if(drawerTab==='vulnerability') body.innerHTML=renderVulnerabilityTab(m,ins);
  else body.innerHTML=renderExploreTab(m,ins);
}'''


def patch(text: str, label: str) -> str:
    def apply(old, new, name, required=True):
        nonlocal text
        if new in text:
            print(f'  [SKIP] {name} already applied')
            return
        if old not in text:
            if required:
                raise RuntimeError(f'Marker not found for {name} in {label}')
            print(f'  [WARN] {name} marker not found (skipping, non-critical)')
            return
        text = text.replace(old, new, 1)
        print(f'  [OK] {name}')

    apply(OLD_INS_HOME, NEW_INS_HOME, 'remove vuln-panel div (home)')
    apply(OLD_INS_AWAY, NEW_INS_AWAY, 'remove vuln-panel div (away)')
    apply(OLD_TAB_BUTTONS, NEW_TAB_BUTTONS, 'add Vulnerability drawer tab button')
    apply(OLD_DH_LABEL, NEW_DH_LABEL, 'update drawer-handle label')
    apply(OLD_VULN_PANEL_CSS, NEW_VULN_COLS_CSS, 'replace vuln-panel overlay CSS with vuln-cols CSS')
    apply(OLD_STATS_CENTRAL, NEW_STATS_CENTRAL, 'fix "Central player" tooltip + label in stat card')
    apply(OLD_INS_CENTRAL, NEW_INS_CENTRAL, 'fix "Most central" tooltip + label in insight line')
    apply(OLD_RENDER_VULN_PANEL, NEW_RENDER_VULNERABILITY_TAB, 'replace renderVulnPanel with renderVulnerabilityTab')
    apply(OLD_TOGGLE_VULN, NEW_TOGGLE_VULN, 'update toggleVulnRemoval to re-render the drawer')
    apply(OLD_RENDER_BOTH_TAIL, NEW_RENDER_BOTH_TAIL, 'remove renderVulnPanel calls from renderBoth')
    apply(OLD_RENDER_DRAWER, NEW_RENDER_DRAWER, 'add vulnerability tab dispatch to renderDrawer')
    return text


def main():
    print('Patching outputs/web/index.html ...')
    html = HTML_PATH.read_text(encoding='utf-8')
    html = patch(html, 'index.html')
    HTML_PATH.write_text(html, encoding='utf-8')

    if APP_JS_PATH.exists():
        print('\\nBest-effort sync to outputs/web/app.js ...')
        app_js = APP_JS_PATH.read_text(encoding='utf-8')
        try:
            app_js = patch(app_js, 'app.js')
            APP_JS_PATH.write_text(app_js, encoding='utf-8')
        except RuntimeError as exc:
            print(f'  [WARN] app.js has diverged from index.html, skipping sync: {exc}')


if __name__ == '__main__':
    main()
