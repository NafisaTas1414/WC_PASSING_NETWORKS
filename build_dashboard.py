#!/usr/bin/env python3
"""
Build the full WC passing-networks dashboard:
  - 2022 + 2018 network JSON
  - Dominance / defensive resistance insights (notebook 08)
  - Interactive HTML shell
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from insights_builder import build_all_insights  # noqa: E402
from network_export import OVERRIDES, build_data_2018  # noqa: E402

WEB_DIR = Path('outputs/web')
HTML_PATH = WEB_DIR / 'index.html'
APP_JS = WEB_DIR / 'app.js'

EXTRA_CSS = """
#sidebar-filters{display:flex;gap:3px;padding:8px 10px;border-bottom:1px solid #30363d;flex-wrap:wrap}
.sf{flex:1;min-width:46px;padding:4px 0;background:#21262d;border:1px solid #30363d;color:#8b949e;border-radius:4px;cursor:pointer;font-size:9px;font-weight:700}
.sf.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
.sf:hover:not(.on){background:#30363d;color:#c9d1d9}
.match-row{display:flex;align-items:center;gap:6px}
.dom-pip{height:4px;border-radius:2px;flex-shrink:0;background:#58a6ff}
.dom-pip.upset{background:#f85149}
.dom-pip.dom-win{background:#3fb950}
.dom-pip.draw{background:#d29922}
.upset-badge{margin-left:4px;font-size:10px}
.empty-side{padding:16px;color:#7d8590;font-size:11px;text-align:center}
#story-bar{display:none;padding:8px 16px;background:#161b22;border-bottom:1px solid #30363d;gap:12px;align-items:center;flex-shrink:0;flex-wrap:wrap}
.story-left{flex:1;min-width:220px}
#dom-meter .meter-labels{display:flex;justify-content:space-between;font-size:10px;color:#8b949e;margin-bottom:3px}
.meter-track{display:flex;height:8px;border-radius:4px;overflow:hidden;background:#21262d}
.meter-home{background:linear-gradient(90deg,#1f6feb,#58a6ff);transition:width .3s}
.meter-away{background:linear-gradient(90deg,#f0a500,#ffca66);transition:width .3s}
.meter-sub{font-size:10px;color:#7d8590;margin-top:4px}
.outcome-badge{padding:4px 10px;border-radius:12px;font-size:10px;font-weight:700;white-space:nowrap}
.outcome-badge.upset{background:rgba(248,81,73,.15);color:#f85149;border:1px solid rgba(248,81,73,.35)}
.outcome-badge.dom-win{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid rgba(63,185,80,.35)}
.outcome-badge.draw{background:rgba(210,153,34,.15);color:#d29922;border:1px solid rgba(210,153,34,.35)}
.outcome-badge.split{background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid rgba(88,166,255,.35)}
#story-insight{flex:2;font-size:11px;color:#8b949e;line-height:1.5;min-width:200px}
#story-insight b{color:#c9d1d9}
#drawer{max-height:0;overflow:hidden;transition:max-height .25s ease;background:#0d1117;border-top:1px solid #21262d;flex-shrink:0}
#drawer.open{max-height:220px}
#drawer-tabs{display:flex;gap:0;border-bottom:1px solid #21262d}
.dt{padding:6px 14px;background:transparent;border:none;color:#7d8590;font-size:10px;font-weight:700;cursor:pointer;border-bottom:2px solid transparent}
.dt.on{color:#f0a500;border-bottom-color:#f0a500}
.dt:hover:not(.on){color:#c9d1d9}
#drawer-body{padding:8px 14px;overflow-y:auto;max-height:170px;font-size:11px}
.metric-bars{display:flex;flex-direction:column;gap:6px}
.mb-row{display:grid;grid-template-columns:72px 1fr;gap:8px;align-items:start}
.mb-label{color:#7d8590;font-size:9px;text-transform:uppercase;padding-top:2px}
.mb-bars{display:flex;flex-direction:column;gap:2px}
.mb-side{display:grid;grid-template-columns:72px 1fr 32px;gap:4px;align-items:center;font-size:9px;color:#8b949e}
.mb-track{height:5px;background:#21262d;border-radius:2px;overflow:hidden}
.mb-fill{height:100%;border-radius:2px}
.mb-fill.home{background:#58a6ff}
.mb-fill.away{background:#f0a500}
.mb-val{text-align:right;color:#c9d1d9;font-weight:700}
.mb-lead{grid-column:2;font-size:8px;color:#7d8590;font-style:italic}
.resistance-card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 10px;margin-bottom:8px}
.rc-title{font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:6px}
.rc-grid{display:flex;gap:8px;flex-wrap:wrap}
.rc-pill{text-align:center;min-width:52px}
.rc-val{font-size:14px;font-weight:700;color:#e6edf3}
.rc-lbl{font-size:8px;color:#7d8590;text-transform:uppercase}
.rc-bench{font-size:10px;color:#7d8590;margin-top:6px}
.def-cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.def-col-title{font-weight:700;color:#c9d1d9;margin-bottom:4px;font-size:10px}
.def-player{padding:4px 0;border-bottom:1px solid #21262d}
.def-pos{color:#7d8590;font-size:9px}
.def-stats{color:#8b949e;font-size:9px;margin-top:1px}
.def-totals{font-size:9px;color:#7d8590;margin-top:4px}
.def-empty{color:#7d8590;font-size:10px}
.ctx-chip{margin-top:8px;padding:6px 8px;background:#161b22;border-radius:4px;color:#8b949e;font-size:10px;border-left:3px solid #f0a500}
.explore-wrap{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}
#scatter-host{flex:1;min-width:200px}
.scatter-svg{background:#161b22;border-radius:4px}
.similar-block{flex:1;min-width:160px}
.sim-title{font-size:9px;color:#7d8590;text-transform:uppercase;margin-bottom:4px}
.sim-btn{display:block;width:100%;text-align:left;padding:5px 8px;margin-bottom:3px;background:#21262d;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;font-size:10px;cursor:pointer}
.sim-btn:hover{background:#30363d}
.sim-btn.upset{border-left:3px solid #f85149}
.sim-btn.dom-win{border-left:3px solid #3fb950}
#mapToggle.on{border-color:#58a6ff;color:#58a6ff}
#tt{display:flex;gap:4px;margin-top:7px}
.ttb{flex:1;padding:5px 0;background:#21262d;border:1px solid #30363d;color:#8b949e;border-radius:5px;cursor:pointer;font-size:11px;font-weight:700;letter-spacing:.3px;transition:background .15s,color .15s}
.ttb.active{background:#f0a500;color:#0d1117;border-color:#f0a500}
.ttb:hover:not(.active){background:#30363d;color:#e6edf3}
"""

BODY_HTML = """
<div id="sidebar">
  <div id="sidebar-title">⚽ WC Matches
    <div id="tt">
      <button class="ttb active" id="btn22" onclick="switchTournament(2022)">2022</button>
      <button class="ttb" id="btn18" onclick="switchTournament(2018)">2018</button>
    </div>
  </div>
  <div id="sidebar-filters">
    <button class="sf on" data-f="all">All</button>
    <button class="sf" data-f="upsets">Upsets ⚡</button>
    <button class="sf" data-f="dominant">Dom. wins</button>
    <button class="sf" data-f="close">Close</button>
  </div>
  <div id="match-list"></div>
</div>

<div id="main">
  <div id="topbar">
    <h1 id="main-title">FIFA World Cup 2022 — Passing Networks</h1>
    <p id="desc">Select a match — explore networks, dominance, and defensive resistance</p>
  </div>
  <div id="story-bar">
    <div class="story-left">
      <div id="dom-meter"></div>
    </div>
    <div id="outcome-badge" class="outcome-badge"></div>
    <div id="story-insight"></div>
  </div>
  <div id="panels">
    <div id="ph">← Select a match to explore passing dominance &amp; networks</div>
    <div class="net-panel" id="ph-home" style="display:none">
      <div class="panel-name" id="nm-home"></div>
      <div class="sc" id="sc-home"></div>
      <svg class="panel-svg" id="sv-home"></svg>
      <div class="ins" id="ins-home"></div>
    </div>
    <div class="net-panel" id="ph-away" style="display:none">
      <div class="panel-name" id="nm-away"></div>
      <div class="sc" id="sc-away"></div>
      <svg class="panel-svg" id="sv-away"></svg>
      <div class="ins" id="ins-away"></div>
    </div>
  </div>
  <div id="drawer">
    <div id="drawer-tabs">
      <button class="dt on" data-t="metrics">Metrics</button>
      <button class="dt" data-t="defense">Defense</button>
      <button class="dt" data-t="explore">Explore</button>
    </div>
    <div id="drawer-body"></div>
  </div>
  <div id="ctrl">
    <label>Min passes:</label>
    <input type="range" id="mw" min="1" max="20" value="3">
    <span id="mwv">3</span>
    <div class="sep"></div>
    <span>Show:</span>
    <button class="fb on" data-f="all">All</button>
    <button class="fb" data-f="openplay" id="btn-op">Open play</button>
    <button class="fb" data-f="prog" id="btn-prog">Progressive</button>
    <button class="fb" data-f="ft" id="btn-ft">Final third</button>
    <button class="fb" data-f="h1">1st half</button>
    <button class="fb" data-f="h2">2nd half</button>
    <div class="sep"></div>
    <button id="lblBtn">Show all labels</button>
    <button id="mapToggle">Upset map</button>
    <div class="leg">
      <div class="li"><div class="lt"></div><span>Key player</span></div>
      <div class="li"><div class="ll"></div><span>Top connection</span></div>
      <div class="li"><div class="le"></div><span>Pass edge</span></div>
    </div>
  </div>
</div>

<div id="tip"></div>
"""

BASE_CSS = r"""*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden}
#sidebar{width:255px;flex-shrink:0;background:#161b22;border-right:1px solid #30363d;display:flex;flex-direction:column;overflow:hidden}
#sidebar-title{padding:13px 16px;font-size:13px;font-weight:700;color:#f0a500;border-bottom:1px solid #30363d}
#match-list{overflow-y:auto;flex:1}
#match-list::-webkit-scrollbar{width:4px}
#match-list::-webkit-scrollbar-thumb{background:#30363d;border-radius:2px}
.stage-label{padding:10px 16px 4px;font-size:10px;font-weight:700;color:#7d8590;text-transform:uppercase;letter-spacing:1px}
.match-item{padding:8px 16px;cursor:pointer;border-left:3px solid transparent;transition:background .1s,border-color .1s;user-select:none}
.match-item:hover{background:#1c2128}
.match-item.active{background:#1c2128;border-left-color:#f0a500}
.match-teams{font-size:12px;color:#c9d1d9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.match-score{font-size:11px;color:#7d8590;margin-top:2px}
.match-item.active .match-teams{color:#f0a500}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#topbar{padding:9px 16px;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}
#topbar h1{font-size:14px;font-weight:600}
#topbar p{font-size:11px;color:#7d8590;margin-top:1px}
#panels{flex:1;display:flex;overflow:hidden;position:relative;min-height:0}
#ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#7d8590;font-size:14px;pointer-events:none}
.net-panel{flex:1;display:flex;flex-direction:column;border-right:1px solid #30363d;overflow:hidden;min-width:0}
.net-panel:last-child{border-right:none}
.panel-name{padding:6px 10px;font-size:12px;font-weight:700;text-align:center;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0;color:#c9d1d9;letter-spacing:.3px}
.sc{display:grid;grid-template-columns:repeat(3,1fr);background:#0d1117;border-bottom:1px solid #21262d;flex-shrink:0}
.si{padding:4px 6px;text-align:center;border-right:1px solid #21262d}
.si:nth-child(3n){border-right:none}
.sv{font-size:12px;font-weight:700;color:#e6edf3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sv.hi{color:#f0a500}
.sl{font-size:8px;color:#7d8590;text-transform:uppercase;letter-spacing:.4px;margin-top:1px}
.panel-svg{flex:1;width:100%;display:block;min-height:0}
.ins{padding:4px 8px;font-size:10px;color:#7d8590;background:#0d1117;border-top:1px solid #21262d;flex-shrink:0;display:flex;gap:12px;justify-content:center;overflow:hidden}
.ins b{color:#c9d1d9}
.ins .te{color:#ff6b6b}
#ctrl{padding:5px 12px;background:#161b22;border-top:1px solid #30363d;display:flex;align-items:center;gap:6px;flex-shrink:0;font-size:11px;color:#7d8590;flex-wrap:wrap}
#ctrl input[type=range]{width:70px;accent-color:#f0a500;cursor:pointer}
#mwv{color:#f0a500;font-weight:700;min-width:16px}
.sep{width:1px;height:15px;background:#30363d;margin:0 1px;flex-shrink:0}
.fb{background:#21262d;border:1px solid #30363d;color:#8b949e;padding:2px 7px;border-radius:4px;cursor:pointer;font-size:11px;transition:background .1s}
.fb:hover:not(:disabled){background:#30363d;color:#c9d1d9}
.fb.on{background:#1f6feb;border-color:#1f6feb;color:white}
.fb:disabled{opacity:.35;cursor:not-allowed}
#lblBtn,#mapToggle{background:#21262d;border:1px solid #30363d;color:#8b949e;padding:2px 7px;border-radius:4px;cursor:pointer;font-size:11px}
#lblBtn:hover,#mapToggle:hover{background:#30363d}
#lblBtn.on{border-color:#f0a500;color:#f0a500}
.leg{margin-left:auto;display:flex;gap:10px;align-items:center}
.li{display:flex;align-items:center;gap:4px;font-size:10px}
.lt{width:9px;height:9px;border-radius:50%;background:#f0a500;border:2px solid white;box-shadow:0 0 0 2.5px rgba(255,255,255,.25)}
.ll{width:18px;height:2.5px;background:#ff6b6b}
.le{width:18px;height:1.5px;background:rgba(255,255,255,.45)}
#tip{position:fixed;background:#1c2128;border:1px solid #444c56;border-radius:6px;padding:9px 11px;font-size:11px;pointer-events:none;z-index:9999;display:none;max-width:250px;line-height:1.65;box-shadow:0 4px 14px rgba(0,0,0,.55)}
#tip table{border-collapse:collapse}
#tip td{padding:0 10px 1px 0;vertical-align:top}
"""


def extract_data_2022(html: str) -> dict:
    m = re.search(r'const DATA_2022 = (\{.*?\});\n', html, re.DOTALL)
    if not m:
        m = re.search(r'const DATA = (\{.*?\});\n', html, re.DOTALL)
    if not m:
        raise RuntimeError('Could not find DATA_2022 / DATA in index.html')
    return json.loads(m.group(1))


def build_data_2018_wrapper() -> dict:
    print('Building 2018 networks...')
    data = build_data_2018()
    print(f'  2018: {len(data["matches"])} matches, {len(data["networks"])} networks')
    return data


def assemble_html(data_2022, data_2018, ins_2022, ins_2018, context, app_js: str) -> str:
    j = lambda o: json.dumps(o, separators=(',', ':'), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WC Passing Networks</title>
<style>
{BASE_CSS}
{EXTRA_CSS}
</style>
</head>
<body>
{BODY_HTML}
<script>
const DATA_2022 = {j(data_2022)};
const DATA_2018 = {j(data_2018)};
const INSIGHTS_2022 = {j(ins_2022)};
const INSIGHTS_2018 = {j(ins_2018)};
const CONTEXT = {j(context)};
let CUR = DATA_2022;
{app_js}
</script>
</body>
</html>
"""


def main():
    print('Reading existing 2022 data from index.html...')
    html = HTML_PATH.read_text(encoding='utf-8')
    data_2022 = extract_data_2022(html)
    print(f'  {len(data_2022["matches"])} matches, {len(data_2022["networks"])} networks')

    data_2018 = build_data_2018_wrapper()

    print('Computing dominance & defensive insights...')
    ins_2022, ins_2018, context = build_all_insights(
        data_2022['matches'],
        data_2018['matches'],
        OVERRIDES,
    )
    print(f'  2022 insights: {len(ins_2022)}')
    print(f'  2018 insights: {len(ins_2018)}')
    print(f'  Dominant win rate: {context["dominant_win_rate"]:.0%}')
    print(f'  Failed dominance: {context["failed_dominance_count"]}')

    app_js = APP_JS.read_text(encoding='utf-8')
    out = assemble_html(data_2022, data_2018, ins_2022, ins_2018, context, app_js)

    HTML_PATH.write_text(out, encoding='utf-8')
    print(f'\nWrote {HTML_PATH} ({len(out)/1024/1024:.2f} MB)')


if __name__ == '__main__':
    main()
