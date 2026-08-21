/* ── Data aliases ───────────────────────────────── */
let INS = INSIGHTS_2022;
let matchFilter = 'all';
let drawerTab = 'metrics';
let scatterOn = false;

const STAGES = [
  {key:'group',label:'Group Stage'},{key:'R16',label:'Round of 16'},
  {key:'QF',label:'Quarter-finals'},{key:'SF',label:'Semi-finals'},
  {key:'Final',label:'Final'},{key:'3rd',label:'3rd Place Final'},
];

/* ── Helpers ───────────────────────────────────── */
function getIns(mid){ return INS[String(mid)] || null; }

function sn(n){
  const p=(n||'').split(' ');
  return p.length>1?p[p.length-1]:n;
}

function passesFilter(m){
  const ins=getIns(m.match_id);
  if(!ins) return matchFilter==='all';
  if(matchFilter==='all') return true;
  if(matchFilter==='upsets') return ins.failed_dominance;
  if(matchFilter==='dominant') return ins.dominant_won;
  if(matchFilter==='close') return ins.close_match;
  return true;
}

function outcomeClass(ins){
  if(!ins) return '';
  if(ins.failed_dominance) return 'upset';
  if(ins.dominant_won) return 'dom-win';
  if(ins.winner===null) return 'draw';
  return 'split';
}

function outcomeLabel(ins){
  if(!ins) return '';
  if(ins.failed_dominance) return 'Failed dominance';
  if(ins.dominant_won) return 'Dominant team won';
  if(ins.winner===null) return 'Draw — balanced outcome';
  return 'Lower-possession team won';
}

/* ── Sidebar ───────────────────────────────────── */
function buildSidebar(){
  const listEl=document.getElementById('match-list');
  listEl.innerHTML='';
  const filtered=CUR.matches.filter(passesFilter);
  if(!filtered.length){
    listEl.innerHTML='<div class="empty-side">No matches match this filter.</div>';
    return;
  }
  STAGES.forEach(st=>{
    const sm=filtered.filter(m=>m.stage===st.key);
    if(!sm.length) return;
    const lbl=document.createElement('div');
    lbl.className='stage-label'; lbl.textContent=st.label;
    listEl.appendChild(lbl);
    sm.forEach(m=>{
      const ins=getIns(m.match_id);
      const oc=outcomeClass(ins);
      const pip=ins?`<span class="dom-pip ${oc}" style="width:${Math.max(8,Math.abs(ins.dominance_pct-50)*1.2)}px"></span>`:'';
      const badge=ins&&ins.failed_dominance?'<span class="upset-badge">⚡</span>':'';
      const it=document.createElement('div');
      it.className='match-item'+(curMatch&&curMatch.match_id===m.match_id?' active':'');
      it.dataset.mid=m.match_id;
      it.innerHTML=
        `<div class="match-row">${pip}<div class="match-teams">${m.home_team} vs ${m.away_team}${badge}</div></div>`+
        `<div class="match-score">${m.home_score}–${m.away_score} · ${m.date}</div>`;
      it.addEventListener('click',()=>selectMatch(m,it));
      listEl.appendChild(it);
    });
  });
}

document.querySelectorAll('.sf').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.sf').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    matchFilter=b.dataset.f;
    buildSidebar();
  });
});

/* ── State ─────────────────────────────────────── */
let cur=null,curMatch=null,minW=3,filt='all',allLbls=false;

/* ── Tooltip ───────────────────────────────────── */
const tip=document.getElementById('tip');
let tipOn=false;
function showTip(h,x,y){tip.innerHTML=h;tip.style.display='block';moveTip(x,y);tipOn=true;}
function moveTip(x,y){
  const tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=Math.min(x+14,window.innerWidth-tw-8)+'px';
  tip.style.top =Math.min(y+14,window.innerHeight-th-8)+'px';
}
function hideTip(){tip.style.display='none';tipOn=false;}
document.addEventListener('mousemove',e=>{if(tipOn)moveTip(e.clientX,e.clientY);});

/* ── Controls ──────────────────────────────────── */
const mwSlider=document.getElementById('mw'),mwLbl=document.getElementById('mwv');
mwSlider.addEventListener('input',()=>{minW=+mwSlider.value;mwLbl.textContent=minW;if(curMatch)renderBoth(curMatch);});
document.querySelectorAll('.fb').forEach(b=>{
  b.addEventListener('click',()=>{
    if(b.disabled) return;
    document.querySelectorAll('.fb').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); filt=b.dataset.f;
    if(curMatch) renderBoth(curMatch);
  });
});
document.getElementById('lblBtn').addEventListener('click',function(){
  allLbls=!allLbls; this.textContent=allLbls?'Top 3 labels only':'Show all labels';
  this.classList.toggle('on',allLbls); if(curMatch) renderBoth(curMatch);
});
document.querySelectorAll('.dt').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.dt').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); drawerTab=b.dataset.t;
    if(curMatch) renderDrawer(curMatch);
  });
});
document.getElementById('mapToggle').addEventListener('click',function(){
  scatterOn=!scatterOn;
  this.classList.toggle('on',scatterOn);
  const d=document.getElementById('drawer');
  d.classList.toggle('open',scatterOn||curMatch);
  if(scatterOn){ drawerTab='explore'; document.querySelectorAll('.dt').forEach(x=>x.classList.toggle('on',x.dataset.t==='explore')); renderScatter(); }
  else if(curMatch) renderDrawer(curMatch);
});

/* ── Match selection ───────────────────────────── */
function selectMatch(m,it){
  if(cur) cur.classList.remove('active');
  it.classList.add('active'); cur=it; curMatch=m;
  scatterOn=false;
  document.getElementById('mapToggle').classList.remove('on');
  document.getElementById('desc').textContent=
    `${m.home_team}  ${m.home_score}–${m.away_score}  ${m.away_team}  ·  ${m.stage}  ·  ${m.date}`;
  document.getElementById('ph').style.display='none';
  document.getElementById('ph-home').style.display='flex';
  document.getElementById('ph-away').style.display='flex';
  document.getElementById('drawer').classList.add('open');
  renderStoryBar(m);
  renderBoth(m);
  renderDrawer(m);
}

/* ── Story bar (dominance meter) ───────────────── */
function renderStoryBar(m){
  const bar=document.getElementById('story-bar');
  const ins=getIns(m.match_id);
  if(!ins){ bar.style.display='none'; return; }
  bar.style.display='flex';
  const oc=outcomeClass(ins);
  document.getElementById('dom-meter').innerHTML=
    `<div class="meter-labels"><span>${m.home_team} ${ins.home_pct}%</span><span>${m.away_team} ${ins.away_pct}%</span></div>`+
    `<div class="meter-track"><div class="meter-home" style="width:${ins.home_pct}%"></div><div class="meter-away" style="width:${ins.away_pct}%"></div></div>`+
    `<div class="meter-sub">Passing dominance · pass leader: <b>${ins.pass_leader||ins.dominant_team}</b>${ins.composite_leader&&ins.composite_leader!==ins.pass_leader?` · composite edge: <b>${ins.composite_leader}</b>`:''}</div>`;
  document.getElementById('outcome-badge').className='outcome-badge '+oc;
  document.getElementById('outcome-badge').textContent=outcomeLabel(ins);
  document.getElementById('story-insight').innerHTML=buildStoryInsight(m,ins);
}

function buildStoryInsight(m,ins){
  const h=ins.metrics.home, a=ins.metrics.away;
  const parts=[];
  const passDiff=Math.abs(h.completed_passes-a.completed_passes);
  const passLeader=ins.pass_leader||ins.dominant_team;
  if(passDiff>25) parts.push(`<b>${passLeader}</b> completed ${passDiff} more passes`);
  if(ins.composite_leader&&ins.composite_leader!==passLeader)
    parts.push(`composite passing metrics favoured <b>${ins.composite_leader}</b> instead`);
  const ftDiff=Math.abs(h.final_third_entries-a.final_third_entries);
  if(ftDiff>8){
    const ftLeader=h.final_third_entries>=a.final_third_entries?m.home_team:m.away_team;
    parts.push(`<b>${ftLeader}</b> made ${ftDiff} more final-third entries`);
  }
  if(ins.failed_dominance && ins.winner){
    const wSide=ins.winner_side;
    const wDef=ins.defensive[wSide].totals;
    const lSide=wSide==='home'?'away':'home';
    const lDef=ins.defensive[lSide].totals;
    if(wDef.clearances+lDef.clearances>0)
      parts.push(`<b>${ins.winner}</b>'s defenders recorded ${wDef.clearances} clearances vs ${lDef.clearances}`);
    if(wDef.blocks+lDef.blocks>0)
      parts.push(`${wDef.blocks} blocks vs ${lDef.blocks}`);
  }
  if(ins.low_poss_winner){
    const bench=CONTEXT.benchmarks.low_poss_winners.clearances||0;
    parts.push(`Lower-possession winners in this dataset averaged <b>${bench}</b> clearances per defender`);
  }
  if(!parts.length) return 'Both teams showed similar passing profiles — explore the networks below.';
  return parts.join(' · ') + '.';
}

/* ── Stats card ────────────────────────────────── */
function fillStats(id,mx){
  const el=document.getElementById(id);
  if(!el||!mx) return;
  const pct=((mx.top_reliance||0)*100).toFixed(1)+'%';
  const dn=(mx.density||0).toFixed(3);
  const sn2=mx.most_involved?mx.most_involved.split(' ').pop():'—';
  el.innerHTML=
    `<div class="si"><div class="sv">${mx.completed_passes}</div><div class="sl">Passes</div></div>`+
    `<div class="si"><div class="sv">${mx.unique_pairs}</div><div class="sl">Connections</div></div>`+
    `<div class="si"><div class="sv">${mx.num_players}</div><div class="sl">Players</div></div>`+
    `<div class="si"><div class="sv">${dn}</div><div class="sl">Density</div></div>`+
    `<div class="si"><div class="sv hi">${sn2}</div><div class="sl">Most involved</div></div>`+
    `<div class="si"><div class="sv">${pct}</div><div class="sl">Hub reliance</div></div>`;
}

function fillIns(id,mx){
  const el=document.getElementById(id); if(!el) return;
  if(!mx||!mx.most_involved){ el.innerHTML=''; return; }
  const mi=mx.most_involved.split(' ').pop();
  const pair=mx.top_pair?`<span class="te">Top connection: <b>${mx.top_pair}</b> (${mx.top_pair_count})</span>`:'';
  el.innerHTML=`<span>Most involved: <b>${mi}</b></span>${pair}`;
}

/* ── Drawer ────────────────────────────────────── */
function renderDrawer(m){
  const body=document.getElementById('drawer-body');
  const ins=getIns(m.match_id);
  if(!ins){ body.innerHTML=''; return; }
  if(drawerTab==='metrics') body.innerHTML=renderMetricsTab(m,ins);
  else if(drawerTab==='defense') body.innerHTML=renderDefenseTab(m,ins);
  else body.innerHTML=renderExploreTab(m,ins);
}

function renderMetricsTab(m,ins){
  const h=ins.metrics.home, a=ins.metrics.away;
  let html='<div class="metric-bars">';
  CONTEXT.display_metrics.forEach(([key,label])=>{
    const hv=h[key]||0, av=a[key]||0;
    const max=Math.max(hv,av,1);
    const hp=Math.round(hv/max*100), ap=Math.round(av/max*100);
    const lead=hv>av?m.home_team:av>hv?m.away_team:'';
    html+=`<div class="mb-row"><div class="mb-label">${label}</div>`+
      `<div class="mb-bars">`+
      `<div class="mb-side home"><span>${m.home_team}</span><div class="mb-track"><div class="mb-fill home" style="width:${hp}%"></div></div><span class="mb-val">${hv}</span></div>`+
      `<div class="mb-side away"><span>${m.away_team}</span><div class="mb-track"><div class="mb-fill away" style="width:${ap}%"></div></div><span class="mb-val">${av}</span></div>`+
      `</div>${lead?`<div class="mb-lead">${lead} leads</div>`:''}</div>`;
  });
  html+='</div>';
  html+=`<div class="ctx-chip">Across 128 WC matches (2018+2022), the passing-dominant team wins only <b>${Math.round(CONTEXT.dominant_win_rate*100)}%</b> of the time.</div>`;
  return html;
}

function renderDefenseTab(m,ins){
  let html='';
  if(ins.low_poss_winner){
    const w=ins.winner, side=ins.winner_side;
    const t=ins.defensive[side].totals;
    html+=`<div class="resistance-card">`+
      `<div class="rc-title">🛡 ${w} won with less possession</div>`+
      `<div class="rc-grid">`+
      defPill('Clearances',t.clearances)+defPill('Blocks',t.blocks)+
      defPill('Interceptions',t.interceptions)+defPill('Pressures',t.pressures)+
      `</div>`;
    const bench=CONTEXT.benchmarks.low_poss_winners;
    html+=`<div class="rc-bench">Winning underdogs averaged <b>${bench.clearances||0}</b> clearances and <b>${bench.blocks||0}</b> blocks per defender across the dataset.</div></div>`;
  }
  html+='<div class="def-cols">';
  ['home','away'].forEach(side=>{
    const team=side==='home'?m.home_team:m.away_team;
    const d=ins.defensive[side];
    html+=`<div class="def-col"><div class="def-col-title">${team} — defensive leaders</div>`;
    if(!d.leaders.length) html+='<div class="def-empty">No defender data</div>';
    else d.leaders.forEach(p=>{
      html+=`<div class="def-player"><b>${p.player}</b> <span class="def-pos">${p.position}</span>`+
        `<div class="def-stats">${p.clearances} clr · ${p.blocks} blk · ${p.interceptions} int · ${p.pressures} prs</div></div>`;
    });
    html+=`<div class="def-totals">Team totals: ${d.totals.clearances||0} clr, ${d.totals.blocks||0} blk, ${d.totals.interceptions||0} int</div></div>`;
  });
  html+='</div>';
  return html;
}

function defPill(label,val){
  return `<div class="rc-pill"><div class="rc-val">${val||0}</div><div class="rc-lbl">${label}</div></div>`;
}

function renderExploreTab(m,ins){
  let html=`<div class="explore-wrap">`;
  html+=`<div id="scatter-host"></div>`;
  html+=`<div class="similar-block"><div class="sim-title">Similar matches</div><div id="similar-list"></div></div>`;
  html+=`<div class="ctx-chip"><b>${CONTEXT.failed_dominance_count}</b> matches saw the passing-dominant team lose — <b>${Math.round(CONTEXT.dominant_win_rate*100)}%</b> dominant-team win rate overall.</div>`;
  html+='</div>';
  setTimeout(()=>{ renderScatter(m&&m.match_id); renderSimilar(m,ins); },0);
  return html;
}

function renderSimilar(m,ins){
  const el=document.getElementById('similar-list');
  if(!el||!ins) return;
  const sim=CUR.matches.map(x=>({m:x,ins:getIns(x.match_id)}))
    .filter(x=>x.ins&&x.m.match_id!==m.match_id)
    .map(x=>({...x,d:Math.abs(x.ins.dominance_pct-ins.dominance_pct)+(x.ins.failed_dominance===ins.failed_dominance?0:20)}))
    .sort((a,b)=>a.d-b.d).slice(0,4);
  el.innerHTML=sim.map(({m:x,ins:xi})=>
    `<button class="sim-btn ${outcomeClass(xi)}" data-mid="${x.match_id}">`+
    `${x.home_team} vs ${x.away_team} · dom ${xi.dominance_pct}%`+
    (xi.failed_dominance?' ⚡':'')+`</button>`
  ).join('');
  el.querySelectorAll('.sim-btn').forEach(b=>{
    b.addEventListener('click',()=>{
      const mid=+b.dataset.mid;
      const match=CUR.matches.find(x=>x.match_id===mid);
      const item=document.querySelector(`.match-item[data-mid="${mid}"]`);
      if(match&&item) selectMatch(match,item);
    });
  });
}

function renderScatter(highlightMid){
  const host=document.getElementById('scatter-host');
  if(!host) return;
  const W=host.clientWidth||560, H=160, P=28;
  const pts=CUR.matches.map(m=>{
    const ins=getIns(m.match_id);
    if(!ins) return null;
    return {m,ins,x:ins.scatter.x,y:ins.scatter.y};
  }).filter(Boolean);
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  const xMin=0,xMax=100, yMin=Math.min(...ys,-3), yMax=Math.max(...ys,3);
  const tx=x=>P+(x-xMin)/(xMax-xMin)*(W-P*2);
  const ty=y=>H-P-((y-yMin)/(yMax-yMin))*(H-P*2);
  let svg=`<svg width="${W}" height="${H}" class="scatter-svg">`;
  svg+=`<line x1="${P}" y1="${ty(0)}" x2="${W-P}" y2="${ty(0)}" stroke="#30363d"/>`;
  svg+=`<line x1="${tx(50)}" y1="${P}" x2="${tx(50)}" y2="${H-P}" stroke="#30363d" stroke-dasharray="4"/>`;
  pts.forEach(({m,ins,x,y})=>{
    const col=ins.failed_dominance?'#f85149':ins.dominant_won?'#3fb950':ins.winner===null?'#d29922':'#58a6ff';
    const r=m.match_id===highlightMid?6:4;
    const sw=m.match_id===highlightMid?2:0;
    svg+=`<circle cx="${tx(x)}" cy="${ty(y)}" r="${r}" fill="${col}" stroke="#fff" stroke-width="${sw}" style="cursor:pointer" data-mid="${m.match_id}">`+
      `<title>${m.home_team} vs ${m.away_team} (${x}% / ${y>0?'+':''}${y} gd)</title></circle>`;
  });
  svg+=`<text x="${P}" y="${H-6}" fill="#7d8590" font-size="9">← less possession dominance</text>`;
  svg+=`<text x="${W-P}" y="${H-6}" fill="#7d8590" font-size="9" text-anchor="end">more →</text>`;
  svg+=`<text x="${P}" y="${12}" fill="#7d8590" font-size="9">goal diff (home)</text></svg>`;
  host.innerHTML=svg;
  host.querySelectorAll('circle[data-mid]').forEach(c=>{
    c.addEventListener('click',()=>{
      const mid=+c.dataset.mid;
      const match=CUR.matches.find(x=>x.match_id===mid);
      const item=document.querySelector(`.match-item[data-mid="${mid}"]`);
      if(match&&item) selectMatch(match,item);
    });
  });
}

/* ── Edge / network rendering (unchanged core) ─── */
function ew(e){
  if(filt==='prog') return e.prog||0;
  if(filt==='ft') return e.ft||0;
  if(filt==='h1') return e.h1||0;
  if(filt==='h2') return e.h2||0;
  if(filt==='openplay') return e.op!==undefined?e.op:e.w;
  return e.w;
}

function topPartner(edges,pid){
  let b=null,bw=0;
  edges.forEach(e=>{if(e.s===pid&&e.w>bw){b=e.t;bw=e.w;}});
  return b?{name:b,count:bw}:null;
}

function syncBtns(net){
  const hasEnd=net.edges.length>0&&net.edges[0].prog!==undefined;
  const hasOp =net.edges.length>0&&net.edges[0].op!==undefined;
  document.getElementById('btn-prog').disabled=!hasEnd;
  document.getElementById('btn-ft').disabled=!hasEnd;
  document.getElementById('btn-op').disabled=!hasOp;
}

function renderBoth(m){
  const hNet=CUR.networks[`${m.match_id}_${m.home_team}`];
  const aNet=CUR.networks[`${m.match_id}_${m.away_team}`];
  document.getElementById('nm-home').textContent=m.home_team;
  document.getElementById('nm-away').textContent=m.away_team;
  fillStats('sc-home',hNet&&hNet.metrics);
  fillStats('sc-away',aNet&&aNet.metrics);
  fillIns('ins-home',hNet&&hNet.metrics);
  fillIns('ins-away',aNet&&aNet.metrics);
  if(hNet) syncBtns(hNet);
  const hVulnInfo=getVulnInfo(m.match_id,m.home_team);
  const aVulnInfo=getVulnInfo(m.match_id,m.away_team);
  renderNet('sv-home',m.match_id,m.home_team,vulnRemoved.home&&hVulnInfo?hVulnInfo.critical_player:null);
  renderNet('sv-away',m.match_id,m.away_team,vulnRemoved.away&&aVulnInfo?aVulnInfo.critical_player:null);
  renderVulnPanel('home',m.match_id,m.home_team);
  renderVulnPanel('away',m.match_id,m.away_team);
}

function renderNet(svgId,matchId,team,removedPlayer){
  const net=CUR.networks[`${matchId}_${team}`];
  const svg=document.getElementById(svgId);
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  if(!net) return;
  const mx=net.metrics||{};
  const W=svg.clientWidth||600, H=svg.clientHeight||360;
  const P=20, pw=W-P*2, ph=H-P*2;
  const tx=x=>P+(x/120)*pw, ty=y=>P+(y/80)*ph;
  const NS='http://www.w3.org/2000/svg';
  const mk=(tag,attrs,txt)=>{
    const e=document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
    if(txt!==undefined) e.textContent=txt;
    return e;
  };
  svg.appendChild(mk('rect',{width:W,height:H,fill:'#1a1a2e'}));
  const lc='rgba(255,255,255,0.13)';
  const ln=(x1,y1,x2,y2)=>mk('line',{x1:tx(x1),y1:ty(y1),x2:tx(x2),y2:ty(y2),stroke:lc,'stroke-width':1});
  [ln(0,0,120,0),ln(0,80,120,80),ln(0,0,0,80),ln(120,0,120,80),ln(60,0,60,80),
   ln(0,18,18,18),ln(18,18,18,62),ln(0,62,18,62),
   ln(102,18,120,18),ln(102,62,120,62),ln(102,18,102,62)
  ].forEach(l=>svg.appendChild(l));
  svg.appendChild(mk('circle',{cx:tx(60),cy:ty(40),r:tx(9.15)-tx(0),fill:'none',stroke:lc,'stroke-width':1}));
  const nm={}; net.nodes.forEach(n=>{nm[n.id]=n;});
  const vinfo=getVulnInfo(matchId,team);
  const criticalId=vinfo?vinfo.critical_player:null;
  // Fixed scale refs: always computed from the FULL original network, so
  // removing the critical player never rescales the remaining nodes/edges.
  const maxS=Math.max(...net.nodes.map(n=>n.sent),1);
  const nR=n=>4+14*(n.sent/maxS);
  const nodesToRender=removedPlayer?net.nodes.filter(n=>n.id!==removedPlayer):net.nodes;
  const edgesSource=removedPlayer?net.edges.filter(e=>e.s!==removedPlayer&&e.t!==removedPlayer):net.edges;
  const scaleVis=net.edges.map(e=>({...e,_w:ew(e)})).filter(e=>e._w>=minW&&nm[e.s]&&nm[e.t]&&nm[e.s].x);
  const maxVW=Math.max(...scaleVis.map(e=>e._w),1);
  const vis=edgesSource.map(e=>({...e,_w:ew(e)})).filter(e=>e._w>=minW&&nm[e.s]&&nm[e.t]&&nm[e.s].x);
  const topE=vis.reduce((b,e)=>e._w>(b?b._w:-1)?e:b,null);
  const totP=mx.completed_passes||1;
  vis.forEach(ed=>{
    const s=nm[ed.s],d=nm[ed.t];
    const x1=tx(s.x),y1=ty(s.y),x2=tx(d.x),y2=ty(d.y);
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy);
    if(len<1) return;
    const r=nR(d)+2;
    const ex2=x2-(dx/len)*r,ey2=y2-(dy/len)*r;
    const isTop=ed===topE;
    const alpha=(0.1+0.75*(ed._w/maxVW)).toFixed(2);
    const lw=isTop?'3':(0.5+5*(ed._w/maxVW)).toFixed(1);
    svg.appendChild(mk('line',{x1,y1,x2:ex2,y2:ey2,stroke:isTop?'rgba(255,107,107,0.95)':`rgba(255,255,255,${alpha})`,'stroke-width':lw,'stroke-linecap':'round'}));
    const edc=ed;
    const hit=mk('line',{x1,y1,x2:ex2,y2:ey2,stroke:'transparent','stroke-width':'12',style:'cursor:crosshair'});
    hit.addEventListener('mouseenter',evt=>{
      let h=`<div style="font-weight:700;color:#c9d1d9;margin-bottom:3px">${sn(edc.s)} → ${sn(edc.t)}</div><table>`;
      h+=`<tr><td style="color:#7d8590">Passes</td><td><b style="color:#f0a500">${edc._w}</b></td></tr>`;
      if(edc.prog!=null) h+=`<tr><td style="color:#7d8590">Progressive</td><td><b>${edc.prog}</b></td></tr>`;
      if(edc.ft!=null) h+=`<tr><td style="color:#7d8590">Final third</td><td><b>${edc.ft}</b></td></tr>`;
      showTip(h,evt.clientX,evt.clientY);
    });
    hit.addEventListener('mouseleave',hideTip);
    svg.appendChild(hit);
  });
  nodesToRender.forEach(n=>{
    if(!n.x||!n.y) return;
    const cx=tx(n.x),cy=ty(n.y),r=nR(n);
    const isTop=n.id===mx.most_involved;
    const isCritical=!removedPlayer&&n.id===criticalId;
    const g=mk('g',{transform:`translate(${cx},${cy})`,style:'cursor:default'});
    if(isTop) g.appendChild(mk('circle',{r:r+5,fill:'none',stroke:'rgba(255,255,255,0.3)','stroke-width':1.5,'stroke-dasharray':'3,2'}));
    if(isCritical) g.appendChild(mk('circle',{r:r+7,fill:'none',stroke:'#f85149','stroke-width':2}));
    g.appendChild(mk('circle',{r,fill:'#f0a500',stroke:isCritical?'#ffcc00':(isTop?'white':'rgba(255,255,255,0.5)'),'stroke-width':isCritical?'2.5':(isTop?'2.5':'1.5')}));
    const nd=n,pt=topPartner(net.edges,n.id);
    g.addEventListener('mouseenter',evt=>{
      const ptRow=pt?`<tr><td style="color:#7d8590">Top partner</td><td><b>${sn(pt.name)}</b> (${pt.count})</td></tr>`:'';
      showTip(`<div style="font-weight:700;color:#f0a500;margin-bottom:3px">${sn(nd.id)}</div><table>`+
        `<tr><td style="color:#7d8590">Sent</td><td><b>${nd.sent}</b></td></tr>`+
        `<tr><td style="color:#7d8590">Received</td><td><b>${nd.recv}</b></td></tr>`+ptRow+'</table>',evt.clientX,evt.clientY);
    });
    g.addEventListener('mouseleave',hideTip);
    if(allLbls||!nd.rank||nd.rank<=3){
      g.appendChild(mk('text',{y:r+10,'text-anchor':'middle',fill:'white','font-size':isTop?'9.5':'8','font-weight':isTop?'700':'500',style:'pointer-events:none'},sn(nd.id)));
    }
    svg.appendChild(g);
  });
}

window.addEventListener('resize',()=>{
  if(curMatch){
    renderBoth(curMatch);
    if(drawerTab==='explore'||scatterOn) renderScatter(curMatch.match_id);
  }
});

function switchTournament(yr){
  CUR=yr===2022?DATA_2022:DATA_2018;
  INS=yr===2022?INSIGHTS_2022:INSIGHTS_2018;
  curMatch=null; cur=null;
  document.getElementById('btn22').classList.toggle('active',yr===2022);
  document.getElementById('btn18').classList.toggle('active',yr===2018);
  document.getElementById('main-title').textContent='FIFA World Cup '+yr+' — Passing Networks';
  document.getElementById('desc').textContent='Select a match — explore networks, dominance, and defensive resistance';
  document.getElementById('ph').style.display='';
  document.getElementById('ph-home').style.display='none';
  document.getElementById('ph-away').style.display='none';
  document.getElementById('story-bar').style.display='none';
  document.getElementById('drawer').classList.remove('open');
  scatterOn=false;
  document.getElementById('mapToggle').classList.remove('on');
  buildSidebar();
}

buildSidebar();

/* ── Vulnerability widget (critical-player removal) ─ */
let vulnRemoved={home:false,away:false};
function vulnKey(matchId,team){return `${matchId}_${team}`;}
function getVulnInfo(matchId,team){return (typeof VULN_DATA!=='undefined'&&VULN_DATA[vulnKey(matchId,team)])||null;}

function renderVulnPanel(side,matchId,team){
  const el=document.getElementById('vuln-'+side);
  if(!el) return;
  const info=getVulnInfo(matchId,team);
  if(!info){ el.innerHTML=''; el.style.display='none'; return; }
  el.style.display='block';
  const removed=vulnRemoved[side];
  const notObvious=info.pass_involvement_rank>1||(info.betweenness_rank&&info.betweenness_rank>1);
  let html=`<div class="vuln-head"><span class="vuln-icon">\ud83c\udfaf</span> Most Critical Player: <b>${sn(info.critical_player)}</b></div>`;
  if(notObvious){
    const btRank=info.betweenness_rank?`#${info.betweenness_rank}`:'n/a';
    html+=`<div class="vuln-rank-contrast">Structural-damage rank <b>#1</b> \u00b7 Pass involvement <b>#${info.pass_involvement_rank}</b> \u00b7 Betweenness <b>${btRank}</b></div>`;
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
}

function toggleVulnRemoval(side,matchId,team){
  vulnRemoved[side]=!vulnRemoved[side];
  const info=getVulnInfo(matchId,team);
  renderNet(side==='home'?'sv-home':'sv-away',matchId,team,vulnRemoved[side]&&info?info.critical_player:null);
  renderVulnPanel(side,matchId,team);
}
