/* Ouro workspace explorer. Native browser UI; no build step or remote assets. */
'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const canonical = text => String(text ?? '').trim().toLocaleLowerCase();
const tokenLabel = text => String(text ?? '').replaceAll('\n','↵').replaceAll('\t','⇥').trim() || '·';
const percent = value => value >= .01 ? `${(value * 100).toFixed(1)}%` : value > .0001 ? `${(value * 100).toFixed(2)}%` : value > 0 ? '<0.01%' : '0%';
const COLORS = ['#d87947','#688c77','#6e88af','#aa8251'];
const EXAMPLES = {
  currency: {prompt:'Fact: The currency used in the country shaped like a boot is the',mode:'raw',max_new_tokens:16},
  capital: {prompt:'The capital of France is the city of',mode:'raw',max_new_tokens:8},
  planet: {prompt:'The color of the planet fourth from the sun is',mode:'raw',max_new_tokens:16}
};
const S = {run:null,runs:[],layers:new Map(),loop:3,site:128,position:0,lens:'jlens',mode:'raw',pins:[],activePin:null,
  traces:{},cellInventory:{},busy:false,job:null,live:true,cellWidth:78,rowHeight:23,gutter:67,header:40,
  rows:[],positions:[],conceptKey:'',cachedConcepts:[],loadSequence:0};
const traceCache = new Map();
let toastTimer,paintFrame;

async function api(path,options={}) {
  const response = await fetch(path,options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path,value) => api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
const delay = ms => new Promise(resolve=>setTimeout(resolve,ms));
function toast(message) { $('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,3800); }
function error(message) { $('error-banner').textContent=message;$('error-banner').hidden=false; }
function clearError() { $('error-banner').hidden=true; }
function textFor(id) { return S.run?.vocabulary[String(id)] ?? `<${id}>`; }
function currentPin() { return S.pins.find(p=>p.word===S.activePin) || null; }

async function status() {
  try {
    const data=await api('/api/status'); S.live=data.live;
    $('connection').classList.remove('offline');$('connection').lastChild.textContent=data.live ? (data.engine==='loading'?'Loading model':data.device==='cpu'?'Live analysis · CPU':'Live analysis ready') : 'Saved runs only';
    $('gpu-label').textContent=data.live?(data.device==='cpu'?`CPU · GPU ${data.gpu} busy`:`GPU ${data.gpu} · auto`):'Read-only';
    $('analyze-button').disabled=!S.live||S.busy;$('intervene-open').disabled=!S.live||S.busy;
  } catch (_) { $('connection').classList.add('offline');$('connection').lastChild.textContent='Server unavailable'; }
}

async function refreshRuns() { S.runs=await api('/api/runs');renderRuns(); }
function renderRuns() {
  const search=canonical($('run-search').value);
  const runs=S.runs.filter(r=>canonical(r.title).includes(search));
  $('run-list').innerHTML=runs.length?runs.map(r=>`<button class="run-item ${S.run?.id===r.id?'active':''}" data-run="${esc(r.id)}" title="${esc(r.title)}"><span class="run-item-icon">${r.kind==='research'?'▧':r.kind==='intervention'?'⇄':'↳'}</span><span><span class="run-item-title">${esc(r.title)}</span><small>${r.kind==='research'?'Research capture':r.kind==='intervention'?'Intervention':'Live analysis'} · ${r.token_count} tokens</small></span></button>`).join(''):'<div class="sidebar-empty">No matching saved runs.</div>';
}

function setMode(mode) { S.mode=mode;document.querySelectorAll('#prompt-mode button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));$('chat-options').hidden=mode!=='chat'; }
function fillComposer(input) { $('prompt').value=input.prompt||'';setMode(input.mode||'raw');$('system').value=input.system||'';$('prefill').value=input.prefill||'';const n=String(input.max_new_tokens??16);if (![...$('max-tokens').options].some(o=>o.value===n))$('max-tokens').add(new Option(`${n} tokens`,n));$('max-tokens').value=n; }
function composerPayload() { return {kind:'analyze',prompt:$('prompt').value,mode:S.mode,system:$('system').value,prefill:$('prefill').value,max_new_tokens:Number($('max-tokens').value)}; }

async function loadRun(runId,view=null) {
  const sequence=++S.loadSequence;clearError();$('run-list').setAttribute('aria-busy','true');$('run-loader').hidden=false;
  try {
    const [run,extras]=await Promise.all([api(`/api/runs/${encodeURIComponent(runId)}`),api(`/api/runs/${encodeURIComponent(runId)}/extras`)]);
    if(sequence!==S.loadSequence)return;
    S.run=run;S.layers=new Map(run.layers.map(l=>[l.site,l]));S.traces=extras.traces||{};S.pins=[];S.activePin=null;S.cellInventory={};S.conceptKey='';
    for(const [key,result] of Object.entries(extras.inventories||{})){S.cellInventory[key]=result.inventory;Object.assign(run.vocabulary,result.vocabulary);}
    S.loop=view&&[0,1,2,3,4].includes(Number(view.loop))?Number(view.loop):3;
    S.site=view&&Number.isInteger(Number(view.site))?Math.max(0,Math.min(191,Number(view.site))):128;
    S.position=view&&Number.isInteger(Number(view.pos))?Math.max(0,Math.min(run.tokens.length-1,Number(view.pos))):Math.max(0,run.prompt_length-1);
    S.lens=view&&['jlens','logit','inventory'].includes(view.lens)?view.lens:'jlens';
    if(view?.band!==undefined)$('workspace-only').checked=view.band==='1';
    if(view?.special!==undefined)$('show-special').checked=view.special==='1';
    if(['all','prompt','completion'].includes(view?.scope))$('position-scope').value=view.scope;
    if(['8','16','25','50'].includes(view?.k))$('inventory-k').value=view.k;
    if(['0','0.01','0.05'].includes(view?.theta))$('inventory-threshold').value=view.theta;
    if(view?.zoom==='105'||view?.zoom==='78')S.cellWidth=Number(view.zoom);
    if(view?.site!==undefined&&!view?.band&&(S.site%48<24||S.site%48>40))$('workspace-only').checked=false;
    fillComposer(run.input);$('analysis').hidden=false;$('welcome').hidden=true;
    $('run-title').textContent=run.title;$('run-kind').textContent=run.kind==='research'?'RESEARCH':run.kind==='intervention'?'INTERVENED':'LIVE RUN';
    $('run-meta').textContent=`${run.tokens.length} tokens · ${run.layers.length} recorded layers · 4 loops${run.elapsed_seconds?` · ${run.elapsed_seconds.toFixed(1)}s`:''}`;
    $('baseline-toggle').hidden=!run.baseline_id;
    const notices=run.notes.filter(n=>/budget|no final|thought block|omit|not captured/.test(n));
    $('run-notices').innerHTML=notices.map(n=>`<span class="notice">${esc(n)}</span>`).join('');
    $('token-count').textContent=`${run.prompt_length} prompt + ${run.tokens.length-run.prompt_length} continuation`;
    const previews=run.baseline_generation?[['Baseline',run.baseline_generation],['Intervened',run.generation]]:[['Model continuation',run.generation]];
    $('continuation-preview').classList.toggle('comparison',!!run.baseline_generation);
    $('continuation-preview').innerHTML=previews.map(([label,g])=>`<div><span class="eyebrow">${label}</span><p>${esc(g.answer_text||(g.unclosed_thought?'No final answer within this token budget.':g.output_tokens?'No visible output.':'Prompt-only readout; no continuation requested.'))}</p>${g.had_thought?`<details><summary>Full trace, including thought tokens</summary><pre>${esc(g.raw_text)}</pre></details>`:''}</div>`).join('');
    $('provenance-content').textContent=JSON.stringify({model:run.model,provenance:run.provenance,generation:run.generation,
      exact_input:run.prompt_text,sequence_ids:run.tokens.map(t=>t.id),intervention:run.intervention,notes:run.notes},null,2);
    renderRuns();renderAll();scrollSelection();updateHash();$('sidebar').classList.remove('open');
    if(view?.pins)for(const word of view.pins.split(',').filter(Boolean).slice(0,4))addPin(word);
  } finally { if(sequence===S.loadSequence){$('run-list').removeAttribute('aria-busy');$('run-loader').hidden=true;} }
}

async function startJob(payload) {
  if(S.busy){toast('An analysis is already running.');return;}
  S.busy=true;clearError();$('analyze-button').disabled=true;$('intervene-open').disabled=true;
  $('job-panel').hidden=false;$('job-phase').textContent='Preparing analysis';$('job-progress').style.width='0%';$('job-percent').textContent='0%';$('partial-output').textContent='';
  try {
    const submitted=await post('/api/jobs',payload);S.job=submitted.id;
    let job=submitted;
    while(!['complete','failed','cancelled'].includes(job.state)) {
      $('job-phase').textContent=job.phase;$('job-percent').textContent=`${Math.round(job.progress*100)}%`;$('job-progress').style.width=`${Math.round(job.progress*100)}%`;
      if(job.partial_text)$('partial-output').textContent=job.partial_text;
      await delay(450);job=await api(`/api/jobs/${submitted.id}`);
    }
    if(job.state==='failed')throw new Error(job.error);
    if(job.state==='cancelled'){toast('Analysis cancelled.');return;}
    if(job.result.kind==='run'){await refreshRuns();await loadRun(job.result.run_id);toast('Analysis saved. Click a cell to explore.');}
    else if(job.result.kind==='inventory'&&job.result.run_id===S.run?.id){
      const result=job.result;S.cellInventory[`${result.site}:${result.position}`]=result.inventory;Object.assign(S.run.vocabulary,result.vocabulary);
      S.lens='inventory';S.site=result.site;S.position=result.position;S.conceptKey='';renderAll();scrollSelection();updateHash();toast('Sparse inventory computed for this cell.');
    }
  } catch(e) { error(e.message); }
  finally { S.busy=false;S.job=null;$('job-panel').hidden=true;$('analyze-button').disabled=!S.live;$('intervene-open').disabled=!S.live;status(); }
}

function viewState() { return {run:S.run?.id||'',lens:S.lens,loop:String(S.loop),site:String(S.site),pos:String(S.position),pins:S.pins.map(p=>p.word).join(','),
  band:$('workspace-only').checked?'1':'0',special:$('show-special').checked?'1':'0',scope:$('position-scope').value,k:$('inventory-k').value,theta:$('inventory-threshold').value,zoom:String(S.cellWidth)}; }
function updateHash() { if(S.run)history.replaceState(null,'','#'+new URLSearchParams(viewState()).toString()); }
function positions() { if(!S.run)return[];const scope=$('position-scope').value;return S.run.tokens.filter(t=>($('show-special').checked||!t.special)&&(scope==='all'||t.segment===scope)).map(t=>t.position); }
function rows() { const band=$('workspace-only').checked;return Array.from({length:192},(_,s)=>s).filter(s=>(S.loop===0||Math.floor(s/48)+1===S.loop)&&(!band||(s%48>=24&&s%48<=40))); }
function inventoryAt(layer,position) {
  const extra=S.cellInventory[`${layer?.site}:${position}`];const value=extra||layer?.inventory;const index=extra?0:position;
  if(!value)return null;
  const k=$('inventory-k').value,snapshot=value.snapshots[k];if(!snapshot)return null;
  const ids=value.ids[index],coeff=snapshot.coefficients[index];if(!ids||!coeff)return null;
  const threshold=Number($('inventory-threshold').value);
  return {items:ids.slice(0,Number(k)).map((id,i)=>({id,value:coeff[i],selection:i+1})).filter(x=>x.id>=0&&x.value>=threshold),residual:snapshot.residual_fraction[index]};
}
function cellItems(site,position,lens=S.lens) {
  const layer=S.layers.get(site);if(!layer)return null;
  if(lens==='inventory')return inventoryAt(layer,position)?.items??null;
  const data=layer[lens];if(!data?.ids[position])return null;
  return data.ids[position].map((id,i)=>({id,value:data.probabilities[position][i],selection:i+1}));
}
function pinRank(site,position,pin,lens=S.lens) {
  if(!pin)return null;
  if(lens==='inventory'){
    const items=cellItems(site,position,lens);if(items===null)return null;
    const hit=items.find(x=>canonical(textFor(x.id))===canonical(pin.word));
    return {rank:hit?hit.selection:null,value:hit?.value||0,exact:false,inventory:true,found:!!hit};
  }
  const exact=S.traces[canonical(pin.word)]?.layers[String(site)]?.[lens];
  if(exact)return {rank:exact.ranks[position],value:exact.probabilities[position],exact:true,found:true};
  const items=cellItems(site,position,lens);if(items===null)return null;
  const matches=items.filter(x=>canonical(textFor(x.id))===canonical(pin.word));
  return {rank:matches.length?matches[0].selection:null,value:matches.reduce((s,x)=>s+x.value,0),exact:false,found:matches.length>0};
}

function renderAll() {
  if(!S.run)return;
  document.querySelectorAll('#lens-mode button').forEach(b=>b.classList.toggle('active',b.dataset.lens===S.lens));
  document.querySelectorAll('#loop-tabs button').forEach(b=>b.classList.toggle('active',Number(b.dataset.loop)===S.loop));
  $('inventory-settings').hidden=S.lens!=='inventory';
  S.positions=positions();S.rows=rows();
  $('grid-viewport').setAttribute('aria-rowcount',S.rows.length);$('grid-viewport').setAttribute('aria-colcount',S.positions.length);
  $('grid-space').style.width=`${S.gutter+S.positions.length*S.cellWidth}px`;$('grid-space').style.height=`${S.header+S.rows.length*S.rowHeight}px`;
  $('grid-subtitle').textContent=S.lens==='jlens'?'Transported to each loop’s own end':S.lens==='logit'?'Raw state, directly unembedded':'Positive sparse reconstruction';
  $('inventory-coverage').textContent=S.run.layers.some(l=>l.inventory)?'Full recorded grid':`${Object.keys(S.cellInventory).length} computed cell(s)`;
  const pin=currentPin();$('grid-legend').innerHTML=pin?`<span style="color:${pin.color}">●</span> ${esc(pin.word)} · ${S.lens==='inventory'?'coefficient':'rank'}`:`<span class="legend-gradient"></span>low → high ${S.lens==='inventory'?'coefficient':'probability'}`;
  renderTranscript();renderInspector();renderPins();renderTrace();renderConcepts();schedulePaint();
}
function renderTranscript() {
  $('transcript').innerHTML=S.run.tokens.filter(t=>$('show-special').checked||!t.special).map(t=>`<button class="token ${t.segment} ${t.special?'special':''} ${t.position===S.position?'selected':''}" data-position="${t.position}" title="Position ${t.position} · token ID ${t.id} · ${esc(t.text)}" aria-label="Token ${t.position}: ${esc(t.text)}" aria-pressed="${t.position===S.position}">${esc(tokenLabel(t.text))}</button>`).join('');
}
function renderInspector() {
  const token=S.run.tokens[S.position],loop=Math.floor(S.site/48)+1,physical=S.site%48;
  $('cell-heading').textContent=`Loop ${loop} · Layer ${String(physical).padStart(2,'0')}`;
  $('cell-position').textContent=`Virtual site ${S.site} · position ${S.position} · ${token.segment}`;
  $('selected-token').textContent=`After  ${JSON.stringify(token.text)}`;
  $('readout-title').textContent=S.lens==='jlens'?'J-lens readout':S.lens==='logit'?'Logit lens readout':'Selected atoms';
  $('readout-units').textContent=S.lens==='inventory'?'coefficient / ‖state‖':'probability';
  const items=cellItems(S.site,S.position);
  if(items===null)$('readout-list').innerHTML=`<div class="readout-row empty">${S.layers.has(S.site)?'No inventory for this cell yet. Compute it below.':'This loop-end target was not recorded in this research run. Live runs include all 192 sites.'}</div>`;
  else if(!items.length)$('readout-list').innerHTML='<div class="readout-row empty">No selected atoms meet this coefficient threshold.</div>';
  else {
    const display=S.lens==='inventory'?items.slice(0,10):items;
    const max=Math.max(...display.map(x=>x.value),1e-9);
    $('readout-list').innerHTML=display.map(x=>`<button class="readout-row" data-pin="${esc(textFor(x.id).trim())}" title="Token ID ${x.id} · click to trace ${esc(textFor(x.id))}"><span class="readout-rank">${x.selection}</span><span class="readout-bar" style="width:${Math.max(1,Math.round(x.value/max*76))}%"></span><span class="readout-word">${esc(tokenLabel(textFor(x.id)))}</span><span class="readout-prob">${S.lens==='inventory'?x.value.toFixed(3):percent(x.value)}</span></button>`).join('');
  }
  const inv=inventoryAt(S.layers.get(S.site),S.position);
  $('compute-inventory').hidden=!!inv;$('compute-inventory').disabled=!S.live||!S.layers.has(S.site)||S.busy;
  $('cell-reconstruction').textContent=S.lens==='inventory'&&inv?`${((1-inv.residual)*100).toFixed(1)}% of state energy reconstructed · ${inv.items.length} retained atoms`:'';
  $('loop-compare').innerHTML=[1,2,3,4].map(l=>{
    const site=(l-1)*48+physical,value=cellItems(site,S.position);const word=value?.length?tokenLabel(textFor(value[0].id)):value?'No retained atoms':'Not recorded';
    return `<button data-site="${site}" class="${l===loop?'active':''}" title="Inspect loop ${l}, layer ${physical}"><small>LOOP ${l}</small><strong>${esc(word)}</strong></button>`;
  }).join('');
  $('routing-note').textContent=S.lens==='jlens'?`Own-loop target: v${loop*48-1}. Layer numbers start at 0.`:S.lens==='logit'?'Raw residual through the model’s own norm and output head.':`Selected from normalized Jᵀu directions. Coefficients are not probabilities.`;
}

function setCell(site,position,{scroll=false}={}) {
  S.site=Math.max(0,Math.min(191,site));S.position=Math.max(0,Math.min(S.run.tokens.length-1,position));
  let filtersChanged=false;
  if($('workspace-only').checked&&(S.site%48<24||S.site%48>40)){$('workspace-only').checked=false;filtersChanged=true;}
  if($('position-scope').value!=='all'&&S.run.tokens[S.position].segment!==$('position-scope').value){$('position-scope').value='all';filtersChanged=true;}
  if(S.loop!==0&&Math.floor(S.site/48)+1!==S.loop){S.loop=Math.floor(S.site/48)+1;filtersChanged=true;}
  if(filtersChanged)renderAll();else{renderTranscript();renderInspector();renderTrace();schedulePaint();}
  if(scroll)scrollSelection();updateHash();
}
function scrollSelection() {
  const c=S.positions.indexOf(S.position),r=S.rows.indexOf(S.site),scroller=$('grid-scroller');
  if(c>=0){const x=c*S.cellWidth;if(x<scroller.scrollLeft)scroller.scrollLeft=x;else if(x+S.cellWidth>scroller.scrollLeft+scroller.clientWidth-S.gutter)scroller.scrollLeft=Math.max(0,x-(scroller.clientWidth-S.gutter)/2);}
  if(r>=0){const y=r*S.rowHeight;if(y<scroller.scrollTop)scroller.scrollTop=y;else if(y+S.rowHeight>scroller.scrollTop+scroller.clientHeight-S.header)scroller.scrollTop=Math.max(0,y-(scroller.clientHeight-S.header)/2);}
  schedulePaint();
}

function schedulePaint() { if(!paintFrame)paintFrame=requestAnimationFrame(()=>{paintFrame=null;paintGrid();}); }
function truncate(ctx,text,width) { text=tokenLabel(text);if(ctx.measureText(text).width<=width)return text;while(text.length>1&&ctx.measureText(text+'…').width>width)text=text.slice(0,-1);return text+'…'; }
function paintGrid() {
  if(!S.run)return;
  const canvas=$('grid-canvas'),scroller=$('grid-scroller'),w=scroller.clientWidth,h=scroller.clientHeight,dpr=window.devicePixelRatio||1;
  if(!w||!h)return;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);canvas.style.width=w+'px';canvas.style.height=h+'px';
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#ffffff';c.fillRect(0,0,w,h);
  const sx=scroller.scrollLeft,sy=scroller.scrollTop,{gutter:g,header:hh,rowHeight:rh,cellWidth:cw}=S;
  const c0=Math.max(0,Math.floor(sx/cw)),c1=Math.min(S.positions.length,Math.ceil((sx+w-g)/cw)+1);
  const r0=Math.max(0,Math.floor(sy/rh)),r1=Math.min(S.rows.length,Math.ceil((sy+h-hh)/rh)+1),pin=currentPin();
  c.save();c.beginPath();c.rect(g,hh,w-g,h-hh);c.clip();
  for(let ri=r0;ri<r1;ri++){
    const site=S.rows[ri],y=hh+ri*rh-sy,band=site%48>=24&&site%48<=40;
    for(let ci=c0;ci<c1;ci++){
      const pos=S.positions[ci],x=g+ci*cw-sx,items=cellItems(site,pos),rank=pinRank(site,pos,pin);
      let label='—',value=0,fill=band?'#fcfcf8':'#f8f9f5',color='#b4bbaa';
      if(pin&&items!==null&&rank){
        if(S.lens==='inventory'){value=rank.value;label=rank.found?value.toFixed(3):'—';}
        else {value=rank.rank?1/(1+Math.log10(rank.rank)):0;label=rank.rank?`#${rank.rank.toLocaleString()}`:'>8';}
        if(value){const intensity=S.lens==='inventory'?Math.min(1,value*4):value;fill=`rgba(219,135,77,${.06+intensity*.45})`;color=intensity>.6?'#75421f':'#967455';}
      }else if(items?.length){
        const top=items[0];label=tokenLabel(textFor(top.id));value=top.value;
        const intensity=S.lens==='inventory'?Math.min(1,value*3):Math.sqrt(Math.min(1,value));
        fill=S.lens==='inventory'?`rgba(134,164,111,${.04+intensity*.48})`:`rgba(212,165,103,${.035+intensity*.45})`;
        color=intensity>.45?'#80552e':intensity>.12?'#6f7058':'#a0a28f';
      }else if(items===null){fill='#f2f3ef';label='·';}
      c.fillStyle=fill;c.fillRect(x+1,y+1,cw-1,rh-1);
      if(pos===S.position){c.fillStyle='#e68c4f08';c.fillRect(x,y,cw,rh);}
      c.font=`${site===S.site&&pos===S.position?'600':'400'} 10px ui-monospace, monospace`;c.fillStyle=color;c.textAlign='center';c.textBaseline='middle';c.fillText(truncate(c,label,cw-10),x+cw/2,y+rh/2+.5);
      if(site===S.site&&pos===S.position){c.strokeStyle='#df7b49';c.lineWidth=1.8;c.strokeRect(x+1,y+1,cw-2,rh-2);}
    }
  }
  c.restore();
  c.fillStyle='#fbfcf8';c.fillRect(0,0,w,hh);c.fillRect(0,hh,g,h-hh);
  c.save();c.beginPath();c.rect(g,0,w-g,hh);c.clip();
  for(let ci=c0;ci<c1;ci++){
    const pos=S.positions[ci],x=g+ci*cw-sx,t=S.run.tokens[pos];
    if(pos===S.position){c.fillStyle='#f8e9db';c.fillRect(x,0,cw,hh);}
    c.textAlign='center';c.textBaseline='alphabetic';c.font='8px ui-monospace,monospace';c.fillStyle='#a5ad98';c.fillText(`${t.segment==='completion'?'↳ ':''}${pos}`,x+cw/2,12);
    c.font='10px ui-monospace,monospace';c.fillStyle=pos===S.position?'#be7346':t.segment==='completion'?'#ae8a61':'#6e7d5b';c.fillText(truncate(c,t.text,cw-10),x+cw/2,28);
    c.strokeStyle='#e7ecdf';c.beginPath();c.moveTo(x+cw,0);c.lineTo(x+cw,hh);c.stroke();
  }
  c.restore();c.save();c.beginPath();c.rect(0,hh,g,h-hh);c.clip();
  for(let ri=r0;ri<r1;ri++){
    const site=S.rows[ri],y=hh+ri*rh-sy,loop=Math.floor(site/48)+1;
    if(site===S.site){c.fillStyle='#f8e9db';c.fillRect(0,y,g,rh);}
    c.fillStyle=COLORS[loop-1];c.globalAlpha=.55;c.fillRect(8,y+8,3,8);c.globalAlpha=1;
    c.font='9px ui-monospace,monospace';c.textBaseline='middle';c.textAlign='left';c.fillStyle=site===S.site?'#b46d43':'#8d9a7c';c.fillText(S.loop===0?`${loop} · ${String(site%48).padStart(2,'0')}`:`L${String(site%48).padStart(2,'0')}`,17,y+rh/2);
    if(site%48===0){c.strokeStyle='#d2ddc5';c.beginPath();c.moveTo(0,y);c.lineTo(g,y);c.stroke();}
  }
  c.restore();c.fillStyle='#f6f8f1';c.fillRect(0,0,g,hh);c.fillStyle='#91a17d';c.font='8px ui-monospace,monospace';c.textAlign='left';c.textBaseline='middle';c.fillText(S.loop?`LOOP ${S.loop}`:'LOOP / L',9,hh/2);
  c.strokeStyle='#dfe6d4';c.lineWidth=1;c.beginPath();c.moveTo(g,0);c.lineTo(g,h);c.moveTo(0,hh);c.lineTo(w,hh);c.stroke();
  if(!S.positions.length){c.fillStyle='#9eaa91';c.font='12px sans-serif';c.textAlign='center';c.fillText('No tokens in this selection',w/2,h/2);}
}

function hitCell(event) {
  const rect=$('grid-viewport').getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  const scroll=$('grid-scroller');if(x<S.gutter||y<S.header||x>scroll.clientWidth||y>scroll.clientHeight)return null;
  const ci=Math.floor((x-S.gutter+scroll.scrollLeft)/S.cellWidth),ri=Math.floor((y-S.header+scroll.scrollTop)/S.rowHeight);
  return S.positions[ci]!==undefined&&S.rows[ri]!==undefined?{site:S.rows[ri],position:S.positions[ci],x,y}:null;
}

function renderPins() {
  $('pins').innerHTML=S.pins.map(p=>`<button class="pin-chip ${S.activePin===p.word?'active':''}" style="--pin-color:${p.color}" data-activate-pin="${esc(p.word)}"><i></i>${esc(p.word)}<small>${S.traces[canonical(p.word)]?'exact':p.loading?'…':'top-8'}</small><span data-remove-pin="${esc(p.word)}" role="button" aria-label="Unpin ${esc(p.word)}">×</span></button>`).join('');
  $('trace-quality').textContent=S.pins.length?(S.pins.every(p=>S.traces[canonical(p.word)])?'EXACT RANKS':'SAVED TOP-8 / LOADING'):'PIN A CONCEPT';
}
async function addPin(raw) {
  const word=raw.trim();if(!word)return;
  if(S.pins.some(p=>canonical(p.word)===canonical(word))){S.activePin=S.pins.find(p=>canonical(p.word)===canonical(word)).word;renderAll();updateHash();return;}
  if(S.pins.length>=4){toast('Unpin a concept before adding another (up to four).');return;}
  if(!S.run){toast('Open a run before tracing a concept.');return;}
  const pin={word,color:COLORS[S.pins.length],loading:false};S.pins.push(pin);S.activePin=word;renderAll();updateHash();
  const runId=S.run.id,key=`${runId}:${canonical(word)}`;
  if(S.traces[canonical(word)])return;
  if(traceCache.has(key)){S.traces[canonical(word)]=traceCache.get(key);renderAll();return;}
  if(!S.live)return;
  pin.loading=true;renderPins();
  try {
    const submitted=await post('/api/jobs',{kind:'trace',run_id:runId,word});let job=submitted;
    while(!['complete','failed','cancelled'].includes(job.state)){await delay(700);job=await api(`/api/jobs/${submitted.id}`);}
    if(job.state!=='complete')throw new Error(job.error||'Trace cancelled');
    traceCache.set(key,job.result);
    if(S.run?.id===runId){S.traces[canonical(word)]=job.result;pin.loading=false;renderAll();}
  }catch(e){pin.loading=false;if(S.run?.id===runId){renderPins();toast(e.message);}}
}
function renderTrace() {
  if(!S.pins.length){$('trace-chart').innerHTML='<div class="chart-empty">Click a readout token or pin a word to follow its rank through the loops.</div>';$('trace-caption').textContent='Exact ranks use the full vocabulary. Saved top-8 matches appear while a trace loads.';return;}
  const width=700,height=160,left=36,right=14,top=22,bottom=26,iw=width-left-right,ih=height-top-bottom;
  const exact=S.pins.some(p=>S.traces[canonical(p.word)]),max=exact?Math.max(10000,...S.pins.flatMap(p=>Object.values(S.traces[canonical(p.word)]?.layers||{}).map(v=>v[S.lens==='logit'?'logit':'jlens'].ranks[S.position]||1))):8;
  const x=s=>left+s/191*iw,y=r=>top+Math.log10(Math.max(1,r))/Math.log10(max)*ih;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Concept rank across virtual layers at token ${S.position}">`;
  for(let loop=0;loop<4;loop++){
    svg+=`<rect x="${x(loop*48+24)}" y="${top}" width="${17/191*iw}" height="${ih}" fill="#f1f4eb"/><text x="${x(loop*48+23)}" y="10" text-anchor="middle" fill="#97a58a" font-size="8" font-family="sans-serif">LOOP ${loop+1}</text>`;
    if(loop)svg+=`<line x1="${x(loop*48)}" y1="${top}" x2="${x(loop*48)}" y2="${height-bottom}" stroke="#dce5d1" stroke-dasharray="2 3"/>`;
  }
  const ticks=exact?[1,10,100,1000,10000,...(max>10000?[max]:[])]:[1,2,4,8];
  for(const tick of ticks)svg+=`<line x1="${left}" y1="${y(tick)}" x2="${width-right}" y2="${y(tick)}" stroke="#edf1e6"/><text x="${left-7}" y="${y(tick)+3}" text-anchor="end" fill="#a1ad94" font-size="8" font-family="monospace">${tick>=1000?Math.round(tick/1000)+'k':tick}</text>`;
  for(const tick of [0,47,95,143,191])svg+=`<text x="${x(tick)}" y="${height-8}" text-anchor="middle" fill="#a1ad94" font-size="8" font-family="monospace">${tick}</text>`;
  for(const pin of S.pins){
    let path='',connected=false;
    for(let site=0;site<192;site++){
      const rank=pinRank(site,S.position,pin,S.lens==='logit'?'logit':'jlens');
      if(rank?.rank){path+=`${connected?'L':'M'}${x(site).toFixed(2)},${y(rank.rank).toFixed(2)} `;connected=true;}
      else connected=false;
    }
    svg+=`<path d="${path}" fill="none" stroke="${pin.color}" stroke-width="1.8" ${S.traces[canonical(pin.word)]?'':'stroke-dasharray="3 2"'}/>`;
    const selected=pinRank(S.site,S.position,pin,S.lens==='logit'?'logit':'jlens');if(selected?.rank)svg+=`<circle cx="${x(S.site)}" cy="${y(selected.rank)}" r="3" fill="${pin.color}" stroke="white" stroke-width="1"/>`;
  }
  svg+=`<line x1="${x(S.site)}" y1="${top-2}" x2="${x(S.site)}" y2="${height-bottom}" stroke="#d99a6f" stroke-width="1" stroke-dasharray="3 3"/></svg>`;
  $('trace-chart').innerHTML=svg;
  $('trace-caption').textContent=exact?'Lower rank is stronger. Solid lines use the full vocabulary; gaps in dashed lines mean outside the saved top-8.': 'Dashed traces show only saved top-8 matches. Gaps mean outside the top-8, not rank zero. Exact traces use local compute.';
  if(S.lens==='inventory')$('trace-caption').textContent+= ' The chart uses J-lens ranks; the inventory grid shows coefficients.';
}

function renderConcepts() {
  const key=[S.run.id,S.lens,S.loop,$('workspace-only').checked,$('show-special').checked,$('position-scope').value,$('inventory-k').value,$('inventory-threshold').value,Object.keys(S.cellInventory).length].join(':');
  if(key!==S.conceptKey){
    const counts=new Map();
    for(const layer of S.run.layers){if(S.loop&&layer.loop!==S.loop)continue;if($('workspace-only').checked&&(layer.physical<24||layer.physical>40))continue;
      for(const pos of S.positions){const items=cellItems(layer.site,pos)||[];for(const item of items){const word=canonical(textFor(item.id));if(!/^[\p{L}]+(?:[-'’][\p{L}]+)*$/u.test(word)||word.length<2)continue;
        if(!counts.has(word))counts.set(word,{word,count:0,loops:[0,0,0,0]});const v=counts.get(word);v.count++;v.loops[layer.loop-1]++;
      }}
    }
    S.cachedConcepts=[...counts.values()].sort((a,b)=>b.count-a.count).slice(0,12);S.conceptKey=key;
  }
  $('concept-caption').textContent=`${S.lens==='inventory'?'Retained inventory atoms':'Saved top-8 word occurrences'} · ${S.loop?'loop '+S.loop:'all loops'} · ${$('workspace-only').checked?'physical layers 24–40':'all physical layers'}`;
  $('concept-list').innerHTML=S.cachedConcepts.length?S.cachedConcepts.map(v=>`<button class="concept-item" data-pin="${esc(v.word)}" title="${v.count} matching slots. Click to trace ${esc(v.word)}."><div><strong>${esc(v.word)}</strong><small>${v.count.toLocaleString()}</small></div><div class="concept-spark" aria-label="Occurrences by loop: ${v.loops.join(', ')}">${v.loops.map(n=>`<span style="height:${Math.max(2,n/Math.max(...v.loops)*20)}px"></span>`).join('')}</div></button>`).join(''):'<div class="tiny muted">No word atoms in this selection. Try another scope or compute an inventory.</div>';
}

async function example(name) {
  const input=EXAMPLES[name];fillComposer(input);$('sidebar').classList.remove('open');
  const cached=S.runs.find(r=>r.kind==='live'&&r.input.prompt===input.prompt&&r.input.mode===input.mode);
  if(cached)await loadRun(cached.id);else await startJob({...input,kind:'analyze'});
}

$('prompt-mode').addEventListener('click',e=>{const b=e.target.closest('[data-mode]');if(b)setMode(b.dataset.mode);});
$('analyze-button').addEventListener('click',()=>startJob(composerPayload()));
$('cancel-job').addEventListener('click',async()=>{if(S.job){try{await post(`/api/cancel/${S.job}`,{});$('job-phase').textContent='Cancelling after the current operation…';}catch(e){error(e.message);}}});
$('new-run').addEventListener('click',()=>{S.run=null;S.pins=[];S.loadSequence++;$('analysis').hidden=true;$('welcome').hidden=false;fillComposer({prompt:'',mode:'raw',max_new_tokens:16});renderRuns();history.replaceState(null,'',location.pathname);$('prompt').focus();$('sidebar').classList.remove('open');});
document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>example(b.dataset.example).catch(e=>error(e.message))));
$('welcome-example').addEventListener('click',()=>example('currency').catch(e=>error(e.message)));
$('refresh-runs').addEventListener('click',()=>refreshRuns().then(()=>toast('Saved runs refreshed.')).catch(e=>error(e.message)));
$('run-search').addEventListener('input',renderRuns);
$('run-list').addEventListener('click',e=>{const b=e.target.closest('[data-run]');if(b)loadRun(b.dataset.run).catch(e=>error(e.message));});
$('transcript').addEventListener('click',e=>{const b=e.target.closest('[data-position]');if(b)setCell(S.site,Number(b.dataset.position),{scroll:true});});
$('lens-mode').addEventListener('click',e=>{const b=e.target.closest('[data-lens]');if(!b)return;S.lens=b.dataset.lens;renderAll();updateHash();});
$('loop-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-loop]');if(!b)return;S.loop=Number(b.dataset.loop);if(S.loop)S.site=(S.loop-1)*48+S.site%48;renderAll();scrollSelection();updateHash();});
for(const name of ['show-special','workspace-only','position-scope','inventory-k','inventory-threshold'])$(name).addEventListener('change',()=>{
  if(!S.run)return;if(name==='position-scope'||name==='show-special'){const available=positions();if(available.length&&!available.includes(S.position))S.position=available[0];}
  if(name==='workspace-only'&&$('workspace-only').checked&&(S.site%48<24||S.site%48>40))S.site=Math.floor(S.site/48)*48+32;
  renderAll();scrollSelection();updateHash();
});
$('zoom-grid').addEventListener('click',()=>{S.cellWidth=S.cellWidth===78?105:78;renderAll();scrollSelection();updateHash();});
$('grid-scroller').addEventListener('scroll',()=>{$('grid-tooltip').hidden=true;schedulePaint();});
new ResizeObserver(schedulePaint).observe($('grid-viewport'));
$('grid-viewport').addEventListener('click',e=>{const hit=hitCell(e);if(hit)setCell(hit.site,hit.position);});
$('grid-viewport').addEventListener('mousemove',e=>{
  if(!S.run)return;const hit=hitCell(e);if(!hit){$('grid-tooltip').hidden=true;return;}
  const items=cellItems(hit.site,hit.position),tip=$('grid-tooltip');
  tip.textContent=`Loop ${Math.floor(hit.site/48)+1} · L${hit.site%48} · token ${hit.position}\n`+(items===null?'Not recorded':items.slice(0,3).map(x=>`${tokenLabel(textFor(x.id))}  ${S.lens==='inventory'?x.value.toFixed(3):percent(x.value)}`).join('\n'));
  tip.style.left=Math.max(4,Math.min(hit.x+12,$('grid-viewport').clientWidth-224))+'px';tip.style.top=Math.max(4,Math.min(hit.y+12,$('grid-viewport').clientHeight-95))+'px';tip.hidden=false;
});
$('grid-viewport').addEventListener('mouseleave',()=>$('grid-tooltip').hidden=true);
$('grid-viewport').addEventListener('keydown',e=>{
  if(!S.run||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();
  let c=Math.max(0,S.positions.indexOf(S.position)),r=Math.max(0,S.rows.indexOf(S.site));
  if(e.key==='ArrowLeft')c--;if(e.key==='ArrowRight')c++;if(e.key==='ArrowUp')r--;if(e.key==='ArrowDown')r++;
  if(S.positions.length&&S.rows.length)setCell(S.rows[Math.max(0,Math.min(S.rows.length-1,r))],S.positions[Math.max(0,Math.min(S.positions.length-1,c))],{scroll:true});
});
$('loop-compare').addEventListener('click',e=>{const b=e.target.closest('[data-site]');if(b)setCell(Number(b.dataset.site),S.position,{scroll:true});});
for(const id of ['readout-list','concept-list'])$(id).addEventListener('click',e=>{const b=e.target.closest('[data-pin]');if(b)addPin(b.dataset.pin);});
$('pin-form').addEventListener('submit',e=>{e.preventDefault();const text=$('pin-input').value;$('pin-input').value='';for(const word of text.split(',').filter(Boolean))addPin(word);});
$('pins').addEventListener('click',e=>{
  const remove=e.target.closest('[data-remove-pin]');if(remove){S.pins=S.pins.filter(p=>p.word!==remove.dataset.removePin);if(S.activePin===remove.dataset.removePin)S.activePin=S.pins[0]?.word??null;renderAll();updateHash();return;}
  const b=e.target.closest('[data-activate-pin]');if(b){S.activePin=S.activePin===b.dataset.activatePin?null:b.dataset.activatePin;renderAll();updateHash();}
});
$('trace-chart').addEventListener('click',e=>{if(!S.run||!S.pins.length)return;const box=$('trace-chart').getBoundingClientRect();const site=Math.round(((e.clientX-box.left)/box.width*700-36)/(700-36-14)*191);setCell(site,S.position,{scroll:true});});
$('compute-inventory').addEventListener('click',()=>startJob({kind:'inventory',run_id:S.run.id,site:S.site,position:S.position}));
$('baseline-toggle').addEventListener('click',()=>{if(S.run.baseline_id)loadRun(S.run.baseline_id).catch(e=>error(e.message));});
$('intervene-open').addEventListener('click',()=>{if(!S.run)return;$('intervention-source').value=currentPin()?.word||(/France/.test(S.run.input.prompt)?'France':'');$('intervention-destination').value=/France/.test(S.run.input.prompt)?'China':'';$('intervention-dialog').showModal();});
$('intervention-operation').addEventListener('change',()=>{const erase=$('intervention-operation').value==='erase';$('destination-field').hidden=erase;$('intervention-destination').required=!erase;});
$('intervention-form').addEventListener('submit',e=>{
  e.preventDefault();const payload={kind:'intervene',run_id:S.run.id,operation:$('intervention-operation').value,source:$('intervention-source').value.trim(),destination:$('intervention-destination').value.trim(),
    loops:[...document.querySelectorAll('.intervention-loops input:checked')].map(x=>Number(x.value)),band:[Number($('intervention-start').value),Number($('intervention-end').value)]};
  $('intervention-dialog').close();startJob(payload);
});
for(const id of ['about-open','help-open'])$(id).addEventListener('click',()=>$('help-dialog').showModal());
document.querySelectorAll('.close-dialog').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
$('menu-button').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
$('copy-link').addEventListener('click',async()=>{updateHash();try{await navigator.clipboard.writeText(location.href);toast('View link copied. Open through the same viewer server.');}catch(_){toast('The address bar contains the link to this view.');}});
$('export-run').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({...S.run,viewer_state:viewState(),exact_traces:S.traces,inspected_inventories:S.cellInventory})],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`ouro-${S.run.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Run exported as JSON.');});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!document.querySelector('dialog[open]')){e.preventDefault();startJob(composerPayload());}if(e.key==='n'&&!/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)&&!document.querySelector('dialog[open]'))$('new-run').click();});
window.addEventListener('hashchange',()=>{const v=Object.fromEntries(new URLSearchParams(location.hash.slice(1)));if(v.run)loadRun(v.run,v).catch(e=>error(e.message));});

async function boot() {
  await status();
  try {
    await refreshRuns();const view=Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
    const initial=view.run||S.runs.find(r=>r.kind==='live'&&r.input.prompt===EXAMPLES.currency.prompt)?.id||S.runs[0]?.id;
    if(initial)await loadRun(initial,view.run?view:null);
  } catch(e){error(e.message);}
}
boot();setInterval(status,15000);
