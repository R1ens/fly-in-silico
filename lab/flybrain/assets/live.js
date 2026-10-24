/* Пульт: панель управления слева, живая картинка в середине, цифры справа.
   Кадры приходят по SSE прямо во время счёта, поэтому вся отрисовка работает
   от потока, а не от заранее готового массива, как в статическом отчёте. */
(() => {
const $ = id => document.getElementById(id);
const fmt = n => Number(n).toLocaleString('ru-RU');

let META = null, RCODE = null, viewer = null;
let N = 0;

/* --- состояние текущего прогона --- */
let act = null;          // Float32Array(N), сглаженная частота
let disp = null;         // Float32Array(N), 0..1 для шейдера
let scale = 60;          // Гц на верх цветовой шкалы, подстраивается
let binMs = 10, tau = 50;
let stimSet = null;
let mode = 'stim';
let arenaCfg = null, track = [], flyNow = null, odourTile = null;
let regionNames = [], regionCounts = null;

/* ---------------------------------------------------------------- загрузка */
async function boot(){
  META = await (await fetch('/api/meta')).json();
  N = META.n;
  regionNames = META.regions;
  const raw = await (await fetch('/api/geometry')).arrayBuffer();
  const pos = new Uint16Array(raw, 0, N * 3);
  RCODE = new Uint8Array(raw, N * 6, N);

  viewer = createBrainViewer({
    canvas: $('gl'), labelsEl: $('labels'),
    n: N, pos, bbox: META.bbox, regionCode: RCODE,
    regions: META.regions, regionColors: META.regionColors,
    regionCounts: META.regionCounts, regionAnchors: META.regionAnchors,
  });
  /* слой «активных» — сразу все нейроны: какие из них загорятся, заранее
     неизвестно, прогон ещё не начался */
  const all = new Int32Array(N);
  for (let i = 0; i < N; i++) all[i] = i;
  viewer.setActive(all);
  act = new Float32Array(N);
  disp = new Float32Array(N);
  regionCounts = META.regionCounts;

  $('readout').innerHTML = '<option value="">все нисходящие</option>' +
    META.readouts.map(r => `<option value="${r}">${r}</option>`).join('');
  $('readout').value = 'ALPN';

  buildPresets();
  wireControls();
  connect();
  requestAnimationFrame(frame);
  addEventListener('resize', () => { viewer.resize(); });
}

const PRESETS = [
  ['сахар', 'class:gustatory+side:left', 200, 100, 400, ''],
  ['запах', 'class:olfactory+side:left', 150, 100, 500, ''],
  ['запах без ГТ', 'class:olfactory+side:left', 150, 100, 500, 'type:KC'],
  ['удар', 'type:LPLC2', 150, 100, 350, ''],
  ['вперёд', 'type=DNp09', 100, 0, '', ''],
];
function buildPresets(){
  $('presets').innerHTML = PRESETS.map((p, i) =>
    `<button data-i="${i}">${p[0]}</button>`).join('');
  $('presets').querySelectorAll('button').forEach(b => b.onclick = () => {
    const p = PRESETS[+b.dataset.i];
    $('q-act').value = p[1]; $('rate').value = p[2];
    $('on').value = p[3]; $('off').value = p[4]; $('q-sil').value = p[5];
    preview('q-act', 'hint-act'); preview('q-sil', 'hint-sil');
  });
}

/* ------------------------------------------------------------ связь с SSE */
function connect(){
  const es = new EventSource('/api/events');
  es.onopen = () => { $('conn').textContent = 'связь есть'; $('conn').className = 'live'; };
  es.onerror = () => { $('conn').textContent = 'нет связи'; $('conn').className = ''; };
  es.addEventListener('run', e => onRun(JSON.parse(e.data)));
  es.addEventListener('frame', e => onFrame(JSON.parse(e.data)));
  es.addEventListener('done', e => onDone(JSON.parse(e.data)));
  es.addEventListener('status', e => onStatus(JSON.parse(e.data)));
  es.addEventListener('comment', e => onComment(JSON.parse(e.data)));
  es.addEventListener('error', e => {
    try { setHint('hint-run', JSON.parse(e.data).message, true); } catch (_) {}
  });
}

function b64(str, Type){
  const bin = atob(str), buf = new ArrayBuffer(bin.length);
  const v = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) v[i] = bin.charCodeAt(i);
  return new Type(buf);
}

function onStatus(s){
  const running = s.state === 'running';
  $('go').disabled = running;
  $('stop').disabled = !running;
  if (!running) $('save').disabled = false;
}

function onRun(r){
  mode = r.mode;
  act.fill(0); disp.fill(0);
  scale = 60;
  binMs = r.bin_ms || 10;
  track = []; flyNow = null;
  $('toptab').innerHTML = '';
  $('summary').innerHTML = '<p class="note">Считается…</p>';
  $('comments').innerHTML = '';
  $('cmtbox').style.display = 'none';
  setHint('hint-run', r.title || '');

  const stim = new Float32Array(N);
  (r.stim || []).forEach(i => stim[i] = 1);
  stimSet = stim;
  const all = new Int32Array(N);
  for (let i = 0; i < N; i++) all[i] = i;
  viewer.setActive(all, stim);

  $('stage').classList.toggle('arena', mode === 'arena');
  if (mode === 'arena') { arenaCfg = r.arena; buildOdourTile(); }
  viewer.resize();
}

function onFrame(f){
  const idx = b64(f.idx, Int32Array), val = b64(f.val, Uint8Array);
  const hzPerSpike = 1000 / binMs;
  const decay = Math.exp(-binMs / tau);
  /* экспоненциальное сглаживание, как у кальциевого индикатора: иначе при
     шаге 10 мс один спайк это сразу 100 Гц и картинка мигает */
  for (let i = 0; i < N; i++) act[i] *= decay;
  let peak = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    act[i] += val[k] * hzPerSpike * (1 - decay);
    if (act[i] > peak) peak = act[i];
  }
  scale = Math.max(scale * 0.99, Math.min(400, Math.max(10, peak)));
  for (let i = 0; i < N; i++) disp[i] = act[i] / scale;
  viewer.setActivity(disp);

  $('hud-t').textContent = Math.round(f.t_ms);
  $('hud-act').textContent = fmt(idx.length);
  $('hud-peak').textContent = Math.round(peak);
  $('hud-scale').textContent = Math.round(scale) + ' Гц';

  updateRegions();
  if (f.fly) { flyNow = f.fly; track.push([f.fly.x, f.fly.y, f.fly.theta]); }
}

function onDone(d){
  const rows = (d.top || []).map(t =>
    `<tr><td>${t.label}</td><td><span class="chip" style="background:${
      META.regionColors[t.region] || '#888'}"></span>${t.region}</td>
     <td class="n">${t.hz}</td></tr>`).join('');
  $('toptab').innerHTML = rows;
  const m = d.metrics;
  let extra = '';
  if (m) {
    extra = `<tr><td style="color:var(--dim)">совпадение с градиентом</td>
      <td class="n">${m.grad_alignment >= 0 ? '+' : ''}${m.grad_alignment.toFixed(3)}</td></tr>
      <tr><td style="color:var(--dim)">до еды</td><td class="n">${
        m.food_dist_first_half_mm} → ${m.food_dist_second_half_mm} мм</td></tr>
      <tr><td style="color:var(--dim)">визитов</td><td class="n">${m.food_visits}</td></tr>`;
  }
  $('summary').innerHTML = `<table>
    <tr><td style="color:var(--dim)">активных нейронов</td><td class="n">${fmt(d.n_active)}</td></tr>
    <tr><td style="color:var(--dim)">всего спайков</td><td class="n">${fmt(d.total_spikes)}</td></tr>
    <tr><td style="color:var(--dim)">счёт</td><td class="n">${d.sim_s} с</td></tr>
    ${extra}</table>`;
  $('save').disabled = false;
}

function onComment(c){
  $('cmtbox').style.display = 'block';
  const el = document.createElement('div');
  el.className = 'cmt';
  el.innerHTML = `<span class="ts">${(c.t_ms / 1000).toFixed(1)} с</span>${c.text}`;
  $('comments').appendChild(el);
  $('comments').scrollTop = 1e9;
}

/* --------------------------------------------------------------- области */
let regionAcc = null;
function updateRegions(){
  if (!regionAcc) regionAcc = new Float64Array(regionNames.length);
  regionAcc.fill(0);
  for (let i = 0; i < N; i++) {
    const a = act[i];
    if (a > 0.01) regionAcc[RCODE[i]] += a;
  }
  const rows = [];
  for (let r = 0; r < regionNames.length; r++) {
    const name = regionNames[r];
    const cnt = regionCounts[name] || 1;
    rows.push({name, v: regionAcc[r] / cnt});
  }
  rows.sort((a, b) => b.v - a.v);
  const top = rows.slice(0, 12);
  const max = top.length ? Math.max(top[0].v, 1e-6) : 1;
  $('hud-reg').textContent = top.length && top[0].v > 0.05 ? top[0].name : '—';
  $('regbars').innerHTML = top.map(o => {
    const col = META.regionColors[o.name] || '#888';
    const w = Math.max(1, o.v / max * 100);
    return `<div class="bar" title="${META.regionLabels[o.name] || o.name}">
      <span class="nm" data-r="${o.name}"><span class="chip"
        style="background:${col}"></span>${o.name}</span>
      <span class="tr"><span class="fl" style="width:${w}%;background:${col}"></span></span>
      <span class="vl">${o.v.toFixed(1)}</span></div>`;
  }).join('');
  $('regbars').querySelectorAll('.nm').forEach(el => el.onclick = () => {
    const r = el.dataset.r;
    viewer.setFocus(viewer.focus === r ? null : r);
  });
  const map = {};
  top.forEach(o => map[o.name] = o.v);
  regionActivity = map;
}
let regionActivity = null;

/* ----------------------------------------------------------------- арена */
function buildOdourTile(){
  if (!arenaCfg) return;
  const R = 96;
  odourTile = document.createElement('canvas');
  odourTile.width = odourTile.height = R;
  const c = odourTile.getContext('2d');
  const img = c.createImageData(R, R);
  for (let j = 0; j < R; j++) {
    const wy = (1 - (j + 0.5) / R) * arenaCfg.height;
    for (let i = 0; i < R; i++) {
      const wx = ((i + 0.5) / R) * arenaCfg.width;
      let v = 0;
      for (const o of arenaCfg.odours) {
        const dx = wx - o.x, dy = wy - o.y;
        v += o.s * Math.exp(-(dx * dx + dy * dy) / (2 * o.sigma * o.sigma));
      }
      const t = Math.pow(Math.min(1, v), 1.4), p = (j * R + i) * 4;
      img.data[p] = 7 + 30 * t; img.data[p + 1] = 13 + 140 * t;
      img.data[p + 2] = 21 + 78 * t; img.data[p + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
}

function drawArena(){
  if (mode !== 'arena' || !arenaCfg || !odourTile) return;
  const cv = $('arena'), dpr = Math.min(devicePixelRatio || 1, 2);
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);

  const pad = 12, gaugeW = Math.min(300, Math.max(190, W * 0.34));
  const side = Math.max(60, Math.min(H - pad * 2, W - gaugeW - pad * 3));
  const ox = pad, oy = (H - side) / 2;
  const sx = v => ox + (v / arenaCfg.width) * side;
  const sy = v => oy + (1 - v / arenaCfg.height) * side;

  c.save(); c.beginPath(); c.rect(ox, oy, side, side); c.clip();
  c.drawImage(odourTile, ox, oy, side, side); c.restore();
  c.strokeStyle = '#243044'; c.lineWidth = 1;
  c.strokeRect(ox + 0.5, oy + 0.5, side, side);

  for (const f of arenaCfg.foods) {
    const r = Math.max(3, f.r / arenaCfg.width * side);
    c.beginPath(); c.arc(sx(f.x), sy(f.y), r, 0, 7);
    c.fillStyle = 'rgba(255,205,120,.9)'; c.fill();
    c.strokeStyle = 'rgba(255,230,180,.6)'; c.lineWidth = 1.5; c.stroke();
  }

  if (track.length) {
    c.lineWidth = 1.6; c.lineJoin = 'round';
    c.strokeStyle = 'rgba(120,170,255,.3)';
    c.beginPath();
    track.forEach((p, i) => i ? c.lineTo(sx(p[0]), sy(p[1])) : c.moveTo(sx(p[0]), sy(p[1])));
    c.stroke();
    c.beginPath(); c.arc(sx(track[0][0]), sy(track[0][1]), 3, 0, 7);
    c.strokeStyle = 'rgba(160,210,255,.8)'; c.lineWidth = 1.5; c.stroke();
  }

  if (flyNow) {
    c.save();
    c.translate(sx(flyNow.x), sy(flyNow.y)); c.rotate(-flyNow.theta);
    c.beginPath(); c.moveTo(12, 0); c.lineTo(-8, 6.5); c.lineTo(-4, 0); c.lineTo(-8, -6.5);
    c.closePath();
    c.fillStyle = flyNow.feeding ? '#7ef0ff' : '#ffd79a'; c.fill();
    c.strokeStyle = 'rgba(0,0,0,.6)'; c.lineWidth = 1; c.stroke();
    const nl = Math.min(1, flyNow.odour_l / 150), nr = Math.min(1, flyNow.odour_r / 150);
    c.beginPath(); c.arc(9, -5, 3, 0, 7);
    c.fillStyle = `rgba(120,255,190,${0.2 + 0.8 * nl})`; c.fill();
    c.beginPath(); c.arc(9, 5, 3, 0, 7);
    c.fillStyle = `rgba(120,255,190,${0.2 + 0.8 * nr})`; c.fill();
    c.restore();

    const gx = ox + side + pad * 1.5;
    let gy = oy + 4;
    const gw = Math.min(gaugeW - pad, W - gx - pad);
    c.font = '11px ui-monospace,monospace'; c.textBaseline = 'middle';
    const bar = (label, val, max, col, extra) => {
      c.fillStyle = '#8b97a8'; c.textAlign = 'left';
      c.fillText(label, gx, gy + 6);
      c.fillStyle = '#0f1620'; c.fillRect(gx + 92, gy, gw - 140, 12);
      c.fillStyle = col;
      c.fillRect(gx + 92, gy, (gw - 140) * Math.max(0, Math.min(1, val / max)), 12);
      c.fillStyle = '#e6edf6'; c.textAlign = 'right';
      c.fillText(extra, gx + gw - 4, gy + 6);
      gy += 19;
    };
    bar('запах L', flyNow.odour_l, 150, '#4fd1c5', flyNow.odour_l.toFixed(0) + ' Гц');
    bar('запах R', flyNow.odour_r, 150, '#4fd1c5', flyNow.odour_r.toFixed(0) + ' Гц');
    bar('вкус', flyNow.taste, 200, '#f6ad55', flyNow.taste.toFixed(0) + ' Гц');
    gy += 5;
    const dmax = Math.max(flyNow.dn_l, flyNow.dn_r, 1) * 1.3;
    bar('выход L', flyNow.dn_l, dmax, '#ed64a6', flyNow.dn_l.toFixed(1) + ' Гц');
    bar('выход R', flyNow.dn_r, dmax, '#ed64a6', flyNow.dn_r.toFixed(1) + ' Гц');
    gy += 5;
    bar('скорость', flyNow.v, 22, '#90cdf4', flyNow.v.toFixed(1) + ' мм/с');
    const om = flyNow.omega * 180 / Math.PI;
    bar('поворот', Math.abs(om), 260, '#b794f4', om.toFixed(0) + ' °/с');
    gy += 6;
    c.fillStyle = '#6f7c8d'; c.textAlign = 'left';
    c.fillText('до еды: ' + flyNow.food_dist.toFixed(1) + ' мм', gx, gy + 6);
  }
}

/* -------------------------------------------------------------- отрисовка */
function frame(){
  viewer.draw(viewer.mode === 1 ? null : regionActivity);
  drawArena();
  requestAnimationFrame(frame);
}

/* -------------------------------------------------------------- элементы */
function setHint(id, text, bad){
  const el = $(id);
  el.innerHTML = text || '&nbsp;';
  el.className = 'hint' + (bad ? ' bad' : '');
}

let timers = {};
function preview(inputId, hintId){
  clearTimeout(timers[inputId]);
  timers[inputId] = setTimeout(async () => {
    const q = $(inputId).value.trim();
    if (!q) { setHint(hintId, ''); return; }
    const r = await (await fetch('/api/select?q=' + encodeURIComponent(q))).json();
    if (r.error) { setHint(hintId, r.error, true); return; }
    if (!r.n) { setHint(hintId, 'ничего не найдено', true); return; }
    const reg = Object.entries(r.regions).slice(0, 3)
      .map(([k, v]) => `${k} ${v}`).join(', ');
    setHint(hintId, `<b>${fmt(r.n)}</b> нейронов · ${reg}`);
  }, 250);
}

function wireControls(){
  $('q-act').addEventListener('input', () => preview('q-act', 'hint-act'));
  $('q-sil').addEventListener('input', () => preview('q-sil', 'hint-sil'));
  preview('q-act', 'hint-act');

  $('tab-stim').onclick = () => switchTab('stim');
  $('tab-arena').onclick = () => switchTab('arena');

  $('go').onclick = async () => {
    const body = currentTab === 'stim' ? {
      mode: 'stim',
      activate: [{query: $('q-act').value.trim(), rate: +$('rate').value,
                  on: +$('on').value || 0, off: $('off').value ? +$('off').value : null}],
      silence: $('q-sil').value.trim(),
      t: +$('t-stim').value, bin: +$('bin').value,
    } : {
      mode: 'arena', readout: $('readout').value,
      dn_tau: +$('dn-tau').value, control_ms: +$('control-ms').value,
      t: +$('t-arena').value, seed: +$('seed').value,
      swap: $('swap').checked, comment: $('comment').checked,
    };
    const r = await (await fetch('/api/run', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)})).json();
    if (r.error) setHint('hint-run', r.error, true);
  };
  $('stop').onclick = () => fetch('/api/stop', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: '{}'});
  $('save').onclick = async () => {
    const r = await (await fetch('/api/save', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: '{}'})).json();
    setHint('hint-run', r.error ? r.error : 'сохранено: ' + r.path, !!r.error);
    $('save').disabled = true;
  };

  $('mode-act').onclick = () => setMode(0);
  $('mode-reg').onclick = () => setMode(1);
  $('only').onclick = e => {
    const on = !e.target.classList.contains('on');
    e.target.classList.toggle('on', on);
    viewer.setOnlyActive(on);
  };
  $('lab').onclick = e => {
    const on = !e.target.classList.contains('on');
    e.target.classList.toggle('on', on);
    viewer.setLabels(on);
  };
  $('view').onclick = e => { e.target.textContent = 'вид: ' + viewer.nextView(); };

  const tip = $('tip');
  viewer.onHover = e => {
    const r = $('gl').getBoundingClientRect();
    const k = viewer.pick(e.clientX - r.left, e.clientY - r.top);
    if (k < 0 || act[k] < 0.05) { tip.style.display = 'none'; return; }
    const reg = regionNames[RCODE[k]];
    tip.innerHTML = `<b>${META.regionLabels[reg] || reg}</b><br>${
      act[k].toFixed(0)} Гц`;
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 250) + 'px';
    tip.style.top = (e.clientY - r.top + 14) + 'px';
  };
  $('gl').addEventListener('pointerleave', () => tip.style.display = 'none');
}

let currentTab = 'stim';
function switchTab(t){
  currentTab = t;
  $('tab-stim').classList.toggle('on', t === 'stim');
  $('tab-arena').classList.toggle('on', t === 'arena');
  $('panel-stim').style.display = t === 'stim' ? '' : 'none';
  $('panel-arena').style.display = t === 'arena' ? '' : 'none';
}
function setMode(m){
  viewer.setMode(m); viewer.mode = m;
  $('mode-act').classList.toggle('on', m === 0);
  $('mode-reg').classList.toggle('on', m === 1);
}

boot().catch(e => {
  document.body.innerHTML =
    '<p style="padding:24px;color:#fc8181">Не удалось запуститься: ' + e + '</p>';
});
})();
