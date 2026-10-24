"""Self-contained HTML report: 3D brain viewer plus region timelines.

One file, no network, no build step.  The viewer is hand-written WebGL2 -- a
point cloud of all 138 639 somata in FlyWire coordinates, with the neurons that
fired drawn on top and coloured by their instantaneous rate, animated over the
simulation's time bins.  Region names float at the centroid of each neuropil so
the cloud reads as a map rather than a blob.

Arrays are quantised to 16 bits and base64-encoded; a report of a one-second
run is about 2.5 MB.
"""

from __future__ import annotations

import base64
import html as html_mod
import json
from pathlib import Path

import numpy as np

from . import regions as R

def _b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def region_colors(region_names):
    """Цвета нейропилей — те же, что в окне.

    Палитра живёт в `ui/theme.py` и строится в OkLCh с проверкой различимости
    числом. Раньше она была здесь отдельным набором, и это означало, что карта
    в окне и карта в отчёте красятся по-разному, а всякая правка делается
    дважды. Прежний набор к тому же плохо различался: ближайшая пара отличалась
    на 0,042 при разбросе по палитре до 0,513.
    """
    from .ui import theme

    names = list(region_names)
    rgb = theme.region_colors({n: R.super_region(n) for n in names})
    return {n: theme.hex_of(rgb[n]) for n in names}


def build_payload(result, min_rate=0.5, max_active=40000):
    """Everything the page needs, as JSON-serialisable data."""
    neurons = result.neurons
    coords = neurons[["x_um", "y_um", "z_um"]].to_numpy(np.float32)
    good = np.isfinite(coords).all(axis=1)
    centre = np.nanmean(coords[good], axis=0)
    coords = np.where(good[:, None], coords, centre[None, :]).astype(np.float32)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    quant = np.clip(((coords - lo) / span) * 65535.0, 0, 65535).astype(np.uint16)

    region_names = sorted(neurons["region"].unique().tolist())
    region_code = {name: i for i, name in enumerate(region_names)}
    codes = neurons["region"].map(region_code).to_numpy(np.uint8)

    rates = result.rates
    order = np.argsort(-rates)
    active = order[rates[order] >= min_rate][:max_active]
    active = np.sort(active)

    bins = result.bins[:, active].astype(np.float32)
    bin_s = result.config.bin_ms / 1000.0
    hz = bins / (bin_s * result.config.n_trials)
    # A driven neuron has no refractory period, so its per-bin rate can reach
    # 1/dt and would flatten the colour map for everybody else.  Scale to the
    # 99th percentile of the bins that actually carry spikes instead.
    # With 10 ms bins a single spike already reads as 100 Hz, so the raw per-bin
    # rate takes only a handful of values and the animation flickers.  Pass it
    # through an exponential kernel, the way a calcium indicator integrates
    # spikes, so the propagating wave is visible as a trail.
    tau_bins = max(1.0, 50.0 / result.config.bin_ms)
    decay = float(np.exp(-1.0 / tau_bins))
    smooth = np.empty_like(hz)
    acc = np.zeros(hz.shape[1], dtype=np.float32)
    for b in range(hz.shape[0]):
        acc = acc * decay + hz[b] * (1.0 - decay)
        smooth[b] = acc

    nz = smooth[smooth > 0]
    scale = float(np.percentile(nz, 99)) if nz.size else 1.0
    scale = float(np.clip(scale, 5.0, 300.0))
    hz_u8 = np.clip(smooth / scale * 255.0, 0, 255).astype(np.uint8)
    mean_scale = float(max(rates.max(), 1.0))

    # region x bin mean rate, for the heatmap
    tl = result.region_timeline()
    tl = tl.loc[[r for r in region_names if r in tl.index]]
    region_peak = tl.max(axis=1)

    stim_idx = sorted({int(i) for s in result.stimuli for i in s.indices})

    labels = neurons.loc[active, "label"].astype(str).tolist()
    classes = neurons.loc[active, "cell_class"].astype(str).tolist()
    sides = neurons.loc[active, "side"].astype(str).tolist()
    nts = neurons.loc[active, "top_nt"].astype(str).tolist()

    # label anchor for each region: median soma position of its neurons
    anchors = {}
    for name in region_names:
        sel = neurons["region"].to_numpy() == name
        if sel.sum() >= 20:
            anchors[name] = np.median(coords[sel], axis=0).round(2).tolist()

    return {
        "n": int(len(neurons)),
        "bbox": {"lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()},
        "pos": _b64(quant),
        "regionCode": _b64(codes),
        "regions": region_names,
        "regionLabels": {k: R.region_label(k) for k in region_names},
        "regionColors": region_colors(region_names),
        "regionSuper": {k: R.super_region(k) for k in region_names},
        "regionCounts": {
            k: int(v) for k, v in neurons["region"].value_counts().items()
        },
        "regionAnchors": anchors,
        "activeIdx": _b64(active.astype(np.int32)),
        "activeRate": _b64(
            np.clip(rates[active] / mean_scale * 255.0, 0, 255).astype(np.uint8)
        ),
        "meanScale": round(mean_scale, 3),
        "peakBinHz": round(float(hz.max()) if hz.size else 0.0, 1),
        "activeLabels": labels,
        "activeClasses": classes,
        "activeSides": sides,
        "activeNts": nts,
        "activeRegionCode": _b64(codes[active]),
        "bins": _b64(hz_u8),
        "nBins": int(hz_u8.shape[0]),
        "nActive": int(len(active)),
        "rateScale": round(scale, 3),
        "binMs": float(result.config.bin_ms),
        "stimIdx": stim_idx,
        "timeline": {
            "regions": tl.index.tolist(),
            "values": [[round(float(v), 3) for v in row] for row in tl.to_numpy()],
            "peak": {k: round(float(v), 3) for k, v in region_peak.items()},
        },
        "meta": result.meta(),
    }


def _title(result):
    parts = []
    for s in result.stimuli:
        label = s.name or "stim"
        window = ""
        if s.t_on_ms or s.t_off_ms is not None:
            end = "конец" if s.t_off_ms is None else f"{s.t_off_ms:g} мс"
            window = f", {s.t_on_ms:g}–{end}"
        parts.append(f"{label}: {len(s.indices)} нейронов @ {s.rate_hz:g} Гц{window}")
    if len(result.silenced):
        parts.append(f"заглушено: {len(result.silenced)} нейронов")
    return " · ".join(parts) or "без стимуляции"


def build_arena_payload(episode):
    """Track, sensor traces and commentary for the arena panel."""
    recs = episode.records
    track = np.array([[r["x"], r["y"], r["theta"]] for r in recs], np.float32)

    def series(key, dtype=np.float32):
        return _b64(np.array([r[key] for r in recs], dtype))

    return {
        "arena": episode.arena.to_dict(),
        "track": _b64(track),
        "n": len(recs),
        "controlMs": episode.config.control_ms,
        "series": {
            "v": series("v"),
            "omega": series("omega"),
            "odourL": series("odour_l"),
            "odourR": series("odour_r"),
            "taste": series("taste"),
            "dnL": series("dn_l"),
            "dnR": series("dn_r"),
            "gradCos": series("grad_cos"),
            "feeding": _b64(np.array([r["feeding"] for r in recs], np.uint8)),
        },
        "ornMaxHz": episode.body_params.orn_max_hz,
        "dnRefHz": episode.body_params.dn_ref_hz,
        "speedMax": episode.body_params.speed_max_mm_s,
        "comments": episode.comments,
        "metrics": episode.metrics,
        "swapped": bool(episode.body_params.swap_antennae),
    }


def render(result, out_path, narrative=None, title=None, min_rate=0.5,
           arena=None, home="../index.html"):
    """Write the report and return its path."""
    payload = build_payload(result, min_rate=min_rate)
    payload["title"] = title or _title(result)
    payload["narrative"] = narrative or ""
    payload["arena"] = arena
    payload["home"] = home
    html = TEMPLATE.replace("/*__PAYLOAD__*/ null", json.dumps(payload, ensure_ascii=False))
    # Put the run's own name in the <title>, so browser tabs and the hub page
    # can tell one report from another without parsing the payload.
    safe = (str(payload["title"]).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))
    html = html.replace("<title>Мозг мухи — активность</title>",
                        f"<title>{safe}</title>", 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    # Ничего, кроме самого отчёта, здесь не пишется. Раньше заголовок
    # дописывался в `meta.json` рядом — то есть построение отчёта меняло
    # файл с измерениями и параметрами опыта. Заголовок и так лежит в
    # <title> готового отчёта, и `title_of` достаёт его оттуда; старый
    # ключ `meta.json: report_title` остаётся читаемым ради папок,
    # отрисованных прежним кодом, но больше никогда не записывается.
    return out_path


def title_of(run_dir):
    """Заголовок, с которым отчёт был построен. Ничего не меняет.

    Сначала <title> самого отчёта, потом — старый ключ в `meta.json` у
    папок, отрисованных до того, как побочная запись была убрана.
    """
    run_dir = Path(run_dir)
    path = run_dir / "report.html"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                head = fh.read(4096)
            i = head.find("<title>")
            j = head.find("</title>", i)
            if i >= 0 and j > i:
                t = html_mod.unescape(head[i + 7:j].strip())
                if t and t != "Мозг мухи — активность":
                    return t
        except OSError:
            pass
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get(
                "report_title")
        except (OSError, json.JSONDecodeError):
            return None
    return None


TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мозг мухи — активность</title>
<style>
  :root{
    --bg:#07090d; --panel:#0e1219; --line:#1e2633; --fg:#e6edf6;
    --dim:#8b97a8; --accent:#ffb259; --hot:#ff6b4a;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font:14px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  #app{display:grid;grid-template-columns:1fr 360px;grid-template-rows:auto 1fr;
    height:100vh;gap:1px;background:var(--line)}
  header{grid-column:1/3;background:var(--panel);padding:10px 16px;
    display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  header h1{margin:0;font-size:15px;font-weight:600;letter-spacing:.2px}
  header .home{color:var(--dim);text-decoration:none;font-size:12.5px;
    border:1px solid var(--line);border-radius:5px;padding:2px 8px}
  header .home:hover{color:var(--accent);border-color:var(--accent)}
  header .sub{color:var(--dim);font-size:12.5px}
  #stage{position:relative;background:#05070a;overflow:hidden;
    display:grid;grid-template-rows:1fr}
  #stage.witharena{grid-template-rows:1fr 320px}
  /* min-height:0 обязателен: строка 1fr не ужимается меньше содержимого,
     а у канваса есть собственный размер */
  #brain{position:relative;overflow:hidden;min-height:0}
  #arenaBox{position:relative;border-top:1px solid var(--line);background:#06080c;
    min-height:0}
  canvas#arena{display:block;width:100%;height:100%}
  canvas#gl{display:block;width:100%;height:100%}
  #labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
  .rlab{position:absolute;transform:translate(-50%,-50%);font-size:10.5px;
    font-weight:600;letter-spacing:.3px;white-space:nowrap;
    text-shadow:0 0 6px #000,0 0 2px #000;opacity:.85;pointer-events:none}
  #hud{position:absolute;left:12px;top:12px;background:rgba(8,11,16,.72);
    border:1px solid var(--line);border-radius:8px;padding:10px 12px;
    font-size:12px;min-width:190px;backdrop-filter:blur(6px)}
  #hud .big{font-size:20px;font-weight:700;color:var(--accent);
    font-variant-numeric:tabular-nums}
  #hud .row{display:flex;justify-content:space-between;gap:12px;color:var(--dim)}
  #hud .row b{color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
  #controls{position:absolute;left:12px;right:12px;bottom:12px;
    background:rgba(8,11,16,.78);border:1px solid var(--line);border-radius:8px;
    padding:10px 12px;display:flex;align-items:center;gap:12px;
    backdrop-filter:blur(6px);flex-wrap:wrap}
  button{background:#172030;color:var(--fg);border:1px solid var(--line);
    border-radius:6px;padding:5px 11px;font:inherit;font-size:12.5px;cursor:pointer}
  button:hover{background:#1e2a3c}
  button.on{background:var(--accent);color:#1a1205;border-color:var(--accent)}
  input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}
  .t{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12px;min-width:92px}
  aside{background:var(--panel);overflow-y:auto;padding:14px}
  aside h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;
    color:var(--dim);margin:18px 0 8px;font-weight:600}
  aside h2:first-child{margin-top:0}
  .bar{display:grid;grid-template-columns:58px 1fr 44px 46px;gap:7px;
    align-items:center;margin:3px 0;font-size:12px}
  .bar .ms{text-align:right;color:#6f7c8d;font-variant-numeric:tabular-nums;
    font-size:11px}
  .barhead{display:grid;grid-template-columns:58px 1fr 44px 46px;gap:7px;
    font-size:10.5px;color:var(--dim);margin-bottom:2px}
  .barhead span:nth-child(3),.barhead span:nth-child(4){text-align:right}
  .bar .nm{color:var(--fg);font-weight:600;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;cursor:default}
  .bar .tr{background:#121925;border-radius:3px;height:11px;overflow:hidden}
  .bar .fl{height:100%;border-radius:3px}
  .bar .vl{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:var(--dim);font-weight:600;padding:3px 4px;
    border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
  td{padding:3px 4px;border-bottom:1px solid #141a24}
  td.n{text-align:right;font-variant-numeric:tabular-nums;color:var(--accent)}
  .chip{display:inline-block;width:8px;height:8px;border-radius:2px;
    margin-right:5px;vertical-align:middle}
  #heat{width:100%;display:block;border-radius:6px;background:#0a0e14}
  #tip{position:absolute;pointer-events:none;background:rgba(10,14,20,.95);
    border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:12px;
    display:none;z-index:5;max-width:240px}
  .note{color:var(--dim);font-size:11.5px;line-height:1.45;margin:6px 0 0}
  .narr{white-space:pre-wrap;font-size:12.5px;line-height:1.55;
    background:#0b1017;border:1px solid var(--line);border-radius:6px;padding:10px}
  .cmt{border-left:2px solid #223049;padding:4px 0 4px 9px;margin:0 0 6px;
    font-size:12px;color:#7f8c9e;cursor:pointer;transition:color .15s}
  .cmt:hover{color:var(--fg)}
  .cmt .ts{color:#5c6878;font-variant-numeric:tabular-nums;margin-right:6px}
  .cmt.now{border-left-color:var(--accent);color:var(--fg)}
  #loopmetrics table td{padding:2px 4px}
  #legend .lg{display:grid;grid-template-columns:14px 46px 1fr 48px;gap:6px;
    align-items:center;padding:2px 4px;border-radius:4px;cursor:pointer;
    font-size:11.5px;color:var(--dim)}
  #legend .lg:hover{background:#151c27}
  #legend .lg.sel{background:#1d2735;color:var(--fg)}
  #legend .lg b{color:var(--fg);font-weight:600}
  #legend .lg .ru{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #legend .lg .n{text-align:right;font-variant-numeric:tabular-nums}
  .scaleBar{height:9px;border-radius:3px;margin:4px 0 2px;
    background:linear-gradient(90deg,#10121a,#3b0f70,#8c2981,#dd4968,#fb8861,#fcfdbf)}
  .scaleLbl{display:flex;justify-content:space-between;color:var(--dim);font-size:11px}
</style>
</head>
<body>
<div id="app">
  <header>
    <a id="h-home" class="home" href="../index.html">← все прогоны</a>
    <h1 id="h-title">Мозг мухи</h1>
    <span class="sub" id="h-sub"></span>
  </header>

  <div id="stage">
   <div id="brain">
    <canvas id="gl"></canvas>
    <div id="labels"></div>
    <div id="hud">
      <div class="big"><span id="hud-t">0</span> мс</div>
      <div class="row"><span>активно сейчас</span><b id="hud-act">0</b></div>
      <div class="row"><span>максимум, Гц</span><b id="hud-peak">0</b></div>
      <div class="row"><span>топ-область</span><b id="hud-reg">—</b></div>
      <div class="scaleBar"></div>
      <div class="scaleLbl"><span>0</span><span id="hud-scale">—</span></div>
    </div>
    <div id="controls">
      <button id="play">▶ Пуск</button>
      <input type="range" id="slider" min="0" value="0" step="1">
      <span class="t" id="tlabel">0 / 0 мс</span>
      <button id="mode-act" class="on">активность</button>
      <button id="mode-reg">области</button>
      <button id="only">только активные</button>
      <button id="lab" class="on">подписи</button>
      <button id="view">вид: фронт</button>
    </div>
    <div id="tip"></div>
   </div>
   <div id="arenaBox" style="display:none"><canvas id="arena"></canvas></div>
  </div>

  <aside>
    <div id="loopbox" style="display:none">
      <h2>Замкнутый цикл</h2>
      <div id="loopmetrics"></div>
      <p class="note">Совпадение курса с градиентом: среднее косинуса между
        направлением движения и направлением «в сторону запаха». Случайное
        блуждание даёт 0.</p>
      <h2>Комментарий модели</h2>
      <div id="comments"></div>
    </div>

    <h2>Области мозга по активности</h2>
    <div id="regbars"></div>
    <p class="note" id="regnote"></p>

    <h2>Карта областей</h2>
    <div id="legend"></div>
    <p class="note">Клик по области — показать только её. Повторный клик снимает.</p>

    <h2>Область × время</h2>
    <canvas id="heat"></canvas>
    <p class="note">Средняя частота по области, Гц. Строки отсортированы по пику.</p>

    <h2>Самые активные нейроны</h2>
    <table><thead><tr><th>тип</th><th>область</th><th class="n">Гц</th></tr></thead>
      <tbody id="toptab"></tbody></table>

    <h2>Параметры прогона</h2>
    <div id="metatab"></div>

    <div id="narrbox" style="display:none">
      <h2>Разбор локальной модели</h2>
      <div class="narr" id="narr"></div>
    </div>
  </aside>
</div>

<script>
const DATA = /*__PAYLOAD__*/ null;

/* ---------- helpers ---------- */
function b64(str, Type){
  const bin = atob(str); const buf = new ArrayBuffer(bin.length);
  const v = new Uint8Array(buf);
  for(let i=0;i<bin.length;i++) v[i] = bin.charCodeAt(i);
  return new Type(buf);
}
const fmt = n => n.toLocaleString('ru-RU');

const POS   = b64(DATA.pos, Uint16Array);           // 3N, quantised
const RCODE = b64(DATA.regionCode, Uint8Array);     // N
const AIDX  = b64(DATA.activeIdx, Int32Array);      // K
const ARATE = b64(DATA.activeRate, Uint8Array);     // K, mean rate
const ARCODE= b64(DATA.activeRegionCode, Uint8Array);
const BINS  = b64(DATA.bins, Uint8Array);           // nBins * K
const N = DATA.n, K = DATA.nActive, NB = DATA.nBins;
const SCALE = DATA.rateScale, MSCALE = DATA.meanScale, BIN_MS = DATA.binMs;

const lo = DATA.bbox.lo, hi = DATA.bbox.hi;
const span = [hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]];
const dequant = (q, ax) => lo[ax] + (q/65535)*span[ax];

/* world coordinates: centre the brain, flip y (FlyWire y grows ventrally)
   and z so that the default camera looks at the anterior face. */
const centre = [ (lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2 ];
const radius = Math.max(span[0], span[1], span[2]) * 0.62;
function world(i, out){
  out[0] =  (dequant(POS[i*3  ],0) - centre[0]);
  out[1] = -(dequant(POS[i*3+1],1) - centre[1]);
  out[2] = -(dequant(POS[i*3+2],2) - centre[2]);
  return out;
}

/* ---------- geometry buffers ---------- */
const allPos = new Float32Array(N*3);
const allReg = new Float32Array(N);
{ const t=[0,0,0];
  for(let i=0;i<N;i++){ world(i,t);
    allPos[i*3]=t[0]; allPos[i*3+1]=t[1]; allPos[i*3+2]=t[2];
    allReg[i]=RCODE[i]; } }

const actPos = new Float32Array(K*3);
const actReg = new Float32Array(K);
for(let k=0;k<K;k++){ const i=AIDX[k];
  actPos[k*3]=allPos[i*3]; actPos[k*3+1]=allPos[i*3+1]; actPos[k*3+2]=allPos[i*3+2];
  actReg[k]=ARCODE[k]; }
const STIM = new Set(DATA.stimIdx);
const actStim = new Float32Array(K);
for(let k=0;k<K;k++) actStim[k] = STIM.has(AIDX[k]) ? 1 : 0;

/* ---------- WebGL ---------- */
const cv = document.getElementById('gl');
const gl = cv.getContext('webgl2', {antialias:true, alpha:false});
if(!gl) document.getElementById('stage').innerHTML =
  '<p style="padding:20px">WebGL2 недоступен в этом браузере.</p>';

const REGCOL = new Float32Array(64*3);
DATA.regions.forEach((r,i)=>{
  const h = DATA.regionColors[r] || '#666';
  REGCOL[i*3]   = parseInt(h.slice(1,3),16)/255;
  REGCOL[i*3+1] = parseInt(h.slice(3,5),16)/255;
  REGCOL[i*3+2] = parseInt(h.slice(5,7),16)/255;
});

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in float aReg;
layout(location=2) in float aAct;
layout(location=3) in float aStim;
uniform mat4 uMVP;
uniform float uPointScale, uMode, uLayer, uDim, uFocus;
uniform vec3 uRegCol[64];
out vec4 vCol;
vec3 magma(float t){
  t = clamp(t,0.0,1.0);
  const vec3 c0=vec3(0.063,0.071,0.102), c1=vec3(0.231,0.059,0.439);
  const vec3 c2=vec3(0.549,0.161,0.506), c3=vec3(0.867,0.288,0.408);
  const vec3 c4=vec3(0.984,0.533,0.380), c5=vec3(0.988,0.992,0.749);
  float s=t*5.0; int i=int(floor(s)); float f=fract(s);
  if(i==0) return mix(c0,c1,f); if(i==1) return mix(c1,c2,f);
  if(i==2) return mix(c2,c3,f); if(i==3) return mix(c3,c4,f);
  return mix(c4,c5,f);
}
void main(){
  gl_Position = uMVP * vec4(aPos,1.0);
  float depth = clamp(1.0 - gl_Position.z/gl_Position.w*0.5, 0.35, 1.0);
  int ri = int(aReg+0.5);
  vec3 rc = uRegCol[ri];
  float focus = (uFocus < -0.5 || abs(aReg - uFocus) < 0.5) ? 1.0 : 0.07;
  if(uLayer < 0.5){
    // background cloud: the brain's shape
    vec3 c = (uMode > 0.5) ? rc*0.95 : vec3(0.40,0.52,0.72);
    float a = (uMode > 0.5) ? 0.34 : 0.115;
    vCol = vec4(c*depth, a*uDim*focus);
    gl_PointSize = uPointScale * (focus > 0.5 ? 1.0 : 0.8);
  } else {
    // active neurons on top
    // A neuron that is quiet in this bin all but disappears, so the frame
    // shows what is firing now rather than everything that ever fires.
    vec3 c = (uMode > 0.5) ? rc : magma(0.18 + 0.82*aAct);
    if(aStim > 0.5) c = mix(c, vec3(0.45,0.95,1.0), 0.55);
    float a = (uMode > 0.5) ? min(1.0, 0.30 + 0.9*aAct)
                            : min(1.0, 0.06 + 1.25*aAct);
    vCol = vec4(c, a * max(focus, 0.12));
    gl_PointSize = uPointScale * (1.3 + 6.0*aAct) * (focus > 0.5 ? 1.0 : 0.55);
  }
}`;

const FS = `#version 300 es
precision mediump float;
in vec4 vCol; out vec4 o;
void main(){
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = dot(d,d);
  if(r > 0.25) discard;
  float soft = exp(-r*7.0);
  o = vec4(vCol.rgb * soft, vCol.a * soft);
}`;

function sh(type, src){
  const s = gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER,VS));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);
if(!gl.getProgramParameter(prog,gl.LINK_STATUS)) throw gl.getProgramInfoLog(prog);
gl.useProgram(prog);
const U = {};
['uMVP','uPointScale','uMode','uLayer','uDim','uFocus']
  .forEach(k=>U[k]=gl.getUniformLocation(prog,k));
U.uRegCol = gl.getUniformLocation(prog,'uRegCol[0]');
gl.uniform3fv(U.uRegCol, REGCOL);

function buf(data, loc, size){
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
  return b;
}
const vaoAll = gl.createVertexArray();
gl.bindVertexArray(vaoAll);
buf(allPos); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(0);
buf(allReg); gl.vertexAttribPointer(1,1,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(1);
gl.vertexAttrib1f(2,0); gl.vertexAttrib1f(3,0);

const vaoAct = gl.createVertexArray();
gl.bindVertexArray(vaoAct);
buf(actPos); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(0);
buf(actReg); gl.vertexAttribPointer(1,1,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(1);
const actBuf = gl.createBuffer();
const actVal = new Float32Array(K);
gl.bindBuffer(gl.ARRAY_BUFFER,actBuf);
gl.bufferData(gl.ARRAY_BUFFER,actVal,gl.DYNAMIC_DRAW);
gl.vertexAttribPointer(2,1,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(2);
buf(actStim); gl.vertexAttribPointer(3,1,gl.FLOAT,false,0,0); gl.enableVertexAttribArray(3);
gl.bindVertexArray(null);

/* ---------- matrices ---------- */
function perspective(fovy, asp, near, far){
  const f = 1/Math.tan(fovy/2), nf = 1/(near-far);
  return new Float32Array([f/asp,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function mul(a,b){
  const o = new Float32Array(16);
  for(let i=0;i<4;i++) for(let j=0;j<4;j++){
    let s=0; for(let k=0;k<4;k++) s += a[k*4+j]*b[i*4+k];
    o[i*4+j]=s; }
  return o;
}
function lookAt(eye,ctr,up){
  const z=[eye[0]-ctr[0],eye[1]-ctr[1],eye[2]-ctr[2]];
  let l=Math.hypot(...z); z[0]/=l;z[1]/=l;z[2]/=l;
  const x=[up[1]*z[2]-up[2]*z[1], up[2]*z[0]-up[0]*z[2], up[0]*z[1]-up[1]*z[0]];
  l=Math.hypot(...x)||1; x[0]/=l;x[1]/=l;x[2]/=l;
  const y=[z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]];
  return new Float32Array([
    x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
    -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
    -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]), 1]);
}

const cam = {yaw:0, pitch:0, dist:radius*2.05};
const VIEWS = [
  {name:'фронт', yaw:0,            pitch:0},
  {name:'сверху',yaw:0,            pitch:-Math.PI/2*0.92},
  {name:'сбоку', yaw:Math.PI/2,    pitch:0},
  {name:'3/4',   yaw:-0.6,         pitch:-0.35},
];
let viewI = 0;
let MVP = null;

function updateMVP(){
  const asp = cv.width/cv.height;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const eye = [ Math.sin(cam.yaw)*cp*cam.dist, -sp*cam.dist,
                Math.cos(cam.yaw)*cp*cam.dist ];
  MVP = mul(perspective(0.9, asp, radius*0.1, radius*14),
            lookAt(eye,[0,0,0],[0,1,0]));
}

function project(p){
  const x = MVP[0]*p[0]+MVP[4]*p[1]+MVP[8]*p[2]+MVP[12];
  const y = MVP[1]*p[0]+MVP[5]*p[1]+MVP[9]*p[2]+MVP[13];
  const w = MVP[3]*p[0]+MVP[7]*p[1]+MVP[11]*p[2]+MVP[15];
  if(w <= 0) return null;
  return [(x/w*0.5+0.5)*cv.clientWidth, (1-(y/w*0.5+0.5))*cv.clientHeight];
}

/* ---------- state ---------- */
let bin = 0, playing = false, mode = 0, onlyActive = false, showLabels = true;
let focusRegion = null;
let lastFrame = performance.now(), binAcc = 0;

function setActivity(b){
  const off = b*K;
  for(let k=0;k<K;k++) actVal[k] = BINS[off+k]/255;
  gl.bindBuffer(gl.ARRAY_BUFFER, actBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, actVal);
}

function resize(){
  const dpr = Math.min(devicePixelRatio||1, 2);
  cv.width = Math.round(cv.clientWidth*dpr);
  cv.height = Math.round(cv.clientHeight*dpr);
  gl.viewport(0,0,cv.width,cv.height);
  updateMVP();
}

const labelBox = document.getElementById('labels');
const labelEls = {};
for(const [r,p] of Object.entries(DATA.regionAnchors)){
  const el = document.createElement('div');
  el.className = 'rlab'; el.textContent = r;
  el.style.color = DATA.regionColors[r] || '#9aa';
  labelBox.appendChild(el);
  labelEls[r] = {el, p:[ p[0]-centre[0], -(p[1]-centre[1]), -(p[2]-centre[2]) ]};
}

function regionActivityNow(){
  const acc = {}, cnt = DATA.regionCounts;
  const off = bin*K;
  for(let k=0;k<K;k++){
    const v = BINS[off+k]; if(!v) continue;
    const r = DATA.regions[ARCODE[k]];
    acc[r] = (acc[r]||0) + v/255*SCALE;
  }
  const out = {};
  for(const r in acc) out[r] = acc[r]/(cnt[r]||1);
  return out;
}

/* Labels are placed biggest-region-first and a label is dropped when its box
   would land on one already placed, so the centre of the brain stays legible
   instead of turning into a pile of overlapping text. */
const labelOrder = Object.keys(labelEls).sort(
  (a,b)=> (DATA.regionCounts[b]||0)-(DATA.regionCounts[a]||0));

function placeLabels(){
  const act = mode===0 ? regionActivityNow() : null;
  let maxA = 0; if(act) for(const r in act) maxA = Math.max(maxA, act[r]);
  const placed = [];
  for(const r of labelOrder){
    const {el,p} = labelEls[r];
    const s = (!showLabels || (focusRegion && focusRegion !== r)) ? null : project(p);
    if(!s){ el.style.display='none'; continue; }
    let size = 10.5, op = 0.85, col = DATA.regionColors[r] || '#9aa';
    if(act){
      const a = maxA>0 ? (act[r]||0)/maxA : 0;
      size = 10 + 3.5*a; op = 0.22 + 0.78*Math.sqrt(a);
      col = a > 0.04 ? '#ffd9a0' : '#5d6a7d';
    }
    const w = r.length*size*0.62 + 6, h = size + 5;
    const box = [s[0]-w/2, s[1]-h/2, s[0]+w/2, s[1]+h/2];
    let hidden = false;
    for(const q of placed){
      if(box[0] < q[2] && box[2] > q[0] && box[1] < q[3] && box[3] > q[1]){
        hidden = true; break;
      }
    }
    if(hidden){ el.style.display='none'; continue; }
    placed.push(box);
    el.style.display='block';
    el.style.left = s[0]+'px'; el.style.top = s[1]+'px';
    el.style.opacity = op.toFixed(3);
    el.style.color = col;
    el.style.fontSize = size.toFixed(1)+'px';
  }
}

function draw(){
  gl.clearColor(0.02,0.027,0.039,1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  const dpr = cv.width/Math.max(cv.clientWidth,1);
  gl.uniformMatrix4fv(U.uMVP,false,MVP);
  gl.uniform1f(U.uPointScale, 1.5*dpr);
  gl.uniform1f(U.uMode, mode);
  gl.uniform1f(U.uFocus, focusRegion===null ? -1 : DATA.regions.indexOf(focusRegion));

  if(!onlyActive){
    gl.uniform1f(U.uLayer, 0);
    gl.uniform1f(U.uDim, mode===0 ? 1.0 : 1.0);
    gl.bindVertexArray(vaoAll);
    gl.drawArrays(gl.POINTS, 0, N);
  }
  gl.uniform1f(U.uLayer, 1);
  gl.uniform1f(U.uDim, 1);
  gl.bindVertexArray(vaoAct);
  gl.drawArrays(gl.POINTS, 0, K);
  gl.bindVertexArray(null);
}

function updateHud(){
  const off = bin*K; let n=0, peak=0;
  for(let k=0;k<K;k++){ const v=BINS[off+k]; if(v){ n++; if(v>peak) peak=v; } }
  document.getElementById('hud-t').textContent = Math.round(bin*BIN_MS);
  document.getElementById('hud-act').textContent = fmt(n);
  document.getElementById('hud-peak').textContent = Math.round(peak/255*SCALE);
  const act = regionActivityNow();
  let best='—', bv=-1;
  for(const r in act) if(act[r]>bv){ bv=act[r]; best=r; }
  document.getElementById('hud-reg').textContent = best;
  document.getElementById('tlabel').textContent =
    Math.round(bin*BIN_MS)+' / '+Math.round(NB*BIN_MS)+' мс';
  document.getElementById('slider').value = bin;
  drawHeat();
}

function frame(now){
  const dt = now - lastFrame; lastFrame = now;
  if(playing){
    binAcc += dt/ (1000/ (NB/4) );     // whole run in ~4 seconds
    while(binAcc >= 1){ binAcc -= 1; bin = (bin+1)%NB; setActivity(bin); updateHud(); }
  }
  placeLabels();
  draw();
  drawArena();
  syncComments();
  requestAnimationFrame(frame);
}

/* ---------- interaction ---------- */
let drag = null;
cv.addEventListener('pointerdown', e => { drag = {x:e.clientX, y:e.clientY}; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointerup', e => { drag = null; });
cv.addEventListener('pointermove', e => {
  if(drag){
    cam.yaw += (e.clientX-drag.x)*0.006;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + (e.clientY-drag.y)*0.006));
    drag = {x:e.clientX, y:e.clientY};
    updateMVP();
  } else hover(e);
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.max(radius*0.7, Math.min(radius*9, cam.dist*(1+Math.sign(e.deltaY)*0.09)));
  updateMVP();
}, {passive:false});

const tip = document.getElementById('tip');
function hover(e){
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let best=-1, bd=144;
  for(let k=0;k<K;k++){
    const s = project([actPos[k*3],actPos[k*3+1],actPos[k*3+2]]);
    if(!s) continue;
    const d = (s[0]-mx)**2 + (s[1]-my)**2;
    if(d < bd){ bd = d; best = k; }
  }
  if(best < 0){ tip.style.display='none'; return; }
  const hz = BINS[bin*K+best]/255*SCALE, avg = ARATE[best]/255*MSCALE;
  const reg = DATA.regions[ARCODE[best]];
  tip.innerHTML = '<b>'+DATA.activeLabels[best]+'</b><br>'
    + (DATA.activeClasses[best]||'—') + ' · ' + (DATA.activeSides[best]||'')
    + '<br>' + (DATA.regionLabels[reg]||reg)
    + '<br>' + (DATA.activeNts[best]||'')
    + '<br>сейчас <b>'+hz.toFixed(0)+' Гц</b>, в среднем '+avg.toFixed(1)+' Гц';
  tip.style.display='block';
  tip.style.left = Math.min(mx+14, r.width-250)+'px';
  tip.style.top = (my+14)+'px';
}
cv.addEventListener('pointerleave', ()=> tip.style.display='none');

document.getElementById('play').onclick = e => {
  playing = !playing;
  e.target.textContent = playing ? '❚❚ Пауза' : '▶ Пуск';
  e.target.classList.toggle('on', playing);
};
const slider = document.getElementById('slider');
slider.max = NB-1;
slider.oninput = e => { bin = +e.target.value; setActivity(bin); updateHud(); };
document.getElementById('mode-act').onclick = () => setMode(0);
document.getElementById('mode-reg').onclick = () => setMode(1);
function setMode(m){
  mode = m;
  document.getElementById('mode-act').classList.toggle('on', m===0);
  document.getElementById('mode-reg').classList.toggle('on', m===1);
}
document.getElementById('only').onclick = e => {
  onlyActive = !onlyActive; e.target.classList.toggle('on', onlyActive);
};
document.getElementById('lab').onclick = e => {
  showLabels = !showLabels; e.target.classList.toggle('on', showLabels);
};
document.getElementById('view').onclick = e => {
  viewI = (viewI+1) % VIEWS.length;
  cam.yaw = VIEWS[viewI].yaw; cam.pitch = VIEWS[viewI].pitch;
  e.target.textContent = 'вид: ' + VIEWS[viewI].name;
  updateMVP();
};
addEventListener('keydown', e => {
  if(e.key === ' '){ e.preventDefault(); document.getElementById('play').click(); }
  if(e.key === 'ArrowRight'){ bin=(bin+1)%NB; setActivity(bin); updateHud(); }
  if(e.key === 'ArrowLeft'){ bin=(bin-1+NB)%NB; setActivity(bin); updateHud(); }
});

/* ---------- side panels ---------- */
const TL = DATA.timeline;
/* onset = first bin where a region reaches 20% of its own peak: the order in
   which the signal arrives, which is the whole point of the cascade. */
const tlOrder = TL.regions.map((r,i)=>{
  const v = TL.values[i], peak = Math.max(...v);
  let onset = null;
  for(let b=0;b<v.length;b++){ if(v[b] >= peak*0.2 && peak>0){ onset=b*BIN_MS; break; } }
  return {r, i, peak, onset};
}).sort((a,b)=> b.peak-a.peak);

function regionBars(){
  const box = document.getElementById('regbars');
  const rows = tlOrder.filter(o=>o.peak>0).slice(0,14);
  const max = rows.length ? rows[0].peak : 1;
  box.innerHTML = '<div class="barhead"><span>область</span><span></span>'
    + '<span>Гц</span><span>приход</span></div>'
    + rows.map(o=>{
    const col = DATA.regionColors[o.r]||'#888';
    const w = Math.max(2, o.peak/max*100);
    const cnt = DATA.regionCounts[o.r]||0;
    const ttl = ((DATA.regionLabels[o.r]||o.r) + ' · ' + fmt(cnt) + ' нейронов').replace(/"/g,'');
    const ms = o.onset===null ? '—' : Math.round(o.onset)+' мс';
    return `<div class="bar" title="${ttl}">
      <span class="nm"><span class="chip" style="background:${col}"></span>${o.r}</span>
      <span class="tr"><span class="fl" style="width:${w}%;background:${col}"></span></span>
      <span class="vl">${o.peak.toFixed(1)}</span>
      <span class="ms">${ms}</span></div>`;
  }).join('');
  document.getElementById('regnote').textContent = rows.length
    ? 'Пик средней частоты по области и момент, когда область вышла на 20 % '
      + 'своего пика — порядок прихода сигнала. Наведи на название — полное имя нейропиля.'
    : 'Ни одна область не превысила порог активности.';
}

const heat = document.getElementById('heat');
function drawHeat(){
  const rows = tlOrder.filter(o=>o.peak>0).slice(0,16);
  const rowH = 14, padL = 52, padB = 16;
  const w = heat.clientWidth || 320;
  const h = rows.length*rowH + padB + 4;
  const dpr = Math.min(devicePixelRatio||1,2);
  heat.width = w*dpr; heat.height = h*dpr; heat.style.height = h+'px';
  const c = heat.getContext('2d');
  c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,w,h);
  const gw = w - padL - 6;
  const max = rows.length ? Math.max(...rows.map(o=>o.peak)) : 1;
  rows.forEach((o,ri)=>{
    const vals = TL.values[o.i];
    for(let b=0;b<vals.length;b++){
      const t = Math.min(1, vals[b]/max);
      c.fillStyle = heatColor(t);
      const x = padL + b/vals.length*gw;
      c.fillRect(x, ri*rowH+2, Math.ceil(gw/vals.length)+0.5, rowH-3);
    }
    c.fillStyle = '#8b97a8'; c.font = '10px ui-monospace,monospace';
    c.textAlign = 'right'; c.textBaseline='middle';
    c.fillText(o.r, padL-6, ri*rowH+2+(rowH-3)/2);
  });
  // playhead
  const x = padL + (bin+0.5)/NB*gw;
  c.strokeStyle = '#ffb259'; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x,2); c.lineTo(x,rows.length*rowH+1); c.stroke();
  c.fillStyle='#8b97a8'; c.font='10px ui-monospace,monospace'; c.textAlign='left';
  c.fillText('0', padL, rows.length*rowH+10);
  c.textAlign='right';
  c.fillText(Math.round(NB*BIN_MS)+' мс', padL+gw, rows.length*rowH+10);
}
function heatColor(t){
  const stops = [[16,18,26],[59,15,112],[140,41,129],[221,73,104],[251,136,97],[252,253,191]];
  const s = Math.max(0,Math.min(1,t))*(stops.length-1);
  const i = Math.floor(s), f = s-i;
  const a = stops[i], b = stops[Math.min(i+1,stops.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;
}

function buildLegend(){
  const box = document.getElementById('legend');
  const order = DATA.regions.slice().sort(
    (a,b)=>(DATA.regionCounts[b]||0)-(DATA.regionCounts[a]||0));
  box.innerHTML = order.map(r=>{
    const col = DATA.regionColors[r]||'#888';
    const ru = (DATA.regionLabels[r]||r).split('—').slice(1).join('—').trim() || '';
    return `<div class="lg" data-r="${r}">
      <span class="chip" style="background:${col};width:10px;height:10px"></span>
      <b>${r}</b><span class="ru">${ru}</span>
      <span class="n">${fmt(DATA.regionCounts[r]||0)}</span></div>`;
  }).join('');
  box.querySelectorAll('.lg').forEach(el=>{
    el.onclick = () => {
      const r = el.dataset.r;
      focusRegion = (focusRegion === r) ? null : r;
      box.querySelectorAll('.lg').forEach(e2=>
        e2.classList.toggle('sel', e2.dataset.r === focusRegion));
    };
  });
}

/* ---------- arena panel (only when the run was a closed loop) ---------- */
const AR = DATA.arena;
let TRACK = null, SER = {}, odourTile = null;
const arenaCv = document.getElementById('arena');
const arenaCtx = arenaCv.getContext('2d');

if(AR){
  document.getElementById('stage').classList.add('witharena');
  document.getElementById('arenaBox').style.display = 'block';
  document.getElementById('loopbox').style.display = 'block';
  TRACK = b64(AR.track, Float32Array);
  for(const k in AR.series){
    SER[k] = (k === 'feeding') ? b64(AR.series[k], Uint8Array)
                               : b64(AR.series[k], Float32Array);
  }
  buildOdourTile();
  loopPanels();
}

/* The odour field never changes, so rasterise it once at low resolution and
   let the browser smooth it up to whatever size the panel ends up. */
function buildOdourTile(){
  const R = 96;
  odourTile = document.createElement('canvas');
  odourTile.width = odourTile.height = R;
  const c = odourTile.getContext('2d');
  const img = c.createImageData(R, R);
  for(let j=0;j<R;j++){
    const wy = (1 - (j + 0.5)/R) * AR.arena.height;
    for(let i=0;i<R;i++){
      const wx = ((i + 0.5)/R) * AR.arena.width;
      let v = 0;
      for(const o of AR.arena.odours){
        const dx = wx-o.x, dy = wy-o.y;
        v += o.s * Math.exp(-(dx*dx+dy*dy)/(2*o.sigma*o.sigma));
      }
      const t = Math.pow(Math.min(1, v), 1.4);
      const p = (j*R+i)*4;
      img.data[p]   = 7  +  30*t;
      img.data[p+1] = 13 + 140*t;
      img.data[p+2] = 21 +  78*t;
      img.data[p+3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
}

function drawArena(){
  if(!AR) return;
  const dpr = Math.min(devicePixelRatio||1, 2);
  const W = arenaCv.clientWidth, H = arenaCv.clientHeight;
  if(arenaCv.width !== Math.round(W*dpr)){
    arenaCv.width = Math.round(W*dpr); arenaCv.height = Math.round(H*dpr);
  }
  const c = arenaCtx;
  c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,W,H);

  const pad = 12, gaugeW = Math.min(300, Math.max(190, W*0.34));
  const side = Math.max(60, Math.min(H - pad*2, W - gaugeW - pad*3));
  const ox = pad, oy = (H - side)/2;
  const sx = v => ox + (v / AR.arena.width) * side;
  const sy = v => oy + (1 - v / AR.arena.height) * side;

  c.save();
  c.beginPath(); c.rect(ox,oy,side,side); c.clip();
  c.imageSmoothingEnabled = true;
  c.drawImage(odourTile, ox, oy, side, side);
  c.restore();
  c.strokeStyle = '#243044'; c.lineWidth = 1;
  c.strokeRect(ox+0.5, oy+0.5, side, side);

  for(const f of AR.arena.foods){
    const r = Math.max(3, f.r / AR.arena.width * side);
    c.beginPath(); c.arc(sx(f.x), sy(f.y), r, 0, 7);
    c.fillStyle = 'rgba(255,205,120,.9)'; c.fill();
    c.strokeStyle = 'rgba(255,230,180,.6)'; c.lineWidth = 1.5; c.stroke();
  }

  // path so far, the recent part brighter
  const upto = Math.min(bin, AR.n-1);
  c.lineWidth = 1.6; c.lineJoin = 'round';
  c.strokeStyle = 'rgba(120,170,255,.28)';
  c.beginPath();
  for(let i=0;i<=upto;i++){
    const x = sx(TRACK[i*3]), y = sy(TRACK[i*3+1]);
    i ? c.lineTo(x,y) : c.moveTo(x,y);
  }
  c.stroke();
  const tail = Math.max(0, upto-60);
  c.strokeStyle = 'rgba(160,210,255,.85)'; c.lineWidth = 2;
  c.beginPath();
  for(let i=tail;i<=upto;i++){
    const x = sx(TRACK[i*3]), y = sy(TRACK[i*3+1]);
    i===tail ? c.moveTo(x,y) : c.lineTo(x,y);
  }
  c.stroke();

  // where it started
  c.beginPath(); c.arc(sx(TRACK[0]), sy(TRACK[1]), 3, 0, 7);
  c.strokeStyle = 'rgba(160,210,255,.8)'; c.lineWidth = 1.5; c.stroke();

  // the fly
  const fx = sx(TRACK[upto*3]), fy = sy(TRACK[upto*3+1]);
  const th = TRACK[upto*3+2];
  const feeding = SER.feeding[upto];
  c.save();
  c.translate(fx, fy); c.rotate(-th);
  c.beginPath(); c.moveTo(12,0); c.lineTo(-8,6.5); c.lineTo(-4,0); c.lineTo(-8,-6.5);
  c.closePath();
  c.fillStyle = feeding ? '#7ef0ff' : '#ffd79a';
  c.fill();
  c.strokeStyle = 'rgba(0,0,0,.6)'; c.lineWidth = 1; c.stroke();
  // antennae, brightness = drive on that side
  const nl = Math.min(1, SER.odourL[upto]/AR.ornMaxHz);
  const nr = Math.min(1, SER.odourR[upto]/AR.ornMaxHz);
  c.beginPath(); c.arc(9, -5, 3.0, 0, 7);
  c.fillStyle = `rgba(120,255,190,${0.2+0.8*nl})`; c.fill();
  c.beginPath(); c.arc(9, 5, 3.0, 0, 7);
  c.fillStyle = `rgba(120,255,190,${0.2+0.8*nr})`; c.fill();
  c.restore();

  // gauges
  const gx = ox + side + pad*1.5;
  let gy2 = oy + 4;
  const gw = Math.min(gaugeW - pad, W - gx - pad);
  c.font = '11px ui-monospace,monospace'; c.textBaseline = 'middle';
  const bar = (label, val, max, col, extra) => {
    c.fillStyle = '#8b97a8'; c.textAlign = 'left';
    c.fillText(label, gx, gy2+6);
    c.fillStyle = '#0f1620';
    c.fillRect(gx+92, gy2, gw-140, 12);
    const f = Math.max(0, Math.min(1, val/max));
    c.fillStyle = col;
    c.fillRect(gx+92, gy2, (gw-140)*f, 12);
    c.fillStyle = '#e6edf6'; c.textAlign = 'right';
    c.fillText(extra, gx+gw-4, gy2+6);
    gy2 += 19;
  };
  bar('запах L', SER.odourL[upto], AR.ornMaxHz, '#4fd1c5',
      SER.odourL[upto].toFixed(0)+' Гц');
  bar('запах R', SER.odourR[upto], AR.ornMaxHz, '#4fd1c5',
      SER.odourR[upto].toFixed(0)+' Гц');
  bar('вкус',    SER.taste[upto], 200, '#f6ad55',
      SER.taste[upto].toFixed(0)+' Гц');
  gy2 += 5;
  bar('нисх. L', SER.dnL[upto], AR.dnRefHz*1.6, '#ed64a6',
      SER.dnL[upto].toFixed(1)+' Гц');
  bar('нисх. R', SER.dnR[upto], AR.dnRefHz*1.6, '#ed64a6',
      SER.dnR[upto].toFixed(1)+' Гц');
  gy2 += 5;
  bar('скорость', SER.v[upto], AR.speedMax, '#90cdf4',
      SER.v[upto].toFixed(1)+' мм/с');
  const om = SER.omega[upto]*180/Math.PI;
  bar('поворот', Math.abs(om), 260, '#b794f4', om.toFixed(0)+' °/с');
  gy2 += 6;
  c.fillStyle = '#6f7c8d'; c.textAlign = 'left';
  c.fillText('курс к запаху: ' + SER.gradCos[upto].toFixed(2), gx, gy2+6);
  if(AR.swapped){
    gy2 += 17;
    c.fillStyle = '#fc8181';
    c.fillText('контроль: антенны переставлены', gx, gy2+6);
  }
}

function loopPanels(){
  const m = AR.metrics;
  const rows = [
    ['совпадение с градиентом', m.grad_alignment.toFixed(3)],
    ['поворот к запаху', m.turn_to_odour === null ? '—'
        : (m.turn_to_odour*100).toFixed(0)+' %'],
    ['до еды, 1-я половина', m.food_dist_first_half_mm + ' мм'],
    ['до еды, 2-я половина', m.food_dist_second_half_mm + ' мм'],
    ['добралась до еды', m.food_visits + ' раз'],
    ['средняя скорость', m.mean_speed_mm_s + ' мм/с'],
    ['нисходящие, средн.', m.mean_dn_hz + ' Гц'],
    ['ударов о стену', m.wall_bounces],
  ];
  document.getElementById('loopmetrics').innerHTML =
    '<table>' + rows.map(r=>`<tr><td style="color:var(--dim)">${r[0]}</td>
      <td class="n">${r[1]}</td></tr>`).join('') + '</table>';

  const box = document.getElementById('comments');
  if(!AR.comments.length){
    box.innerHTML = '<p class="note">Комментариев нет — модель была отключена '
      + 'или недоступна.</p>';
    return;
  }
  box.innerHTML = AR.comments.map((cm,i)=>
    `<div class="cmt" data-i="${i}" data-t="${cm.t_ms}">
      <span class="ts">${(cm.t_ms/1000).toFixed(1)} с</span>${cm.text}</div>`
  ).join('');
  box.querySelectorAll('.cmt').forEach(el=>{
    el.onclick = () => {
      bin = Math.min(NB-1, Math.round(+el.dataset.t / BIN_MS));
      setActivity(bin); updateHud();
    };
  });
}

let lastCmt = -1;
function syncComments(){
  if(!AR || !AR.comments.length) return;
  const t = bin*BIN_MS;
  let k = -1;
  for(let i=0;i<AR.comments.length;i++) if(AR.comments[i].t_ms <= t) k = i;
  if(k === lastCmt) return;
  lastCmt = k;
  document.querySelectorAll('#comments .cmt').forEach((el,i)=>
    el.classList.toggle('now', i === k));
}

function topTable(){
  const ord = Array.from({length:K}, (_,k)=>k).sort((a,b)=>ARATE[b]-ARATE[a]).slice(0,30);
  document.getElementById('toptab').innerHTML = ord.map(k=>{
    const r = DATA.regions[ARCODE[k]];
    const col = DATA.regionColors[r]||'#888';
    return `<tr><td>${DATA.activeLabels[k]}</td>
      <td><span class="chip" style="background:${col}"></span>${r}</td>
      <td class="n">${(ARATE[k]/255*MSCALE).toFixed(0)}</td></tr>`;
  }).join('');
}

function metaTable(){
  const m = DATA.meta;
  const rows = [
    ['нейронов', fmt(m.n_neurons)],
    ['активных (&ge;0.5 Гц)', fmt(m.n_active)],
    ['всего спайков', fmt(m.total_spikes)],
    ['длительность', m.config.t_run_s + ' с'],
    ['шаг бина', m.config.bin_ms + ' мс'],
    ['повторов', m.config.n_trials],
    ['заглушено', fmt(m.n_silenced)],
    ['устройство', m.timing.device],
    ['счёт', m.timing.sim_s + ' с'],
  ];
  document.getElementById('metatab').innerHTML =
    '<table>' + rows.map(r=>`<tr><td style="color:var(--dim)">${r[0]}</td>
      <td class="n">${r[1]}</td></tr>`).join('') + '</table>';
}

document.getElementById('h-title').textContent = DATA.title;
if(DATA.home){ document.getElementById('h-home').href = DATA.home; }
else { document.getElementById('h-home').style.display = 'none'; }
document.getElementById('h-sub').textContent =
  fmt(DATA.n) + ' нейронов · ' + fmt(DATA.nActive) + ' активных · '
  + DATA.regions.length + ' областей · шкала ' + DATA.rateScale.toFixed(0)
  + ' Гц (пик ' + DATA.peakBinHz.toFixed(0) + ')';
document.getElementById('hud-scale').textContent = DATA.rateScale.toFixed(0)+' Гц';
if(DATA.narrative){
  document.getElementById('narrbox').style.display='block';
  document.getElementById('narr').textContent = DATA.narrative;
}
regionBars(); buildLegend(); topTable(); metaTable();

addEventListener('resize', ()=>{ resize(); drawHeat(); });
resize();
cam.yaw = VIEWS[0].yaw; cam.pitch = VIEWS[0].pitch;
updateMVP();
setActivity(0); updateHud();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
