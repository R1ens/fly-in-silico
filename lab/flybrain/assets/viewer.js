/* Общий 3D-вьюер мозга: облако из всех сом в координатах FlyWire, поверх —
   сработавшие нейроны, раскрашенные по частоте. Используется и в статическом
   отчёте, и в живом окне управления, поэтому ничего не знает ни про прогон,
   ни про сервер: ему дают геометрию один раз, а дальше кадры активности.

   createBrainViewer(opts) -> объект с методами setActive/setActivity/...
   opts: canvas, labelsEl, n, pos (Uint16Array 3n), bbox {lo,hi},
         regionCode (Uint8Array n), regions [имена], regionColors {имя:#hex},
         regionCounts, regionAnchors {имя:[x,y,z]} */
function createBrainViewer(opts) {
  const cv = opts.canvas;
  const gl = cv.getContext('webgl2', {antialias: true, alpha: false});
  if (!gl) throw new Error('WebGL2 недоступен');

  const N = opts.n;
  const POS = opts.pos, RCODE = opts.regionCode;
  const lo = opts.bbox.lo, hi = opts.bbox.hi;
  const span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const radius = Math.max(span[0], span[1], span[2]) * 0.62;
  const dequant = (q, ax) => lo[ax] + (q / 65535) * span[ax];

  /* мир: центрируем мозг, переворачиваем y (во FlyWire он растёт вниз) и z,
     чтобы камера по умолчанию смотрела на переднюю поверхность */
  const allPos = new Float32Array(N * 3);
  const allReg = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    allPos[i * 3] = dequant(POS[i * 3], 0) - centre[0];
    allPos[i * 3 + 1] = -(dequant(POS[i * 3 + 1], 1) - centre[1]);
    allPos[i * 3 + 2] = -(dequant(POS[i * 3 + 2], 2) - centre[2]);
    allReg[i] = RCODE[i];
  }

  const REGCOL = new Float32Array(64 * 3);
  opts.regions.forEach((r, i) => {
    const h = opts.regionColors[r] || '#666';
    REGCOL[i * 3] = parseInt(h.slice(1, 3), 16) / 255;
    REGCOL[i * 3 + 1] = parseInt(h.slice(3, 5), 16) / 255;
    REGCOL[i * 3 + 2] = parseInt(h.slice(5, 7), 16) / 255;
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
    vec3 c = (uMode > 0.5) ? rc*0.95 : vec3(0.40,0.52,0.72);
    float a = (uMode > 0.5) ? 0.34 : 0.115;
    vCol = vec4(c*depth, a*uDim*focus);
    gl_PointSize = uPointScale * (focus > 0.5 ? 1.0 : 0.8);
  } else {
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

  function sh(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw gl.getProgramInfoLog(prog);
  gl.useProgram(prog);

  const U = {};
  ['uMVP', 'uPointScale', 'uMode', 'uLayer', 'uDim', 'uFocus']
    .forEach(k => U[k] = gl.getUniformLocation(prog, k));
  gl.uniform3fv(gl.getUniformLocation(prog, 'uRegCol[0]'), REGCOL);

  function buf(data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    return b;
  }

  const vaoAll = gl.createVertexArray();
  gl.bindVertexArray(vaoAll);
  buf(allPos); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(0);
  buf(allReg); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(1);
  gl.vertexAttrib1f(2, 0); gl.vertexAttrib1f(3, 0);

  /* слой активных нейронов пересоздаётся при каждой смене набора */
  const vaoAct = gl.createVertexArray();
  let K = 0, actIdx = null, actVal = null;
  let bufActPos = null, bufActReg = null, bufActVal = null, bufActStim = null;

  function setActive(indices, stim) {
    K = indices.length;
    actIdx = indices;
    const p = new Float32Array(K * 3), rg = new Float32Array(K);
    const st = new Float32Array(K);
    const stimSet = stim || null;
    for (let k = 0; k < K; k++) {
      const i = indices[k];
      p[k * 3] = allPos[i * 3]; p[k * 3 + 1] = allPos[i * 3 + 1];
      p[k * 3 + 2] = allPos[i * 3 + 2];
      rg[k] = allReg[i];
      st[k] = stimSet ? stimSet[k] : 0;
    }
    actVal = new Float32Array(K);
    gl.bindVertexArray(vaoAct);
    [bufActPos, bufActReg, bufActVal, bufActStim].forEach(b => b && gl.deleteBuffer(b));
    bufActPos = buf(p); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(0);
    bufActReg = buf(rg); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(1);
    bufActVal = buf(actVal); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(2);
    bufActStim = buf(st); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(3);
    gl.bindVertexArray(null);
    viewer.actPos = p;
  }

  function setActivity(values) {
    if (!K || !actVal) return;
    actVal.set(values.subarray ? values.subarray(0, K) : values);
    gl.bindBuffer(gl.ARRAY_BUFFER, bufActVal);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, actVal);
  }

  /* ---- матрицы и камера ---- */
  function perspective(fovy, asp, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / asp, 0, 0, 0, 0, f, 0, 0,
      0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  }
  function lookAt(eye, ctr, up) {
    const z = [eye[0] - ctr[0], eye[1] - ctr[1], eye[2] - ctr[2]];
    let l = Math.hypot(...z); z[0] /= l; z[1] /= l; z[2] /= l;
    const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
    l = Math.hypot(...x) || 1; x[0] /= l; x[1] /= l; x[2] /= l;
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    return new Float32Array([
      x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
      -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
      -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1]);
  }

  const VIEWS = [
    {name: 'фронт', yaw: 0, pitch: 0},
    {name: 'сверху', yaw: 0, pitch: -Math.PI / 2 * 0.92},
    {name: 'сбоку', yaw: Math.PI / 2, pitch: 0},
    {name: '3/4', yaw: -0.6, pitch: -0.35},
  ];
  let viewI = 0;
  const cam = {yaw: 0, pitch: 0, dist: radius * 2.05};
  let MVP = null;

  function updateMVP() {
    const asp = cv.width / Math.max(cv.height, 1);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const eye = [Math.sin(cam.yaw) * cp * cam.dist, -sp * cam.dist,
      Math.cos(cam.yaw) * cp * cam.dist];
    MVP = mul(perspective(0.9, asp, radius * 0.1, radius * 14),
      lookAt(eye, [0, 0, 0], [0, 1, 0]));
  }

  function project(p) {
    const x = MVP[0] * p[0] + MVP[4] * p[1] + MVP[8] * p[2] + MVP[12];
    const y = MVP[1] * p[0] + MVP[5] * p[1] + MVP[9] * p[2] + MVP[13];
    const w = MVP[3] * p[0] + MVP[7] * p[1] + MVP[11] * p[2] + MVP[15];
    if (w <= 0) return null;
    return [(x / w * 0.5 + 0.5) * cv.clientWidth, (1 - (y / w * 0.5 + 0.5)) * cv.clientHeight];
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = Math.round(cv.clientWidth * dpr);
    cv.height = Math.round(cv.clientHeight * dpr);
    gl.viewport(0, 0, cv.width, cv.height);
    updateMVP();
  }

  /* ---- подписи областей ---- */
  const labelEls = {};
  if (opts.labelsEl) {
    for (const [r, p] of Object.entries(opts.regionAnchors || {})) {
      const el = document.createElement('div');
      el.className = 'rlab';
      el.textContent = r;
      el.style.color = opts.regionColors[r] || '#9aa';
      opts.labelsEl.appendChild(el);
      labelEls[r] = {el, p: [p[0] - centre[0], -(p[1] - centre[1]), -(p[2] - centre[2])]};
    }
  }
  const labelOrder = Object.keys(labelEls).sort(
    (a, b) => (opts.regionCounts[b] || 0) - (opts.regionCounts[a] || 0));

  /* Подписи ставятся от самых крупных областей, и подпись пропускается, если
     её рамка легла бы на уже поставленную: иначе центр мозга превращается в
     кучу наложенного текста. */
  function placeLabels(regionActivity) {
    let maxA = 0;
    if (regionActivity) for (const r in regionActivity) maxA = Math.max(maxA, regionActivity[r]);
    const placed = [];
    for (const r of labelOrder) {
      const {el, p} = labelEls[r];
      const s = (!showLabels || (focusRegion && focusRegion !== r)) ? null : project(p);
      if (!s) { el.style.display = 'none'; continue; }
      let size = 10.5, op = 0.85, col = opts.regionColors[r] || '#9aa';
      if (regionActivity) {
        const a = maxA > 0 ? (regionActivity[r] || 0) / maxA : 0;
        size = 10 + 3.5 * a; op = 0.22 + 0.78 * Math.sqrt(a);
        col = a > 0.04 ? '#ffd9a0' : '#5d6a7d';
      }
      const w = r.length * size * 0.62 + 6, h = size + 5;
      const box = [s[0] - w / 2, s[1] - h / 2, s[0] + w / 2, s[1] + h / 2];
      let hidden = false;
      for (const q of placed) {
        if (box[0] < q[2] && box[2] > q[0] && box[1] < q[3] && box[3] > q[1]) { hidden = true; break; }
      }
      if (hidden) { el.style.display = 'none'; continue; }
      placed.push(box);
      el.style.display = 'block';
      el.style.left = s[0] + 'px'; el.style.top = s[1] + 'px';
      el.style.opacity = op.toFixed(3);
      el.style.color = col;
      el.style.fontSize = size.toFixed(1) + 'px';
    }
  }

  let mode = 0, onlyActive = false, showLabels = true, focusRegion = null;

  function draw(regionActivity) {
    gl.clearColor(0.02, 0.027, 0.039, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    const dpr = cv.width / Math.max(cv.clientWidth, 1);
    gl.uniformMatrix4fv(U.uMVP, false, MVP);
    gl.uniform1f(U.uPointScale, 1.5 * dpr);
    gl.uniform1f(U.uMode, mode);
    gl.uniform1f(U.uFocus, focusRegion === null ? -1 : opts.regions.indexOf(focusRegion));

    if (!onlyActive) {
      gl.uniform1f(U.uLayer, 0);
      gl.uniform1f(U.uDim, 1);
      gl.bindVertexArray(vaoAll);
      gl.drawArrays(gl.POINTS, 0, N);
    }
    if (K) {
      gl.uniform1f(U.uLayer, 1);
      gl.uniform1f(U.uDim, 1);
      gl.bindVertexArray(vaoAct);
      gl.drawArrays(gl.POINTS, 0, K);
    }
    gl.bindVertexArray(null);
    placeLabels(regionActivity);
  }

  /* ---- мышь ---- */
  let drag = null;
  cv.addEventListener('pointerdown', e => {
    drag = {x: e.clientX, y: e.clientY};
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointerup', () => { drag = null; });
  cv.addEventListener('pointermove', e => {
    if (drag) {
      cam.yaw += (e.clientX - drag.x) * 0.006;
      cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + (e.clientY - drag.y) * 0.006));
      drag = {x: e.clientX, y: e.clientY};
      updateMVP();
    } else if (viewer.onHover) {
      viewer.onHover(e);
    }
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    cam.dist = Math.max(radius * 0.7, Math.min(radius * 9,
      cam.dist * (1 + Math.sign(e.deltaY) * 0.09)));
    updateMVP();
  }, {passive: false});

  function pick(mx, my) {
    let best = -1, bd = 144;
    if (!viewer.actPos) return -1;
    for (let k = 0; k < K; k++) {
      const s = project([viewer.actPos[k * 3], viewer.actPos[k * 3 + 1], viewer.actPos[k * 3 + 2]]);
      if (!s) continue;
      const d = (s[0] - mx) ** 2 + (s[1] - my) ** 2;
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }

  const viewer = {
    radius, centre, allPos,
    setActive, setActivity, draw, resize, updateMVP, project, pick,
    get activeCount() { return K; },
    get activeIndices() { return actIdx; },
    setMode(m) { mode = m; },
    setOnlyActive(v) { onlyActive = v; },
    setLabels(v) { showLabels = v; },
    setFocus(r) { focusRegion = r; },
    get focus() { return focusRegion; },
    views: VIEWS,
    get viewName() { return VIEWS[viewI].name; },
    nextView() {
      viewI = (viewI + 1) % VIEWS.length;
      cam.yaw = VIEWS[viewI].yaw; cam.pitch = VIEWS[viewI].pitch;
      updateMVP();
      return VIEWS[viewI].name;
    },
  };
  resize();
  cam.yaw = VIEWS[0].yaw; cam.pitch = VIEWS[0].pitch;
  updateMVP();
  return viewer;
}
