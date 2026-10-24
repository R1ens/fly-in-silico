"""Живое окно управления: браузер как пульт, а не как просмотр постфактум.

Весь смысл в том, что коннектом и движок живут в памяти процесса между
прогонами. Загрузка матрицы связей занимает несколько секунд, и в командной
строке её платишь каждый раз; здесь — один раз за сессию, а дальше кнопка
«Пуск» отвечает сразу.

Кадры активности уходят в браузер по мере счёта через Server-Sent Events:
на каждый шаг записи сервер отдаёт только сработавшие нейроны (индекс плюс
счётчик спайков), браузер разворачивает их в полный массив и отдаёт в WebGL.
Для секунды симуляции это сотня кадров по несколько десятков килобайт —
на localhost незаметно.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from . import data as D, regions as R, report, sim, world

ASSETS = Path(__file__).resolve().parent / "assets"


class Hub:
    """Состояние сессии: коннектом, веса и текущий прогон."""

    def __init__(self, device=None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("загружаю коннектом...")
        self.neurons = D.load_neurons()
        self.weights = D.load_weights(device=self.device)
        print(f"готово: {len(self.neurons)} нейронов на {self.device}")

        # Разбор селекторов — регулярка по 138 тысячам строк, а список
        # популяций для рулёжки это полторы сотни таких разборов. Считаем
        # один раз при старте, иначе каждая загрузка страницы ждёт минуту.
        self._geom = None
        self._meta = None
        print("готовлю геометрию и список популяций...")
        self._readouts = sorted(world.probe_candidates(self.neurons))

        self.subs: list[queue.Queue] = []
        self.sub_lock = threading.Lock()
        self.thread = None
        self.stop_flag = threading.Event()
        self.state = "idle"
        self.last_result = None
        self.last_episode = None

    # ---------------------------------------------------------- рассылка

    def subscribe(self):
        q = queue.Queue(maxsize=256)
        with self.sub_lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.sub_lock:
            if q in self.subs:
                self.subs.remove(q)

    def publish(self, event, payload):
        msg = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        data = msg.encode("utf-8")
        with self.sub_lock:
            targets = list(self.subs)
        for q in targets:
            try:
                q.put_nowait(data)
            except queue.Full:
                # Браузер не успевает — пропускаем кадр, а не копим очередь.
                pass

    # ------------------------------------------------------------ данные

    def geometry(self):
        if self._geom is not None:
            return self._geom
        self._geom = self._build_geometry()
        return self._geom

    def _build_geometry(self):
        n = self.neurons
        coords = n[["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        good = np.isfinite(coords).all(axis=1)
        centre = np.nanmean(coords[good], axis=0)
        coords = np.where(good[:, None], coords, centre[None, :]).astype(np.float32)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        quant = np.clip(((coords - lo) / span) * 65535.0, 0, 65535).astype(np.uint16)
        names = sorted(n["region"].unique().tolist())
        codes = n["region"].map({r: i for i, r in enumerate(names)}).to_numpy(np.uint8)
        return quant, codes, lo, hi, names

    def meta(self):
        if self._meta is not None:
            return self._meta
        quant, codes, lo, hi, names = self.geometry()
        n = self.neurons
        anchors = {}
        coords = n[["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        for name in names:
            sel = (n["region"].to_numpy() == name)
            if sel.sum() >= 20:
                anchors[name] = np.nanmedian(coords[sel], axis=0).round(2).tolist()
        self._meta = {
            "n": int(len(n)),
            "bbox": {"lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()},
            "regions": names,
            "regionColors": report.region_colors(names),
            "regionLabels": {k: R.region_label(k) for k in names},
            "regionCounts": {k: int(v) for k, v in n["region"].value_counts().items()},
            "regionAnchors": anchors,
            "readouts": self._readouts,
            "device": self.device,
        }
        return self._meta

    def geometry_blob(self):
        quant, codes, *_ = self.geometry()
        return quant.tobytes() + codes.tobytes()

    def select(self, query):
        try:
            idx = D.select(self.neurons, query)
        except ValueError as exc:
            return {"error": str(exc)}
        if len(idx) == 0:
            return {"n": 0}
        sub = self.neurons.loc[idx]
        return {
            "n": int(len(idx)),
            "regions": sub["region"].value_counts().head(6).to_dict(),
            "types": sub["label"].value_counts().head(8).to_dict(),
        }

    # ------------------------------------------------------------ прогоны

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self.stop_flag.set()

    def start(self, params):
        if self.busy():
            return {"error": "прогон уже идёт"}
        self.stop_flag.clear()
        mode = params.get("mode", "stim")
        target = self._run_stim if mode == "stim" else self._run_arena
        self.thread = threading.Thread(target=self._guard, args=(target, params),
                                       daemon=True)
        self.thread.start()
        return {"ok": True}

    def _guard(self, fn, params):
        self.state = "running"
        self.publish("status", {"state": "running", "mode": params.get("mode")})
        try:
            fn(params)
        except Exception as exc:  # noqa: BLE001
            self.publish("error", {"message": str(exc)[:400]})
        finally:
            self.state = "idle"
            self.publish("status", {"state": "idle"})

    def _frame(self, t_ms, counts, extra=None):
        """Кадр: только сработавшие нейроны, индексы и счётчики спайков."""
        idx = np.nonzero(counts)[0].astype(np.int32)
        if len(idx) > 60000:
            idx = idx[np.argsort(-counts[idx])][:60000]
        vals = np.clip(counts[idx], 0, 255).astype(np.uint8)
        payload = {
            "t_ms": round(float(t_ms), 1),
            "k": int(len(idx)),
            "idx": base64.b64encode(idx.tobytes()).decode("ascii"),
            "val": base64.b64encode(vals.tobytes()).decode("ascii"),
        }
        if extra:
            payload.update(extra)
        self.publish("frame", payload)

    def _run_stim(self, p):
        neurons = self.neurons
        stimuli = []
        for spec in p.get("activate", []):
            q = (spec.get("query") or "").strip()
            if not q:
                continue
            idx = D.select(neurons, q)
            if len(idx) == 0:
                raise ValueError(f"запрос {q!r} ничего не нашёл")
            stimuli.append(sim.Stimulus(
                idx, float(spec.get("rate", 200.0)),
                t_on_ms=float(spec.get("on", 0.0)),
                t_off_ms=(float(spec["off"]) if spec.get("off") else None),
                name=q))
        if not stimuli:
            raise ValueError("не задан ни один стимул")
        silenced = (D.select(neurons, p["silence"])
                    if p.get("silence") else np.array([], np.int64))

        stim_set = sorted({int(i) for s in stimuli for i in s.indices})
        self.publish("run", {
            "mode": "stim",
            "stim": stim_set[:60000],
            "silenced": int(len(silenced)),
            "t_run_s": float(p.get("t", 1.0)),
            "bin_ms": float(p.get("bin", 10.0)),
            "title": ", ".join(f"{s.name} ({len(s.indices)} @ {s.rate_hz:g} Гц)"
                               for s in stimuli),
        })

        cfg = sim.RunConfig(t_run_s=float(p.get("t", 1.0)),
                            bin_ms=float(p.get("bin", 10.0)),
                            device=self.device)
        res = sim.run(stimuli, silenced=silenced, config=cfg, neurons=neurons,
                      weights=self.weights, progress=False,
                      on_bin=lambda b, t, c: self._frame(t, c),
                      should_stop=self.stop_flag.is_set)
        self.last_result = res
        self.publish("done", self._summary(res))

    def _run_arena(self, p):
        from . import loop

        neurons = self.neurons
        kw = {"swap_antennae": bool(p.get("swap")),
              "dn_tau_s": float(p.get("dn_tau", 0.12))}
        readout = p.get("readout")
        if readout:
            cands = world.probe_candidates(neurons)
            if readout not in cands:
                raise ValueError(f"нет популяции {readout!r}")
            kw["motor_left"], kw["motor_right"] = cands[readout]
            kw["dn_ref_hz"] = None
            kw["turn_gain_rad_s"] = None
        body = world.BodyParams(**kw)

        arena = world.default_arena(int(p.get("seed", 0)))
        self.publish("run", {
            "mode": "arena",
            "arena": arena.to_dict(),
            "stim": [],
            "t_run_s": float(p.get("t", 5.0)),
            "bin_ms": float(p.get("control_ms", 20.0)),
            "readout": readout or "все нисходящие",
            "title": f"Арена {arena.width:.0f}×{arena.height:.0f} мм"
                     + (" — контроль" if kw["swap_antennae"] else ""),
        })

        cfg = loop.EpisodeConfig(
            duration_s=float(p.get("t", 5.0)),
            control_ms=float(p.get("control_ms", 20.0)),
            seed=int(p.get("seed", 0)), device=self.device,
            comment=bool(p.get("comment")),
            comment_every_ms=float(p.get("comment_every", 1500.0)))

        seen = {"n": 0}

        def on_step(step, rec, counts, commentator):
            extra = {"fly": {k: rec[k] for k in
                             ("x", "y", "theta", "v", "omega", "odour_l",
                              "odour_r", "taste", "dn_l", "dn_r", "feeding",
                              "food_dist", "grad_cos")}}
            self._frame(rec["t_ms"], counts, extra)
            out = commentator.out if commentator else []
            if len(out) > seen["n"]:
                for c in out[seen["n"]:]:
                    self.publish("comment", c)
                seen["n"] = len(out)

        ep = loop.run_episode(arena=arena, config=cfg, body_params=body,
                              neurons=neurons, weights=self.weights,
                              progress=False, on_step=on_step,
                              should_stop=self.stop_flag.is_set)
        self.last_episode = ep
        for c in ep.comments[seen["n"]:]:
            self.publish("comment", c)
        self.publish("done", {**self._summary(ep.as_result()),
                              "metrics": ep.metrics})

    def _summary(self, res):
        rt = res.rate_table()
        g = (rt.groupby("region")
               .agg(n=("rate_hz", "size"), mean_hz=("rate_hz", "mean"))
               .sort_values("mean_hz", ascending=False).head(12))
        return {
            "n_active": int(len(rt)),
            "total_spikes": int(res.counts.sum()),
            "sim_s": res.timing.get("sim_s"),
            "regions": [{"region": r, "n": int(v.n), "mean_hz": round(float(v.mean_hz), 1)}
                        for r, v in g.iterrows()],
            "top": [{"label": str(r.label), "region": str(r.region),
                     "hz": round(float(r.rate_hz), 1)}
                    for r in rt.head(25).itertuples()],
        }

    def save_last(self, name):
        """Сохранить последний прогон так же, как это делает командная строка."""
        if self.last_episode is not None and self.state == "idle":
            from . import loop

            out = D.OUT / name
            loop.save(self.last_episode, out)
            html = report.render(self.last_episode.as_result(), out / "report.html",
                                 arena=report.build_arena_payload(self.last_episode))
            self.last_episode = None
            self.last_result = None
            return {"ok": True, "path": str(html)}
        if self.last_result is not None:
            out = D.OUT / name
            sim.save(self.last_result, out)
            html = report.render(self.last_result, out / "report.html")
            self.last_result = None
            return {"ok": True, "path": str(html)}
        return {"error": "нечего сохранять"}


# --------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8",
              extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _asset(self, name, ctype):
        path = ASSETS / name
        if not path.exists():
            self._send(404, {"error": "нет файла"})
            return
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._asset("live.html", "text/html; charset=utf-8")
        elif u.path == "/viewer.js":
            self._asset("viewer.js", "text/javascript; charset=utf-8")
        elif u.path == "/live.js":
            self._asset("live.js", "text/javascript; charset=utf-8")
        elif u.path == "/api/meta":
            self._send(200, self.hub.meta())
        elif u.path == "/api/geometry":
            self._send(200, self.hub.geometry_blob(), "application/octet-stream")
        elif u.path == "/api/select":
            self._send(200, self.hub.select(q.get("q", [""])[0]))
        elif u.path == "/api/events":
            self._events()
        else:
            self._send(404, {"error": "нет такого пути"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "плохой JSON"})
            return
        if u.path == "/api/run":
            self._send(200, self.hub.start(body))
        elif u.path == "/api/stop":
            self.hub.stop()
            self._send(200, {"ok": True})
        elif u.path == "/api/save":
            name = body.get("name") or time.strftime("%Y%m%d-%H%M%S") + "-ui"
            self._send(200, self.hub.save_last(name))
        else:
            self._send(404, {"error": "нет такого пути"})

    def _events(self):
        q = self.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b": ok\n\n")
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=15)
                except queue.Empty:
                    data = b": ping\n\n"
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.hub.unsubscribe(q)


def serve(port=8750, open_browser=True, device=None):
    Handler.hub = Hub(device=device)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\nокно управления: {url}")
    print("закрыть — Ctrl+C в этом окне\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        httpd.server_close()
