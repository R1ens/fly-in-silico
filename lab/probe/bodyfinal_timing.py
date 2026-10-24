"""Пункт 7: из чего складывается время эпизода в теле.

`flybrain.body.run_episode` зовётся напрямую с коротким эпизодом, а замеры
навешиваются снаружи: научный алгоритм не меняется, меняются только
обёртки, которые отмечают время.

Что меряется и чем:

* загрузка нейронов и матрицы связей — своими вызовами до эпизода;
* `world.probe_candidates` и `sim.build_engine` внутри эпизода — обёрнутые
  ссылки в модулях (`body.py` берёт их по имени модуля на каждом вызове);
* сборка тела MuJoCo — `timing["setup_s"]` самого эпизода плюс отметка в
  `on_body`;
* разогрев нисходящих — от `on_body` до первого шага физики;
* динамика LIF — обёртка `sim.Engine.step` (число вызовов) и отдельный
  замер блока из `brain_steps` шагов с синхронизацией после эпизода;
* шаг физики MuJoCo — обёртка метода `step` у собранного тела;
* снятие кадра картинки — свой `mujoco.Renderer` 960×720, тот же, что у
  работника (`worker._WorkerCtx.step`);
* обмен через IPC — `ipc.pack_frame` плюс `Channel.send` на настоящих
  очередях multiprocessing, как это делает `jobs.Ctx.frame`.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\bodyfinal_timing.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
T_EPISODE = 0.4            # модельное время эпизода, с


def main():
    import torch

    from flybrain import body as B, data as D, ipc, sim as SIM, world as W

    rep = {"эпизод, с модельного": T_EPISODE}
    st = {
        "lif_calls": 0, "lif_cpu_ms": 0.0,
        "phys_calls": 0, "phys_ms": 0.0,
        "phys_first_t": None, "phys_last_end": None,
        "render_ms": 0.0, "render_calls": 0,
        "ipc_ms": 0.0, "ipc_calls": 0, "ipc_bytes": 0,
        "t_on_body": None, "t_first_phys": None,
        "prev_end": None, "rows": [], "engine": None, "body": None,
        "pc_s": 0.0, "be_s": 0.0,
    }

    # ------------------------------------------------- подготовка снаружи
    t = time.perf_counter()
    neurons = D.load_neurons()
    rep["загрузка нейронов, с"] = round(time.perf_counter() - t, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter()
    weights = D.load_weights(device=device)
    rep["загрузка матрицы связей, с"] = round(time.perf_counter() - t, 2)
    rep["устройство"] = device
    rep["нейронов"] = int(len(neurons))
    rep["связей"] = int(weights._nnz()) if hasattr(weights, "_nnz") else None
    print("нейроны %.2f с, матрица %.2f с, устройство %s, связей %s"
          % (rep["загрузка нейронов, с"], rep["загрузка матрицы связей, с"],
             device, rep["связей"]))

    # ------------------------------------------------- обёртки замеров
    orig_pc = W.probe_candidates

    def pc(*a, **kw):
        t0 = time.perf_counter()
        out = orig_pc(*a, **kw)
        st["pc_s"] += time.perf_counter() - t0
        return out

    W.probe_candidates = pc

    orig_be = SIM.build_engine

    def be(*a, **kw):
        t0 = time.perf_counter()
        eng = orig_be(*a, **kw)
        st["be_s"] += time.perf_counter() - t0
        st["engine"] = eng
        return eng

    SIM.build_engine = be

    orig_lif = SIM.Engine.step

    def lif(self, rates):
        t0 = time.perf_counter()
        out = orig_lif(self, rates)
        st["lif_cpu_ms"] += (time.perf_counter() - t0) * 1000.0
        st["lif_calls"] += 1
        return out

    SIM.Engine.step = lif

    chan = ipc.Channel()
    renderer = {"ren": None, "cam": None, "data": None}

    def on_body(body):
        """Тело собрано: вешаем замер физики и поднимаем свой рендерер."""
        import mujoco

        st["t_on_body"] = time.perf_counter()
        st["body"] = body
        orig_body_step = body.step

        def phys(action):
            t0 = time.perf_counter()
            if st["phys_first_t"] is None:
                st["phys_first_t"] = t0
            if st["t_first_phys"] is None:
                st["t_first_phys"] = t0
            out = orig_body_step(action)
            t1 = time.perf_counter()
            st["phys_ms"] += (t1 - t0) * 1000.0
            st["phys_calls"] += 1
            st["phys_last_end"] = t1
            return out

        body.step = phys

        t0 = time.perf_counter()
        model = body.physics.model.ptr
        ren = mujoco.Renderer(model, 720, 960)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        from flybrain.worker import _thorax_id

        cam.trackbodyid = _thorax_id(model)
        cam.distance = 6.5
        cam.elevation = -18.0
        renderer.update(ren=ren, cam=cam, data=body.physics.data.ptr)
        rep["создание рендерера, с"] = round(time.perf_counter() - t0, 2)

    def on_step(step, rec, bins_row):
        now = time.perf_counter()
        # снятие кадра: ровно то, что делает работник на шаге управления
        t0 = time.perf_counter()
        ren, cam = renderer["ren"], renderer["cam"]
        if ren is not None:
            cam.azimuth = float(np.degrees(rec.get("theta", 0.0)))
            ren.update_scene(renderer["data"], camera=cam)
            img = ren.render()
            st["render_ms"] += (time.perf_counter() - t0) * 1000.0
            st["render_calls"] += 1
        else:
            img = None
        # обмен: упаковка кадра активности и отправка в настоящие очереди
        t1 = time.perf_counter()
        idx, val = ipc.pack_frame(bins_row)
        chan.send(ipc.Frame("timing", rec["t_ms"], idx, val, rec))
        chan.send(ipc.Series("timing", rec["t_ms"],
                             {k: rec[k] for k in ("dn_l", "dn_r", "odour_l",
                                                  "odour_r", "amp_l", "amp_r",
                                                  "dist", "grad_cos")
                              if k in rec}))
        chan.send(ipc.Status("timing", "running", "эпизод", 0.5))
        t2 = time.perf_counter()
        st["ipc_ms"] += (t2 - t1) * 1000.0
        st["ipc_calls"] += 1
        st["ipc_bytes"] += int(idx.nbytes + val.nbytes)
        row = {
            "шаг": step,
            "всего, мс": None if st["prev_end"] is None
            else round((now - st["prev_end"]) * 1000.0, 1),
            "мозг (LIF+синхр+учёт), мс": None if st["prev_end"] is None
            else round((st["phys_first_t"] - st["prev_end"]) * 1000.0, 1),
            "физика, мс": round(st["phys_ms"], 1),
            "шагов физики": st["phys_calls"],
            "хвост шага, мс": round((now - st["phys_last_end"]) * 1000.0, 1),
            "кадр картинки, мс": round(st["render_ms"], 2),
            "обмен, мс": round(st["ipc_ms"], 2),
            "ненулевых нейронов в кадре": int(len(idx)),
        }
        # счётчики накопительные — переводим в «за этот шаг»
        prev = st["rows"][-1] if st["rows"] else None
        for k, tot in (("физика, мс", "phys_ms"), ("кадр картинки, мс",
                                                   "render_ms"),
                       ("обмен, мс", "ipc_ms")):
            pass
        if prev is not None:
            row["физика, мс"] = round(row["физика, мс"] - prev["_phys_tot"], 1)
            row["кадр картинки, мс"] = round(
                row["кадр картинки, мс"] - prev["_ren_tot"], 2)
            row["обмен, мс"] = round(row["обмен, мс"] - prev["_ipc_tot"], 2)
            row["шагов физики"] -= prev["_phys_n"]
        row["_phys_tot"] = round(st["phys_ms"], 1)
        row["_ren_tot"] = round(st["render_ms"], 2)
        row["_ipc_tot"] = round(st["ipc_ms"], 2)
        row["_phys_n"] = st["phys_calls"]
        st["rows"].append(row)
        st["prev_end"] = time.perf_counter()
        st["phys_first_t"] = None

    # ------------------------------------------------- сам эпизод
    cfg = B.BodyConfig(duration_s=T_EPISODE, device=device)
    rep["настройки"] = {"control_ms": cfg.control_ms,
                        "physics_dt": cfg.physics_dt,
                        "dn_tau_s": cfg.dn_tau_s,
                        "readout": cfg.readout,
                        "шагов LIF на шаг управления":
                            int(round(cfg.control_ms / SIM.DT)),
                        "шагов физики на шаг управления":
                            int(round(cfg.control_ms / 1000.0 / cfg.physics_dt)),
                        "шагов разогрева":
                            max(5, int(round(3000.0 * cfg.dn_tau_s
                                             / cfg.control_ms)))}
    print("настройки: " + json.dumps(rep["настройки"], ensure_ascii=False))
    print("эпизод пошёл...")
    t_all = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, on_step=on_step, on_body=on_body)
    wall = time.perf_counter() - t_all
    rep["эпизод целиком, с"] = round(wall, 2)
    rep["timing из body.py"] = dict(run.timing)
    rep["timing содержит ключи"] = sorted(run.timing)

    # ------------------------------------------------- чистый замер LIF
    eng = st["engine"]
    nb = int(round(cfg.control_ms / SIM.DT))
    if eng is not None:
        rates = torch.zeros(1, len(neurons), device=device)
        for _ in range(20):                       # прогрев кэшей
            eng.step(rates)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nb):
            eng.step(rates)
        if device == "cuda":
            torch.cuda.synchronize()
        blk = (time.perf_counter() - t0) * 1000.0
        rep["чистый блок LIF (%d шагов), мс" % nb] = round(blk, 1)
        rep["чистый шаг LIF, мс"] = round(blk / nb, 4)
        # Накладные на обёртку: вызовов было много, и это надо знать.
        rep["обёртка замера на шаг LIF, мкс"] = None

    # ------------------------------------------------- разборка
    rows = [r for r in st["rows"] if r["всего, мс"] is not None]
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    warm_wall = (st["t_first_phys"] - st["t_on_body"]
                 if st["t_first_phys"] and st["t_on_body"] else None)
    lif_warm = None
    prep = {
        "загрузка нейронов": rep["загрузка нейронов, с"],
        "загрузка матрицы связей": rep["загрузка матрицы связей, с"],
        "перебор популяций (world.probe_candidates внутри эпизода)":
            round(st["pc_s"], 2),
        "сборка движка LIF (sim.build_engine)": round(st["be_s"], 2),
        "сборка тела MuJoCo (timing[setup_s])": run.timing.get("setup_s"),
        "создание рендерера MuJoCo": rep.get("создание рендерера, с"),
        "разогрев нисходящих (от on_body до первого шага физики)":
            None if warm_wall is None else round(warm_wall, 2),
    }
    rep["подготовка по частям, с"] = prep
    rep["подготовка всего (без загрузки данных), с"] = round(
        st["pc_s"] + st["be_s"] + (run.timing.get("setup_s") or 0)
        + (warm_wall or 0), 2)

    step_tot = avg("всего, мс")
    ep = {
        "шагов управления": len(st["rows"]),
        "на шаг управления всего, мс": step_tot,
        "из них мозг (LIF + синхронизация + учёт), мс": avg(
            "мозг (LIF+синхр+учёт), мс"),
        "из них физика MuJoCo, мс": avg("физика, мс"),
        "шагов физики на шаг управления": rows[0]["шагов физики"] if rows
        else None,
        "из них хвост шага (поза, метрики, запись), мс": avg("хвост шага, мс"),
        "кадр картинки (мой рендерер), мс": avg("кадр картинки, мс"),
        "обмен через IPC, мс": avg("обмен, мс"),
        "ненулевых нейронов в кадре": avg("ненулевых нейронов в кадре"),
    }
    if step_tot:
        ep["доля реального времени на шаге"] = round(
            cfg.control_ms / step_tot, 5)
        for k in ("из них мозг (LIF + синхронизация + учёт), мс",
                  "из них физика MuJoCo, мс",
                  "из них хвост шага (поза, метрики, запись), мс",
                  "кадр картинки (мой рендерер), мс",
                  "обмен через IPC, мс"):
            v = ep.get(k)
            if v is not None:
                ep["доля: " + k] = round(v / step_tot, 4)
    rep["эпизод по частям"] = ep
    rep["вызовов LIF всего"] = st["lif_calls"]
    rep["время в обёртке LIF (CPU, без синхронизации), с"] = round(
        st["lif_cpu_ms"] / 1000.0, 2)
    rep["шаги"] = [{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in st["rows"]]

    print("\nподготовка по частям, с:")
    for k, v in prep.items():
        print("   %-62s %s" % (k, v))
    print("\nэпизод по частям:")
    for k, v in ep.items():
        print("   %-62s %s" % (k, v))
    print("\ndoля реального времени из body.timing: %s"
          % run.timing.get("realtime_ratio"))

    # очереди не оставляем полными
    chan.drain()
    chan.close()
    (OUT / "bodyfinal_timing.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/bodyfinal_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
