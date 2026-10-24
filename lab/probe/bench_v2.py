"""Эталон производительности FlyBrain. Снимается ДО любых оптимизаций.

Смысл эталона не в секундах, а в том, чтобы потом можно было доказать:
ускорение ничего не сломало. Поэтому каждый сценарий сохраняет не только
время, но и сам результат — отпечатки активности, моторных команд и
траектории. Сравнение с эталоном — это сравнение чисел, а не ощущений.

Что записывается на сценарий:

* сид, устройство, шаг интегрирования, частота управления, длительность;
* полная конфигурация прогона (из чего он собран);
* активность: суммарные спайки, число активных, отпечаток растра по бинам;
* моторные команды: отпечаток рядов и их статистика;
* положение и ориентация тела, пройденный путь, отпечаток траектории;
* контакты, если вид прогона их даёт;
* время по стене, разложенное по этапам;
* пик памяти видеокарты и процесса.

Эталон НЕ меняется после каждой оптимизации — в этом его смысл. Новый
снимается только тогда, когда сознательно меняется сама модель, и тогда
старый остаётся рядом под своим именем.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\bench_v2.py
    ..\\.venv\\Scripts\\python.exe probe\\bench_v2.py --tag после-правки
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401  прогоны — во временный каталог

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
VERSION = 1


# ------------------------------------------------------------- измерения

def fingerprint(a):
    """Отпечаток массива: по нему видно изменение в последнем бите.

    Берётся от сырых байтов, а не от округлённых чисел: уровень 1 брифа
    требует совпадения либо заранее названной погрешности, и для первого
    нужен именно побайтовый отпечаток.
    """
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def stats(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    if not a.size:
        return {}
    return {"n": int(a.size), "сумма": float(a.sum()),
            "среднее": float(a.mean()), "sd": float(a.std()),
            "мин": float(a.min()), "макс": float(a.max())}


def rss_mb():
    """Память процесса, МБ. Без psutil: его в окружении нет."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc),
            pmc.cb)
        return round(pmc.PeakWorkingSetSize / 1e6, 1)
    except Exception:                                         # noqa: BLE001
        return None


def gpu_peak_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:                                         # noqa: BLE001
        pass
    return None


def gpu_reset():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:                                         # noqa: BLE001
        pass


# ------------------------------------------------------------- сценарии

def sc_lif(ctx):
    """Только коннектом: ни тела, ни физики, ни сенсоров."""
    from flybrain import data as D, sim

    cfg = sim.RunConfig(t_run_s=1.0, bin_ms=10.0, n_trials=1, seed=0,
                        device=ctx["device"])
    idx = D.select(ctx["neurons"], "class:gustatory+side:left")
    gpu_reset()
    t0 = time.perf_counter()
    res = sim.run([sim.Stimulus(idx, 200.0, name="sugar")], config=cfg,
                  neurons=ctx["neurons"], weights=ctx["weights"],
                  progress=False)
    wall = time.perf_counter() - t0
    return {
        "условия": {"сид": cfg.seed, "устройство": res.timing["device"],
                    "шаг движка, мс": sim.DT, "бин, мс": cfg.bin_ms,
                    "длительность, с": cfg.t_run_s,
                    "нейронов под стимулом": int(len(idx)),
                    "частота стимула, Гц": 200.0},
        "активность": {
            "суммарные спайки": int(res.counts.sum()),
            "активных от 0,5 Гц": int(len(res.active())),
            "отпечаток counts": fingerprint(res.counts),
            "отпечаток bins": fingerprint(res.bins),
            "бинов": int(res.bins.shape[0]),
        },
        "время": {"всего, с": round(wall, 3), **res.timing},
    }


def sc_body(ctx, duration=1.0, label="тело"):
    """Тело в физике: MuJoCo, контакты, походка. Сенсоры по умолчанию."""
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=duration, control_ms=20.0, seed=0,
                       device=ctx["device"], readout="ALPN")
    gpu_reset()
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=ctx["neurons"],
                        weights=ctx["weights"], progress=False)
    wall = time.perf_counter() - t0
    recs = run.records
    xy = np.array([[r["x"], r["y"]] for r in recs], dtype=np.float64)
    dn = np.array([[r["dn_l"], r["dn_r"]] for r in recs], dtype=np.float64)
    amp = np.array([[r["amp_l"], r["amp_r"]] for r in recs], dtype=np.float64)
    path = float(np.abs(np.diff(xy, axis=0)).sum()) if len(xy) > 1 else 0.0
    return {
        "условия": {"сид": cfg.seed, "устройство": run.timing["device"],
                    "шаг управления, мс": cfg.control_ms,
                    "шаг физики, с": cfg.physics_dt,
                    "длительность, с": cfg.duration_s,
                    "съём": cfg.readout, "сглаживание, с": cfg.dn_tau_s},
        "активность": {
            "суммарные спайки": int(run.counts.sum()),
            "отпечаток counts": fingerprint(run.counts),
            "отпечаток bins": fingerprint(run.bins),
        },
        "моторные команды": {
            "отпечаток dn": fingerprint(dn), "dn": stats(dn),
            "отпечаток амплитуд": fingerprint(amp), "амплитуды": stats(amp),
        },
        "тело": {
            "шагов": len(recs),
            "конец x, мм": float(recs[-1]["x"]) if recs else None,
            "конец y, мм": float(recs[-1]["y"]) if recs else None,
            "конец курса, рад": float(recs[-1]["theta"]) if recs else None,
            "путь, мм": round(path, 4),
            "отпечаток траектории": fingerprint(xy),
            "метрики": run.metrics,
            "калибровка": run.calib,
        },
        "время": {"всего, с": round(wall, 3), **run.timing},
    }


def sc_arena(ctx, probes=False, duration=2.0):
    """Замкнутый контур на плоской арене: запах -> мозг -> поворот."""
    from flybrain import loop, world

    body = world.BodyParams(dn_tau_s=0.5)
    cfg = loop.EpisodeConfig(duration_s=duration, control_ms=20.0, seed=0,
                             device=ctx["device"], comment=False)
    gpu_reset()
    t0 = time.perf_counter()
    ep = loop.run_episode(config=cfg, body_params=body,
                          neurons=ctx["neurons"], weights=ctx["weights"],
                          progress=False, probes=probes)
    wall = time.perf_counter() - t0
    recs = ep.records
    xy = np.array([[r["x"], r["y"]] for r in recs], dtype=np.float64)
    dn = np.array([[r["dn_l"], r["dn_r"]] for r in recs], dtype=np.float64)
    od = np.array([[r["odour_l"], r["odour_r"]] for r in recs],
                  dtype=np.float64)
    out = {
        "условия": {"сид": cfg.seed, "устройство": ep.timing["device"],
                    "шаг управления, мс": cfg.control_ms,
                    "длительность, с": cfg.duration_s,
                    "зонды популяций": bool(probes),
                    "сглаживание, с": body.dn_tau_s},
        "активность": {
            "суммарные спайки": int(ep.counts.sum()),
            "отпечаток counts": fingerprint(ep.counts),
            "отпечаток bins": fingerprint(ep.bins),
        },
        "сенсоры": {"отпечаток запаха": fingerprint(od), "запах": stats(od)},
        "моторные команды": {"отпечаток dn": fingerprint(dn), "dn": stats(dn)},
        "тело": {
            "шагов": len(recs),
            "конец x, мм": float(recs[-1]["x"]) if recs else None,
            "конец y, мм": float(recs[-1]["y"]) if recs else None,
            "путь, мм": round(float(np.abs(np.diff(xy, axis=0)).sum()), 4)
            if len(xy) > 1 else 0.0,
            "отпечаток траектории": fingerprint(xy),
            "метрики": ep.metrics,
        },
        "время": {"всего, с": round(wall, 3), **ep.timing},
    }
    if probes:
        cols = [k for k in recs[0] if k.startswith("p_")] if recs else []
        pr = np.array([[r[c] for c in cols] for r in recs], dtype=np.float64)
        out["популяции по сторонам"] = {
            "колонок": len(cols), "отпечаток": fingerprint(pr),
            "статистика": stats(pr)}
    return out


SCENARIOS = {
    "lif": ("LIF без тела", sc_lif),
    "body": ("Тело без дополнительных сенсоров",
             lambda c: sc_body(c, 1.0, "тело")),
    "odour": ("Запах: замкнутый обонятельный контур",
              lambda c: sc_arena(c, probes=False, duration=2.0)),
    "closed": ("Замкнутый контур с записью популяций",
               lambda c: sc_arena(c, probes=True, duration=2.0)),
    "contacts": ("Движение с контактами (тело, длиннее)",
                 lambda c: sc_body(c, 2.0, "контакты")),
}

# Сценарии, которых в v1 нет. Перечислены нарочно: эталон должен
# показывать и дырки, иначе потом покажется, что их и не было.
MISSING = {
    "vision": "зрительного пути в коннектом нет: сенсор омматидиев в "
              "NeuroMechFly есть, но в LIF он не заведён — см. аудит",
    "intervention": "интерактивного вмешательства в v1 нет: подавление "
                    "популяции делается только пересборкой матрицы перед "
                    "прогоном (sim._silence)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="эталон",
                    help="как назвать снимок: эталон, после-правки, …")
    ap.add_argument("--only", nargs="*", default=None,
                    help="какие сценарии гнать")
    args = ap.parse_args()

    import torch

    from flybrain import data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("устройство:", device)
    t0 = time.perf_counter()
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    load_s = time.perf_counter() - t0
    print("коннектом загружен за %.1f с" % load_s)

    ctx = {"device": device, "neurons": neurons, "weights": weights}
    rep = {
        "версия эталона": VERSION,
        "метка": args.tag,
        "когда": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "машина": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "устройство": device,
            "видеокарта": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
            "нейронов": int(len(neurons)),
            "связей": int(weights._nnz()) if weights.is_sparse
            else int((weights != 0).sum()),
        },
        "загрузка коннектома, с": round(load_s, 2),
        "сценарии": {},
        "нет в v1": MISSING,
    }

    names = args.only or list(SCENARIOS)
    for name in names:
        title, fn = SCENARIOS[name]
        print("  %-10s %s …" % (name, title), flush=True)
        t1 = time.perf_counter()
        try:
            got = fn(ctx)
        except Exception as exc:                              # noqa: BLE001
            import traceback

            rep["сценарии"][name] = {"название": title,
                                     "ошибка": str(exc)[:200],
                                     "след": traceback.format_exc()[-800:]}
            print("     упал:", str(exc)[:120])
            continue
        got["название"] = title
        got["память"] = {"пик видеокарты, МБ": gpu_peak_mb(),
                         "пик процесса, МБ": rss_mb()}
        rep["сценарии"][name] = got
        w = got["время"]["всего, с"]
        print("     %.2f с" % w, flush=True)

    path = OUT / ("bench_v2_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nснимок:", path)
    print("сценариев:", len(rep["сценарии"]),
          "| упало:", sum(1 for v in rep["сценарии"].values()
                          if "ошибка" in v))
    for name, got in rep["сценарии"].items():
        if "ошибка" in got:
            continue
        print("  %-10s %7.2f с  видеокарта %s МБ"
              % (name, got["время"]["всего, с"],
                 got["память"]["пик видеокарты, МБ"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
