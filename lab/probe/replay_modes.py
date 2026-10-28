"""Три режима и подпись стоимости: ноль вызовов движка против двухсот.

Режимов ровно три, и отличаются они не словом в отчёте, а числом вызовов
``sim.Engine.step`` на шаг управления. Проба мерит это число у каждого и
к каждому замеру ставит встречный:

======================  =========================  ========================
режим                   чем мерится                встречный замер
======================  =========================  ========================
``просмотр``            счётчик не двигается       рядом стоит живой
                                                   движок, у которого
                                                   счётчик двигаться
                                                   умеет — и не двигается;
                                                   плюс просмотр
                                                   открывается в процессе,
                                                   где импорт ``torch``
                                                   запрещён
``пересчёт``            200 вызовов на шаг         записи шагов побитово
                        (канал записи)             те же, что у первого
                                                   счёта: пересчёт — тот
                                                   же прогон, а не другой
``продолжение``         200 вызовов на шаг         вызовов ровно
                                                   ``шагов × 200``, то
                                                   есть разогрев
                                                   калибровки не
                                                   повторялся втихую
======================  =========================  ========================

Двести — это ``round(control_ms / sim.DT)`` при шаге управления 20 мс и
шаге движка 0,1 мс; число берётся из кода, а не из текста.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\replay_modes.py
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

LAB = HERE.parents[1]
PY = Path(sys.executable)
OUT = HERE.parent
CONTROL_MS = 20.0
N_STEPS = 41
TAIL = 20
ASK = 2000

fails = []
rep = {"проверки": {}, "замеры": {}, "режимы": {}, "встречный замер": {},
       "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def rec_fp(rec):
    """Отпечаток записи шага без ``t_ms``: у продолжения он свой (см.
    `probe/resume_state.py`), и сравнивать по нему значило бы мерить
    нумерацию, а не модель."""
    body = json.dumps({k: v for k, v in rec.items() if k != "t_ms"},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _episode(device, n_steps, mode, engine=None, motor=None, start=None,
             counts0=None, arena=None, params=None, t_offset=0.0,
             capture_at_ms=None, probe_idx=None, seed=0):
    from flybrain import loop, replay as RP, world

    from flybrain import data as D

    neurons = _episode.neurons
    weights = _episode.weights
    params = params or world.BodyParams()
    arena = arena if arena is not None else world.default_arena(seed)
    if probe_idx is None:
        probe_idx = {"съём L": np.asarray(
            D.select(neurons, params.motor_left), np.int64)}
    cfg = loop.EpisodeConfig(duration_s=n_steps * CONTROL_MS / 1000.0,
                             control_ms=CONTROL_MS, seed=seed, device=device,
                             comment=False)
    tape = RP.Tape(kind="arena", control_ms=CONTROL_MS, keep_s=20.0,
                   probe_idx=probe_idx, mode=mode, t_offset_ms=t_offset,
                   capture_at_ms=capture_at_ms, meta={"семя": seed})
    t0 = time.perf_counter()
    ep = loop.run_episode(arena=arena, config=cfg, body_params=params,
                          neurons=neurons, weights=weights, progress=False,
                          engine=engine, motor=motor, start=start,
                          counts0=counts0, tape=tape)
    wall = time.perf_counter() - t0
    col = tape.tl.col.get("вызовов_движка")
    per_step = ([int(v) for v in tape.tl.rows[:, col]] if col is not None
                else [])
    return {"ep": ep, "tape": tape, "wall_s": wall, "per_step": per_step,
            "recs": [rec_fp(r) for r in ep.records],
            "engine": tape.engine}


def part_record(device):
    """Первый счёт: он же запись, с которой потом сравнивают остальные."""
    from flybrain import data as D, replay as RP

    _episode.neurons = D.load_neurons()
    _episode.weights = D.load_weights(device=device)

    print("  счёт с записью: %d шагов, снимок на шаге %d"
          % (N_STEPS, N_STEPS - TAIL - 1))
    a = _episode(device, N_STEPS, "пересчёт",
                 capture_at_ms=(N_STEPS - TAIL - 1) * CONTROL_MS)
    where = Path(probe._isolate.OUT) / "_modes_run"
    if where.exists():
        shutil.rmtree(where)
    from flybrain import loop

    loop.save(a["ep"], where)
    rep["замеры"]["шагов записано"] = len(a["recs"])
    rep["замеры"]["папка записи"] = str(where)
    want = RP.lif_per_step(CONTROL_MS)
    rep["замеры"]["вызовов движка на шаг управления, из кода"] = want
    check("вызовов на шаг управления считается из кода, а не задано текстом",
          want == 200, "round(%g / %g) = %d"
          % (CONTROL_MS, __import__("flybrain.sim",
                                    fromlist=["DT"]).DT, want))
    check("запись и снимок легли на диск",
          (where / "timeline.npz").exists()
          and (where / "timeline.json").exists()
          and (where / "state.npz").exists(),
          "файлов в папке %d" % len(list(where.iterdir())))
    return a, where


def part_view(a, where):
    """Просмотр: ноль вызовов движка. Со встречным замером на счётчик."""
    from flybrain import replay as RP, sim, timeline as TL

    view = RP.View.open(where)
    tl = view.tl
    times = tl.times()
    keys = [c.key for c in tl.chans]

    # Встречный замер: рядом стоит ЖИВОЙ движок. Его счётчик двигаться
    # умеет (это проверяется тут же, одним вызовом), и если просмотр
    # втихую считает, счётчик это покажет.
    eng = a["engine"]
    before = int(eng.calls)
    import torch

    eng.step(torch.zeros(1, eng.n, device=eng.device))
    moved = int(eng.calls) - before
    base = int(eng.calls)

    t0 = time.perf_counter()
    asked = holes = 0
    for i in range(ASK):
        t = float(times[i % len(times)])
        if i % 3 == 2:                 # нарочно между кадрами
            t += CONTROL_MS / 2.0
        v, hole = view.value(t, keys[i % len(keys)])
        asked += 1
        holes += int(hole is not None)
    ms = (time.perf_counter() - t0) * 1000.0
    after = int(eng.calls)

    cost = view.cost()
    rep["режимы"]["просмотр"] = {
        "вызовов движка LIF": cost["вызовов движка LIF"],
        "ответов": asked, "из них «нет данных»": holes,
        "мс на ответ": round(ms / asked, 5),
        "всего, мс": round(ms, 2),
    }
    check("просмотр: вызовов движка ноль",
          cost["вызовов движка LIF"] == 0 and after == base,
          "счётчик живого движка: %d до, %d после %d ответов"
          % (base, after, asked))
    rep["встречный замер"]["счётчик движка умеет двигаться"] = moved
    check("встречный замер: тот же счётчик от одного вызова движка растёт "
          "на единицу", moved == 1, "вырос на %d" % moved)
    check("просмотр отвечает за микросекунды, а не за миллисекунды",
          ms / asked < 1.0, "%d ответов за %.1f мс — %.4f мс на ответ"
          % (asked, ms, ms / asked))
    check("на моменты между кадрами просмотр ответил «нет данных»",
          holes >= asked // 3 - 1,
          "«нет данных» %d из %d вопросов (каждый третий спрошен между "
          "кадрами)" % (holes, asked))


def part_no_torch(where):
    """Встречный замер: просмотр открывается там, где torch запрещён."""
    code = (
        "import sys, json, importlib.abc\n"
        "class NoTorch(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch запрещён в этом замере')\n"
        "        return None\n"
        "sys.meta_path.insert(0, NoTorch())\n"
        "sys.path.insert(0, %r)\n"
        "from flybrain import replay as RP\n"
        "v = RP.View.open(%r)\n"
        "t = float(v.tl.times()[0])\n"
        "val, hole = v.value(t, 'dn_l')\n"
        "mid, hmid = v.value(t + %r, 'dn_l')\n"
        "print(json.dumps({'значение': val, 'дырка': hole is not None,\n"
        "  'между кадрами': mid is None and hmid is not None,\n"
        "  'torch в модулях': 'torch' in sys.modules,\n"
        "  'вызовов движка': v.lif_calls}, ensure_ascii=False))\n"
        % (str(LAB), str(where), CONTROL_MS / 2.0))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([str(PY), "-c", code], capture_output=True, env=env)
    err = p.stderr.decode("utf-8", "replace")[-600:]
    got = None
    if p.returncode == 0:
        try:
            got = json.loads(p.stdout.decode("utf-8", "replace").strip()
                             .splitlines()[-1])
        except Exception:                                     # noqa: BLE001
            got = None
    rep["встречный замер"]["просмотр без torch"] = {
        "код возврата": p.returncode, "ответ": got,
        "ошибка": "" if p.returncode == 0 else err}
    ok = bool(got) and got.get("значение") is not None \
        and not got.get("torch в модулях") and got.get("между кадрами") \
        and got.get("вызовов движка") == 0
    check("встречный замер: просмотр записи работает в процессе, где импорт "
          "torch запрещён", ok,
          ("ответ %r, torch в модулях %s, вызовов движка %s"
           % (got.get("значение"), got.get("torch в модулях"),
              got.get("вызовов движка"))) if got else
          ("процесс упал, код %d: %s" % (p.returncode, err)))
    if not ok and got is None:
        rep["находки"].append(
            "просмотр записи не открывается без torch: %s" % err)


def part_recompute(device, a):
    """Пересчёт: 200 вызовов на шаг и побитово та же запись."""
    print("  пересчёт того же прогона с начала")
    b = _episode(device, N_STEPS, "пересчёт")
    per = sorted(set(b["per_step"]))
    rep["режимы"]["пересчёт"] = {
        "вызовов движка на шаг": per,
        "вызовов движка всего": int(b["engine"].calls),
        "шагов": len(b["recs"]),
        "мс на шаг управления": round(b["wall_s"] * 1000.0
                                      / max(len(b["recs"]), 1), 2),
        "всего, с": round(b["wall_s"], 2),
    }
    from flybrain import replay as RP

    want = RP.lif_per_step(CONTROL_MS)
    check("пересчёт: ровно %d вызовов движка на шаг управления" % want,
          per == [want], "по записи: %s" % per)
    check("пересчёт: это тот же прогон — записи шагов побитово те же",
          a["recs"] == b["recs"],
          "шагов %d, отпечатков разошлось %d"
          % (len(a["recs"]),
             sum(1 for x, y in zip(a["recs"], b["recs"]) if x != y)))
    check("пересчёт: счётчик движка не ноль, то есть файл не подменяет счёт",
          int(b["engine"].calls) >= want * len(b["recs"]),
          "вызовов всего %d при %d шагах" % (int(b["engine"].calls),
                                             len(b["recs"])))
    return b


def part_resume(device, a, where):
    """Продолжение: 200 на шаг и ни одного лишнего вызова."""
    from flybrain import replay as RP

    st = RP.State.load(where)
    engine, motor, fly, arena, params, counts, report = RP.restore_arena(
        st, _episode.neurons, weights=_episode.weights, device=device)
    print("  продолжение из снимка на %g мс" % st.t_ms)
    calls_before = int(engine.calls)
    c = _episode(device, TAIL, "продолжение", engine=engine, motor=motor,
                 start=fly, counts0=counts, arena=arena, params=params,
                 t_offset=st.t_ms + CONTROL_MS)
    per = sorted(set(c["per_step"]))
    want = RP.lif_per_step(CONTROL_MS)
    total = int(engine.calls) - calls_before
    rep["режимы"]["продолжение"] = {
        "вызовов движка на шаг": per,
        "вызовов движка всего": total,
        "ожидалось ровно": want * TAIL,
        "шагов": len(c["recs"]),
        "мс на шаг управления": round(c["wall_s"] * 1000.0
                                      / max(len(c["recs"]), 1), 2),
        "модельное время снимка, мс": st.t_ms,
    }
    check("продолжение: ровно %d вызовов движка на шаг управления" % want,
          per == [want], "по записи: %s" % per)
    check("продолжение: вызовов ровно шагов×%d — разогрев калибровки не "
          "повторялся" % want,
          total == want * TAIL,
          "вызовов %d, ожидалось %d (разогрев стоил бы ещё %d)"
          % (total, want * TAIL, 75 * want))
    tail_ref = a["recs"][len(a["recs"]) - TAIL:]
    check("продолжение: записи шагов побитово те же, что у непрерывного "
          "счёта",
          tail_ref == c["recs"],
          "шагов %d, разошлось %d"
          % (len(tail_ref), sum(1 for x, y in zip(tail_ref, c["recs"])
                                if x != y)))

    # Подпись стоимости рядом, числами: во сколько раз просмотр дешевле.
    v = rep["режимы"]["просмотр"]
    r = rep["режимы"]["пересчёт"]
    rep["замеры"]["во сколько раз просмотр дешевле пересчёта, на единицу "
                  "модельного времени"] = round(
        r["мс на шаг управления"] / max(v["мс на ответ"], 1e-9), 1)
    print("    подпись стоимости: просмотр %.4f мс на ответ и 0 вызовов; "
          "пересчёт %.2f мс на шаг и %d вызовов; продолжение %.2f мс на "
          "шаг и %d вызовов"
          % (v["мс на ответ"], r["мс на шаг управления"],
             r["вызовов движка на шаг"][0] if r["вызовов движка на шаг"]
             else -1,
             rep["режимы"]["продолжение"]["мс на шаг управления"],
             per[0] if per else -1))


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep["замеры"]["устройство"] = device
    print("— запись —")
    a, where = part_record(device)
    print("\n— режим «просмотр» —")
    part_view(a, where)
    print("\n— встречный замер: без torch —")
    part_no_torch(where)
    print("\n— режим «пересчёт» —")
    part_recompute(device, a)
    print("\n— режим «продолжение» —")
    part_resume(device, a, where)

    (OUT / "replay_modes.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    for n in rep["находки"]:
        print("  замечание: " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
