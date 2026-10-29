"""Семь уровней замера: от одной физики до долгого сеанса песочницы.

Каждый уровень мерится ОТДЕЛЬНО и по одному и тому же правилу, иначе
сравнивать их между собой нельзя:

1. ``phys``   — только физика: ``physics.step()`` без походки и наблюдений;
2. ``gait``   — физика плюс контроллер походки (``body.step``: CPG,
   прослойка dm_control, наблюдения);
3. ``lif``    — то же плюс движок LIF, 200 шагов на шаг управления,
   без сенсорного входа и без моторного декодера;
4. ``closed`` — полный сенсорно-нейро-моторный контур
   (``body.run_episode``) с кадром живого вида и обменом через IPC: это
   та самая постановка, которой снята базовая линия;
5. ``vision`` — то же со включённым зрением: глаза в модели, кадр
   ``obs[vision]`` через привязку омматидиев на фоторецепторы коннектома;
6. ``live``   — то же с интерактивными правками (``live.BodyLive``) и
   записью ленты (``replay.Tape``);
7. ``long``   — долгий сеанс песочницы без заранее заданного конца, с
   кольцом записи (отдельный режим ``--long``, он идёт минутами).

Что печатается по каждому уровню: модельных секунд на настенную, медиана
И хвост задержки на шаг управления, разложение по составляющим с
невязкой СО ЗНАКОМ, загрузка процессора и видеокарты (своя и машинная),
память видеокарты и процесса, первый шаг против разогретого и счётчики на
точное равенство.

Разложение построено так, чтобы невязку можно было проверить. Окно шага
управления — от конца обработчика предыдущего шага до конца обработчика
этого, то есть **весь** шаг, включая кадр и обмен. У базовой линии кадр и
обмен лежали вне окна, и ровно отсюда её −2,44 мс: разложение сходилось
не потому, что всё учтено, а потому что две составляющие считались мимо
окна. Рядом поэтому печатается и «шаг по определению базовой линии» —
окно без кадра и обмена, прямо сравнимое с 337,41 мс.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\bench_seven.py --levels phys gait lif
    ..\\.venv\\Scripts\\python.exe probe\\bench_seven.py --levels closed
    ..\\.venv\\Scripts\\python.exe probe\\bench_seven.py --long --минут 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe import bench_stand as BS  # noqa: E402

OUT = Path(__file__).parent
BENCH = Path(__file__).resolve().parents[1] / "bench"
BENCH.mkdir(parents=True, exist_ok=True)

# Сколько первых шагов считать холодными и не брать в разогретую медиану.
# Число объявлено, а не подобрано: в базовой линии кадр живого вида на
# нулевом шаге стоил 14,66 мс против 2,0 разогретых, обмен 1,78 против
# 0,35, и к третьему шагу оба выходили на полку.
COLD_STEPS = 3


# --------------------------------------------------------------- обёртки

class Wrap:
    """Обёртки-счётчики на тело и на движок. Снимаются в :meth:`off`."""

    def __init__(self, cnt):
        self.cnt = cnt
        self.phys_ms = 0.0        # сумма вызовов physics.step
        self.body_ms = 0.0        # сумма вызовов body.step
        self.eye_ms = 0.0         # сумма пересъёмов глаз
        self.first_phys = None    # когда начался первый шаг физики шага
        self.last_phys_end = None
        self._undo = []

    def body(self, body, eyes=False):
        phys = body.physics
        o_pstep = phys.step

        def pstep(*a, **kw):
            t0 = time.perf_counter()
            r = o_pstep(*a, **kw)
            t1 = time.perf_counter()
            self.phys_ms += (t1 - t0) * 1000.0
            self.cnt.phys_inner += 1
            if self.first_phys is None:
                self.first_phys = t0
            self.last_phys_end = t1
            return r

        phys.step = pstep
        self._undo.append(lambda: setattr(phys, "step", o_pstep))

        o_bstep = body.step

        def bstep(action):
            t0 = time.perf_counter()
            if self.first_phys is None:
                self.first_phys = t0
            r = o_bstep(action)
            t1 = time.perf_counter()
            self.body_ms += (t1 - t0) * 1000.0
            self.cnt.phys += 1
            self.last_phys_end = t1
            return r

        body.step = bstep
        self._undo.append(lambda: setattr(body, "step", o_bstep))

        if eyes:
            fly = body.fly
            o_vis = fly._update_vision

            def vis(sim):
                was = getattr(fly, "_last_vision_update_time", None)
                t0 = time.perf_counter()
                r = o_vis(sim)
                self.eye_ms += (time.perf_counter() - t0) * 1000.0
                if getattr(fly, "_last_vision_update_time", None) != was:
                    # Считается не вызов, а СЪЁМ: при 100 Гц обновления
                    # вызов приходит на каждый шаг физики, а глаза
                    # пересняты только когда вышел срок.
                    self.cnt.eye += 1
                return r

            fly._update_vision = vis
            self._undo.append(lambda: setattr(fly, "_update_vision", o_vis))
        return self

    def engine(self, SIM):
        o_step = SIM.Engine.step
        o_poi = SIM.Engine._poisson

        def step(s, rates):
            self.cnt.lif += 1
            return o_step(s, rates)

        def poi(s, rates):
            self.cnt.poisson += 1
            return o_poi(s, rates)

        SIM.Engine.step = step
        SIM.Engine._poisson = poi
        self._undo.append(lambda: setattr(SIM.Engine, "step", o_step))
        self._undo.append(lambda: setattr(SIM.Engine, "_poisson", o_poi))
        return self

    def new_step(self):
        self.first_phys = None

    def off(self):
        for f in reversed(self._undo):
            try:
                f()
            except Exception:                                 # noqa: BLE001
                pass
        self._undo = []


# ----------------------------------------------------- уровни 1..3: свой цикл

def level_bare(ctx, kind, steps):
    """Уровни 1–3: шаг управления собран здесь, по частям и без контура.

    Порядок внутри шага тот же, что в `body.run_episode`: сначала мозг,
    потом сто шагов тела. Иначе уровни не складывались бы с четвёртым.
    """
    import torch

    from flybrain import body as B, sim as SIM

    dev = ctx["device"]
    neurons, weights = ctx["neurons"], ctx["weights"]
    cnt = BS.Counters()
    w = Wrap(cnt)
    rep = {"что мерится": {
        "phys": "только физика: physics.step(), ровно сто на шаг управления",
        "gait": "физика плюс контроллер походки: body.step() сто раз",
        "lif": "физика, походка и движок LIF (200 шагов) без сенсоров и "
               "моторного декодера",
    }[kind]}

    cfg = B.BodyConfig(duration_s=steps * BS.CONTROL_MS / 1000.0,
                       device=dev, seed=0)
    phys_per = int(round(cfg.control_ms / 1000.0 / cfg.physics_dt))
    lif_per = int(round(cfg.control_ms / SIM.DT))
    rep["устройство замера"] = {
        "шагов управления": steps,
        "шаг управления, мс": cfg.control_ms,
        "шагов физики на шаг управления (из кода)": phys_per,
        "шагов движка LIF на шаг управления (из кода)": lif_per,
        "устройство": dev,
    }

    BS.gpu_reset()
    t0 = time.perf_counter()
    body, _ = B._build_body(cfg, B.DEFAULT_SOURCES.copy())
    obs, _ = body.reset(seed=cfg.seed)
    t_build = time.perf_counter() - t0

    engine = None
    t_eng = t_graphs = 0.0
    if kind == "lif":
        o_graphs = SIM.Engine.use_graphs
        box = {"ms": 0.0}

        def ug(s, enable=True):
            t = time.perf_counter()
            r = o_graphs(s, enable)
            box["ms"] += (time.perf_counter() - t) * 1000.0
            return r

        SIM.Engine.use_graphs = ug
        t = time.perf_counter()
        engine = SIM.build_engine(weights=weights, neurons=neurons,
                                  device=dev, seed=cfg.seed)
        t_eng = time.perf_counter() - t
        t_graphs = box["ms"] / 1000.0
        SIM.Engine.use_graphs = o_graphs
        rep["устройство замера"]["графов захвачено"] = (
            len(engine._graphs) if engine._graphs is not None else 0)

    w.body(body)
    if engine is not None:
        w.engine(SIM)

    names = {
        "phys": ["чистая физика MuJoCo"],
        "gait": ["чистая физика MuJoCo", "походка и прослойка dm_control"],
        "lif": ["мозг: LIF, синхронизация, учёт", "чистая физика MuJoCo",
                "походка и прослойка dm_control"],
    }[kind]
    split = BS.Split(names)
    cpu = BS.Cpu()
    action = np.array([1.0, 1.0])
    rates = (torch.zeros(1, len(neurons), device=dev)
             if engine is not None else None)

    per_step, cold = [], {}
    t_win0 = time.perf_counter()
    m_cpu = cpu.mark()
    prev_exit = None
    for s in range(steps):
        w.new_step()
        p0, b0, e0 = w.phys_ms, w.body_ms, w.eye_ms
        t_brain0 = time.perf_counter()
        if engine is not None:
            acc = torch.zeros(len(neurons), device=dev)
            for _ in range(lif_per):
                acc += engine.step(rates)[0]
            float(acc.sum())            # синхронизация: цифру надо прочесть
        t_brain1 = time.perf_counter()
        if kind == "phys":
            for _ in range(phys_per):
                body.physics.step()
        else:
            for _ in range(phys_per):
                obs, _, term, trunc, _ = body.step(action)
                if term or trunc:
                    break
        t_end = time.perf_counter()
        cnt.snap(engine)
        parts = {"чистая физика MuJoCo": w.phys_ms - p0}
        if kind != "phys":
            parts["походка и прослойка dm_control"] = (
                (w.body_ms - b0) - (w.phys_ms - p0))
        if kind == "lif":
            parts["мозг: LIF, синхронизация, учёт"] = (
                (t_brain1 - t_brain0) * 1000.0)
        total = ((t_end - prev_exit) * 1000.0 if prev_exit is not None
                 else (t_end - t_brain0) * 1000.0)
        row = {"шаг": s, "всего, мс": round(total, 3),
               **{k: round(v, 3) for k, v in parts.items()}}
        per_step.append(row)
        if s >= COLD_STEPS:
            split.add(total, parts)
        if s == 0:
            cold = dict(row)
        prev_exit = t_end
    wall = time.perf_counter() - t_win0
    t_win1 = time.perf_counter()

    rep["подготовка, с"] = {
        "сборка тела MuJoCo и сброс": round(t_build, 2),
        "сборка движка LIF": round(t_eng, 2) if engine is not None else None,
        "из неё захват CUDA-графов": (round(t_graphs, 2)
                                      if engine is not None else None),
    }
    rep["разложение (разогретые шаги)"] = split.report()
    rep["шаг целиком"] = BS.stat([r["всего, мс"] for r in per_step[COLD_STEPS:]])
    med = rep["шаг целиком"].get("медиана")
    rep["скорость"] = {
        "доля реального времени по медиане шага":
            round(cfg.control_ms / med, 5) if med else None,
        "модельных секунд на настенную (по всему окну)":
            round(steps * cfg.control_ms / 1000.0 / wall, 5) if wall else None,
        "во сколько раз медленнее реального времени":
            round(med / cfg.control_ms, 2) if med else None,
    }
    rep["первый шаг против разогретого"] = {
        "шаг 0": cold,
        "медиана разогретых (с шага %d)" % COLD_STEPS: {
            k: BS.stat([r[k] for r in per_step[COLD_STEPS:]]).get("медиана")
            for k in per_step[0] if k != "шаг"},
    }
    rep["счётчики"] = cnt.verdict({
        "шагов physics.step": phys_per,
        "шагов body.step": 0 if kind == "phys" else phys_per,
        "вызовов движка LIF": lif_per if kind == "lif" else 0,
        "взятий генератора": lif_per if kind == "lif" else 0,
    })
    rep["загрузка"] = cpu.since(m_cpu)
    rep["видеокарта"] = dict(ctx["gpu"].window(t_win0, t_win1), **BS.gpu_mine())
    rep["память процесса"] = {"пик, МБ": BS.rss_peak_mb(),
                              "сейчас, МБ": BS.rss_now_mb()}
    rep["шаги"] = per_step

    w.off()
    try:
        body.close()
    except Exception:                                         # noqa: BLE001
        pass
    del engine, body
    return rep


# ------------------------------------------- уровни 4..6: настоящий контур

class VisionLive:
    """Зрение в контур через существующую ручку живых условий.

    Своего протокола нет нарочно: `body.run_episode` зовёт у состояния
    живых условий ровно четыре метода, и зрение подаётся тем же
    `add_drive`, которым подаются прямые нейронные стимулы. Так уровень 5
    отличается от уровня 4 только зрением, а не другим циклом.

    Кадр берётся из `fly._curr_visual_input` — того самого массива,
    который `Fly.get_observation` кладёт в `obs["vision"]`. Пересчитывать
    его отдельным вызовом было бы лишней работой, которой в настоящем
    сеансе нет.
    """

    def __init__(self, cnt, max_hz=150.0, ref=0.5):
        self.cnt = cnt
        self.max_hz = max_hz
        self.ref = ref
        self.body = None
        self.maps = None
        self.t_map = None
        self.ms = 0.0
        self.frames = 0
        self.frame_fp = []

    def attach(self, body, sources, cfg, neurons, device, engine=None):
        import torch

        from flybrain import retina as R

        self.body = body
        self.maps = R.build_all(neurons, mode="whiten", assign="capacity")
        self.t_map = {k: torch.as_tensor(m.neurons, device=device)
                      for k, m in self.maps.items()}
        self.R = R
        self.torch = torch
        self.device = device
        self.fingerprint = R.fingerprint_all(self.maps)
        self.photoreceptors = int(len(np.unique(np.concatenate(
            [self.maps[k].neurons for k in self.maps]))))

    def add_drive(self, rates):
        fr = getattr(self.body.fly, "_curr_visual_input", None)
        if fr is None:
            return
        t0 = time.perf_counter()
        hz = self.R.rates(self.maps, np.asarray(fr, dtype=np.float32),
                          max_hz=self.max_hz, ref=self.ref)
        for key, val in hz.items():
            rates[0, self.t_map[key]] = self.torch.as_tensor(
                np.asarray(val, dtype=np.float32), device=self.device)
        self.ms += (time.perf_counter() - t0) * 1000.0
        self.frames += 1
        self.cnt.retina += 1
        if len(self.frame_fp) < 400:
            self.frame_fp.append(BS.fp(np.asarray(fr, dtype=np.float32)))

    def active_sources(self):
        return None

    def _read_sources(self):
        return []


def against_reference(kind, got, steps, cfg):
    """Сверка выхода уровня с ЗАМОРОЖЕННЫМ эталоном `bench_v2_эталон.json`.

    Это не формальность, а проверка самого стенда. Уровень 4 считается с
    навешенными обёртками, со своим рендерером и с отправкой в настоящие
    очереди IPC — и если наблюдение за сеансом сдвигает числа, то стенд
    мерит не ту систему, которую сравнивают с эталоном. Сценарий `body`
    эталона — это ровно те же настройки (семя 0, шаг управления 20 мс,
    съём ALPN) на одной секунде модельного времени, то есть 50 шагов.

    У уровней 5 и 6 равенства не ожидается **по построению**, и это
    сказано здесь, а не замечено потом: со зрением в модели появляются
    камеры глаз (другая модель), а правки живых условий меняют условия
    опыта. Сверка для них печатается, но провалом не считается.
    """
    ref_path = OUT / "bench_v2_эталон.json"
    out = {"эталон": str(ref_path.name),
           "сценарий эталона": "body (1,0 с модельного = 50 шагов)"}
    if not ref_path.exists():
        out["почему не сверено"] = ("снимка эталона нет в каталоге проб: "
                                    "он не в истории кода (gitignore), и "
                                    "его надо принести с машины")
        return out
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    sc = (ref.get("сценарии") or {}).get("body")
    if sc is None:
        out["почему не сверено"] = "в снимке эталона нет сценария body"
        return out
    out["метка эталона"] = ref.get("метка")
    out["когда снят эталон"] = ref.get("когда")
    if steps != 50:
        out["почему не сверено"] = (
            "у эталона сценарий body на 50 шагов, а здесь %d: сверять "
            "отпечатки разной длины нельзя" % steps)
        return out
    want = {
        "отпечаток counts": (sc.get("активность") or {}).get("отпечаток counts"),
        "отпечаток bins": (sc.get("активность") or {}).get("отпечаток bins"),
        "отпечаток траектории": (sc.get("тело") or {}).get(
            "отпечаток траектории"),
        "отпечаток моторных команд": (sc.get("моторные команды") or {}).get(
            "отпечаток dn"),
    }
    rows = {}
    for k, v in want.items():
        rows[k] = {"эталон": v, "стенд": got.get(k),
                   "совпало побитово": bool(v and v == got.get(k))}
    out["поля"] = rows
    out["совпало всего"] = sum(1 for v in rows.values()
                               if v["совпало побитово"])
    out["полей сверено"] = len(rows)
    out["равенство ожидается"] = (kind == "closed")
    out["вывод"] = (
        "уровень 4 считается под обёртками стенда, со своим рендерером и "
        "с отправкой в настоящие очереди — и выход обязан быть ПОБИТОВО "
        "эталонным, иначе стенд мерит не ту систему"
        if kind == "closed" else
        "равенства не ожидается по построению: " + (
            "со зрением в модели появляются камеры глаз, то есть модель "
            "другая" if kind == "vision" else
            "правки живых условий меняют условия опыта"))
    return out


def level_loop(ctx, kind, steps):
    """Уровни 4–6: `body.run_episode` целиком, замеры навешены снаружи.

    Научный алгоритм не меняется ни на операцию: меняются только
    обёртки, отмечающие время. Это то же правило, по которому снята
    базовая линия (`probe/bodyfinal_timing.py`), и взято оно нарочно —
    иначе числа стенда нельзя было бы приложить к её числам.
    """
    import torch

    from flybrain import body as B, ipc, sim as SIM

    dev = ctx["device"]
    neurons, weights = ctx["neurons"], ctx["weights"]
    cnt = BS.Counters()
    w = Wrap(cnt)
    rep = {"что мерится": {
        "closed": "полный сенсорно-нейро-моторный контур, кадр живого вида "
                  "и обмен через IPC: постановка базовой линии",
        "vision": "то же со включённым зрением: глаза в модели, кадр "
                  "obs[vision] через привязку омматидиев в фоторецепторы",
        "live": "то же с интерактивными правками и записью ленты",
    }[kind]}

    cfg = B.BodyConfig(duration_s=steps * BS.CONTROL_MS / 1000.0,
                       device=dev, seed=0, readout="ALPN")
    phys_per = int(round(cfg.control_ms / 1000.0 / cfg.physics_dt))
    lif_per = int(round(cfg.control_ms / SIM.DT))
    warm_steps = max(5, int(round(3000.0 * cfg.dn_tau_s / cfg.control_ms)))
    rep["устройство замера"] = {
        "шагов управления": steps,
        "шаг управления, мс": cfg.control_ms,
        "шагов физики на шаг управления (из кода)": phys_per,
        "шагов движка LIF на шаг управления (из кода)": lif_per,
        "шагов разогрева нисходящих (из кода)": warm_steps,
        "съём": cfg.readout, "устройство": dev,
    }

    # Зрение: модель собирается с глазами. Это ПРАВКА МОДЕЛИ, а не
    # настройка — у `Fly` добавляются камеры глаз, — и поэтому уровень 5
    # численно не равен четвёртому по построению. Объявлено здесь, а не
    # замечено потом.
    o_fly_class = B._fly_class
    if kind == "vision":
        def fly_class(fast):
            base = o_fly_class(fast)

            class FlyEyes(base):
                def __init__(self, *a, **kw):
                    kw.setdefault("enable_vision", True)
                    kw.setdefault("vision_refresh_rate", 100)
                    super().__init__(*a, **kw)

            return FlyEyes

        B._fly_class = fly_class

    # Захват графов считается отдельно: это секунды, и в шаг они не входят.
    o_graphs = SIM.Engine.use_graphs
    box = {"ms": 0.0, "n": 0}

    def ug(s, enable=True):
        t = time.perf_counter()
        r = o_graphs(s, enable)
        box["ms"] += (time.perf_counter() - t) * 1000.0
        box["n"] = 0 if s._graphs is None else len(s._graphs)
        return r

    SIM.Engine.use_graphs = ug
    w.engine(SIM)

    live = tape = None
    edits = []
    if kind == "vision":
        live = VisionLive(cnt)
    if kind == "live":
        from flybrain import live as LV, replay as RP

        live = LV.BodyLive()
        # Окно записи обязано быть названо числом: без него таймлайн
        # отказывается собираться, и правильно — ряды и растр растут по
        # строке на шаг управления. Берём с запасом на весь замер.
        tape = RP.Tape(kind="body", control_ms=cfg.control_ms,
                       keep_s=max(2.0, steps * cfg.control_ms / 1000.0 + 1.0),
                       mode="пересчёт",
                       meta={"стенд": "уровень 6", "семя": cfg.seed})

    chan = ipc.Channel()
    ren = {"ren": None, "cam": None, "data": None}
    marks = {"build": None, "renderer_s": None, "first_phys": None}
    split = BS.Split([
        "мозг: LIF, синхронизация, учёт",
        "чистая физика MuJoCo",
        "походка и прослойка dm_control",
        "хвост шага: поза, метрики, запись",
        "кадр живого вида",
        "обмен через IPC",
        "прочее в обработчике шага",
    ])
    rows, state = [], {"prev_exit": None, "cold": None, "p_ms": 0.0,
                       "b_ms": 0.0, "e_ms": 0.0}

    def on_body(body):
        import mujoco

        from flybrain.worker import _thorax_id

        marks["build"] = time.perf_counter()
        w.body(body, eyes=(kind == "vision"))
        t0 = time.perf_counter()
        model = body.physics.model.ptr
        r = mujoco.Renderer(model, 720, 960)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = _thorax_id(model)
        cam.distance = 6.5
        cam.elevation = -18.0
        ren.update(ren=r, cam=cam, data=body.physics.data.ptr)
        marks["renderer_s"] = time.perf_counter() - t0
        marks["first_phys"] = None

    def on_step(step, rec, bins_row):
        t_entry = time.perf_counter()
        if marks["first_phys"] is None and w.first_phys is not None:
            marks["first_phys"] = w.first_phys
        # Приросты считаются от среза, снятого в КОНЦЕ предыдущего
        # обработчика, а не от начала этого: вся работа шага уже сделана
        # к моменту, когда обработчик позвали, и разница «вход минус
        # выход этого же обработчика» была бы нулём.
        p_ms = w.phys_ms - state["p_ms"]
        b_ms = w.body_ms - state["b_ms"]
        e_ms = w.eye_ms - state["e_ms"]
        first_phys, last_end = w.first_phys, w.last_phys_end

        # кадр живого вида — ровно то, что делает работник на шаге
        t0 = time.perf_counter()
        r, cam = ren["ren"], ren["cam"]
        if r is not None:
            cam.azimuth = float(np.degrees(rec.get("theta", 0.0)))
            r.update_scene(ren["data"], camera=cam)
            r.render()
        t_ren = (time.perf_counter() - t0) * 1000.0

        # обмен: упаковка кадра активности и отправка в настоящие очереди
        t1 = time.perf_counter()
        idx, val = ipc.pack_frame(bins_row)
        chan.send(ipc.Frame("стенд", rec["t_ms"], idx, val, rec))
        chan.send(ipc.Series("стенд", rec["t_ms"],
                             {k: rec[k] for k in
                              ("dn_l", "dn_r", "odour_l", "odour_r", "amp_l",
                               "amp_r", "dist", "grad_cos") if k in rec}))
        chan.send(ipc.Status("стенд", "running", "эпизод", 0.5))
        t_ipc = (time.perf_counter() - t1) * 1000.0

        # интерактивные правки: ровно тем протоколом, которым правит
        # песочница, и на границе шага управления — там, где состояние
        # модели целостно
        t2 = time.perf_counter()
        if kind == "live":
            live.tick(rec["t_ms"])
            for i, (at, payload, must) in enumerate(EDITS):
                if step == at:
                    t3 = time.perf_counter()
                    try:
                        got = live.apply(payload, cmd_id=i + 1,
                                         t_ms=rec["t_ms"])
                        ok, why = True, got
                    except Exception as exc:                  # noqa: BLE001
                        ok, why = False, str(exc)[:160]
                    edits.append({
                        "шаг": step, "модельное, мс": rec["t_ms"],
                        "правка": json.dumps(payload, ensure_ascii=False)[:120],
                        "обязана примениться": must,
                        "применилась": ok,
                        "цена, мс": round((time.perf_counter() - t3) * 1000.0, 2),
                        "исход": (json.dumps(why, ensure_ascii=False,
                                             default=str)[:200]
                                  if ok else why),
                    })
        t_edit = (time.perf_counter() - t2) * 1000.0

        cnt.snap(engine_box.get("eng"))
        t_exit = time.perf_counter()
        brain = ((first_phys - state["prev_exit"]) * 1000.0
                 if (state["prev_exit"] is not None and first_phys) else None)
        phys_inner = p_ms
        body_tot = b_ms
        parts = {
            "мозг: LIF, синхронизация, учёт": brain or 0.0,
            "чистая физика MuJoCo": phys_inner,
            "походка и прослойка dm_control": body_tot - phys_inner,
            "хвост шага: поза, метрики, запись":
                (t_entry - last_end) * 1000.0 if last_end else 0.0,
            "кадр живого вида": t_ren,
            "обмен через IPC": t_ipc,
            "прочее в обработчике шага":
                (t_exit - t_entry) * 1000.0 - t_ren - t_ipc,
        }
        row = {"шаг": step,
               "всего, мс": (round((t_exit - state["prev_exit"]) * 1000.0, 3)
                             if state["prev_exit"] is not None else None),
               "по определению базовой линии, мс":
                   (round((t_entry - state["prev_exit"]) * 1000.0, 3)
                    if state["prev_exit"] is not None else None),
               **{k: round(v, 3) for k, v in parts.items()},
               "съём глаз, мс": round(e_ms, 3),
               "ненулевых нейронов в кадре": int(len(idx))}
        rows.append(row)
        if row["всего, мс"] is not None and step >= COLD_STEPS:
            split.add(row["всего, мс"], parts)
        if state["cold"] is None:
            state["cold"] = dict(row)
        state["prev_exit"] = t_exit
        state["p_ms"], state["b_ms"], state["e_ms"] = (
            w.phys_ms, w.body_ms, w.eye_ms)
        w.new_step()

    # Правки для уровня 6. Последняя обязана быть ОТКЛОНЕНА: без
    # отказа, который ничего не меняет, проверка «правки доходят»
    # прошла бы и на механизме, который применяет что попало.
    EDITS = [
        (3, {"orn_max_hz": 200.0}, True),
        (5, {"sources": [{"x": 20.0, "y": 6.0, "strength": 1.0, "on": True},
                         {"x": 38.0, "y": -16.0, "strength": 1.0,
                          "on": False}]}, True),
        (7, {"stim": [{"query": "celltype=DNp01", "rate_hz": 50.0}]}, True),
        (9, {"lesion": [{"query": "celltype=T4a", "kind": "hold",
                         "ms": 100.0}]}, True),
        (11, {"orn_max_hz": 1e9}, False),
    ]

    engine_box = {}
    o_build = SIM.build_engine

    def build(*a, **kw):
        t = time.perf_counter()
        eng = o_build(*a, **kw)
        engine_box["eng"] = eng
        engine_box["s"] = time.perf_counter() - t
        return eng

    SIM.build_engine = build

    BS.gpu_reset()
    cpu = BS.Cpu()
    m_cpu = cpu.mark()
    t_win0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, on_step=on_step, on_body=on_body,
                        live=live, tape=tape)
    wall = time.perf_counter() - t_win0
    t_win1 = time.perf_counter()

    SIM.build_engine = o_build
    SIM.Engine.use_graphs = o_graphs
    B._fly_class = o_fly_class
    w.off()
    chan.drain()
    chan.close()

    warm = [r for r in rows if r["всего, мс"] is not None
            and r["шаг"] >= COLD_STEPS]
    rep["подготовка, с"] = {
        "сборка движка LIF": round(engine_box.get("s", 0.0), 2),
        "из неё захват CUDA-графов": round(box["ms"] / 1000.0, 2),
        "графов захвачено": box["n"],
        "сборка тела MuJoCo (timing[setup_s])": run.timing.get("setup_s"),
        "создание рендерера MuJoCo": round(marks["renderer_s"] or 0.0, 2),
        "разогрев нисходящих (от сборки тела до первого шага физики)":
            (round(marks["first_phys"] - marks["build"], 2)
             if marks["first_phys"] and marks["build"] else None),
        "эпизод целиком по стене": round(wall, 2),
    }
    rep["разложение (разогретые шаги)"] = split.report()
    rep["шаг целиком"] = BS.stat([r["всего, мс"] for r in warm])
    rep["шаг по определению базовой линии"] = BS.stat(
        [r["по определению базовой линии, мс"] for r in warm])
    med = rep["шаг целиком"].get("медиана")
    rep["скорость"] = {
        "доля реального времени по медиане шага":
            round(cfg.control_ms / med, 5) if med else None,
        # По сумме окон шагов, а не по всей стене: в стену входят сборка
        # тела и разогрев нисходящих, а они не часть шага и считаются
        # отдельно, в «подготовке».
        "модельных секунд на настенную (по сумме окон шагов)":
            (round(len(warm) * cfg.control_ms / 1000.0
                   / (sum(r["всего, мс"] for r in warm) / 1000.0), 5)
             if warm else None),
        "realtime_ratio из самого прогона": run.timing.get("realtime_ratio"),
        "во сколько раз медленнее реального времени":
            round(med / cfg.control_ms, 2) if med else None,
    }
    rep["первый шаг против разогретого"] = {
        "шаг 0": state["cold"],
        "медиана разогретых (с шага %d)" % COLD_STEPS: {
            k: BS.stat([r[k] for r in warm]).get("медиана")
            for k in warm[0] if k != "шаг"} if warm else {},
    }
    want = {"шагов physics.step": phys_per, "шагов body.step": phys_per,
            "вызовов движка LIF": lif_per, "взятий генератора": lif_per,
            "вызовов по счётчику движка": lif_per}
    if kind == "vision":
        want["кадров сетчатки в сеть"] = 1
        want["пересъёмов глаз"] = 2    # 100 Гц на шаге управления 20 мс
    rep["счётчики"] = cnt.verdict(want)
    rep["счётчики"]["вызовов движка всего за прогон"] = cnt.lif
    rep["счётчики"]["арифметика вызовов"] = {
        "обязано": lif_per * (warm_steps + len(rows)),
        "как считается": "200 × (разогрев %d + шагов %d)"
                         % (warm_steps, len(rows)),
        "сошлось": cnt.lif == lif_per * (warm_steps + len(rows)),
    }
    rep["загрузка"] = cpu.since(m_cpu)
    rep["видеокарта"] = dict(ctx["gpu"].window(t_win0, t_win1), **BS.gpu_mine())
    rep["память процесса"] = {"пик, МБ": BS.rss_peak_mb(),
                              "сейчас, МБ": BS.rss_now_mb()}
    rep["выход прогона"] = {
        "шагов": len(run.records),
        "суммарные спайки": int(run.counts.sum()),
        "отпечаток counts": BS.fp(run.counts),
        "отпечаток bins": BS.fp(run.bins),
        "отпечаток траектории": BS.fp(np.array(
            [[r["x"], r["y"]] for r in run.records], dtype=np.float64)),
        "отпечаток моторных команд": BS.fp(np.array(
            [[r["dn_l"], r["dn_r"]] for r in run.records], dtype=np.float64)),
        "конец x, мм": run.records[-1]["x"],
        "конец y, мм": run.records[-1]["y"],
        "конец курса, рад": run.records[-1]["theta"],
        "метрики": run.metrics,
        "оговорки": run.timing.get("оговорки"),
    }
    rep["сверка с сохранённым эталоном"] = against_reference(
        kind, rep["выход прогона"], len(run.records), cfg)
    if kind == "vision":
        uniq = len(set(live.frame_fp))
        rep["зрение"] = {
            "фоторецепторов под привязкой": live.photoreceptors,
            "отпечаток привязки": live.fingerprint,
            "кадров подано в сеть": live.frames,
            "различных кадров": uniq,
            "кадр менялся": uniq > 1,
            "цена привязки и подачи, мс на шаг":
                round(live.ms / max(live.frames, 1), 3),
            "съём глаз, мс на шаг медиана":
                BS.stat([r["съём глаз, мс"] for r in warm]).get("медиана"),
        }
    if kind == "live":
        rep["правки"] = edits
        rep["правки сводка"] = {
            "подано": len(edits),
            "применилось": sum(1 for e in edits if e["применилась"]),
            "отклонено": sum(1 for e in edits if not e["применилась"]),
            "все ли вели себя как обязаны":
                all(e["применилась"] == e["обязана примениться"]
                    for e in edits),
            "цена правки, мс": BS.stat([e["цена, мс"] for e in edits], 3),
        }
        tl = run.timeline
        rep["запись ленты"] = {
            "строк в ленте": len(tl.times()) if tl is not None else None,
            "шагов в прогоне": len(run.records),
            "вызовов движка на шаг управления по ленте":
                tl.meta.get("вызовов движка на шаг управления")
                if tl is not None else None,
            "журнал правок в прогоне": len(run.timing.get("правки") or [])
            if isinstance(run.timing.get("правки"), list) else
            bool(run.timing.get("правки")),
        }
    rep["шаги"] = rows
    try:
        del run
    except Exception:                                         # noqa: BLE001
        pass
    return rep


# ------------------------------------------------------------------ счётчики

def part_counters(ctx):
    """Что ловят счётчики, чего не ловят отпечатки. Встречный замер.

    Нападение, от которого отпечатки не защищают: прогон не считает, а
    отдаёт записанное. Отпечатки выходов сходятся ПОБИТОВО — они и взяты
    из той же записи, — а движок при этом не позван ни разу. Здесь это
    сделано буквально: сначала честный прогон с записью спайков, потом
    «ускоренный», который отдаёт записанное.
    """
    import torch

    from flybrain import sim as SIM

    dev = ctx["device"]
    neurons, weights = ctx["neurons"], ctx["weights"]
    n = len(neurons)
    steps, lif_per = 3, int(round(BS.CONTROL_MS / SIM.DT))
    rep = {"что проверяется": "счётчики против отпечатков на подлоге "
                              "«ускорились, потому что перестали считать»"}

    eng = SIM.build_engine(weights=weights, neurons=neurons, device=dev,
                           seed=0)
    rates = torch.zeros(1, n, device=dev)

    honest, tape = [], []
    c0 = eng.calls
    t0 = time.perf_counter()
    for _ in range(steps):
        acc = torch.zeros(n, device=dev)
        for _ in range(lif_per):
            s = eng.step(rates)[0]
            acc += s
        tape.append(acc.detach().cpu().numpy().copy())
        honest.append(BS.fp(tape[-1]))
    t_honest = (time.perf_counter() - t0) * 1000.0 / steps
    calls_honest = eng.calls - c0

    # «Ускоренный»: ни одного вызова движка, выход из записи.
    c1 = eng.calls
    t0 = time.perf_counter()
    fake = [BS.fp(row) for row in tape]
    t_fake = (time.perf_counter() - t0) * 1000.0 / steps
    calls_fake = eng.calls - c1

    rep["честный прогон"] = {
        "вызовов движка на шаг управления": calls_honest // steps,
        "мс на шаг управления": round(t_honest, 3),
        "отпечатки": honest,
    }
    rep["подлог «выход из записи»"] = {
        "вызовов движка на шаг управления": calls_fake // steps,
        "мс на шаг управления": round(t_fake, 4),
        "отпечатки": fake,
    }
    rep["итог"] = {
        "отпечатки совпали побитово": honest == fake,
        "ускорение по стене, раз": round(t_honest / max(t_fake, 1e-9), 1),
        "счётчик вызовов движка поймал": calls_fake == 0 and calls_honest > 0,
        "что это значит": (
            "отпечатки на этом подлоге сходятся побитово и пропускают его, "
            "а счётчик вызовов движка равен нулю против %d и бракует сразу. "
            "Поэтому счётчики на точное равенство стоят рядом с "
            "отпечатками, а не вместо них."
            % (calls_honest // steps)),
    }
    del eng
    return rep


# ------------------------------------------------------------- долгий сеанс

def level_long(ctx, minutes, keep_s, every_s=10.0):
    """Уровень 7: сеанс без заранее заданного конца, с кольцом записи.

    Сеанс кончается остановкой, а не исчерпанием шагов: `duration_s <= 0`
    и `keep_s > 0`. Мерится не «прошло ли», а то, за чем можно сидеть:
    не ползёт ли шаг управления к концу сеанса, не растёт ли память, и
    совпадает ли кольцо записи с хвостом сеанса.
    """
    import torch

    from flybrain import body as B, live as LV, replay as RP, sim as SIM

    dev = ctx["device"]
    neurons, weights = ctx["neurons"], ctx["weights"]
    cnt = BS.Counters()
    w = Wrap(cnt)
    cfg = B.BodyConfig(duration_s=0.0, keep_s=keep_s, device=dev, seed=0,
                       odour_ref=0.01)
    lif_per = int(round(cfg.control_ms / SIM.DT))
    phys_per = int(round(cfg.control_ms / 1000.0 / cfg.physics_dt))
    live = LV.BodyLive()
    tape = RP.Tape(kind="body", control_ms=cfg.control_ms, keep_s=keep_s,
                   mode="пересчёт", meta={"стенд": "уровень 7"})
    rep = {"что мерится": "долгий сеанс песочницы: без заданного конца, "
                          "кольцо записи, правки на ходу",
           "устройство замера": {
               "минут по стене заказано": minutes,
               "окно записи, с модельного": keep_s,
               "шаг управления, мс": cfg.control_ms,
               "шагов движка LIF на шаг управления": lif_per,
               "шагов физики на шаг управления": phys_per}}

    w.engine(SIM)
    deadline = {"t": None}
    rows, mem, edits = [], [], []
    state = {"prev": None}
    EDITS = [
        (0.20, {"orn_max_hz": 220.0}),
        (0.40, {"sources": [{"x": 15.0, "y": 10.0, "strength": 2.0,
                             "on": True},
                            {"x": 38.0, "y": -16.0, "strength": 1.0,
                             "on": False}]}),
        (0.60, {"stim": [{"query": "celltype=DNa02", "rate_hz": 30.0}]}),
        (0.80, {"lesion": [{"query": "celltype=L1", "kind": "hold",
                            "ms": 400.0}]}),
    ]
    done = set()

    def on_body(body):
        w.body(body)
        deadline["t"] = time.perf_counter() + minutes * 60.0

    def on_step(step, rec, bins_row):
        now = time.perf_counter()
        live.tick(rec["t_ms"])
        if state["prev"] is not None:
            rows.append({"шаг": step, "мс": (now - state["prev"]) * 1000.0,
                         "модельное, мс": rec["t_ms"]})
        state["prev"] = now
        if step % 25 == 0:
            mem.append({"шаг": step, "модельное, с": rec["t_ms"] / 1000.0,
                        "процесс, МБ": BS.rss_now_mb(),
                        "видеокарта своя, МБ": round(
                            torch.cuda.memory_allocated() / 1e6, 1)
                        if torch.cuda.is_available() else None,
                        "стена, с": round(now - (deadline["t"]
                                                 - minutes * 60.0), 1)})
        frac = ((now - (deadline["t"] - minutes * 60.0))
                / (minutes * 60.0)) if deadline["t"] else 0.0
        for at, payload in EDITS:
            if at not in done and frac >= at:
                done.add(at)
                t0 = time.perf_counter()
                try:
                    got = live.apply(payload, cmd_id=len(done),
                                     t_ms=rec["t_ms"])
                    ok = True
                except Exception as exc:                      # noqa: BLE001
                    got, ok = str(exc)[:160], False
                edits.append({"доля сеанса": at, "шаг": step,
                              "модельное, мс": rec["t_ms"], "применилось": ok,
                              "цена, мс": round(
                                  (time.perf_counter() - t0) * 1000.0, 2),
                              "исход": json.dumps(got, ensure_ascii=False,
                                                  default=str)[:200]})
        cnt.snap(None)

    def should_stop():
        return deadline["t"] is not None and time.perf_counter() > deadline["t"]

    BS.gpu_reset()
    cpu = BS.Cpu()
    m_cpu = cpu.mark()
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, on_step=on_step, on_body=on_body,
                        should_stop=should_stop, live=live, tape=tape)
    wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    w.off()

    ms = [r["мс"] for r in rows]
    third = max(len(ms) // 3, 1)
    rep["шаг управления, мс"] = BS.stat(ms)
    rep["дрейф по сеансу"] = {
        "первая треть, медиана мс": BS.stat(ms[:third]).get("медиана"),
        "последняя треть, медиана мс": BS.stat(ms[-third:]).get("медиана"),
        "отношение последняя/первая": round(
            BS.stat(ms[-third:]).get("медиана", 0)
            / max(BS.stat(ms[:third]).get("медиана", 1), 1e-9), 4),
        "шагов в трети": third,
    }
    med = rep["шаг управления, мс"].get("медиана")
    rep["скорость"] = {
        "шагов управления прожито": len(run.records),
        "модельного времени, с": run.timing.get("duration_s"),
        "по стене, с": round(wall, 1),
        "модельных секунд на настенную":
            round((run.timing.get("duration_s") or 0) / wall, 5),
        "realtime_ratio из прогона": run.timing.get("realtime_ratio"),
        "доля реального времени по медиане шага":
            round(cfg.control_ms / med, 5) if med else None,
    }
    rep["память по сеансу"] = mem
    if mem:
        rep["память: рост"] = {
            "процесс в начале, МБ": mem[0]["процесс, МБ"],
            "процесс в конце, МБ": mem[-1]["процесс, МБ"],
            "прирост, МБ": (round(mem[-1]["процесс, МБ"]
                                  - mem[0]["процесс, МБ"], 1)
                            if mem[0]["процесс, МБ"] else None),
            "видеокарта своя в начале, МБ": mem[0]["видеокарта своя, МБ"],
            "видеокарта своя в конце, МБ": mem[-1]["видеокарта своя, МБ"],
        }
    rep["правки"] = edits
    rep["счётчики"] = cnt.verdict({"вызовов движка LIF": lif_per,
                                   "шагов body.step": phys_per,
                                   "шагов physics.step": phys_per})
    rep["кольцо записи"] = {
        "строк в записи": int(run.bins.shape[0]),
        "шагов прожито": len(run.records),
        "окно заказано, с": keep_s,
        "строк обязано быть": min(len(run.records),
                                  int(round(keep_s / (cfg.control_ms / 1000)))),
        "сошлось": int(run.bins.shape[0]) == min(
            len(run.records), int(round(keep_s / (cfg.control_ms / 1000)))),
        "запись начинается с, мс": run.timing.get("запись с, мс"),
        "счётчики за весь сеанс, а не за окно": int(run.counts.sum()),
    }
    rep["загрузка"] = cpu.since(m_cpu)
    rep["видеокарта"] = dict(ctx["gpu"].window(t0, t1), **BS.gpu_mine())
    rep["память процесса"] = {"пик, МБ": BS.rss_peak_mb(),
                              "сейчас, МБ": BS.rss_now_mb()}
    rep["оговорки прогона"] = run.timing.get("оговорки")
    return rep


# ------------------------------------------------------------------- запуск

LEVELS = {"phys": 1, "gait": 2, "lif": 3, "closed": 4, "vision": 5,
          "live": 6, "long": 7}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="*", default=["phys", "gait", "lif"],
                    help="какие уровни гнать: " + ", ".join(LEVELS))
    ap.add_argument("--steps", type=int, default=20,
                    help="шагов управления на уровень")
    ap.add_argument("--tag", default="стенд")
    ap.add_argument("--long", action="store_true", help="уровень 7")
    ap.add_argument("--минут", type=float, default=10.0)
    ap.add_argument("--окно", type=float, default=10.0)
    ap.add_argument("--счётчики", action="store_true",
                    help="только встречный замер на счётчики")
    args = ap.parse_args()

    import torch

    gpu = BS.Gpu()
    if gpu.ok:
        gpu.start()
    cpu = BS.Cpu()

    # Фон ДО замера, при простаивающем стенде. Без этого числа уровней
    # не с чем сопоставить: на этой машине одновременно считают другие
    # ветки, и «медленно» может значить «фон занял видеокарту».
    print("снимаю фон до замера (стенд простаивает)…", flush=True)
    m = cpu.mark()
    t0 = time.perf_counter()
    time.sleep(6.0)
    bg = {"процессор": cpu.since(m),
          "видеокарта": gpu.window(t0, time.perf_counter())}
    print("  фон: видеокарта %s %%, машина %s %% ядер"
          % (bg["видеокарта"].get("загрузка видеокарты, % медиана"),
             bg["процессор"].get("вся машина, %")), flush=True)

    from flybrain import data as D, sim as SIM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter()
    neurons = D.load_neurons()
    t_neu = time.perf_counter() - t
    t = time.perf_counter()
    weights = D.load_weights(device=dev)
    t_w = time.perf_counter() - t
    ctx = {"device": dev, "neurons": neurons, "weights": weights, "gpu": gpu}

    rep = {
        "метка": getattr(args, "tag"),
        "когда": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ветка": os.environ.get("FLYBRAIN_BRANCH", ""),
        "машина": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "видеокарта": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
            "ядер": os.cpu_count(),
            "нейронов": int(len(neurons)),
            "связей": int(weights._nnz()),
            "шаг движка LIF, мс": SIM.DT,
        },
        "фон до замера (стенд простаивал 6 с)": bg,
        "загрузка данных, с": {"нейроны": round(t_neu, 2),
                               "матрица связей": round(t_w, 2)},
        "каталог прогонов": os.environ.get("FLYBRAIN_OUT"),
        "уровни": {},
    }

    if getattr(args, "счётчики"):
        rep["встречный замер на счётчики"] = part_counters(ctx)
        BS.save(BENCH / ("bench7_счётчики_%s.json" % getattr(args, "tag")),
                rep)
        gpu.stop()
        return 0

    if args.long:
        print("— уровень 7: долгий сеанс песочницы —", flush=True)
        rep["уровни"]["7 long"] = level_long(
            ctx, getattr(args, "минут"), getattr(args, "окно"))
        BS.save(BENCH / ("bench7_long_%s.json" % getattr(args, "tag")), rep)
        gpu.stop()
        return 0

    for name in args.levels:
        if name not in LEVELS:
            print("нет уровня %r" % name)
            continue
        print("— уровень %d: %s —" % (LEVELS[name], name), flush=True)
        t = time.perf_counter()
        if name in ("phys", "gait", "lif"):
            got = level_bare(ctx, name, args.steps)
        else:
            got = level_loop(ctx, name, args.steps)
        got["уровень"] = LEVELS[name]
        rep["уровни"]["%d %s" % (LEVELS[name], name)] = got
        print("   шаг %s мс (медиана), хвост p95 %s, невязка %s мс (%s %% шага)"
              % (got.get("шаг целиком", {}).get("медиана"),
                 got.get("шаг целиком", {}).get("p95"),
                 got["разложение (разогретые шаги)"]["невязка"]["медиана, мс"],
                 got["разложение (разогретые шаги)"]["невязка"][
                     "медиана со знаком, % шага"]), flush=True)
        print("   счётчики сошлись: %s | за %.1f с"
              % (got["счётчики"]["всё сошлось"], time.perf_counter() - t),
              flush=True)
        BS.save(BENCH / ("bench7_%s.json" % getattr(args, "tag")), rep)

    rep["встречный замер на счётчики"] = part_counters(ctx)
    BS.save(BENCH / ("bench7_%s.json" % getattr(args, "tag")), rep)
    gpu.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
