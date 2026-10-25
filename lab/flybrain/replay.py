"""Три режима работы с сеансом и состояние, из которого его продолжают.

Режимов ровно три, и они различаются **не на слово, а по подписи
стоимости, которую нельзя подделать, не посчитав**:

=====================  ==========================  =========================
режим                  вызовов `sim.Engine.step`   чем он вообще является
=====================  ==========================  =========================
``просмотр``           0                           чтение записи с диска
``пересчёт``           200 на шаг управления        тот же счёт заново
``продолжение``        200 на шаг управления        счёт с сохранённого места
=====================  ==========================  =========================

Двести — это ``round(control_ms / sim.DT)`` при шаге управления 20 мс и
шаге движка 0,1 мс. Счётчик растёт внутри самого шага движка
(``sim.Engine.calls``), а не рядом с ним, и попадает **в саму запись**
(канал ``вызовов_движка`` таймлайна) и в проверку. Поэтому «просмотр»,
который втихую посчитал бы, виден по ненулевому счётчику, а «пересчёт»,
который втихую прочитал бы файл, — по нулевому.

У просмотра подпись сильнее счётчика: :class:`View` не держит ни движка,
ни весов, ни тела, и ``torch`` ему не нужен вовсе. Позвать движок из него
нечем.

Второе здесь — **состояние**. Продолжение сеанса с сохранённого места
имеет смысл только тогда, когда состояние полно: не «похоже», а
побитово. Состав состояния перечислен в :data:`PARTS`, по строке на
элемент, и полнота его проверяется не рассуждением, а **выбрасыванием
каждого элемента по одному** (``probe/resume_state.py``,
:func:`restore_arena` с ``drop=``). Элемент, выбрасывание которого ничего
не меняет, в состояние не входит — и узнать это можно только замером.

И главное правило, из-за которого всё это и написано: **неполное
состояние не восстанавливается, а отклоняется.** :func:`restore_arena` и
:func:`restore_body` без ``allow_incomplete=True`` отказывают с
перечислением того, чего в записи нет. Сделать вид, что состояние
восстановлено точно, и посчитать дальше — это тот же сорт выдумки, что
интерполяция между кадрами.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sim, timeline as TL

__all__ = ["MODES", "Mode", "PARTS", "Part", "Tape", "State", "View",
           "capture_arena", "restore_arena", "capture_body", "restore_body",
           "parts_of", "lif_per_step", "open_run"]


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    what: str
    lif_calls: str          # "ноль" | "шаг управления"
    how: str                # чем подпись проверяется


MODES = {m.key: m for m in (
    Mode("просмотр", "Просмотр записанного",
         "ряды и растр читаются с диска; модель не считается вовсе",
         "ноль",
         "`sim.Engine.calls` не растёт, потому что движка в этом режиме "
         "нет: `View` не импортирует torch и не держит ни весов, ни тела"),
    Mode("пересчёт", "Пересчёт заново",
         "тот же прогон считается с начала по сохранённой спеке и семени",
         "шаг управления",
         "`sim.Engine.calls` растёт на round(control_ms / sim.DT) за шаг "
         "управления — 200 при шаге 20 мс; число пишется в сам таймлайн, "
         "канал `вызовов_движка`"),
    Mode("продолжение", "Продолжение из состояния",
         "состояние ставится на место и счёт идёт дальше с того же шага",
         "шаг управления",
         "то же, что у пересчёта, плюс отказ при неполном состоянии: "
         "`restore_*` без allow_incomplete перечисляет недостающее"),
)}


# Сколько шагов управления продолжение держится ПОБИТОВО. Это замеры, а
# не обещания, и получены они `probe/resume_state.py`:
#
# * арена — 170 шагов из 170 проверенных (3,4 с модельного времени), то
#   есть расхождения не нашлось вовсе на всей проверенной длине;
# * тело — 21 шаг (420 мс), дальше расходится в последнем записанном
#   знаке. При этом ВСЕ 18 полей состояния, которые удалось назвать и
#   замерить, в миг восстановления совпадают побитово (qpos, qvel, act,
#   ctrl, mocap_pos, qacc_warmstart, sensordata, qacc, фазы и амплитуды
#   CPG, счётчики поправок, прилипание). Значит состояние сеанса в теле
#   НЕ исчерпывается этим перечнем, и чего в нём не хватает — не
#   установлено. Поэтому продолжение в теле честно называет свой
#   горизонт оговоркой, а не выдаёт себя за точное.
EXACT_STEPS = {"arena": 170, "body": 21}


def lif_per_step(control_ms=20.0):
    """Сколько вызовов движка стоит один шаг управления. Из кода, не из вкуса."""
    return max(1, int(round(float(control_ms) / sim.DT)))


# ------------------------------------------------------------- состояние


@dataclass(frozen=True)
class Part:
    """Один элемент состояния: что это, где лежит и чем он важен.

    ``needed`` — обязателен ли он для продолжения. У необязательных
    элементов отсутствие не отказ, а оговорка.
    """

    name: str
    where: str              # engine | motor | smooth | calib | pose | env | cpg | mujoco | counts
    fields: tuple
    kinds: tuple
    why: str
    needed: bool = True


PARTS = (
    Part("состояние движка", "engine", sim.Engine.STATE_TENSORS, ("arena", "body"),
         "мембрана, проводимость, спайки, рефрактерный счётчик и кольцо "
         "аксональной задержки — всё, что шаг читает и пишет"),
    Part("указатель кольца задержек", "engine", ("ptr",), ("arena", "body"),
         "положение кольца `delay_buffer`. Живёт питоновским целым и в "
         "CUDA-граф не попадает (графов 19, по одному на слот); без него "
         "продолжение читает не тот слот, и события задержки разъезжаются"),
    Part("состояние генератора", "engine", ("gen",), ("arena", "body"),
         "поток случайных чисел пуассоновского входа. Единственное место "
         "со случайностью в шаге; без него продолжение идёт по другим "
         "числам"),
    Part("маски вмешательств", "engine", sim.Engine.MASK_TENSORS,
         ("arena", "body"),
         "четыре вектора по нейрону: не выдаёт спайк, исходящие "
         "отключены, входящие отключены, мембрана удержана"),
    Part("накопитель сглаживания", "motor",
         ("_dn_l", "_dn_r", "_v", "_w", "_feed_timer"), ("arena",),
         "низкочастотный фильтр выхода популяции и сглаживание скорости "
         "и поворота. При tau 0,5 с он помнит шесть постоянных, то есть "
         "150 шагов управления"),
    Part("накопитель сглаживания", "smooth", ("el", "er"), ("body",),
         "низкочастотный фильтр выхода популяции: el, er"),
    Part("калибровка разогрева", "calib",
         ("dn_ref_hz", "asym_bias", "turn_gain_rad_s"), ("arena",),
         "опора, собственный перекос сети и усиление поворота, снятые на "
         "разогреве. Пересчитать их заново нельзя: разогрев двигает "
         "движок, то есть меняет состояние, из которого продолжают"),
    Part("калибровка разогрева", "calib",
         ("bias", "noise", "odour_ref", "dn_ref_hz"), ("body",),
         "перекос, шум, яркость насыщения и опора разогрева"),
    Part("поза", "pose", ("x", "y", "theta", "v", "omega", "eaten",
                          "feeding"), ("arena",),
         "где муха и куда смотрит"),
    Part("среда", "env", ("arena",), ("arena",),
         "кормушки и источники запаха: то, что модель читает на каждом шаге"),
    Part("среда", "env", ("sources", "slots"), ("body",),
         "слоты источника запаха: положение, сила, включённость"),
    Part("фаза CPG", "cpg", ("curr_phases", "curr_magnitudes",
                             "random_state", "retraction_persistence_counter",
                             "retraction_correction",
                             "stumbling_correction"), ("body",),
         "состояние генератора походки HybridTurningController: фазы и "
         "амплитуды шести ног и счётчики поправок"),
    Part("физика MuJoCo", "mujoco", ("mj_state",), ("body",),
         "полное состояние интегрирования (mjSTATE_INTEGRATION): время, "
         "qpos, qvel, act, warmstart, ctrl, приложенные силы, mocap"),
    Part("прилипание ног", "mujoco", ("_last_adhesion", "_active_adhesion"),
         ("body",),
         "какие ноги прилипли: Fly держит это питоновскими массивами, "
         "вне состояния MuJoCo", needed=False),
    Part("счётчики сеанса", "counts", ("counts",), ("arena", "body"),
         "накопленные спайки за весь сеанс: частоты считаются делением "
         "на длительность, и без них они поехали бы", needed=False),
)


def parts_of(kind):
    return tuple(p for p in PARTS if kind in p.kinds)


class State:
    """Снятое состояние: массивы плюс паспорт. Знает, чего в нём нет.

    Хранится двумя файлами, как и таймлайн: ``state.npz`` с массивами и
    ``state.json`` с паспортом — видом прогона, модельным временем,
    отпечатками и перечнем того, что в снимок вошло. Паспорт нужен не
    для порядка: продолжение обязано отказаться, если состояние снято с
    другого прогона, а узнать это можно только по нему.
    """

    VERSION = 1

    def __init__(self, kind, t_ms, arrays=None, meta=None, have=None,
                 allow_incomplete=False):
        self.kind = str(kind)
        self.t_ms = float(t_ms)
        self.arrays = dict(arrays or {})
        self.meta = dict(meta or {})
        self.have = set(have or ())
        # Разрешение продолжать на неполном состоянии несёт сам снимок, и
        # ставится оно только :meth:`drop` — то есть когда элемент
        # выброшен НАРОЧНО, замером полноты. Снимок, в котором элемента не
        # было изначально, такого разрешения не несёт и отклоняется: это и
        # есть правило «не утверждай точного восстановления из неполной
        # записи».
        self.allow_incomplete = bool(allow_incomplete)

    # -- полнота --------------------------------------------------------

    def missing(self):
        """Каких обязательных элементов в снимке нет."""
        return tuple(p.name + " (" + p.where + ")"
                     for p in parts_of(self.kind)
                     if p.needed and p.where not in self.have)

    def complete(self):
        return not self.missing()

    def describe(self):
        return {
            "version": self.VERSION,
            "вид": self.kind,
            "модельное, мс": self.t_ms,
            "элементы": sorted(self.have),
            "нет обязательных": list(self.missing()),
            "паспорт": self.meta,
        }

    # -- диск -----------------------------------------------------------

    def save(self, out_dir, name="state"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        flat = {}
        for where, d in self.arrays.items():
            for k, v in d.items():
                flat["%s/%s" % (where, k)] = np.asarray(v)
        np.savez_compressed(out_dir / (name + ".npz"), **flat)
        (out_dir / (name + ".json")).write_text(
            json.dumps(self.describe(), ensure_ascii=False, indent=1,
                       default=str), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, out_dir, name="state"):
        out_dir = Path(out_dir)
        sch = json.loads((out_dir / (name + ".json")).read_text(
            encoding="utf-8"))
        z = np.load(out_dir / (name + ".npz"), allow_pickle=False)
        arrays = {}
        for key in z.files:
            where, _, field = key.partition("/")
            arrays.setdefault(where, {})[field] = z[key]
        # Перечень элементов берётся из паспорта, а не из имён массивов:
        # часть элементов (среда, настройка) лежит не массивами, и
        # вывести их наличие по `npz` нельзя — снимок после чтения с
        # диска выглядел бы неполным, и продолжение отказывало бы на
        # ровном месте.
        have = sch.get("элементы")
        return cls(sch["вид"], sch["модельное, мс"], arrays,
                   sch.get("паспорт"),
                   have if have is not None else arrays.keys())

    def drop(self, *where):
        """Копия без названных элементов. Для проверки полноты.

        Нужна одному — замеру «что меняется, если элемент потерять».
        Копия, а не правка на месте: исходный снимок остаётся целым, и
        сравнивать есть с чем.
        """
        out = State(self.kind, self.t_ms,
                    {k: dict(v) for k, v in self.arrays.items()
                     if k not in where},
                    dict(self.meta),
                    [k for k in self.have if k not in where],
                    allow_incomplete=True)
        out.meta["выброшено"] = sorted(where)
        # Выбрасывание внутри движка (ptr, gen, маски) — это не раздел, а
        # поля: их надо убрать из массивов движка, иначе `engine.restore`
        # вернёт их на место, и замер ничего не измерит.
        eng = out.arrays.get("engine")
        if eng:
            for f in list(eng):
                if f in where:
                    del eng[f]
        return out


# ------------------------------------------------------- снятие: арена


def capture_arena(engine, motor, fly, arena, counts=None, t_ms=0.0,
                  meta=None):
    """Снять состояние эпизода на арене. Зовётся на границе шага.

    Снимается ровно то, что перечислено в :data:`PARTS` для вида
    ``arena``, и ничего сверх: веса, таблица нейронов и селекторы — это
    устройство прогона, а не его состояние, и лежат они в спеке.
    """
    from dataclasses import asdict

    p = motor.p
    arrays = {
        "engine": engine.snapshot(),
        "motor": {k: np.float64(0.0 if getattr(motor, k) is None
                                else getattr(motor, k))
                  for k in ("_dn_l", "_dn_r", "_v", "_w", "_feed_timer")},
        "calib": {k: np.float64(getattr(p, k))
                  for k in ("dn_ref_hz", "asym_bias", "turn_gain_rad_s")},
        "pose": {"x": np.float64(fly.x), "y": np.float64(fly.y),
                 "theta": np.float64(fly.theta), "v": np.float64(fly.v),
                 "omega": np.float64(fly.omega),
                 "eaten": np.int64(fly.eaten),
                 "feeding": np.int64(bool(fly.feeding))},
        # Копия, а не вид: `np.asarray` на массиве нужного типа копии не
        # делает, и счётчики продолжали бы расти внутри снимка.
        "counts": {"counts": np.array(
            counts.cpu().numpy() if hasattr(counts, "cpu") else
            (counts if counts is not None else np.zeros(0)),
            dtype=np.int64, copy=True)},
    }
    # Накопитель сглаживания до первого шага равен None, и это отдельное
    # состояние, а не ноль: `Body.act` на None берёт первое значение как
    # есть, без фильтра. Признак едет рядом числом.
    arrays["motor"]["свежий"] = np.int64(motor._dn_l is None)
    # Среда уходит в паспорт, а не в массивы: она маленькая и читается
    # глазами, а восстанавливать её надо целиком, а не по полям.
    info = dict(meta or {})
    info["среда"] = arena.to_dict()
    info["параметры тела"] = {k: v for k, v in asdict(p).items()}
    st = State("arena", t_ms, arrays, info,
               ["engine", "motor", "calib", "pose", "counts", "env"])
    st.meta.setdefault("шаг управления, мс", None)
    return st


def restore_arena(state, neurons, weights=None, device=None, drop=(),
                  allow_incomplete=False, graphs=True):
    """Поставить состояние арены на место. Возвращает то, чем продолжать.

    Отдаёт ``(engine, motor, fly, arena, body_params, counts, отчёт)`` —
    то, что ``loop.run_episode`` принимает параметрами ``engine=``,
    ``motor=``, ``start=``, ``arena=`` и ``body_params=``.

    Без ``allow_incomplete`` неполное состояние **отклоняется**: сделать
    вид, что продолжение точное, и посчитать дальше — выдумка. С
    ``allow_incomplete`` отказ снимается, но в отчёт ложится, чего не
    хватало; это нужно одному — замеру полноты выбрасыванием.
    """
    from . import data as D, world

    if state.kind != "arena":
        raise ValueError("это состояние вида %r, а не арены" % state.kind)
    want = set(drop)
    short = [p.name + " (" + p.where + ")" for p in parts_of("arena")
             if p.needed and (p.where not in state.have or p.where in want)]
    # Выброшенные внутри движка элементы (ptr, gen, маски) живут в том же
    # `engine`, поэтому ищутся по имени поля, а не по разделу.
    for p in parts_of("arena"):
        if p.where != "engine" or not p.needed:
            continue
        if any(f in want for f in p.fields):
            short.append(p.name + " (" + ", ".join(p.fields) + ")")
    short = sorted(set(short))
    if short and not (allow_incomplete or state.allow_incomplete):
        raise ValueError(
            "продолжение из состояния невозможно: в записи нет %s. "
            "Восстановить это нечем, и считать дальше, назвав результат "
            "продолжением, значит выдать другой прогон за тот же. Либо "
            "допишите состояние, либо пересчитайте прогон с начала"
            % "; ".join(short))

    device = device or ("cuda" if _cuda() else "cpu")
    a = state.meta.get("среда") or {}
    arena = world.Arena(
        float(a.get("width", 60.0)), float(a.get("height", 60.0)),
        [world.Food(f["x"], f["y"], f["r"]) for f in (a.get("foods") or [])],
        [world.Odour(o["x"], o["y"], o["s"], o["sigma"])
         for o in (a.get("odours") or [])])

    raw = dict(state.meta.get("параметры тела") or {})
    known = _fields(world.BodyParams)
    params = world.BodyParams(**{k: v for k, v in raw.items() if k in known})
    cal = state.arrays.get("calib") or {}
    if "calib" not in want:
        for k in ("dn_ref_hz", "asym_bias", "turn_gain_rad_s"):
            if k in cal:
                setattr(params, k, float(cal[k]))
    else:
        # Калибровку выбросили нарочно: тогда `run_episode` снимет её
        # заново на разогреве — и это и есть тот случай, который проверка
        # должна увидеть как расхождение, а не как успех.
        params.dn_ref_hz = None
        params.turn_gain_rad_s = None
        params.asym_bias = 0.0

    motor = world.Body(params)
    m = state.arrays.get("motor") or {}
    if "motor" not in want and m:
        fresh = bool(int(m.get("свежий", 0)))
        motor._dn_l = None if fresh else float(m["_dn_l"])
        motor._dn_r = None if fresh else float(m["_dn_r"])
        motor._v = float(m["_v"])
        motor._w = float(m["_w"])
        motor._feed_timer = float(m["_feed_timer"])

    po = state.arrays.get("pose") or {}
    if "pose" in want or not po:
        fly = world.FlyState(arena.width * 0.5, arena.height * 0.12,
                             float(np.radians(80.0)))
    else:
        fly = world.FlyState(float(po["x"]), float(po["y"]),
                             float(po["theta"]), float(po["v"]),
                             float(po["omega"]), int(po["eaten"]),
                             bool(int(po["feeding"])))

    sensors = np.unique(np.concatenate(
        [D.select(neurons, q) for q in world.SENSORS.values()]))
    engine = sim.build_engine(
        weights=weights, neurons=neurons, device=device,
        seed=int((state.meta.get("семя") or 0)), no_refractory=sensors,
        graphs=graphs)
    done = []
    if "engine" not in want:
        done = engine.restore(state.arrays.get("engine") or {}, drop=want)
    # Счётчик вызовов у продолжения начинается с нуля: это подпись
    # стоимости ЭТОГО отрезка, а не всей истории прогона.
    engine.calls = 0

    counts = None
    c = (state.arrays.get("counts") or {}).get("counts")
    if c is not None and len(c) and "counts" not in want:
        counts = np.asarray(c, np.int64)

    report = {
        "режим": "продолжение",
        "модельное, мс": state.t_ms,
        "восстановлено": sorted(set(done) | (
            {"motor", "calib", "pose", "env"} - want)),
        "выброшено нарочно": sorted(want),
        "не хватало в записи": short,
        "устройство": device,
    }
    return engine, motor, fly, arena, params, counts, report


# -------------------------------------------------------- снятие: тело


def capture_body(controller, engine, cfg, calib, smooth, sources, slots=None,
                 counts=None, t_ms=0.0, meta=None):
    """Снять состояние сеанса в теле, включая MuJoCo и фазу походки.

    ``mjSTATE_INTEGRATION`` взят не наугад: ``mjSTATE_PHYSICS`` не несёт
    ни времени, ни разогрева решателя (``qacc_warmstart``), ни
    управляющих сигналов, ни ``mocap_pos`` — а маркеры источников запаха
    это именно mocap-тела, и без них источник после продолжения стоял бы
    в исходном месте.
    """
    import mujoco

    phys = controller.physics
    model, data = phys.model.ptr, phys.data.ptr
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    buf = np.zeros(size, np.float64)
    mujoco.mj_getState(model, data, buf,
                       mujoco.mjtState.mjSTATE_INTEGRATION)

    cpg = controller.cpg_network
    rs = cpg.random_state.get_state()
    # `np.asarray` на массиве нужного типа копии НЕ делает, и снимок,
    # собранный им, остаётся живым видом на состояние модели: к концу
    # сеанса в нём лежат последние фазы походки, а не те, что были в
    # снятый момент. Поймано замером (`probe/resume_state.py`,
    # расхождение фаз на 6,23 рад при совпавших qpos и qvel), а не
    # рассуждением. Поэтому ниже всюду `np.array(..., copy=True)`.
    def _copy(v, dtype=np.float64):
        return np.array(v, dtype=dtype, copy=True)

    arrays = {
        "engine": engine.snapshot(),
        "mujoco": {"mj_state": _copy(buf),
                   # Разогрев решателя — отдельным массивом, хотя он и
                   # входит в mjSTATE_INTEGRATION. Причина измерена:
                   # после `mj_setState` наблюдение надо перечитать, а
                   # любое чтение зовёт `mj_forward`, и тот ПЕРЕПИСЫВАЕТ
                   # `qacc_warmstart` своим решением. Без повторной
                   # установки положение через 100 шагов физики
                   # расходилось на 0,17 мм при совпавших спайках.
                   "qacc_warmstart": _copy(data.qacc_warmstart),
                   # Показания датчиков — производная величина, и
                   # `mj_setState` их не несёт. Пересчитать их
                   # `mj_forward` недостаточно: после `mj_step` в них
                   # лежат значения ДО интегрирования, и пересчёт на
                   # восстановленной позе даёт другое число. Расхождение
                   # замерено: 4e-12 по датчикам — и этого хватает,
                   # чтобы через 21 шаг разошлись спайки, потому что
                   # показания идут в частоты рецепторов и дальше в
                   # пуассоновский вход.
                   "sensordata": _copy(data.sensordata),
                   "qacc": _copy(data.qacc),
                   "_last_adhesion": _copy(
                       getattr(controller.fly, "_last_adhesion",
                               np.zeros(6))),
                   "_active_adhesion": _copy(
                       getattr(controller.fly, "_active_adhesion",
                               np.zeros(6)))},
        "cpg": {"curr_phases": _copy(cpg.curr_phases),
                "curr_magnitudes": _copy(cpg.curr_magnitudes),
                "random_state": _copy(rs[1], np.uint32),
                "random_pos": np.int64(rs[2]),
                "random_has_gauss": np.int64(rs[3]),
                "random_cached": np.float64(rs[4]),
                "retraction_persistence_counter": _copy(
                    controller.retraction_persistence_counter),
                "retraction_correction": _copy(
                    controller.retraction_correction),
                "stumbling_correction": _copy(
                    controller.stumbling_correction)},
        "smooth": {"el": np.float64(0.0 if smooth[0] is None else smooth[0]),
                   "er": np.float64(0.0 if smooth[1] is None else smooth[1]),
                   "свежий": np.int64(smooth[0] is None)},
        # Имена канонические, а не те, что в `calib` эпизода: там перекос
        # и шум зовутся `asym_bias` и `asym_noise`, а в самом цикле —
        # `bias` и `noise`. Разойтись этим двум именам нельзя: при
        # промахе продолжение считает с перекосом ноль и шумом 1e-4, то
        # есть упирается в ±3 сигмы на первом же шаге и разводит ноги в
        # обратную сторону. Это и случилось на первом замере.
        "calib": {
            "bias": np.float64(calib.get("asym_bias",
                                         calib.get("bias", 0.0))),
            "noise": np.float64(calib.get("asym_noise",
                                          calib.get("noise", 1e-4))),
            "odour_ref": np.float64(calib.get("odour_ref", 0.0)),
            "dn_ref_hz": np.float64(calib.get("dn_ref_hz", 0.0)),
        },
        "counts": {"counts": _copy(
            counts.cpu().numpy() if hasattr(counts, "cpu")
            else (counts if counts is not None else np.zeros(0)),
            np.int64)},
    }
    info = dict(meta or {})
    info["источники"] = np.asarray(sources, float).tolist()
    info["слоты"] = list(slots or [])
    info["настройка"] = _asdict(cfg)
    st = State("body", t_ms, arrays, info,
               ["engine", "mujoco", "cpg", "smooth", "calib", "counts",
                "env"])
    return st


def restore_body(state, controller, engine, drop=(), allow_incomplete=False):
    """Поставить состояние тела на место. Возвращает (сглаживание, отчёт).

    Тело приходит уже собранным: модель MuJoCo на ходу не пересобирается
    (это условие песочницы), и состояние ставится в неё, а не вместо неё.
    """
    import mujoco

    if state.kind != "body":
        raise ValueError("это состояние вида %r, а не тела" % state.kind)
    want = set(drop)
    short = [p.name + " (" + p.where + ")" for p in parts_of("body")
             if p.needed and (p.where not in state.have or p.where in want)]
    for p in parts_of("body"):
        if p.where != "engine" or not p.needed:
            continue
        if any(f in want for f in p.fields):
            short.append(p.name + " (" + ", ".join(p.fields) + ")")
    short = sorted(set(short))
    if short and not (allow_incomplete or state.allow_incomplete):
        raise ValueError(
            "продолжение сеанса в теле невозможно: в состоянии нет %s. "
            "Продолжить без этого можно только назвав результат другим "
            "прогоном" % "; ".join(short))

    done = []
    if "engine" not in want:
        done = engine.restore(state.arrays.get("engine") or {}, drop=want)
    engine.calls = 0

    mj = state.arrays.get("mujoco") or {}
    if "mujoco" not in want and "mj_state" in mj:
        phys = controller.physics
        mujoco.mj_setState(phys.model.ptr, phys.data.ptr,
                           np.asarray(mj["mj_state"], np.float64),
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        # Пересчёт обязателен: `mj_setState` ставит позу и скорости, но
        # ускорения оставляет те, что были, и dm_control на первом же
        # шаге бракует состояние (mjWARN_BADQACC). Проверено снятием
        # этого вызова.
        mujoco.mj_forward(phys.model.ptr, phys.data.ptr)
        # Именно в этом порядке: `mj_forward` переписывает и разогрев
        # решателя, и показания датчиков своим решением, поэтому
        # сохранённые ставятся ПОСЛЕ него, а не до.
        for k in ("qacc_warmstart", "sensordata", "qacc"):
            if k in mj:
                getattr(phys.data.ptr, k)[:] = np.asarray(mj[k], np.float64)
        for k in ("_last_adhesion", "_active_adhesion"):
            if k in mj and hasattr(controller.fly, k):
                getattr(controller.fly, k)[:] = np.asarray(mj[k])
        done.append("mujoco")

    cp = state.arrays.get("cpg") or {}
    if "cpg" not in want and cp:
        cpg = controller.cpg_network
        cpg.curr_phases[:] = np.asarray(cp["curr_phases"], np.float64)
        cpg.curr_magnitudes[:] = np.asarray(cp["curr_magnitudes"], np.float64)
        cpg.random_state.set_state((
            "MT19937", np.asarray(cp["random_state"], np.uint32),
            int(cp["random_pos"]), int(cp["random_has_gauss"]),
            float(cp["random_cached"])))
        controller.retraction_persistence_counter[:] = np.asarray(
            cp["retraction_persistence_counter"], np.float64)
        controller.retraction_correction[:] = np.asarray(
            cp["retraction_correction"], np.float64)
        controller.stumbling_correction[:] = np.asarray(
            cp["stumbling_correction"], np.float64)
        done.append("cpg")

    sm = state.arrays.get("smooth") or {}
    if "smooth" in want or not sm:
        smooth = (None, None)
    else:
        fresh = bool(int(sm.get("свежий", 0)))
        smooth = ((None, None) if fresh
                  else (float(sm["el"]), float(sm["er"])))
        done.append("smooth")
    if "calib" not in want and (state.arrays.get("calib") or {}):
        # Калибровку ставит не этот вызов, а сам `body.run_episode`: она
        # читается из снимка там, где иначе шёл бы разогрев. Здесь она
        # только отмечается восстановленной, чтобы отчёт не врал.
        done.append("calib")
    report = {
        "режим": "продолжение",
        "модельное, мс": state.t_ms,
        "восстановлено": sorted(set(done)),
        "выброшено нарочно": sorted(want),
        "не хватало в состоянии": short,
    }
    return smooth, report


# ------------------------------------------------------------ лента записи


class Tape:
    """Лента: пишет таймлайн на границе шага и умеет снять состояние.

    Контракт из трёх вызовов, и все три на виду:

    * :meth:`begin` — один раз, когда движок и среда собраны;
    * :meth:`step` — на каждом шаге управления, после записи шага;
    * :meth:`finish` — один раз, в конце: проворачивает кольцо.

    Лента не считает модель и не меняет её. Единственное, что она берёт
    у движка, — счётчик вызовов: он и есть подпись стоимости режима.
    """

    def __init__(self, kind="arena", control_ms=20.0, keep_s=TL.DEFAULT_KEEP_S,
                 probe_idx=None, mode="пересчёт", t_offset_ms=0.0, meta=None,
                 capture_at_ms=None):
        if mode not in MODES:
            raise ValueError("режим %r не из списка: %s"
                             % (mode, ", ".join(MODES)))
        self.mode = mode
        self.t_offset_ms = float(t_offset_ms)
        self.tl = TL.Timeline(kind=kind, control_ms=control_ms, keep_s=keep_s,
                              probe_idx=probe_idx or {},
                              meta=dict(meta or {}, режим=mode))
        self.engine = None
        self._calls = 0
        self._wall = None
        self.capture_at_ms = (None if capture_at_ms is None
                              else float(capture_at_ms))
        self.state = None
        self._grab = None

    def begin(self, engine, grab=None):
        """Движок собран. ``grab(t_ms)`` снимает состояние, если попросят."""
        self.engine = engine
        self._calls = int(getattr(engine, "calls", 0))
        self._grab = grab
        self._wall = time.perf_counter()
        self.tl.meta["вызовов движка на шаг управления"] = lif_per_step(
            self.tl.control_ms)
        return self

    def step(self, t_ms, rec, counts=None):
        """Один шаг управления: ряд, растр и подпись стоимости."""
        now = time.perf_counter()
        calls = int(getattr(self.engine, "calls", 0))
        d_calls = calls - self._calls
        self._calls = calls
        wall_ms = 0.0 if self._wall is None else (now - self._wall) * 1000.0
        self._wall = now
        t = self.t_offset_ms + float(t_ms)
        self.tl.put(t, rec, counts=counts, lif_calls=d_calls,
                    wall_ms=wall_ms)
        if (self.capture_at_ms is not None and self.state is None
                and self._grab is not None and t >= self.capture_at_ms - 1e-9):
            self.state = self._grab(t)
        return d_calls

    def event(self, t_ms, group, what, detail="", payload=None):
        return self.tl.event(None if t_ms is None
                             else self.t_offset_ms + float(t_ms),
                             group, what, detail, payload)

    def finish(self, n_steps=None):
        self.tl.close(n_steps)
        self.tl.meta["вызовов движка всего"] = int(
            getattr(self.engine, "calls", 0))
        return self.tl


# ----------------------------------------------------- режим просмотра


class View:
    """Просмотр записанного: ноль вызовов движка, единицы миллисекунд.

    Движка здесь нет совсем — ни весов, ни тела, ни ``torch``. Это и есть
    подпись стоимости, которую невозможно подделать: позвать движок из
    этого объекта нечем. Все ответы приходят из таймлайна, и на моменты,
    которых в записи нет, он отвечает «нет данных» (:class:`~
    flybrain.timeline.Hole`), а не числом.
    """

    mode = "просмотр"

    def __init__(self, tl):
        self.tl = tl
        self.asked = 0
        self.holes = 0
        self.ms = 0.0

    @classmethod
    def open(cls, out_dir):
        return cls(TL.load(out_dir))

    @property
    def lif_calls(self):
        """Вызовов движка за всё время просмотра. Ноль по построению."""
        return 0

    def value(self, t_ms, channel):
        t0 = time.perf_counter()
        v, hole = self.tl.value(t_ms, channel)
        self.ms += (time.perf_counter() - t0) * 1000.0
        self.asked += 1
        self.holes += int(hole is not None)
        return v, hole

    def neuron(self, t_ms, neuron):
        t0 = time.perf_counter()
        v, hole = self.tl.neuron(t_ms, neuron)
        self.ms += (time.perf_counter() - t0) * 1000.0
        self.asked += 1
        self.holes += int(hole is not None)
        return v, hole

    def row(self, t_ms):
        t0 = time.perf_counter()
        out = self.tl.row(t_ms)
        self.ms += (time.perf_counter() - t0) * 1000.0
        self.asked += 1
        self.holes += int(isinstance(out, TL.Hole))
        return out

    def cost(self):
        return {"режим": self.mode, "вызовов движка LIF": self.lif_calls,
                "ответов": self.asked, "из них «нет данных»": self.holes,
                "мс на ответ": round(self.ms / max(self.asked, 1), 4)}


def open_run(out_dir):
    """Сеанс с диска в режиме просмотра, если он там есть."""
    out_dir = Path(out_dir)
    if not (out_dir / "timeline.json").exists():
        return None
    return View.open(out_dir)


# ------------------------------------------------------------- мелочи


def _cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:                                         # noqa: BLE001
        return False


def _fields(cls):
    from dataclasses import fields

    return {f.name for f in fields(cls)}


def _asdict(obj):
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                    else str(v)) for k, v in asdict(obj).items()}
    return {}
