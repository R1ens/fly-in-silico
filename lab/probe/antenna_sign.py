"""Почему перекос `(левая − правая)` на идущей мухе меняет знак.

Задача пробы — не починить знак, а измерить и объяснить. На неподвижном
теле источник слева даёт более сильный сигнал левой антенне, а в двух
наблюдениях на идущей мухе при источнике слева ярче оказывалась правая.
Разнос антенн 0,17 мм, и спор идёт о величине порядка процента от самой
яркости, поэтому решается он только числами.

Цепочка, по которой яркость попадает в мозг, такая:

1. `Fly._add_odor_sensors` вешает на `LFuniculus` и `RFuniculus` по
   дочернему телу со смещением `rel_pos` из `config["olfaction"]` и ставит
   на каждое датчик `framepos` — то есть МИРОВОЕ положение этого тела;
2. `Fly.get_observation` читает четыре датчика
   (`physics.bind(self._antennae_sensors).sensordata`), складывает их в
   массив (4, 3) и отдаёт `OdorArena.get_olfaction`;
3. арена суммирует по источникам `peak * diffuse_func(расстояние)`,
   `diffuse_func` по умолчанию `x ** -2`, и приводит ответ к float32;
4. `body._odour_channels` находит столбцы антенн по имени (2 и 3), а
   `read_odour` берёт из них два числа.

Проба повторяет шаги 1–3 своей арифметикой по мировым координатам
сенсоров и сверяет с наблюдением. Порог взят из разрядности: наблюдение
приводится к float32, и предельная ошибка такого приведения — половина
единицы младшего разряда, то есть 5,96e-8 относительных. Порог вдвое
больше (`2 * eps`), а встречный замер показывает, сколько дала бы
НЕВЕРНАЯ цепочка: те же смещения сенсоров, перенесённые с телом, но не
повёрнутые вместе с ним.

Проверки знака здесь только там, где знак задан соглашением (неподвижное
тело, курс ноль). Там, где знак — предмет спора, проверяется
СОГЛАСОВАННОСТЬ: совпадает ли наблюдаемый перекос с тем, что следует из
самих координат сенсоров и источника.

    set PYTHONIOENCODING=utf-8
    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\antenna_sign.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "неподвижное тело": [],
       "ориентации и положения": [], "наклон тела": [], "динамика": []}

# Предельная относительная ошибка приведения float64 -> float32 — половина
# единицы младшего разряда. Порог берём вдвое больше: своя арифметика
# считается в float64 и расходится с аренной на единицы младшего разряда
# float64, то есть на девять порядков меньше.
EPS32 = float(np.finfo(np.float32).eps)
TOL_REL = 2.0 * EPS32

# Ниже этого уровня перекос в наблюдении — шум округления float32, и знак
# у него не определён. Такие замеры в проверках знака не участвуют, и их
# число называется отдельно.
SKEW_FLOOR = 4.0 * EPS32


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# ------------------------------------------------------------ арифметика

def field(p, sources, peaks):
    """Яркость в точке своей арифметикой: сумма peak * расстояние ** -2."""
    d = np.linalg.norm(np.asarray(p, float)[None, :] - sources, axis=1)
    return float(np.sum(peaks * d ** -2.0))


def grad(p, sources, peaks):
    """Градиент той же яркости: нужен для предсказания перекоса."""
    v = np.asarray(p, float)[None, :] - sources
    d = np.linalg.norm(v, axis=1)
    return (-2.0 * peaks[:, None] * v / d[:, None] ** 4.0).sum(axis=0)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def equiv_bearing_deg(skew, i0, d, sep):
    """Перекос, пересказанный в градусах пеленга.

    В первом порядке по разносу перекос равен ``2 * I / d * разнос *
    sin(пеленг)``, где пеленг отсчитывается от оси антенн. Обратный
    пересчёт даёт угол, на который надо было бы отвести источник в
    сторону, чтобы получить такой же перекос, — в этих градусах удобно
    сравнивать вклад курса и вклад позы.
    """
    s = skew * d / (2.0 * i0 * sep)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


# ------------------------------------------------------- тело и сенсоры

def build(sources):
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=0.1, control_ms=20.0, seed=0, device="cpu")
    sim, _ = B._build_body(cfg, np.asarray(sources, float))
    sim.reset(seed=0)
    return sim


def sensors_world(sim):
    """Мировые координаты четырёх обонятельных сенсоров.

    Тем же способом, которым их берёт `Fly.get_observation`: датчики
    `framepos`, прочитанные через `physics.bind(...).sensordata` и
    сложенные в (4, 3). Другого способа нет — именно это число уходит в
    `OdorArena.get_olfaction`.
    """
    raw = np.asarray(sim.physics.bind(sim.fly._antennae_sensors).sensordata,
                     float)
    return raw.reshape(4, 3).copy()


def free_adr(sim):
    import mujoco

    jt = np.asarray(sim.physics.model.jnt_type)
    free = int(np.flatnonzero(jt == mujoco.mjtJoint.mjJNT_FREE)[0])
    return int(sim.physics.model.jnt_qposadr[free])


def set_pose(sim, x, y, z, yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Поставить тело в заданное место и ориентацию прямо в MuJoCo.

    Свободный сустав мухи хранит мировое положение и мировой кватернион
    корневого тела, поэтому ориентация задаётся здесь, а не пересборкой
    сцены: `spawn_orientation` потребовал бы нового тела на каждый угол.
    """
    from scipy.spatial.transform import Rotation as R

    ph = sim.physics
    adr = free_adr(sim)
    q = R.from_euler("ZYX", [math.radians(yaw_deg), math.radians(pitch_deg),
                             math.radians(roll_deg)]).as_quat()  # x, y, z, w
    ph.data.qpos[adr:adr + 3] = [x, y, z]
    ph.data.qpos[adr + 3:adr + 7] = [q[3], q[0], q[1], q[2]]
    ph.data.qvel[:] = 0.0
    ph.forward()


def row_of(sim, sources, peaks, ant, tag):
    """Один замер: координаты сенсоров, своя арифметика, наблюдение."""
    from flybrain import body as B

    al, ar = ant
    obs = sim.fly.get_observation(sim)
    p = sensors_world(sim)
    oi = np.asarray(obs["odor_intensity"], float)[0]
    x, y, theta = B._pose(obs)

    pl, pr = p[al], p[ar]
    cl, cr = field(pl, sources, peaks), field(pr, sources, peaks)
    mid = (pl + pr) / 2.0
    dp = pl - pr
    sep = float(np.linalg.norm(dp))
    e = dp / sep
    pred = float(grad(mid, sources, peaks) @ dp)   # первый порядок по разносу

    # Пеленг источника: боковая составляющая направления на ближайший
    # источник в кадре антенн и азимут того же источника от курса тела.
    d_src = np.linalg.norm(sources - mid[None, :], axis=1)
    near = int(np.argmin(d_src))
    u = (sources[near] - mid) / d_src[near]
    lat = float(u @ e)
    az_src = math.degrees(math.atan2(sources[near][1] - y,
                                     sources[near][0] - x))
    ax = math.degrees(math.atan2(dp[1], dp[0]))
    return {
        "замер": tag,
        "x": round(x, 4), "y": round(y, 4),
        "курс, °": round(math.degrees(theta), 3),
        "лев. антенна": [round(float(v), 6) for v in pl],
        "прав. антенна": [round(float(v), 6) for v in pr],
        "разнос, мм": round(sep, 6),
        "ось антенн, °": round(ax, 3),
        "отклонение оси от курса+90°, °": round(
            wrap_deg(ax - math.degrees(theta) - 90.0), 3),
        "яркость L ариф.": cl, "яркость R ариф.": cr,
        "яркость L obs": float(oi[al]), "яркость R obs": float(oi[ar]),
        "перекос obs": float(oi[al]) - float(oi[ar]),
        "перекос ариф.": cl - cr,
        "перекос 1-го порядка": pred,
        "боковая на источник": round(lat, 8),
        "пеленг от курса, °": round(wrap_deg(az_src - math.degrees(theta)), 3),
        "до источника, мм": round(float(d_src[near]), 4),
        "невязка L": abs(cl - float(oi[al])) / max(cl, 1e-300),
        "невязка R": abs(cr - float(oi[ar])) / max(cr, 1e-300),
        "_p": p, "_mid": mid,
        "_thorax": np.asarray(obs["fly"], float)[0].copy(),
    }


def pub(row):
    """Строка таблицы без служебных полей и с разумным числом знаков."""
    out = {}
    for k, v in row.items():
        if k.startswith("_"):
            continue
        out[k] = (float("%.10g" % v) if isinstance(v, float) else v)
    return out


def small_skew(r):
    """Перекос ниже разрядности float32: знак у него не определён."""
    return abs(r["перекос obs"]) <= SKEW_FLOOR * r["яркость L obs"]


def sign_misfits(rows):
    """Замеры, где знак перекоса разошёлся с геометрией координат."""
    bad, used, small = [], 0, 0
    for r in rows:
        if small_skew(r):
            small += 1
            continue
        used += 1
        if math.copysign(1.0, r["перекос obs"]) != math.copysign(
                1.0, r["перекос 1-го порядка"]):
            bad.append(r["замер"])
    return bad, used, small


# --------------------------------------- 1. неподвижное тело: три места

def part_fixed(store):
    """Слева, справа и по центру — на теле в позе сброса, курс ноль."""
    from flybrain import body as B

    peaks = np.ones(1)
    places = {
        "источник слева": np.array([[0.0, 25.0, 1.5]]),
        "источник справа": np.array([[0.0, -25.0, 1.5]]),
        "источник по центру (прямо вперёд)": np.array([[25.0, 0.0, 1.5]]),
    }
    got = {}
    for tag, src in places.items():
        sim = build(src)
        ant = B._odour_channels(sim.fly)
        r = row_of(sim, src, peaks, ant, tag)
        got[tag] = r
        rep["неподвижное тело"].append(pub(r))
        store["rows"].append(r)
        if tag == "источник слева":
            # Смещения антенн от груди в позе сброса при курсе ноль. Ими
            # считается «жёсткая» оценка в динамике: где стояли бы
            # сенсоры, если бы их разворачивал только курс.
            store["off"] = (r["_p"][ant[0]] - r["_thorax"],
                            r["_p"][ant[1]] - r["_thorax"])
            # Столбцы: антенны, а не щупики. Разнос антенн 0,17 мм,
            # щупиков — 0,60 мм, и прежняя ошибка (индексы 0 и 1) была
            # именно в этом. Замер прямой: берём оба разноса из модели.
            names = list(sim.fly.config["olfaction"]["sensor_positions"])
            pal = [i for i, n in enumerate(names) if "palp" in n.lower()]
            sep_ant = r["разнос, мм"]
            sep_pal = float(np.linalg.norm(r["_p"][pal[0]] - r["_p"][pal[1]]))
            rep["замеры"]["разнос сенсоров, мм"] = {
                "антенны": sep_ant, "щупики": round(sep_pal, 6)}
            check("столбцы яркости — антенны, а не щупики",
                  ant == (2, 3) and abs(sep_ant - 0.1713) < 0.002
                  and abs(sep_pal - 0.600) < 0.01,
                  "столбцы %s, разнос антенн %.4f мм против %.4f мм у "
                  "щупиков (имена %s)" % (str(ant), sep_ant, sep_pal,
                                          ", ".join(names)))
            check("своя арифметика считается по тем же источникам, что арена",
                  np.array_equal(np.asarray(sim.arena.odor_source, float), src)
                  and np.array_equal(
                      np.asarray(sim.arena.peak_odor_intensity,
                                 float).ravel(), peaks),
                  "odor_source %s, peak %s"
                  % (sim.arena.odor_source.tolist(),
                     sim.arena.peak_odor_intensity.ravel().tolist()))
        sim.close()

    left = got["источник слева"]
    right = got["источник справа"]
    mid = got["источник по центру (прямо вперёд)"]
    rep["замеры"]["неподвижное тело, перекос"] = {
        t: float("%.6g" % got[t]["перекос obs"]) for t in got}
    rep["замеры"]["неподвижное тело, перекос в долях яркости"] = {
        t: float("%.6g" % (got[t]["перекос obs"] / got[t]["яркость L obs"]))
        for t in got}

    # Знак здесь задан соглашением и проверяется прямо: это та опора, от
    # которой считается расхождение. Провалится, если лево и право в
    # цепочке переставлены.
    check("на неподвижном теле источник слева ярче левой антенне",
          left["перекос obs"] > 0.0 and right["перекос obs"] < 0.0,
          "слева %+.6g (%+.3f %% яркости), справа %+.6g (%+.3f %%)"
          % (left["перекос obs"],
             100.0 * left["перекос obs"] / left["яркость L obs"],
             right["перекос obs"],
             100.0 * right["перекос obs"] / right["яркость L obs"]))
    # Источник прямо впереди: перекос обязан утонуть в разрядности
    # float32. Порог не с потолка — это четыре единицы младшего разряда
    # от самой яркости. Боковой источник даёт в сотни раз больше.
    floor = SKEW_FLOOR * mid["яркость L obs"]
    check("источник по центру не даёт перекоса выше разрядности",
          abs(mid["перекос obs"]) <= floor,
          "перекос %+.3g при пороге %.3g; боковой источник на том же "
          "расстоянии даёт %+.3g, то есть в %.0f раз больше порога"
          % (mid["перекос obs"], floor, left["перекос obs"],
             abs(left["перекос obs"]) / floor))


# ------------------------- 2. ориентации тела и положения в арене

YAWS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
PLACES = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (-15.0, -8.0)]


def part_orient(store):
    """Один источник слева по миру, тело — восемь курсов, четыре места.

    Встречный замер считается тут же: та же арифметика, но по смещениям
    сенсоров, которые перенесли вместе с телом и забыли повернуть. Так
    выглядела бы ошибка преобразования, и видно, насколько проверка на
    совпадение к ней чувствительна.
    """
    from flybrain import body as B

    src = np.array([[0.0, 25.0, 1.5]])
    peaks = np.ones(1)
    sim = build(src)
    ant = B._odour_channels(sim.fly)
    z0 = float(sim.physics.data.qpos[free_adr(sim) + 2])

    base = row_of(sim, src, peaks, ant, "курс 0")
    p0, mid0 = base["_p"], base["_mid"]

    worst_naive = 0.0
    rows = []
    for (x, y) in PLACES:
        for yaw in YAWS:
            set_pose(sim, x, y, z0, yaw)
            r = row_of(sim, src, peaks, ant,
                       "x=%.0f y=%.0f курс %.0f°" % (x, y, yaw))
            rows.append(r)
            store["rows"].append(r)
            rep["ориентации и положения"].append(pub(r))
            # Встречное предположение: сенсоры перенесли, но не повернули.
            shift = r["_mid"] - mid0
            for i in (ant[0], ant[1]):
                naive = field(p0[i] + shift, src, peaks)
                real = field(r["_p"][i], src, peaks)
                worst_naive = max(worst_naive,
                                  abs(naive - real) / max(real, 1e-300))
    sim.close()

    rep["замеры"]["встречный замер: сенсоры не повёрнуты вместе с телом"] = {
        "худшая относительная невязка": float("%.4g" % worst_naive),
        "порог совпадения": float("%.4g" % TOL_REL),
    }
    store["naive"] = worst_naive

    # Знак здесь не требуется, требуется согласованность: наблюдаемый
    # перекос обязан совпасть по знаку с тем, что следует из самих
    # координат сенсоров и источника (перекос в первом порядке по
    # разносу). Провалится, если знак берётся не из геометрии.
    bad, used, small = sign_misfits(rows)
    check("знак перекоса на неподвижном теле следует из координат сенсоров",
          used >= 24 and not bad,
          "совпал на %d замерах из %d (%d ниже разрядности); расхождения: %s"
          % (used - len(bad), used, small, ", ".join(bad) or "нет"))

    # Тот же набор, пересказанный через курс: при источнике слева по миру
    # знак перекоса определяется тем, куда смотрит муха, а не тем, где
    # источник в арене. Замеры, где источник почти строго впереди или
    # строго позади (|пеленг| < 5° или > 175°), в счёт не идут: там
    # боковая составляющая сама ниже разрешения.
    bad2, used2 = [], 0
    for r in rows:
        b = abs(r["пеленг от курса, °"])
        if small_skew(r) or b < 5.0 or b > 175.0:
            continue
        used2 += 1
        if math.copysign(1.0, r["перекос obs"]) != math.copysign(
                1.0, r["пеленг от курса, °"]):
            bad2.append((r["замер"], r["пеленг от курса, °"],
                         r["перекос obs"]))
    check("перекос и пеленг источника от курса одного знака",
          used2 >= 24 and not bad2,
          "совпал на %d замерах из %d (прочие — источник почти по курсу); "
          "курс 0°: пеленг %+.1f°, перекос %+.3g; курс 180°: пеленг "
          "%+.1f°, перекос %+.3g"
          % (used2 - len(bad2), used2,
             rows[0]["пеленг от курса, °"], rows[0]["перекос obs"],
             rows[4]["пеленг от курса, °"], rows[4]["перекос obs"])
          if not bad2 else "расхождения: %s" % (bad2[:4],))

    by_yaw = {}
    for r in rows:
        if r["замер"].startswith("x=0 y=0"):
            by_yaw[int(round(r["курс, °"])) % 360] = r
    rep["замеры"]["перекос по курсу в начале арены, источник слева"] = {
        "%d°" % k: float("%.6g" % by_yaw[k]["перекос obs"])
        for k in sorted(by_yaw)}
    store["by_yaw"] = by_yaw


# --------------------------------- 3. наклон тела: крен и тангаж

TILTS = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0), (10.0, 10.0),
         (20.0, 20.0), (-10.0, 10.0), (-20.0, 20.0)]


def part_tilt(store):
    """Сколько перекоса даёт сама поза: источник строго впереди.

    При курсе ноль и источнике прямо по курсу боковая составляющая
    направления на источник равна нулю, и всё, что остаётся в перекосе, —
    это вклад наклона тела. Его и меряем, а потом переводим в градусы
    пеленга: столько же перекоса дал бы источник, отведённый в сторону на
    этот угол.
    """
    from flybrain import body as B

    src = np.array([[25.0, 0.0, 1.5]])
    peaks = np.ones(1)
    sim = build(src)
    ant = B._odour_channels(sim.fly)
    z0 = float(sim.physics.data.qpos[free_adr(sim) + 2])

    rows = []
    for roll, pitch in TILTS:
        set_pose(sim, 0.0, 0.0, z0, 0.0, pitch_deg=pitch, roll_deg=roll)
        r = row_of(sim, src, peaks, ant,
                   "крен %+.0f° тангаж %+.0f°" % (roll, pitch))
        r["равносильный пеленг, °"] = round(equiv_bearing_deg(
            r["перекос ариф."], r["яркость L ариф."],
            r["до источника, мм"], r["разнос, мм"]), 4)
        rows.append(r)
        store["rows"].append(r)
        rep["наклон тела"].append(pub(r))
    sim.close()

    flat = rows[0]
    worst = max(rows[1:], key=lambda r: abs(r["равносильный пеленг, °"]))
    rep["замеры"]["наклон тела: равносильный пеленг, °"] = {
        r["замер"]: r["равносильный пеленг, °"] for r in rows}
    store["tilt_worst"] = worst

    # Проверяется не знак, а то, что наклон вообще сдвигает перекос, и на
    # сколько: при строго переднем источнике у ровного тела перекос лежит
    # в разрядности, а у наклонённого — выше неё. Провалится, если наклон
    # на перекос не влияет (тогда объяснение «поза разворачивает голову»
    # было бы пустым) либо если он влияет и на ровном теле (тогда
    # неверной была бы опора всего сравнения).
    floor = SKEW_FLOOR * flat["яркость L obs"]
    check("наклон тела сдвигает перекос, ровное тело — нет",
          abs(flat["перекос obs"]) <= floor
          and abs(worst["перекос obs"]) > 10.0 * floor,
          "ровное тело %+.3g при пороге %.3g; %s даёт %+.3g, это "
          "равносильно пеленгу %+.2f°"
          % (flat["перекос obs"], floor, worst["замер"],
             worst["перекос obs"], worst["равносильный пеленг, °"]))


# ------------------------------------------- 4. идущая муха, по шагам

DYN_SRC = np.array([[25.0, 12.0, 1.5]])
DYN_S = 4.0


def part_dynamic(neurons, store):                             # noqa: C901
    """Эпизод в теле: на каждом шаге поза, координаты сенсоров, сигналы.

    Снимок берётся из `on_step`, то есть после того, как шаг управления
    досчитан. Этот же снимок — начало следующего шага, поэтому яркость из
    него обязана совпасть с той, что эпизод записал в СЛЕДУЮЩЕЙ строке
    (`odour_l`, `odour_r`): так проверяется, что `read_odour` берёт те же
    столбцы в тот же момент, а не другие и не на шаг раньше.
    """
    import torch

    from flybrain import body as B, data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = D.load_weights(device=device)
    peaks = np.ones(1)
    cfg = B.BodyConfig(duration_s=DYN_S, control_ms=20.0, seed=0,
                       device=device, readout="ALPN")
    holder, rows = {}, []

    def on_body(body):
        holder["body"] = body
        holder["ant"] = B._odour_channels(body.fly)

    def on_step(step, rec, _row):
        r = row_of(holder["body"], DYN_SRC, peaks, holder["ant"],
                   "шаг %d" % step)
        r["шаг"] = int(step)
        r["t, мс"] = float(rec["t_ms"])
        r["записано L"] = float(rec["odour_l"])
        r["записано R"] = float(rec["odour_r"])
        rows.append(r)

    print("  эпизод %.1f с модельного времени на %s …" % (DYN_S, device),
          flush=True)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        sources=DYN_SRC, progress=False,
                        on_body=on_body, on_step=on_step)
    rep["замеры"]["эпизод, с по стене"] = round(time.perf_counter() - t0, 1)
    rep["замеры"]["эпизод, шагов"] = len(rows)
    store["rows"].extend(rows)
    for r in rows:
        rep["динамика"].append(pub(r))

    check("эпизод прошёл и дал запись по шагам",
          len(rows) == int(round(DYN_S / 0.02))
          and len(run.records) == len(rows),
          "шагов %d, строк записи %d, путь %.1f мм"
          % (len(rows), len(run.records), run.metrics.get("path_mm", -1)))
    if not rows:
        return

    # --- снимок совпал с тем, что записал эпизод ---
    #
    # Запись округлена до седьмого знака, значит предельное расхождение —
    # половина единицы последнего разряда, 5e-8. Порог ровно оттуда.
    worst_rec, n_rec = 0.0, 0
    for a, b in zip(rows[:-1], rows[1:]):
        n_rec += 1
        worst_rec = max(worst_rec,
                        abs(a["яркость L ариф."] - b["записано L"]),
                        abs(a["яркость R ариф."] - b["записано R"]))
    rep["замеры"]["динамика: расхождение с записью эпизода"] = float(
        "%.3g" % worst_rec)
    check("яркость из координат совпала с записью эпизода",
          n_rec > 0 and worst_rec <= 5.1e-8,
          "худшее расхождение %.2g при пределе округления записи 5,0e-8 "
          "(на %d парах шагов)" % (worst_rec, n_rec))

    # --- цепочка преобразования точна на каждом шаге ---
    worst = max(max(r["невязка L"], r["невязка R"]) for r in rows)
    rep["замеры"]["динамика: худшая невязка арифметики и obs"] = float(
        "%.4g" % worst)
    check("на идущей мухе арифметика по координатам совпала с obs",
          worst <= TOL_REL,
          "худшая относительная невязка %.3g при пороге %.3g (две единицы "
          "младшего разряда float32); неповёрнутые сенсоры дали бы %.3g"
          % (worst, TOL_REL, store.get("naive", float("nan"))))

    # --- знак перекоса следует из координат на каждом шаге ---
    bad, used, small = sign_misfits(rows)
    check("знак перекоса на идущей мухе следует из координат сенсоров",
          used > 0.8 * len(rows) and not bad,
          "совпал на %d шагах из %d (%d ниже разрядности); расхождения: %s"
          % (used - len(bad), used, small,
             ", ".join(str(b) for b in bad) or "нет"))

    # --- смена знака совпадает со сменой стороны источника ---
    #
    # Это и есть ответ на вопрос: перекос переворачивается ровно там, где
    # источник переходит с одной стороны оси антенн на другую. Сравнение
    # пошаговое, поэтому подогнать нечего.
    sign_sk = [math.copysign(1.0, r["перекос obs"]) for r in rows]
    sign_lat = [math.copysign(1.0, r["боковая на источник"]) for r in rows]
    flips_sk = [r["шаг"] for i, r in enumerate(rows)
                if i and sign_sk[i] != sign_sk[i - 1]]
    flips_lat = [r["шаг"] for i, r in enumerate(rows)
                 if i and sign_lat[i] != sign_lat[i - 1]]
    rep["замеры"]["динамика: смены знака"] = {
        "перекос, на шагах": flips_sk,
        "сторона источника, на шагах": flips_lat,
        "шагов с отрицательным перекосом": sum(1 for s in sign_sk if s < 0),
        "шагов с положительным перекосом": sum(1 for s in sign_sk if s > 0),
    }
    check("смена знака перекоса шаг в шаг совпала со сменой стороны "
          "источника в кадре антенн",
          flips_sk == flips_lat,
          "смен знака перекоса %d, смен стороны %d, шаги %s против %s"
          % (len(flips_sk), len(flips_lat), flips_sk[:8], flips_lat[:8]))

    # --- что в положении сенсоров даёт курс, а что остальная поза ---
    #
    # «Жёсткая» оценка: сенсоры стоят в тех же смещениях от груди, что в
    # позе сброса, и разворачивает их только курс. Разница с
    # действительным положением, пересказанная в градусах пеленга, и есть
    # вклад позы — наклона тела, шейных и антеннальных суставов.
    off_l, off_r = store["off"]
    d_rig = []
    for r in rows:
        th = math.radians(r["курс, °"])
        c, s = math.cos(th), math.sin(th)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pl = r["_thorax"] + rz @ off_l
        pr = r["_thorax"] + rz @ off_r
        il, ir = field(pl, DYN_SRC, peaks), field(pr, DYN_SRC, peaks)
        m = (pl + pr) / 2.0
        b_rig = equiv_bearing_deg(
            il - ir, il, float(np.linalg.norm(DYN_SRC[0] - m)),
            float(np.linalg.norm(pl - pr)))
        b_real = equiv_bearing_deg(
            r["перекос ариф."], r["яркость L ариф."],
            r["до источника, мм"], r["разнос, мм"])
        r["пеленг по перекосу, °"] = round(b_real, 4)
        r["пеленг жёсткой оценки, °"] = round(b_rig, 4)
        r["вклад позы, ° пеленга"] = round(b_real - b_rig, 4)
        d_rig.append(b_real - b_rig)

    dev = [r["отклонение оси от курса+90°, °"] for r in rows]
    yaws = [r["курс, °"] for r in rows]
    bear = [r["пеленг от курса, °"] for r in rows]
    seps = [r["разнос, мм"] for r in rows]
    rep["замеры"]["динамика: поза"] = {
        "курс, ° от": round(min(yaws), 2), "курс, ° до": round(max(yaws), 2),
        "пеленг от курса, ° от": round(min(bear), 2),
        "пеленг от курса, ° до": round(max(bear), 2),
        "отклонение оси антенн от курса+90°, ° от": round(min(dev), 3),
        "отклонение оси антенн от курса+90°, ° до": round(max(dev), 3),
        "разнос антенн, мм от": round(min(seps), 5),
        "разнос антенн, мм до": round(max(seps), 5),
        "вклад позы в пеленг, ° от": round(min(d_rig), 3),
        "вклад позы в пеленг, ° до": round(max(d_rig), 3),
        "вклад позы в пеленг, ° по модулю средний": round(
            float(np.mean(np.abs(d_rig))), 3),
    }
    # Разрешение этого сравнения считается из разрядности: перекос ниже
    # SKEW_FLOOR * яркость не различим, а в градусах пеленга это
    # floor * d / (2 * I * разнос). Порог ставим в пятьдесят раз выше
    # найденного разрешения. Второе условие — про сам разнос антенн: он
    # записан с шагом 1e-6 мм, и значимым размахом считается сотня таких
    # шагов. Провалится, если положение сенсоров — жёсткая функция курса:
    # тогда вклад позы утонет в разрешении, разнос окажется постоянным, и
    # объяснение «поза разворачивает голову» будет пустым.
    r0 = rows[0]
    res_deg = math.degrees(SKEW_FLOOR * r0["до источника, мм"]
                           / (2.0 * r0["разнос, мм"]))
    rep["замеры"]["разрешение пеленга по разрядности float32, °"] = float(
        "%.3g" % res_deg)
    worst_pose = max(abs(v) for v in d_rig)
    check("поза, а не только курс, задаёт положение сенсоров",
          worst_pose > 50.0 * res_deg
          and max(seps) - min(seps) > 1e-4,
          "вклад позы до %+.2f° пеленга (среднее по модулю %.2f°) при "
          "разрешении %.4f°; ось антенн отходит от курса+90° на %+.2f…%+.2f°, "
          "разнос гуляет %.5f…%.5f мм (размах %.2g при пороге 1e-4)"
          % (max(d_rig, key=abs), float(np.mean(np.abs(d_rig))), res_deg,
             min(dev), max(dev), min(seps), max(seps),
             max(seps) - min(seps)))

    store["dyn"] = rows


# ------------------------------------------------------------------ свод

def main():
    store = {"rows": []}
    print("— 1. неподвижное тело: слева, справа, по центру —")
    part_fixed(store)
    print("— 2. восемь курсов и четыре места в арене —")
    part_orient(store)
    print("— 3. наклон тела при источнике строго впереди —")
    part_tilt(store)
    print("— 4. идущая муха, по шагам —")
    from flybrain import data as D

    neurons = D.load_neurons()
    part_dynamic(neurons, store)

    # Общая невязка по всем замерам: цепочка «координаты сенсоров ->
    # формула поля -> столбцы obs» либо точна везде, либо ошибочна.
    worst = max(max(r["невязка L"], r["невязка R"]) for r in store["rows"])
    rep["замеры"]["худшая невязка по всем замерам"] = float("%.4g" % worst)
    check("цепочка преобразования точна на всех замерах пробы",
          worst <= TOL_REL,
          "замеров %d, худшая относительная невязка %.3g при пороге %.3g; "
          "неповёрнутые сенсоры дали бы %.3g"
          % (len(store["rows"]), worst, TOL_REL,
             store.get("naive", float("nan"))))

    (OUT / "antenna_sign.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
