"""Песочница целиком: сценарий из четырнадцати шагов и контроль на зерне.

Проба отвечает на два вопроса, и оба — настоящим прогоном через
работника, а не разбором объектов в своём процессе.

**Сценарий.** Муха запускается сеансом в теле без заранее заданного
конца, источник включается слева от неё, переезжает вправо без
перезапуска, выключается; дальше пауза, один шаг, продолжение,
остановка, сохранение. Проверяется не отправка команд, а то, что
работник подтвердил применение и что сенсорный вход изменился ровно
там, где его меняли.

**Контроль.** Главная ловушка этой песочницы в том, что одна
траектория всегда чем-нибудь отличается от другой: тело считается с
шумом, и любой сеанс выглядит «другим». Поэтому здесь гоняются три
коротких сеанса одинаковой длины — с источником и без него на одном
зерне и ещё один с источником на другом зерне, — и разница между
условиями сравнивается с разбросом между зёрнами. Что можно утверждать
по одной паре, а что нельзя, проба пишет словами: различие сенсорного
входа — прямое следствие включения источника, различие траектории по
одной паре эпизодов эффектом не является.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sandbox_run.py
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
rep = {"проверки": {}, "замеры": {}, "шаги": {}, "подтверждения": [],
       "контроль": {}, "находки": []}

# Шаг управления сеанса. Все подтверждения обязаны ложиться на его
# границу: шаг физики посреди вычисления не прерывается.
CONTROL_MS = 20.0

# Четыре слота модели. Их число задаётся при сборке и на ходу не
# меняется, поэтому правка всегда несёт ровно четыре записи.
SLOTS = [[12.0, 8.0, 1.5], [12.0, -8.0, 1.5],
         [-12.0, 8.0, 1.5], [-12.0, -8.0, 1.5]]

# Куда ставить источник относительно самой мухи: вперёд и в сторону.
# Поперечного относа нарочно много — иначе перекос яркости на антеннах
# тонет в том, что муха успевает повернуться между замерами.
AHEAD_MM, ASIDE_MM = 10.0, 14.0

KEYS = ("odour_l", "odour_r", "orn_l", "orn_r", "dn_l", "dn_r", "asym",
        "amp_l", "amp_r", "x", "y", "theta", "dist", "source", "spikes")


def num(value, default=0.0):
    """Число из записи шага; `None` — только настоящее отсутствие ключа.

    Нарочно не `value or default`: у половины этих ключей ноль — это
    осмысленное значение, а не пустота. `odour_l == 0` значит «не чует
    ничего», `source == 0` — «ближайший слот нулевой», и подстановка
    по `or` превращала бы и то и другое в своё отрицание.
    """
    return float(default) if value is None else float(value)


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    """Находка в `flybrain`, которую проба не правит, а называет.

    Отдельно от проверок нарочно: это не провал пробы и не её успех, а
    то, о чём читающий должен узнать, не разбирая код.
    """
    print("  замечание  " + text)
    rep["находки"].append(text)


def slots(active=None, pos=None, strength=1.0):
    """Правка источников: ровно четыре слота, включён не более одного.

    `active` — номер включаемого слота или `None` (все выключены),
    `pos` — куда его поставить; `None` значит «оставить на месте».
    """
    out = []
    for i, s in enumerate(SLOTS):
        x, y = s[0], s[1]
        if i == active and pos is not None:
            x, y = float(pos[0]), float(pos[1])
        out.append({"x": x, "y": y, "strength": strength,
                    "on": i == active})
    return out


def lateral(ex, src):
    """Относ источника поперёк оси мухи, мм; плюс — слева от неё."""
    th = num(ex.get("theta"))
    dx = float(src[0]) - num(ex.get("x"))
    dy = float(src[1]) - num(ex.get("y"))
    return -math.sin(th) * dx + math.cos(th) * dy


def aside(ex, sign):
    """Точка впереди мухи и в стороне от неё: `sign` +1 слева, -1 справа."""
    th = float(ex.get("theta") or 0.0)
    x, y = num(ex.get("x")), num(ex.get("y"))
    fx, fy = math.cos(th), math.sin(th)
    lx, ly = -math.sin(th), math.cos(th)
    return (round(x + AHEAD_MM * fx + sign * ASIDE_MM * lx, 3),
            round(y + AHEAD_MM * fy + sign * ASIDE_MM * ly, 3))


class Run:
    """Сеанс плюс приём подтверждений: оба замера живут в одном месте.

    Время прихода подтверждения запоминается в тот момент, когда оно
    пришло, а не после ожидания: иначе настенная задержка показывала бы
    длину самого ожидания, а не работы работника.
    """

    def __init__(self, ses):
        self.ses = ses
        self.applied = []
        self.arrived = {}
        self.worst_ms = 0.0

    def once(self):
        a = time.perf_counter()
        self.ses.pump(dt=0.02)
        self.worst_ms = max(self.worst_ms, (time.perf_counter() - a) * 1000.0)
        for rec in self.ses.take_applied():
            self.applied.append(rec)
            self.arrived[rec.cmd_id] = time.perf_counter()
            rep["подтверждения"].append(
                {"действие": rec.action, "модельное, мс": rec.t_model_ms,
                 "по стене, с": rec.wall_s, "исход": rec.outcome,
                 "почему": rec.why})

    def pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.once()
            time.sleep(0.01)

    def wait_ready(self, limit=240.0):
        t0 = time.perf_counter()
        while not self.ses.ready and time.perf_counter() - t0 < limit:
            self.once()
            time.sleep(0.01)
        return time.perf_counter() - t0

    def wait_state(self, state, limit=420.0):
        t0 = time.perf_counter()
        while self.ses.state != state and self.ses.error is None \
                and time.perf_counter() - t0 < limit:
            self.once()
            time.sleep(0.01)
        return time.perf_counter() - t0

    def wait_frame(self, limit=120.0):
        t0 = time.perf_counter()
        while not self.ses.extra and time.perf_counter() - t0 < limit:
            self.once()
            time.sleep(0.01)
        return bool(self.ses.extra)

    def send(self, action, payload=None):
        """Подать команду. Возвращает номер, модельное и настенное время.

        Модельное время здесь — время последнего ПОЛУЧЕННОГО кадра, оно
        отстаёт от модели на несколько шагов. Поэтому по нему проверяется
        не равенство, а то, что применение не раньше наблюдённого.
        """
        t_ms, wall = self.ses.t_ms, time.perf_counter()
        cid = self.ses.control(action, payload)
        return cid, t_ms, wall

    def wait_applied(self, cid, limit=120.0):
        end = time.perf_counter() + limit
        while time.perf_counter() < end:
            self.once()
            for r in self.applied:
                if r.cmd_id == cid:
                    return r
            time.sleep(0.01)
        return None

    def wait_model(self, t_ms, limit=120.0):
        """Дождаться кадра не раньше заданного модельного времени.

        Без этого «после» мерилось бы по кадру, снятому до применения
        правки: кадры приходят в теле на 2–2,6 Гц и отстают.
        """
        end = time.perf_counter() + limit
        while self.ses.t_ms < t_ms and time.perf_counter() < end:
            self.once()
            time.sleep(0.01)
        return self.ses.t_ms >= t_ms

    def wait_idle(self, limit=300.0):
        t0 = time.perf_counter()
        while self.ses.busy and time.perf_counter() - t0 < limit:
            self.once()
            time.sleep(0.01)
        return time.perf_counter() - t0

    def wait_summary(self, limit=180.0):
        t0 = time.perf_counter()
        while self.ses.summary is None and self.ses.error is None \
                and time.perf_counter() - t0 < limit:
            self.once()
            time.sleep(0.01)
        return time.perf_counter() - t0

    def snap(self, tag, src=None):
        ex = dict(self.ses.extra or {})
        m = {k: ex.get(k) for k in KEYS}
        m["t_ms кадра"] = self.ses.t_ms
        if src is not None:
            m["источник"] = [src[0], src[1]]
            m["поперёк оси, мм"] = round(lateral(ex, src), 2)
        rep["шаги"][tag] = m
        print("    %-26s яркость %s/%s, ORN %s/%s Гц, выход %s/%s, "
              "dist %s" % (tag, m["odour_l"], m["odour_r"], m["orn_l"],
                           m["orn_r"], m["dn_l"], m["dn_r"], m["dist"]))
        return m


def latency(r, rec, sent_ms, sent_wall, tag):
    """Задержка применения: модельная и настенная, двумя числами.

    Их нельзя путать. Модельная — сколько модельного времени прошло
    между подачей и применением; настенная — сколько прождал человек.
    При скорости тела около 0,055 от настенной один шаг управления в
    20 мс занимает по стене около трети секунды.
    """
    lag_ms = rec.t_model_ms - sent_ms
    wall = round(r.arrived.get(rec.cmd_id, time.perf_counter()) - sent_wall, 3)
    rep["замеры"].setdefault("задержки применения", {})[tag] = {
        "модельная, мс": lag_ms, "по стене, с": wall,
        "подано на кадре, мс": sent_ms, "применено, мс": rec.t_model_ms}
    return lag_ms, wall


# ------------------------------------------------- часть первая: сценарий

def part_scenario():                                          # noqa: C901
    from flybrain.ui.session import Session

    ses = Session()
    r = Run(ses)
    took = r.wait_ready()
    check("шаг 1: работник поднялся", ses.ready, "за %.1f с" % took)
    if not ses.ready:
        ses.close()
        return

    spec = {"kind": "body", "t": 0.0, "seed": 0, "open_report": False,
            "out": "probe-sandbox-run", "bin_ms": 10.0,
            "body": {"readout": "ALPN", "control_ms": CONTROL_MS,
                     "keep_s": 5.0, "odour_ref": 0.005,
                     "sources": [list(s) for s in SLOTS],
                     "live_initial": {"sources": slots(None)}}}
    job = ses.submit(spec)
    took = r.wait_state("running")
    check("шаг 1: муха пошла — сеанс в теле идёт",
          ses.state == "running" and job is not None,
          "%s за %.1f с, ошибка %s" % (ses.state, took, ses.error))
    if ses.state != "running":
        ses.close()
        return
    check("шаг 1: кадры шага доходят до окна", r.wait_frame(),
          "модельное время %.0f мс" % ses.t_ms)
    # Худший разбор почты считается от начала сеанса. Разогрев в счёт не
    # идёт: в `Hello` едут позиции всех нейронов, их распаковка занимает
    # сотни миллисекунд и ничего не говорит про отзывчивость окна.
    r.worst_ms = 0.0
    m0 = r.snap("шаг 1: первый шаг управления")
    # Первый шаг эпизода несёт яркость от арены ДО применения исходных
    # условий: `body.run_episode` читает `obs` из `body.reset` (строка
    # 363) раньше, чем `live.attach` применит `live_initial` (строка
    # 368). Это ошибка в flybrain, она названа ниже и в отчёте пробы;
    # проверка поэтому смотрит со второго шага, а утечка меряется
    # отдельно и в своих числах, а не выдаётся за успех.
    leak = max(num(m0["odour_l"]), num(m0["odour_r"]))
    rep["замеры"]["яркость первого шага при выключенных источниках"] = leak
    r.wait_model(2 * CONTROL_MS)
    m1 = r.snap("шаг 1: со второго шага управления")
    check("шаг 1: без включённых источников чуять нечего — со второго "
          "шага управления",
          num(m1["odour_l"]) == 0.0 and num(m1["odour_r"]) == 0.0
          and num(m1["dist"]) == -1.0,
          "яркость %s/%s, dist %s (на первом шаге было %.6g)"
          % (m1["odour_l"], m1["odour_r"], m1["dist"], leak))
    if leak > 0.0:
        note("flybrain/body.py, `run_episode`: наблюдение `obs` снято "
             "`body.reset(seed=...)` ДО того, как `live.attach` применил "
             "исходные условия `live_initial`, и на первом шаге эпизода "
             "яркость берётся из этого же устаревшего `obs`. Поэтому при "
             "выключенных источниках первый шаг управления всё равно "
             "чует все слоты на полной силе — здесь %.6g, что при "
             "`odour_ref` 0,005 насыщает рецепторы до потолка. Разогрев "
             "калибровки идёт на той же устаревшей яркости, то есть "
             "опора `dn_ref_hz`, перекос и шум мерятся при насыщенном "
             "обонянии, а не при тех условиях, с которых сеанс "
             "начинается. Лечится перечитыванием `obs` сразу после "
             "`live.attach`." % leak)

    # --- скорость сеанса: числом, а не словами «в реальном времени» ---
    t_a, w_a = ses.t_ms, time.perf_counter()
    r.pump(12.0)
    t_b, w_b = ses.t_ms, time.perf_counter()
    ratio = ((t_b - t_a) / 1000.0) / max(w_b - w_a, 1e-9)
    rep["замеры"]["отношение модельного к настенному"] = round(ratio, 4)
    check("скорость сеанса измерена", ratio > 0.0,
          "%.4f (%.2f с модельного за %.1f с по стене)"
          % (ratio, (t_b - t_a) / 1000.0, w_b - w_a))

    # --- шаг 2: источник слева от мухи ---
    ex = dict(ses.extra or {})
    left = aside(ex, +1.0)
    before = r.snap("шаг 2: до включения", src=left)
    cid, sent_ms, sent_wall = r.send("set", {"sources": slots(0, left)})
    check("шаг 2: источник слева от мухи заказан",
          lateral(ex, left) > 3.0,
          "поза (%.1f, %.1f) θ %.2f, источник (%.1f, %.1f), поперёк %+.1f мм"
          % (ex.get("x") or 0, ex.get("y") or 0, ex.get("theta") or 0,
             left[0], left[1], lateral(ex, left)))

    # --- шаг 3: подтверждение от работника ---
    rec = r.wait_applied(cid)
    check("шаг 3: работник подтвердил применение",
          rec is not None and rec.outcome == "applied",
          ("исход %s, %s" % (rec.outcome, rec.why)) if rec else "не пришло")
    if rec is None:
        ses.close()
        return
    lag_ms, wall = latency(r, rec, sent_ms, sent_wall, "включение")
    check("шаг 3: применено на границе шага управления",
          abs(rec.t_model_ms / CONTROL_MS
              - round(rec.t_model_ms / CONTROL_MS)) < 1e-6,
          "%.1f мс кратно %g" % (rec.t_model_ms, CONTROL_MS))
    check("шаг 3: применено не раньше наблюдённого времени, в пределах "
          "нескольких шагов", 0.0 <= lag_ms <= 20 * CONTROL_MS,
          "%+.0f мс модельного (%.1f шага), %.2f с по стене"
          % (lag_ms, lag_ms / CONTROL_MS, wall))

    # --- шаг 4: изменение сенсорного сигнала ---
    r.wait_model(rec.t_model_ms + 3 * CONTROL_MS)
    after = r.snap("шаг 4: после включения слева", src=left)
    check("шаг 4: включение изменило сенсорный вход",
          num(after["odour_l"]) > 0.0 and num(after["odour_r"]) > 0.0,
          "яркость %s/%s -> %s/%s, ORN %s/%s -> %s/%s Гц"
          % (before["odour_l"], before["odour_r"], after["odour_l"],
             after["odour_r"], before["orn_l"], before["orn_r"],
             after["orn_l"], after["orn_r"]))
    check("шаг 4: цель появилась у включённого слота",
          num(after["dist"]) > 0.0 and num(after["source"], -1) == 0.0,
          "dist %s мм, слот %s" % (after["dist"], after["source"]))
    lat1 = lateral(dict(ses.extra or {}), left)
    asy1 = num(after["odour_l"]) - num(after["odour_r"])

    # --- шаг 5: переезд вправо без перезапуска ---
    ex = dict(ses.extra or {})
    right = aside(ex, -1.0)
    cid2, sent_ms2, sent_wall2 = r.send("set", {"sources": slots(0, right)})
    rec2 = r.wait_applied(cid2)
    check("шаг 5: перемещение подтверждено",
          rec2 is not None and rec2.outcome == "applied",
          ("исход %s на %.0f мс" % (rec2.outcome, rec2.t_model_ms))
          if rec2 else "не пришло")
    if rec2 is None:
        ses.close()
        return
    lag2, wall2 = latency(r, rec2, sent_ms2, sent_wall2, "перемещение")
    check("шаг 5: сеанс не перезапускался",
          ses.state == "running" and ses.job_id == job
          and rec2.t_model_ms > rec.t_model_ms,
          "та же задача %s, модельное время %.0f -> %.0f мс"
          % (ses.job_id, rec.t_model_ms, rec2.t_model_ms))

    # --- шаг 6: снова измерить ---
    r.wait_model(rec2.t_model_ms + 3 * CONTROL_MS)
    m6 = r.snap("шаг 6: после переезда вправо", src=right)
    lat2 = lateral(dict(ses.extra or {}), right)
    asy2 = num(m6["odour_l"]) - num(m6["odour_r"])
    check("шаг 6: источник встал по другую сторону оси мухи",
          lat1 * lat2 < 0.0 and min(abs(lat1), abs(lat2)) > 3.0,
          "поперёк %+.1f -> %+.1f мм" % (lat1, lat2))
    check("шаг 6: перекос яркости следует за стороной источника",
          asy1 != 0.0 and asy2 != 0.0
          and math.copysign(1.0, asy1 * lat1)
          == math.copysign(1.0, asy2 * lat2),
          "слева-справа %+.3g при %+.1f мм, %+.3g при %+.1f мм"
          % (asy1, lat1, asy2, lat2))
    check("шаг 6: сенсорный вход изменился от переезда",
          abs(num(m6["odour_l"]) - num(after["odour_l"]))
          + abs(num(m6["odour_r"]) - num(after["odour_r"])) > 0.0,
          "яркость %s/%s -> %s/%s" % (after["odour_l"], after["odour_r"],
                                      m6["odour_l"], m6["odour_r"]))

    # --- шаг 7: отключение ---
    cid3, sent_ms3, sent_wall3 = r.send("set", {"sources": slots(None)})
    rec3 = r.wait_applied(cid3)
    check("шаг 7: отключение подтверждено",
          rec3 is not None and rec3.outcome == "applied",
          ("исход %s" % rec3.outcome) if rec3 else "не пришло")
    if rec3 is None:
        ses.close()
        return
    latency(r, rec3, sent_ms3, sent_wall3, "отключение")
    r.wait_model(rec3.t_model_ms + 3 * CONTROL_MS)
    m7 = r.snap("шаг 7: после отключения")
    check("шаг 7: выключенный источник не чует ни одна антенна",
          num(m7["odour_l"], 1.0) == 0.0 and num(m7["odour_r"], 1.0) == 0.0,
          "яркость %s/%s -> %s/%s, ORN %s/%s -> %s/%s Гц"
          % (m6["odour_l"], m6["odour_r"], m7["odour_l"], m7["odour_r"],
             m6["orn_l"], m6["orn_r"], m7["orn_l"], m7["orn_r"]))
    check("шаг 7: выключенный источник едой не считается",
          num(m7["dist"]) == -1.0 and num(m7["source"]) == -1.0,
          "dist %s, слот %s" % (m7["dist"], m7["source"]))

    # --- шаг 8: пауза ---
    cid4, _, _ = r.send("pause")
    rec4 = r.wait_applied(cid4)
    check("шаг 8: пауза подтверждена",
          rec4 is not None and rec4.outcome == "applied",
          ("на %.0f мс" % rec4.t_model_ms) if rec4 else "не пришло")
    r.pump(3.0)
    frozen = ses.t_ms
    r.pump(4.0)
    check("шаг 8: на паузе модельное время стоит",
          abs(ses.t_ms - frozen) < 1e-9,
          "%.0f мс и через четыре секунды те же %.0f мс" % (frozen, ses.t_ms))
    check("шаг 8: окно живо на паузе", r.worst_ms < 300.0,
          "худший разбор почты %.0f мс" % r.worst_ms)

    # --- шаг 9: один шаг ---
    before_step = ses.t_ms
    cid5, _, _ = r.send("step")
    rec5 = r.wait_applied(cid5)
    r.pump(5.0)
    check("шаг 9: одиночный шаг подтверждён",
          rec5 is not None and rec5.outcome == "applied",
          ("исход %s" % rec5.outcome) if rec5 else "не пришло")
    check("шаг 9: шаг сдвинул ровно на один шаг управления",
          abs((ses.t_ms - before_step) - CONTROL_MS) < 1e-6,
          "%.0f -> %.0f мс (%+.0f)"
          % (before_step, ses.t_ms, ses.t_ms - before_step))
    stepped = ses.t_ms

    # --- шаг 10: продолжение ---
    cid6, _, _ = r.send("resume")
    rec6 = r.wait_applied(cid6)
    r.pump(8.0)
    check("шаг 10: продолжение подтверждено",
          rec6 is not None and rec6.outcome == "applied",
          ("исход %s" % rec6.outcome) if rec6 else "не пришло")
    check("шаг 10: после продолжения модельное время снова растёт",
          ses.t_ms > stepped + CONTROL_MS,
          "%.0f -> %.0f мс" % (stepped, ses.t_ms))

    # --- шаг 11: остановка ---
    cid7, _, _ = r.send("stop")
    took = r.wait_idle()
    check("шаг 11: остановка довела сеанс до конца", not ses.busy,
          "за %.1f с, состояние %s" % (took, ses.state))
    rec7 = next((a for a in r.applied if a.cmd_id == cid7), None)
    check("шаг 11: остановка подтверждена работником",
          rec7 is not None and rec7.outcome == "applied",
          ("на %.0f мс модельного" % rec7.t_model_ms) if rec7 else "не пришло")

    # --- шаг 12: сохранение делает работник ---
    # Выход из цикла по `not busy` и приход сводки — разные события.
    took = r.wait_summary()
    path = (ses.summary or {}).get("path") or next(
        (p for k, p in ses.artifacts if k == "run"), None)
    out = Path(path) if path else None
    check("шаг 12: работник отдал сводку с путём прогона", out is not None,
          "за %.1f с, %s; ошибка %s" % (took, out, ses.error))
    if out is None:
        ses.close()
        return
    want = ("embodied.json", "meta.json", "rates.parquet", "bins.npz",
            "track.parquet", "spec.json", "run.json")
    missing = [f for f in want if not (out / f).exists()]
    check("шаг 12: файлы прогона легли на диск", not missing,
          "%s, нет: %s" % (out, ", ".join(missing) or "ничего"))
    rep["замеры"]["папка прогона"] = str(out)

    # --- шаг 13: журнал правок в сохранённом ---
    meta = json.loads((out / "embodied.json").read_text(encoding="utf-8"))
    tm = meta.get("timing") or {}
    rep["замеры"]["timing сохранённого"] = {
        k: v for k, v in tm.items() if k not in ("живые правки",)}
    edits = tm.get("живые правки") or []
    rep["контроль"]["журнал правок"] = edits
    kinds = sorted({e.get("что") for e in edits})
    check("шаг 13: в сохранённом есть журнал правок",
          len(edits) >= 4 and "правка" in kinds
          and any(e.get("что") == "исходные условия" for e in edits),
          "записей %d, виды %s" % (len(edits), kinds))
    check("шаг 13: сохранённый прогон несёт исходные условия",
          bool(tm.get("исходные условия")),
          str(sorted((tm.get("исходные условия") or {}))))
    # Модельное время применения есть в подтверждении работника
    # (`ipc.Applied.t_model_ms`), и окно его видит, но в журнал
    # `live.BodyLive.log` оно не попадает: запись несёт только «что»,
    # «было» и «стало». Поэтому проверяется то, что в файле ЕСТЬ, а
    # отсутствие времени называется находкой, а не выдаётся за успех.
    stamped = [e for e in edits
               if any("мс" in str(k) or "время" in str(k) or k == "t_ms"
                      for k in e)]
    check("шаг 13: модельное время правок прослеживается — "
          "в подтверждениях работника",
          all(a.t_model_ms >= 0.0 for a in r.applied if a.action == "set")
          and len([a for a in r.applied if a.action == "set"]) >= 3,
          "правок с временем в подтверждениях %d: %s"
          % (len([a for a in r.applied if a.action == "set"]),
             ", ".join("%.0f мс" % a.t_model_ms
                       for a in r.applied if a.action == "set")))
    if not stamped:
        note("flybrain/live.py, `BodyLive.apply`: запись журнала правок "
             "несёт только «что», «было» и «стало» — модельного времени "
             "в ней нет, и в `timing[\"живые правки\"]` оно не попадает "
             "(`jobs._record_edits`). По сохранённому файлу нельзя "
             "сказать, на какой миллисекунде правка применилась; это "
             "время живёт только в `ipc.Applied` и уходит в окно. "
             "Проверка выше поэтому смотрит на подтверждения работника, "
             "а не на файл.")
    rep["замеры"]["правок со временем в файле"] = len(stamped)

    # --- шаг 14: отчёт по сеансу ---
    from flybrain import body as B, report, sim

    err = ""
    html = out / "report.html"
    try:
        res = sim.load(out)
        report.render(res, html, title="песочница: сеанс в теле")
    except Exception as exc:                                  # noqa: BLE001
        err = repr(exc)[:200]
    check("шаг 14: отчёт по сеансу собрался и не упал",
          not err and html.exists() and html.stat().st_size > 10000,
          err or "%s, %d КБ" % (html.name, html.stat().st_size // 1024))

    # Папку с телом `cli.cmd_report` читает не `sim.load`, а `body.load`:
    # через `sim.load` теряются длительность по факту и имена сенсорных
    # каналов. Раз отчёт строится и так, проверяются оба пути — иначе
    # проба говорила бы про тот, которым пользуются реже.
    err2, dur = "", None
    html2 = out / "report-body.html"
    try:
        run = B.load(out)
        res2 = run.as_result()
        dur = float(res2.config.t_run_s)
        report.render(res2, html2, title="песочница: сеанс в теле (body.load)")
    except Exception as exc:                                  # noqa: BLE001
        err2 = repr(exc)[:200]
    check("шаг 14: отчёт собирается и через `body.load` — тот путь, "
          "которым идёт командная строка",
          not err2 and html2.exists() and html2.stat().st_size > 10000,
          err2 or "%d КБ, длительность по факту %s с"
          % (html2.stat().st_size // 1024,
             "—" if dur is None else "%g" % dur))

    ses.close()


# -------------------------------------------- часть вторая: контроль

def series(ses, name):
    return np.asarray(ses.series_array(name), dtype=float)


def one_episode(r, tag, seed, on, out_name, t_s=2.0):
    """Короткий сеанс конечной длины. Прочие условия у всех одни и те же."""
    ses = r.ses
    spec = {"kind": "body", "t": t_s, "seed": seed, "open_report": False,
            "out": out_name, "bin_ms": 10.0,
            "body": {"readout": "ALPN", "control_ms": CONTROL_MS,
                     "keep_s": 0.0, "odour_ref": 0.005,
                     "sources": [list(s) for s in SLOTS],
                     "live_initial": {
                         "sources": slots(0 if on else None)}}}
    job = ses.submit(spec)
    r.wait_state("running")
    if job is None or ses.state not in ("running", "saving", "done"):
        return {"tag": tag, "ошибка": "не пошёл: %s %s" % (ses.state,
                                                           ses.error)}
    took = r.wait_idle(limit=600.0)
    r.wait_summary()
    if ses.summary is None:
        return {"tag": tag,
                "ошибка": "сводки нет: %s %s" % (ses.state, ses.error)}
    ep = {"tag": tag, "зерно": seed, "источник включён": bool(on),
          "длительность, с": t_s, "по стене, с": round(took, 1),
          "путь": (ses.summary or {}).get("path"),
          "шагов записи": len(series(ses, "dn_l")),
          "odour_l": series(ses, "odour_l"), "odour_r": series(ses, "odour_r"),
          "dn_l": series(ses, "dn_l"), "dn_r": series(ses, "dn_r")}
    print("    %-22s %d шагов за %.0f с по стене" % (tag, ep["шагов записи"],
                                                     took))
    return ep


def pair(a, b, key_l, key_r, skip=1):
    """Среднее расхождение двух сеансов по каналу: обе половины вместе.

    `skip` отбрасывает первые шаги. По умолчанию один: первый шаг
    эпизода несёт яркость от арены до применения исходных условий (см.
    находку про `body.run_episode`), и сравнивать по нему — значит
    сравнивать то, чего в заказанных условиях нет.
    """
    n = min(len(a[key_l]), len(b[key_l]), len(a[key_r]), len(b[key_r]))
    if n <= skip:
        return float("nan"), 0
    d = (np.abs(a[key_l][skip:n] - b[key_l][skip:n])
         + np.abs(a[key_r][skip:n] - b[key_r][skip:n])) / 2.0
    return float(d.mean()), n - skip


def part_control():                                           # noqa: C901
    from flybrain.ui.session import Session

    ses = Session()
    r = Run(ses)
    took = r.wait_ready()
    check("контроль: работник поднялся", ses.ready, "за %.1f с" % took)
    if not ses.ready:
        ses.close()
        return

    # Четыре сеанса, а не два. Пара «с источником / без» сама по себе
    # ничего не разделяет: нужен пол шума (повтор того же зерна и того
    # же условия) и разброс между зёрнами, иначе любое расхождение
    # траекторий можно объявить действием источника.
    eps = []
    for tag, seed, on, name in (
            ("с источником, зерно 0", 0, True, "probe-sandbox-run-on"),
            ("без источника, зерно 0", 0, False, "probe-sandbox-run-off"),
            ("повтор: с источником, зерно 0", 0, True,
             "probe-sandbox-run-on-again"),
            ("с источником, зерно 1", 1, True, "probe-sandbox-run-on-s1")):
        eps.append(one_episode(r, tag, seed, on, name))
    ses.close()

    bad = [e for e in eps if e.get("ошибка")]
    check("контроль: все четыре сеанса посчитались", not bad,
          "; ".join("%s: %s" % (e["tag"], e["ошибка"]) for e in bad)
          or "четыре")
    if bad:
        return
    on0, off0, on0b, on1 = eps
    lens = [e["шагов записи"] for e in eps]
    check("контроль: сеансы одной длины — сравнивать есть что",
          len(set(lens)) == 1 and lens[0] > 50, "шагов %s" % lens)
    if min(lens) < 2:
        # Дальше идут замеры по шагам, и на пустой записи они упали бы
        # с IndexError, похоронив и остальные проверки.
        check("контроль: записи хватает на сравнение", False,
              "шагов %s — сравнивать нечего" % lens)
        return

    # Первый шаг управления отбрасывается у всех сеансов разом: он несёт
    # яркость от арены до применения исходных условий (находка про
    # `body.run_episode`). Размер утечки мерится отдельными числами.
    off_both = np.maximum(off0["odour_l"], off0["odour_r"])
    leak_off = float(off_both[0])
    leak_on = float(max(on0["odour_l"][0], on0["odour_r"][0]))
    n_leak = int(np.count_nonzero(off_both))
    off_max = float(max(off0["odour_l"][1:].max(), off0["odour_r"][1:].max()))
    on_mean = float((on0["odour_l"][1:].mean()
                     + on0["odour_r"][1:].mean()) / 2.0)
    check("контроль: утечка яркости ограничена первым шагом управления",
          n_leak <= 1,
          "ненулевых шагов яркости в сеансе без источника %d из %d, "
          "на первом %.6g" % (n_leak, len(off_both), leak_off))
    check("контроль: первый шаг либо чист, либо несёт одно и то же в "
          "обоих условиях — эффектом источника он быть не может",
          leak_off == 0.0 or abs(leak_on - leak_off) < 1e-9,
          "яркость первого шага: с источником %.6g, без источника %.6g"
          % (leak_on, leak_off))
    rep["контроль"]["утечка первого шага"] = {
        "с источником": leak_on, "без источника": leak_off,
        "ненулевых шагов без источника": n_leak}
    d_sens, n_s = pair(on0, off0, "odour_l", "odour_r")
    d_dn_cond, n_d = pair(on0, off0, "dn_l", "dn_r")
    d_dn_seed, n_z = pair(on0, on1, "dn_l", "dn_r")
    d_dn_rep, n_r = pair(on0, on0b, "dn_l", "dn_r")
    d_sens_seed, _ = pair(on0, on1, "odour_l", "odour_r")
    d_sens_rep, _ = pair(on0, on0b, "odour_l", "odour_r")
    ratio = d_dn_seed / d_dn_cond if d_dn_cond > 0 else float("inf")
    r_rep = d_dn_rep / d_dn_cond if d_dn_cond > 0 else float("inf")
    rep["контроль"]["замеры"] = {
        "шагов сравнения": n_s,
        "яркость без источника, максимум": off_max,
        "яркость с источником, средняя": on_mean,
        "сенсорный вход: условия, среднее |разн.|": d_sens,
        "сенсорный вход: повтор зерна, среднее |разн.|": d_sens_rep,
        "сенсорный вход: зёрна, среднее |разн.|": d_sens_seed,
        "выход популяции: условия, среднее |разн.|": d_dn_cond,
        "выход популяции: повтор зерна, среднее |разн.|": d_dn_rep,
        "выход популяции: зёрна, среднее |разн.|": d_dn_seed,
        "зёрна/условия": None if math.isinf(ratio) else round(ratio, 3),
        "повтор/условия": None if math.isinf(r_rep) else round(r_rep, 3)}

    check("контроль: без источника сенсорный вход ровно нулевой — "
          "со второго шага управления",
          off_max == 0.0,
          "максимум яркости за сеанс %.8g" % off_max)
    check("контроль: с источником сенсорный вход положителен",
          on_mean > 0.0, "средняя яркость %.6g" % on_mean)
    check("контроль: различие сенсорного входа — прямое следствие "
          "включения источника",
          off_max == 0.0 and on_mean > 0.0 and d_sens > 0.0,
          "среднее |разн.| %.6g на %d шагах; без источника ровно ноль, "
          "и взяться яркости больше неоткуда" % (d_sens, n_s))
    check("контроль: выход популяции различается между условиями — "
          "измерено, но это ещё не эффект",
          math.isfinite(d_dn_cond),
          "среднее |разн.| %.4g на %d шагах" % (d_dn_cond, n_d))
    check("контроль: повтор того же зерна и того же условия даёт пол шума",
          math.isfinite(d_dn_rep) and n_r > 50,
          "среднее |разн.| %.4g на %d шагах, повтор/условия %s"
          % (d_dn_rep, n_r,
             "—" if math.isinf(r_rep) else "%.2f" % r_rep))
    check("контроль: разброс между зёрнами измерен тем же способом",
          math.isfinite(d_dn_seed) and n_z > 50,
          "среднее |разн.| %.4g на %d шагах, зёрна/условия %s"
          % (d_dn_seed, n_z,
             "—" if math.isinf(ratio) else "%.2f" % ratio))
    # Повтор одного зерна при одном условии — единственный замер, который
    # отделяет «посчиталось иначе» от «условия другие». Если он нулевой,
    # прогон воспроизводим, и разницу между условиями хотя бы не съедает
    # собственный шум счёта; если нет — съедает, и это надо сказать.
    check("контроль: воспроизводимость прогона названа числом",
          math.isfinite(d_dn_rep),
          "повтор одного зерна при одном условии: %.4g (ноль — прогон "
          "повторяется в точности)" % d_dn_rep)

    if math.isinf(ratio):
        how = ("условия не дали расхождения выхода вовсе, а зёрна дали: "
               "разницы между условиями по этой паре не видно")
    elif ratio >= 0.5:
        how = ("разброс между зёрнами того же порядка, что разница между "
               "условиями (%.2f от неё) — одна пара эпизодов одно от "
               "другого не отличает" % ratio)
    else:
        how = ("разброс между зёрнами в %.2f от разницы между условиями: "
               "меньше, но одна пара всё равно не даёт отличить влияние "
               "источника от того, как легло это конкретное зерно" % ratio)
    rep_say = ("повтор того же зерна при том же условии дал ровно ноль, "
               "то есть сам счёт воспроизводим и в разницу не мешается"
               if d_dn_rep == 0.0 else
               "повтор того же зерна при том же условии дал %.4g — это "
               "пол шума самого счёта, и разницу между условиями надо "
               "мерить от него, а не от нуля" % d_dn_rep)
    verdict = (
        "Сенсорный вход: без источника яркость на антеннах ровно нулевая "
        "(максимум %.8g за сеанс, со второго шага управления), с "
        "источником — %.6g в среднем. Это прямое следствие включения "
        "источника, и утверждать это можно: выключенный слот не чует ни "
        "одна антенна, других источников яркости в сеансе нет. Выход "
        "популяции и траектория: разница между условиями %.4g, %s, "
        "разброс между зёрнами %.4g — %s. Поэтому изменение траектории "
        "по ОДНОЙ паре эпизодов эффектом называть нельзя: для этого "
        "нужна серия эпизодов на разных зёрнах и сравнение разброса "
        "внутри условия с разницей между условиями. %s"
        % (off_max, on_mean, d_dn_cond, rep_say, d_dn_seed, how,
           ("Отдельная оговорка: первый шаг управления у всех четырёх "
            "сеансов несёт яркость от арены до применения исходных "
            "условий (%.6g с источником и %.6g без) — он отброшен во "
            "всех сравнениях, и это ошибка в flybrain, а не свойство "
            "опыта." % (leak_on, leak_off)) if leak_off > 0.0 else
           ("Первый шаг управления отброшен во всех сравнениях на всякий "
            "случай, но утечки в нём нет: при выключенных источниках он "
            "чист (%.6g), то есть исходные условия действуют с нулевого "
            "шага." % leak_off)))
    rep["контроль"]["вывод"] = verdict
    print("\n  вывод контроля:\n    " + verdict.replace(". ", ".\n    "))
    check("контроль: вывод записан словами и траектория не объявлена "
          "эффектом",
          "нельзя" in verdict and "прямое следствие" in verdict,
          "вывод в отчёте пробы, %d знаков" % len(verdict))


def main():
    print("— сценарий из четырнадцати шагов —")
    part_scenario()
    print("\n— контроль: одно зерно, с источником и без —")
    part_control()
    (OUT / "sandbox_run.json").write_text(
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
