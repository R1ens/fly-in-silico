"""Пятое пространство «Песочница»: рисуется в простое и не ломает прочие.

Проверяется ровно то, что ломается первым. Песочница живёт на сеансе, но
в окне она появляется раньше любого сеанса: кадров MuJoCo нет, позы
камеры нет, подтверждений нет, журнал пуст. Это и есть главный случай
падения, и именно он здесь главный.

Работник не поднимается нарочно: окно обязано рисоваться и без него, а
видеокарту в это время может занимать настоящий сеанс. Поэтому сеанс
здесь поддельный — тот же `probe.app_shot.FakeSession`, дополненный
управляющей частью, которой песочница спрашивает подтверждения.

Второе, что проверяется, — целость остальных четырёх пространств:
`App.frame` разбирает ответы ВСЕХ пространств на каждом кадре, в каком
бы из них мы ни были, поэтому сломанная песочница валила бы и «Тело», и
«Мозг».

Время кадра меряется после разогрева: первые кадры уходят на сборку
ядер Taichi, и считать их временем рисования нельзя.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sandbox_shot.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"снимки": [], "проверки": {}, "замеры": {}, "трассировки": {},
       "заметки": []}

# Замечено по ходу и НЕ правлено: правка в `flybrain/` в задачу не
# входила. Пишется в отчёт, чтобы не потерялось. Номера строк — на
# ws_sandbox.py в 792 строки; файл в это время правили рядом, так что
# смотреть стоит по имени, а не по номеру.
NOTES = (
    "flybrain/desk/ws_sandbox.py:128 — `answers` зовёт `ses.take_applied()` "
    "без проверки, а `App.frame` (app.py:292) разбирает ответы ВСЕХ "
    "пространств на каждом кадре. Значит сеанс без управляющей части валит "
    "не песочницу, а любое пространство. Этим уже сломаны пробы, которые "
    "берут `probe/app_shot.py:FakeSession`: ana_shot, app_shot, app_states, "
    "bodyfinal_dpi, bodyfinal_events, body_fixes, integrity — все падают с "
    "AttributeError: 'FakeSession' object has no attribute 'take_applied'. "
    "Там же `_meters` читает `ses._born` и `ses.sent` так же без проверки.",
    "flybrain/desk/ws_sandbox.py:624 — «по стене» считается как "
    "`perf_counter() - ses._born`, то есть от создания сеанса, а не от "
    "начала сеанса песочницы. В простое это возраст окна: на снимке "
    "sandbox-idle-1600x900.png рядом с «модельное 0,00 с» и «состояние "
    "простой» стоит «по стене 0,9 с», и число растёт само. Либо «—», пока "
    "сеанса не было.",
    "flybrain/desk/app.py:360 — при ширине около 900 px переключатель "
    "пространств не рисуется вовсе (`if nav.w >= wide`), и сменить "
    "пространство мышью нечем: другого органа в окне нет. Видно на "
    "sandbox-900x700.png.",
    "flybrain/desk/app.py:696 — подпись группы «ЧТО ДАЛЬШЕ» рисуется "
    "всегда, а текст под ней — только при ширине больше 140 px. На "
    "900×700 остаётся пустая подписанная колонка.",
)

SIZES = ((1600, 900), (1280, 720), (900, 700))


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


class SandboxSession:
    """Поддельный сеанс с управляющей частью.

    `probe.app_shot.FakeSession` знает только про прогон, а песочница
    спрашивает ещё и подтверждения: `take_applied`, `sent`, `_born`,
    `busy`, `control`. Простой — это когда все они пусты, и именно так
    они здесь и заведены.
    """

    def __init__(self, ready=False):
        from probe.app_shot import FakeSession

        self._base = FakeSession()
        self.alive = bool(ready)
        self.ready = bool(ready)
        self.busy = False
        self.extra = {}
        self.t_ms = 0.0
        self.body_frame = None
        self.body_seq = 0
        self.sent = []
        self.applied = []
        self._born = time.perf_counter()
        self.state = "idle"
        self.phase = ""
        self.progress = 0.0
        self.job_id = None
        self.hello = None
        self.error = None
        self.summary = None
        self.boot_phase = "коннектом"
        self.boot_progress = 0.0 if ready else 0.35
        self.series = {}
        self.log = []
        self.act = None

    def pump(self, dt=None):
        return []

    def ask(self, what, arg=None):
        return ""

    def submit(self, spec):
        raise AssertionError("проба не запускает расчётов")

    def cancel(self):
        pass

    def control(self, action, payload=None, at_t_ms=None):
        raise AssertionError("проба не правит идущий сеанс")

    def take_applied(self):
        got, self.applied = self.applied, []
        return got

    def n_active(self):
        return 0

    def geometry(self):
        return None

    def close(self):
        pass


def main():                                                   # noqa: C901
    from flybrain.desk import layout as L
    from flybrain.desk import ws_sandbox as SB
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.desk.state import WORKSPACES
    from flybrain.desk.tokens import C

    keys = [k for k, _n in WORKSPACES]
    names = dict(WORKSPACES)

    sh = Offscreen(SIZES[0])
    ses = SandboxSession(ready=False)
    app = App(sh, ses)
    check("окно открылось в простое", (sh.w, sh.h) == SIZES[0],
          "%d×%d, масштаб %.2f" % (sh.w, sh.h, sh.sc.k))
    check("пять пространств в заявленном порядке",
          keys == ["experiment", "brain", "body", "sandbox", "analysis"],
          ", ".join(keys))
    check("у «Песочницы» три закладки",
          SB.TABS == ("Показания", "События", "Команды"),
          ", ".join(SB.TABS))

    # --- простое: сеанса не было, и это видно по состоянию ---
    sb = app.st.sb
    check("простой настоящий: ни кадра, ни позы, ни подтверждений",
          ses.body_frame is None and sb.cam is None
          and not sb.applied_slots and not sb.applied_stim
          and not sb.pending and not sb.events,
          "кадр %s, поза %s, подтверждений %d, событий %d"
          % (ses.body_frame, sb.cam,
             len(sb.applied_slots) + len(sb.applied_stim), len(sb.events)))

    def draw(n=1):
        """Нарисовать n кадров. Вернуть худшее время кадра, мс."""
        worst = 0.0
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            app.frame()
            worst = max(worst, (time.perf_counter() - t0) * 1000.0)
        return worst

    def ink():
        """Доля пикселей, отличных от фона: пустой кадр так и видно."""
        a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
        bg = np.asarray(C.bg[:3], dtype=np.float32)
        return float((np.abs(a - bg).max(axis=2) > 0.02).mean())

    def shot(what, note=""):
        name = "sandbox-%s.png" % what
        sh.shot(OUT / name)
        rep["снимки"].append({"снимок": name, "что": note,
                              "мс на кадр": round(sh.frame_ms, 2)})
        print("  снимок %-28s %s" % (name, note))
        return name

    def zones(space):
        """Зоны раскладки этого пространства: ищем наложения."""
        lay = L.compute(sh.w, sh.h, sh.sc, app.st.ws.dock(space))
        zs = [(n, getattr(lay, n)) for n in
              ("topbar", "inspector", "centre", "secondary", "timeline",
               "footer") if lay.has(n)]
        bad = [(a, b) for i, (a, ra) in enumerate(zs)
               for (b, rb) in zs[i + 1:] if ra.overlaps(rb)]
        return [n for n, _ in zs], bad

    def visit(space, frames=3, tab=None):
        """Открыть пространство и нарисовать. Вернуть (ошибка, худший, чернил).

        Разогрев отдельно: первые кадры в новом пространстве собирают
        ядра Taichi, и их время — не время рисования.
        """
        app.st.ws.switch(space)
        if tab is not None:
            app.st.ws.set_tab(tab, space)
        try:
            draw(2)
            worst = draw(frames)
        except Exception:                                     # noqa: BLE001
            tb = traceback.format_exc()
            rep["трассировки"][space + ("/%s" % tab if tab is not None
                                        else "")] = tb
            print(tb)
            return tb, 0.0, 0.0
        return "", worst, ink()

    # --- 1. все пять пространств по очереди, в простое ---
    worst_space = {}
    for key in keys:
        err, worst, filled = visit(key)
        worst_space[key] = round(worst, 1)
        check("пространство «%s» рисуется в простое" % names[key],
              not err and filled > 0.05,
              err.strip().splitlines()[-1] if err else
              "худший кадр %.1f мс, чернил %.0f %%" % (worst, filled * 100))
    rep["замеры"]["худший кадр по пространствам при 1600×900, мс"] = \
        worst_space

    # --- 2. «Песочница» в простое: снимок и зоны ---
    app.st.ws.switch("sandbox")
    app.st.ws.set_tab(0, "sandbox")
    draw(3)
    zs, bad = zones("sandbox")
    check("зоны «Песочницы» не налезают друг на друга", not bad,
          ",".join(zs) + ("  НАЛОЖЕНИЕ %s" % bad if bad else ""))
    shot("idle-1600x900", "«Песочница» в простое, закладка «Показания»")

    # --- 3. все три закладки правой колонки ---
    for i, tab in enumerate(SB.TABS):
        err, worst, filled = visit("sandbox", tab=i)
        check("закладка «%s» рисуется" % tab, not err and filled > 0.05,
              err.strip().splitlines()[-1] if err else
              "худший кадр %.1f мс, чернил %.0f %%" % (worst, filled * 100))
        if i == 1:
            shot("events", "закладка «События»: журнал пуст")
        elif i == 2:
            shot("commands", "закладка «Команды»: ничего не отправлено")
    app.st.ws.set_tab(0, "sandbox")

    # --- 4. простое не превратилось в подделку по ходу рисования ---
    check("рисование не придумало подтверждений",
          not sb.applied_slots and not sb.applied_stim and not sb.pending,
          "подтверждений %d, ожидающих %d"
          % (len(sb.applied_slots) + len(sb.applied_stim), len(sb.pending)))
    check("рисование не отправило ни одной команды",
          not ses.sent and not sb.events,
          "отправлено %d, событий %d" % (len(ses.sent), len(sb.events)))
    check("четыре слота на месте и выключены",
          len(sb.slots) == 4 and not any(s.on for s in sb.slots),
          ", ".join("(%.0f, %.0f)" % (s.x, s.y) for s in sb.slots))
    check("скорость не объявлена, пока её не измерили", sb.ratio == 0.0,
          "отношение %.3f" % sb.ratio)

    # --- 5. остальные четыре целы после «Песочницы» ---
    cam = app.st.camera("body")
    before = (cam.view, cam.yaw, cam.pitch, round(cam.dist, 6), cam.free)
    def dock_of(k):
        d = app.st.ws.dock(k)
        return (d.inspector_w, d.secondary_w, d.timeline_h, d.maximized)

    docks = {k: dock_of(k) for k in keys}
    for key in [k for k in keys if k != "sandbox"]:
        err, worst, filled = visit(key)
        check("«%s» цело после «Песочницы»" % names[key],
              not err and filled > 0.05,
              err.strip().splitlines()[-1] if err else
              "худший кадр %.1f мс, чернил %.0f %%" % (worst, filled * 100))
        shot(key, "пространство «%s»" % names[key])
    after = (cam.view, cam.yaw, cam.pitch, round(cam.dist, 6), cam.free)
    check("«Песочница» не сдвинула камеру «Тела»", before == after,
          "%s -> %s" % (before, after))
    check("раскладка прочих пространств не тронута",
          all(docks[k] == dock_of(k) for k in keys),
          "рейка, колонка и нижняя полоса у всех пяти прежние")
    check("ни одно переключение ничего не запустило",
          app.st.run.phase == "idle" and app.st.last is None
          and app.st.run.job_id == "",
          "фаза %s, последний прогон %s"
          % (app.st.run.phase, app.st.last))

    # --- 6. сужение окна ---
    worst_size = {}
    for (w, h) in SIZES:
        sh.resize(w, h, 1.0)
        app.st.ws.switch("sandbox")
        try:
            draw(2)
            worst = draw(4)
        except Exception:                                     # noqa: BLE001
            tb = traceback.format_exc()
            rep["трассировки"]["sandbox/%dx%d" % (w, h)] = tb
            print(tb)
            check("«Песочница» при %d×%d" % (w, h), False,
                  tb.strip().splitlines()[-1])
            continue
        zs, bad = zones("sandbox")
        worst_size["%d×%d" % (w, h)] = round(worst, 1)
        check("«Песочница» при %d×%d рисуется без наложений" % (w, h),
              not bad and ink() > 0.05,
              "зоны: %s, худший кадр %.1f мс, чернил %.0f %%"
              % (",".join(zs), worst, ink() * 100))
        check("худший кадр при %d×%d меньше 100 мс" % (w, h), worst < 100.0,
              "%.1f мс" % worst)
        if (w, h) != SIZES[0]:
            shot("%dx%d" % (w, h), "раскладка при %d×%d" % (w, h))
    rep["замеры"]["худший кадр «Песочницы» по размерам, мс"] = worst_size
    sh.resize(*SIZES[0], dpi=1.0)

    # --- 7. «Пуск» в «Песочнице» ---
    from flybrain import catalog as K

    app.st.ws.switch("sandbox")
    draw(2)
    kind, title = app.run_kind()
    check("«Пуск» в «Песочнице» запускает сеанс песочницы",
          (kind, title) == ("sandbox", "сеанс песочницы"),
          "%s / %s" % (kind, title))
    spec = app.sandbox_spec()
    bad = K.validate(spec)
    rep["спека"] = spec
    check("спека сеанса проходит справочник без замечаний", not bad,
          "; ".join(bad) if bad else
          "вид %s, длительность %g с, окно записи %g с, слотов %d"
          % (spec.get("kind"), spec.get("t"),
             (spec.get("body") or {}).get("keep_s"),
             len((spec.get("body") or {}).get("sources") or ())))
    check("спека описывает сеанс без заранее заданного конца",
          float(spec.get("t", -1)) == 0.0
          and float((spec["body"]).get("keep_s", 0)) > 0.0
          and (spec["body"].get("live_initial") or {}).get("sources"),
          "t = %g, окно записи %g с, исходные условия слотов есть"
          % (spec.get("t"), spec["body"].get("keep_s")))

    ok_idle, why_idle = app.run_ready()
    check("без работника «Пуск» отказывает именно из-за работника",
          not ok_idle and "работник" in why_idle, why_idle)

    # Тот же вопрос при готовом работнике. Работник не поднимается:
    # окну достаточно, чтобы сеанс назвался живым и готовым.
    ses.alive = ses.ready = True
    ses.boot_progress = 1.0
    draw(2)
    ok_run, why_run = app.run_ready()
    check("при готовом работнике «Пуск» в «Песочнице» возможен",
          ok_run and why_run == "сеанс песочницы",
          "%s / %s" % (ok_run, why_run))
    check("нижняя полоса говорит, что запустится",
          "сеанс песочницы" in app.st.ready_line, app.st.ready_line)
    check("ничего не запущено и после этого",
          app.st.run.phase == "idle" and not ses.sent,
          "фаза %s, отправлено %d" % (app.st.run.phase, len(ses.sent)))
    shot("ready", "«Песочница» с готовым работником: «Пуск» доступен")

    ses.close()
    rep["заметки"] = list(NOTES)
    print("\nзамечено и не правлено:")
    for note in NOTES:
        print("  · " + note)
    (OUT / "sandbox_shot.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
