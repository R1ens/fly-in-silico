"""Рабочее пространство «Песочница»: условия меняются, пока муха живёт.

Что здесь настоящее, а что нет — важнее устройства панелей, поэтому
сначала об этом.

**Сцена настоящая.** В центре те же кадры MuJoCo, что показывает «Тело»:
их снимает работник своим ``mujoco.Renderer`` на шаге управления. Ни
одной декоративной анимации здесь нет; если кадров нет, панель так и
говорит.

**Источник берётся в сцене, и координаты — координаты модели.** Работник
кладёт рядом с каждым кадром позу камеры, которой кадр снят, а
:mod:`flybrain.scene` переводит щелчок в точку плоскости ``z = 1,5 мм``.
Сверено с настоящим кадром (``probe/scene_pick.py``): предсказание
попадает в маркер с точностью до единиц пикселей, обратный перевод — до
десятых миллиметра.

**Слоты источника создаются при сборке модели.** Маркер — это
``mocap``-тело MJCF, и добавить тело в собранную модель нельзя, не
пересобрав её. Поэтому «добавить источник» здесь — это включить
свободный слот. Четыре слота; сказано это прямо, а не спрятано.

**Отправлено, ожидает, применено — три разных состояния.** Между
нажатием и применением лежит шаг управления: правка разбирается на его
границе, в работнике. Поэтому окно показывает свой замысел отдельно от
подтверждённого состояния модели, и подтверждение приходит только от
работника, с модельным временем применения.

**Скорость не объявляется, а измеряется.** Отношение модельного времени
к настенному считается по выборке и показывается числом; слова «в
реальном времени» здесь нет, потому что это неправда — у тела в MuJoCo
отношение около 0,06.

Чего в первой итерации нет нарочно: испуга, бегства и зрения. Контур
здесь один — обонятельный, и он замкнут по-настоящему.
"""

import math
import time

from . import layout as L
from . import widgets as W
from .state import BODY_VIEWS, SANDBOX_Z
from .tokens import C, R, S

__all__ = ["SPACE"]

TABS = ("Показания", "События", "Команды")
FRAME_AR = 960.0 / 720.0

SLOT_COLORS = (C.accent, C.info, C.ok, C.plot_alt)

# Радиус взятия метки источника, в пикселях кадра работника. Примерно
# вдвое больше маркера на среднем отдалении: целиться в полумиллиметровую
# капсулу мышью нельзя.
PICK_PX = 36.0

# Через сколько секунд по стене команда без подтверждения считается
# задержавшейся, и через сколько — потерянной. Первое — несколько шагов
# управления при измеренной скорости около 0,06; второе — столько, что
# ждать уже нечего.
PENDING_WARN_S = 10.0
PENDING_DROP_S = 60.0

SCENE_HELP = (
    "Кадры снимает работник на шаге управления — вид откликается, пока "
    "сеанс идёт. Источник берётся щелчком по его метке и переносится "
    "протяжкой; координаты считаются по позе камеры этого самого кадра, "
    "а не угадываются. Отпустили — ушла команда, и до подтверждения "
    "метка показана штрихами, а не кольцом: это замысел, а не состояние модели.")

SESSION_HELP = (
    "Длительность 0 — сеанс без заранее заданного конца: его закончит "
    "«Стоп». Такой сеанс обязан держать окно записи, иначе запись "
    "активности растёт без предела — одна строка это 138 639 × 2 байта "
    "= 277 КБ, при шаге 20 мс 13,9 МБ на секунду модельного времени. "
    "Счётчики и частоты при этом считаются по всему сеансу, а растр — "
    "только по окну, и в сохранённом прогоне это сказано оговоркой.")

SLOT_HELP = (
    "Слотов четыре, и это предел модели: маркер источника — тело MJCF, "
    "оно создаётся при сборке и на ходу не добавляется. «Включить» "
    "ставит слоту заказанную силу, «выключить» — ноль: выключенный слот "
    "не чует ни одна антенна, где бы он ни стоял, и едой он тоже не "
    "считается.")

STIM_HELP = (
    "Прямая стимуляция: выбранным нейронам подаётся пуассоновский вход "
    "заданной частоты, тем же путём, каким подаются рецепторы. Разница "
    "есть, и о ней надо знать: сенсорные каналы собраны в "
    "`no_refractory` при сборке движка, а нейроны, выбранные запросом на "
    "ходу, — нет. Значит у них остаётся рефрактерный период, и та же "
    "частота даст меньше спайков, чем на антеннах. Менять это на ходу "
    "нельзя: это пересборка движка.")

ODOUR_HELP = (
    "Яркость, которая считается насыщением рецепторов. Задана числом, а "
    "не измерена на разогреве: сеанс начинается с выключенными "
    "источниками, и измеренная по началу яркость была бы нулевой — "
    "первый же включённый источник насытил бы рецепторы до потолка и "
    "градиент стал бы не виден.")


# ------------------------------------------------- разбор подтверждений

def answers(app):
    """Подтверждения работника и измерение скорости. Зовётся каждый кадр.

    Здесь же меряется отношение модельного времени к настенному: по
    выборке длиной около секунды, а не по мгновенной разнице — кадры
    приходят раз в несколько сотен миллисекунд, и мгновенное отношение
    прыгало бы от нуля до единиц.
    """
    ses, sb = app.ses, app.st.sb
    if ses is None:
        return
    # Разбор ответов оболочка зовёт для ВСЕХ пространств на каждом кадре,
    # в каком бы пространстве мы ни были (`app.frame`). Значит сеанс без
    # управляющей части валил бы не песочницу, а любое пространство: так
    # и вышло — поддельный сеанс проб (`probe/app_shot.py:FakeSession`)
    # не умеет `take_applied`, и первый же кадр в «Теле» падал. Поэтому
    # проверяется не вид сеанса, а наличие самого умения.
    take = getattr(ses, "take_applied", None)
    if take is None:
        return
    now = time.perf_counter()

    # Поза камеры того кадра, что сейчас на экране. Приходит вместе с
    # записью шага, потому что считать её в окне нечем: охват зависит от
    # габарита мухи, а его знает только работник.
    cam = (ses.extra or {}).get("cam")
    if cam:
        sb.cam = cam
        # Номер кадра нужен, чтобы знать, не отстала ли поза: кадры идут
        # кольцом и всегда свежие, а поза — сообщением, и сообщение при
        # переполнении очереди теряется.
        sb.cam_seq = int(ses.body_seq)

    # Скорость: модельное время против настенного на выборке.
    if sb.speed_wall <= 0.0:
        sb.speed_ms, sb.speed_wall = ses.t_ms, now
    elif now - sb.speed_wall >= 1.0:
        d_model = (ses.t_ms - sb.speed_ms) / 1000.0
        d_wall = now - sb.speed_wall
        if d_wall > 0:
            r = d_model / d_wall
            # Сглаживание ровно затем, чтобы число не дёргалось между
            # кадрами; скрывать им простой нельзя, поэтому ноль проходит
            # без сглаживания.
            sb.ratio = r if (sb.ratio <= 0.0 or r <= 0.0) else (
                0.7 * sb.ratio + 0.3 * r)
        sb.speed_ms, sb.speed_wall = ses.t_ms, now

    for rec in take():
        row = sb.pending.pop(rec.cmd_id, None)
        what = (row or {}).get("что", rec.action)
        if rec.outcome == "applied":
            if rec.action == "set" and isinstance(rec.values, dict):
                if "sources" in rec.values:
                    sb.applied_slots = list(rec.values["sources"])
                if "stim" in rec.values:
                    sb.applied_stim = list(rec.values["stim"])
                    sb.stim_note = ""
                    sb.stim_bad = False
            if rec.action == "pause":
                sb.paused = True
            elif rec.action in ("resume", "step"):
                sb.paused = (rec.action == "step")
            # Измеренный результат: насколько изменился сенсорный вход
            # между отправкой и применением. Это замер, а не обещание,
            # поэтому в журнале он стоит рядом с модельным временем.
            det = "применено"
            was = (row or {}).get("до") or {}
            now_s = _sense(ses)
            if was and rec.action == "set":
                parts = []
                for k, title in (("odour_l", "запах L"),
                                 ("odour_r", "запах R"),
                                 ("dn_l", "выход L"), ("dn_r", "выход R")):
                    if k in was and k in now_s:
                        d = now_s[k] - was[k]
                        if abs(d) > 1e-9:
                            parts.append("%s %+.6g" % (title, d))
                det += ("; " + ", ".join(parts)) if parts else                     "; сенсорный вход пока не изменился"
            sb.note(what, rec.t_model_ms, det, "ok")
        else:
            if rec.action == "set" and "stim" in (
                    (row or {}).get("поля") or ()):
                sb.stim_note = rec.why
                sb.stim_bad = True
            sb.note(what, rec.t_model_ms,
                    "%s: %s" % (rec.outcome, rec.why or "без причины"),
                    "danger")

    # Команда, ушедшая и не подтверждённая дольше десяти секунд по
    # стене, — не «ожидает», а потерялась. Молчать об этом нельзя.
    for cid, row in list(sb.pending.items()):
        age = now - row["подана"]
        if age > PENDING_WARN_S and not row.get("предупредили"):
            row["предупредили"] = True
            sb.note(row["что"], None,
                    "подтверждения нет больше %.0f с" % PENDING_WARN_S,
                    "warn")
        if age > PENDING_DROP_S or (not ses.busy and age > PENDING_WARN_S):
            # Держать её в «ждут применения» дальше — врать: ждать
            # больше нечего. Запись уходит из ожидания в журнал с
            # исходом, и счётчик перестаёт расти навсегда.
            sb.pending.pop(cid, None)
            sb.note(row["что"], None,
                    "подтверждения так и не пришло"
                    + ("" if ses.busy else ": сеанс кончился"), "danger")


def _sense(ses):
    """Сенсорный вход и выход популяции на этот миг — для замера правки."""
    ex = (ses.extra or {}) if ses is not None else {}
    out = {}
    for k in ("odour_l", "odour_r", "orn_l", "orn_r", "dn_l", "dn_r"):
        v = ex.get(k)
        if v is not None:
            out[k] = float(v)
    out["t_ms"] = float(getattr(ses, "t_ms", 0.0) or 0.0)
    return out


def _send(app, what, payload=None, action="set", fields=()):
    """Отправить команду и завести ей запись «ожидает применения».

    Вместе с командой запоминается сенсорный вход на миг отправки: по
    нему потом считается измеренный результат правки. Без этого в
    журнале осталось бы «источник включён», и чем это кончилось —
    неизвестно.
    """
    ses, sb = app.ses, app.st.sb
    if ses is None or not ses.busy:
        return None
    # «Не раньше следующего шага» — это модельное время, а не задержка по
    # стене: работник придержит команду в своей очереди, пока модельное
    # время не дойдёт до названного. На паузе время стоит, поэтому
    # правка дождётся именно того шага, который сделает «Один шаг».
    at = None
    if sb.on_next and action == "set":
        at = float(ses.t_ms) + float(sb.control_ms)
    cid = ses.control(action, payload, at_t_ms=at)
    sb.pending[cid] = {"что": what, "подана": time.perf_counter(),
                       "действие": action, "поля": tuple(fields),
                       "до": _sense(ses), "не раньше, мс": at}
    sb.note(what, None,
            "отправлено" if at is None
            else ("отправлено, придержано до %s мс модельного времени"
                  % _fmt(at, 0)),
            "info")
    return cid


def _send_slots(app, what):
    return _send(app, what, {"sources": app.st.sb.payload()},
                 fields=("sources",))


# ---------------------------------------------------------------- центр

def centre(app, ctx, r):                                      # noqa: C901
    from .. import scene as SC

    dl, sc, f = ctx.dl, app.sh.sc, app.sh.fonts
    st, sb = app.st, app.st.sb
    cam3 = st.camera("sandbox")
    dock = st.ws.dock()

    dl.rect(r.x, r.y, r.x1, r.y1, C.sunken, 1.0, sc.px(R.panel))
    view = r.inset(sc.px(S.sm))
    frame = app.ses.body_frame if app.ses is not None else None
    if st.run.phase == "prep":
        frame = None
    box = view.fit_aspect(FRAME_AR)

    if frame is not None:
        dl.image(frame, box.x, box.y, box.w, box.h)
    else:
        dl.rect(box.x, box.y, box.x1, box.y1, C.bg, 1.0, sc.px(R.control))
        mid = L.Rect(box.x + sc.px(S.xl), box.cy - sc.px(34),
                     box.w - sc.px(S.xl) * 2, sc.px(70))
        W.label(ctx, mid, "Сеанса ещё не было", role="heading",
                color=C.dim, align="centre")
        W.label(ctx, L.Rect(mid.x + mid.w // 6,
                            mid.y + f.heading.gh + sc.px(S.sm),
                            mid.w - mid.w // 3, sc.px(90)),
                "Нажмите «Пуск» внизу. Тело MuJoCo собирается около "
                "тридцати секунд, первый кадр придёт после сборки. "
                "Источники начинают выключенными — включите слот в "
                "рейке слева или возьмите его метку в сцене.",
                role="small", color=C.faint, wrapped=True)

    # --- полоса вида: та же, что у «Тела» ---
    pad = sc.px(S.md)
    keys = [v[0] for v in BODY_VIEWS]
    names = [v[1] for v in BODY_VIEWS]
    idx = keys.index(cam3.view) if cam3.view in keys else 0
    tb = L.Rect(view.x + pad, view.y + pad, view.w - pad * 2,
                sc.px(S.control) + sc.px(S.xs) * 2)
    hit, picks = W.viewport_toolbar(ctx, "sb/tb", tb, [
        ("seg", "view", names, idx,
         ["камера позади мухи", "профиль, головой вправо",
          "сверху, головой вверх", "лицом к мухе"]),
        ("-",),
        ("zoom_out", "minus", "отдалить: источники стоят в десяти "
                              "миллиметрах, а охват считается по мухе"),
        ("zoom_in", "plus", "приблизить"),
        ("reset", "reset", "вернуть вид по умолчанию"),
        ("-",),
        ("full", "fullscreen",
         "свернуть вид (F11)" if dock.maximized else "развернуть вид (F11)",
         dock.maximized),
    ])
    if "view" in picks:
        cam3.view = keys[picks["view"]]
        cam3.yaw = cam3.pitch = 0.0
        cam3.free = False
    if hit == "reset":
        cam3.reset()
    elif hit == "zoom_in":
        cam3.zoom(0.8)
    elif hit == "zoom_out":
        cam3.zoom(1.25)
    elif hit == "full":
        dock.maximized = not dock.maximized

    # --- метки слотов и перетаскивание ---
    cam = sb.cam
    live = bool(app.ses is not None and app.ses.busy)
    # Поза могла остаться от более старого кадра: кадры идут кольцом в
    # общей памяти и всегда свежие, а поза — сообщением, которое при
    # переполнении очереди теряется. Пока они расходятся, брать источник
    # нельзя: щелчок переведётся по чужой камере.
    stale = (sb.cam_seq >= 0 and app.ses is not None
             and app.ses.body_seq - sb.cam_seq > 1)
    hover, held, clicked, _ = ctx.probe("sb/view", view, True)
    if cam and frame is not None and live and not stale:
        # Кадр вписан в `box` с сохранением отношения сторон, поэтому
        # пиксель кадра считается от его угла, а не от угла вьюпорта.
        kx = float(cam["w"]) / max(box.w, 1)
        ky = float(cam["h"]) / max(box.h, 1)

        def px_of(mx, my):
            return (mx - box.x) * kx, (my - box.y) * ky

        model = [(s.x, s.y, SANDBOX_Z) for s in sb.slots]
        shown = list(sb.applied_slots) or None

        # Взятие и протяжка. Команда уходит ОДНА, на отпускание: иначе
        # каждый кадр протяжки слал бы правку, и очередь работника
        # забилась бы десятками команд на один жест.
        if held and hover:
            fx, fy = px_of(ctx.inp.mx, ctx.inp.my)
            if sb.drag < 0:
                got = SC.pick_slot(cam, model, fx, fy,
                                   radius_px=PICK_PX * kx)
                if got >= 0:
                    sb.drag = got
                    sb.picked = got
                    sb.drag_from = (sb.slots[got].x, sb.slots[got].y)
            if sb.drag >= 0:
                p = SC.unproject(cam, fx, fy, z=SANDBOX_Z)
                if p is not None:
                    s = sb.slots[sb.drag]
                    s.x, s.y = round(p[0], 3), round(p[1], 3)
        elif sb.drag >= 0:
            i = sb.drag
            sb.drag = -1
            moved = (not sb.drag_from
                     or abs(sb.slots[i].x - sb.drag_from[0]) > 1e-6
                     or abs(sb.slots[i].y - sb.drag_from[1]) > 1e-6)
            if moved:
                _send_slots(app, "слот %d перенесён в (%s, %s) мм"
                                 % (i + 1, _fmt(sb.slots[i].x, 1),
                                    _fmt(sb.slots[i].y, 1)))
        elif clicked:
            fx, fy = px_of(ctx.inp.mx, ctx.inp.my)
            sb.picked = SC.pick_slot(cam, model, fx, fy, radius_px=PICK_PX * kx)

        # Рисование меток. Сплошное кольцо — подтверждённое работником
        # положение, четыре штриха — замысел окна, пока правка не
        # применена. Разница видна глазом, а не только в журнале.
        for i, s in enumerate(sb.slots):
            col = SLOT_COLORS[i % len(SLOT_COLORS)]
            real = None
            if shown and i < len(shown):
                real = (float(shown[i]["x"]), float(shown[i]["y"]),
                        SANDBOX_Z)
            waiting = (real is None
                       or abs(real[0] - s.x) > 1e-3
                       or abs(real[1] - s.y) > 1e-3)
            for p, firm in ((real, True), ((s.x, s.y, SANDBOX_Z), False)):
                if p is None or (not firm and not waiting):
                    continue
                at = SC.project(cam, p)
                if at is None:
                    continue
                cx = box.x + at[0] / kx
                cy = box.y + at[1] / ky
                m = sc.px(S.xl)
                if not (box.x - m <= cx <= box.x1 + m
                        and box.y - m <= cy <= box.y1 + m):
                    continue
                on = s.on if not firm else bool(
                    real is not None and shown and i < len(shown)
                    and shown[i].get("on"))
                k = sc.px(9 if (sb.picked == i) else 7)
                a = 1.0 if on else 0.35
                if firm:
                    dl.outline(cx - k, cy - k, cx + k, cy + k, col,
                               width=sc.px(2), alpha=a, radius=k)
                    if sb.picked == i:
                        dl.seg(cx - k * 2, cy, cx - k, cy, col,
                               sc.px(1), 0.9)
                        dl.seg(cx + k, cy, cx + k * 2, cy, col,
                               sc.px(1), 0.9)
                else:
                    # Замысел: четыре коротких штриха вместо кольца.
                    for dx, dy in ((-k, 0), (k, 0), (0, -k), (0, k)):
                        dl.seg(cx + dx * 0.55, cy + dy * 0.55,
                               cx + dx, cy + dy, col, sc.px(2), 0.95)
                lab = "%d" % (i + 1)
                dl.text(f.micro, lab, cx + k + sc.px(3),
                        cy - f.micro.gh // 2, col, a)

    if hover and ctx.inp.wheel:
        cam3.zoom(0.9 ** ctx.inp.wheel)
    # Вращение мышью — только когда не тянем источник: иначе один жест
    # означал бы сразу две вещи.
    last = getattr(app, "_sb_drag_last", None)
    if held and sb.drag < 0:
        if last is not None:
            dx, dy = ctx.inp.mx - last[0], ctx.inp.my - last[1]
            if dx or dy:
                cam3.nudge(dx * 0.4, -dy * 0.3)
        app._sb_drag_last = (ctx.inp.mx, ctx.inp.my)
    else:
        app._sb_drag_last = None

    # --- подпись: что видно и насколько честно ---
    bits = []
    if app.ses is not None and app.ses.busy:
        bits.append(("модельное %.2f с" % (app.ses.t_ms / 1000.0)
                     ).replace(".", ","))
        if sb.ratio > 0:
            bits.append(("скорость %.3f от настенной" % sb.ratio
                         ).replace(".", ","))
        else:
            bits.append("скорость ещё не измерена")
        if sb.paused:
            bits.append("пауза")
    else:
        bits.append("сеанс не идёт")
    if cam is None and frame is not None:
        bits.append("позы камеры нет: взять источник в сцене нельзя")
    elif stale:
        bits.append("поза камеры отстала от кадра: источник не взять")
    elif cam is not None and frame is not None and not live:
        bits.append("сеанс не идёт: кадр последний снятый, источник "
                    "не взять")
    if sb.pending:
        bits.append("ожидают применения: %d" % len(sb.pending))
    note = "  ·  ".join(bits)
    back = ctx.layer(ctx.overlay)
    at = f.small
    cw = at.width(note) + sc.px(S.md) * 2
    cap = L.Rect(view.x + pad, view.y1 - pad - at.gh - sc.px(S.sm),
                 min(cw, view.w - pad * 2), at.gh + sc.px(S.sm))
    dl.rect(cap.x, cap.y, cap.x1, cap.y1, C.bg, 0.78, sc.px(R.control))
    W.label(ctx, L.Rect(cap.x + sc.px(S.md), cap.y + sc.px(S.xs) // 2 + 1,
                        cap.w - sc.px(S.md) * 2, at.gh), note,
            role="small", color=C.dim)
    ctx.layer(back)


# --------------------------------------------------------------- рейка

def inspector(app, ctx, r):                                   # noqa: C901
    sc, f = app.sh.sc, app.sh.fonts
    st, sb = app.st, app.st.sb
    ses = app.ses
    live = bool(ses is not None and ses.busy)
    content, token = W.scroll(ctx, "sb/insp", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    # --- сеанс ---
    W.section(ctx, s, "Сеанс", first=True, wid="sb/ses", info=SESSION_HELP)
    sb.t_run = W.numeric(ctx, "sb/t", s.take(sc.px(S.control)), sb.t_run,
                         0.0, 7200.0, 10.0, "с", label="длина",
                         enabled=not live,
                         tip="0 — без заранее заданного конца")
    sb.keep_s = W.numeric(ctx, "sb/keep", s.take(sc.px(S.control)),
                          sb.keep_s, 1.0, 600.0, 5.0, "с",
                          label="окно записи", enabled=not live,
                          tip="сколько последних секунд активности держать")
    sb.seed = int(W.numeric(ctx, "sb/seed", s.take(sc.px(S.control)),
                            sb.seed, 0, 10000, 1, "", label="зерно",
                            fmt="{:.0f}", enabled=not live,
                            tip="одно зерно — один и тот же сеанс"))
    W.label(ctx, s.take(f.small.gh + sc.px(S.xs)), "популяция для рулёжки",
            role="small", color=C.dim)
    opts = list(st.worker.readouts) or [sb.readout or "ALPN"]
    cur = opts.index(sb.readout) if sb.readout in opts else 0
    got = W.combo(ctx, "sb/out", s.take(sc.px(S.control)), opts, cur,
                  enabled=bool(st.worker.readouts) and not live,
                  tip="её разницу по полушариям читают как поворот; она же "
                      "показана справа в «Выбранной популяции»")
    if st.worker.readouts and opts[got] != sb.readout:
        sb.readout = opts[got]
    W.section(ctx, s, "Запах", wid="sb/odour", info=ODOUR_HELP)
    sb.odour_ref = W.numeric(ctx, "sb/ref", s.take(sc.px(S.control)),
                             sb.odour_ref, 1e-6, 1.0, 0.001, "",
                             label="насыщение", fmt="{:.4f}",
                             enabled=not live,
                             tip="яркость, считающаяся насыщением")

    # --- время сеанса ---
    W.section(ctx, s, "Время сеанса", wid="sb/time",
              info="На паузе модельное время стоит, и кадр замирает: "
                   "работник снимает его на шаге управления, а шагов "
                   "нет. Окно при этом живо. Правка, поданная на паузе, "
                   "применяется на той же границе — модельное время в "
                   "подтверждении будет замороженным; чтобы она ждала "
                   "следующего шага, включите «применять со следующего "
                   "шага» ниже.")
    row = s.take(sc.px(S.control))
    q = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "sb/pause", q[0], "Пауза", icon="pause",
                enabled=live and not sb.paused):
        _send(app, "пауза", action="pause")
    if W.button(ctx, "sb/resume", q[1], "Продолжить", icon="play",
                enabled=live and sb.paused):
        _send(app, "продолжить", action="resume")
    row = s.take(sc.px(S.control))
    q = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "sb/step", q[0], "Один шаг", icon="step",
                enabled=live):
        _send(app, "один шаг", action="step")
    if W.button(ctx, "sb/stop", q[1], "Остановить", kind="danger",
                icon="stop", enabled=live):
        _send(app, "остановить сеанс", action="stop")
    sb.on_next = W.toggle(ctx, "sb/onnext", s.take(sc.px(S.row)), sb.on_next,
                          "применять со следующего шага", enabled=live,
                          tip="правка уйдёт работнику сразу, но он "
                              "придержит её, пока модельное время не "
                              "дойдёт до следующего шага. На паузе время "
                              "стоит — значит правку можно подготовить и "
                              "применить кнопкой «Один шаг»")

    # --- источники ---
    W.section(ctx, s, "Сцена", wid="sb/scene", info=SCENE_HELP)
    cam3 = st.camera("sandbox")
    cam3.dist = W.slider(ctx, "sb/dist", s.take(sc.px(40)), cam3.dist,
                         0.3, cam3.dist_max, "отдаление", "×",
                         fmt="{:.2f}",
                         tip="множитель к охвату, который считается от "
                             "габарита мухи; источники стоят в 14 мм, и "
                             "ближе четырёх они у края кадра")
    row = s.take(sc.px(S.control))
    if W.button(ctx, "sb/camreset", row.cols(2, gap=sc.px(S.sm))[0],
                "Сбросить вид", icon="reset"):
        cam3.reset()

    W.section(ctx, s, "Источники запаха", wid="sb/src", info=SLOT_HELP,
              right="слотов %d" % len(sb.slots))
    for i, slot in enumerate(sb.slots):
        if not s.fits(sc.px(S.row) + sc.px(S.control) * 2):
            break
        col = SLOT_COLORS[i % len(SLOT_COLORS)]
        head = s.take(sc.px(S.row))
        mark = L.Rect(head.x, head.cy - sc.px(5), sc.px(10), sc.px(10))
        ctx.dl.rect(mark.x, mark.y, mark.x1, mark.y1, col,
                    1.0 if slot.on else 0.3, sc.px(2))
        hit = L.Rect(head.x + sc.px(S.lg), head.y,
                     head.w - sc.px(S.lg), head.h)
        was = slot.on
        slot.on = W.toggle(ctx, "sb/on%d" % i, hit, slot.on,
                           "слот %d" % (i + 1), enabled=live,
                           tip="включить — поставить слоту заказанную силу")
        if slot.on != was:
            _send_slots(app, "слот %d %s" % (i + 1,
                                             "включён" if slot.on
                                             else "выключен"))
        if sb.picked == i:
            # Внутри области, а не левее её: `scroll` ставит обрезку по
            # краю содержимого, и засечка снаружи не рисовалась вовсе.
            ctx.dl.vline(head.x + sc.px(1), head.y, head.y1, col,
                         sc.px(2), 0.9)
        row = s.take(sc.px(S.control))
        half = row.cols(2, gap=sc.px(S.sm))
        nx = W.numeric(ctx, "sb/x%d" % i, half[0], slot.x, -150.0, 150.0,
                       1.0, "мм", label="x", enabled=live)
        ny = W.numeric(ctx, "sb/y%d" % i, half[1], slot.y, -150.0, 150.0,
                       1.0, "мм", label="y", enabled=live)
        row = s.take(sc.px(40))
        nst = W.slider(ctx, "sb/s%d" % i, row, slot.strength, 0.0, 10.0,
                       "сила", "", fmt="{:.2f}", enabled=live)
        if (abs(nx - slot.x) > 1e-9 or abs(ny - slot.y) > 1e-9
                or abs(nst - slot.strength) > 1e-9):
            slot.x, slot.y, slot.strength = nx, ny, nst
            _send_slots(app, "слот %d: (%s, %s) мм, сила %s"
                             % (i + 1, _fmt(nx, 1), _fmt(ny, 1),
                                _fmt(nst, 2)))
        if sb.applied_slots and i < len(sb.applied_slots):
            a = sb.applied_slots[i]
            diff = (abs(float(a["x"]) - slot.x) > 1e-3
                    or abs(float(a["y"]) - slot.y) > 1e-3)
            if diff and s.fits(f.micro.gh):
                W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                        "в модели: (%s, %s) мм; ваша правка ещё не "
                        "применена" % (_fmt(a["x"], 1), _fmt(a["y"], 1)),
                        role="micro", color=C.warn)

    # --- прямой стимул ---
    W.section(ctx, s, "Прямой стимул", wid="sb/stim", info=STIM_HELP)
    ti = ctx.editor("sb/q", sb.query)
    W.text_field(ctx, "sb/q", s.take(sc.px(S.control)), ti, "селектор",
                 icon="search", enabled=live)
    sb.query = ti.text
    if sb.stim_note and s.fits(f.small.gh * 2):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    sb.stim_note, role="small",
                    color=C.danger if sb.stim_bad else C.dim, wrapped=True)
        s.take(h, sc.px(S.xs))
    sb.stim_hz = W.slider(ctx, "sb/hz", s.take(sc.px(40)), sb.stim_hz,
                          0.0, 1000.0, "частота", "Гц", fmt="{:.0f}",
                          enabled=live,
                          tip="предел 1000 Гц: выше пуассоновский вход "
                              "перестаёт быть пуассоновским")
    row = s.take(sc.px(S.control))
    half = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "sb/stimon", half[0], "Подать", kind="primary",
                enabled=live and bool(sb.query.strip())):
        sb.stim_on = True
        _send(app, "стимул %s, %s Гц" % (sb.query.strip(),
                                         _fmt(sb.stim_hz, 0)),
              {"stim": [{"query": sb.query.strip(),
                         "rate_hz": float(sb.stim_hz)}]},
              fields=("stim",))
    if W.button(ctx, "sb/stimoff", half[1], "Снять",
                enabled=live and bool(sb.applied_stim)):
        sb.stim_on = False
        _send(app, "стимул снят", {"stim": []}, fields=("stim",))
    if sb.applied_stim and s.fits(f.micro.gh):
        a = sb.applied_stim[0]
        W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                "подан: %s; нейронов %d, %s Гц"
                % (a.get("query", "?"), int(a.get("нейронов", 0)),
                   _fmt(a.get("rate_hz"), 0)),
                role="micro", color=C.ok)

    W.scroll_done(ctx, "sb/insp", r, s.used, token)


# ------------------------------------------------------- правая колонка

def secondary(app, ctx, r, tab):
    if tab == 0:
        _meters(app, ctx, r)
    elif tab == 1:
        _events(app, ctx, r)
    else:
        _commands(app, ctx, r)


def _fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if nd == 0:
        return "{:,.0f}".format(x).replace(",", " ")
    return (("{:.%df}" % nd).format(x)).replace(".", ",")


def _meters(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    sb, ses = app.st.sb, app.ses
    ex = dict((ses.extra or {}) if ses is not None else {})
    content, token = W.scroll(ctx, "sb/met", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, "Время", first=True,
              info="Скорость измерена по выборке модельного времени "
                   "против настенного. Слова «в реальном времени» здесь "
                   "нет: у тела в MuJoCo отношение около 0,06.")
    W.kv(ctx, s.take(sc.px(S.row)), "модельное",
         _fmt((ses.t_ms / 1000.0) if ses is not None else None, 2), "с")
    # Настенное время СЕАНСА, а не возраст объекта сеанса: первая
    # версия брала `ses._born` и в простое показывала растущее число
    # рядом с «модельное 0,00 с» — то есть врала о сеансе, которого не
    # было. `run.t_wall` до пуска равен нулю.
    wall = app.st.run.t_wall
    W.kv(ctx, s.take(sc.px(S.row)), "по стене",
         _fmt(wall, 1) if wall > 0.0 else "—", "с" if wall > 0.0 else "")
    W.kv(ctx, s.take(sc.px(S.row)), "скорость",
         _fmt(sb.ratio, 3) if sb.ratio > 0 else "не измерена", "× настенной")
    if sb.ratio > 0 and s.fits(f.micro.gh):
        W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                "шаг управления 20 мс — это %s с по стене"
                % _fmt(0.02 / max(sb.ratio, 1e-9), 2),
                role="micro", color=C.faint)
    W.kv(ctx, s.take(sc.px(S.row)), "состояние",
         "пауза" if sb.paused else ("идёт" if (ses is not None and ses.busy)
                                    else "простой"))

    W.section(ctx, s, "Сенсорный вход")
    W.kv(ctx, s.take(sc.px(S.row)), "запах слева",
         _fmt(ex.get("odour_l"), 6))
    W.kv(ctx, s.take(sc.px(S.row)), "запах справа",
         _fmt(ex.get("odour_r"), 6))
    W.kv(ctx, s.take(sc.px(S.row)), "рецепторы слева",
         _fmt(ex.get("orn_l"), 0), "Гц")
    W.kv(ctx, s.take(sc.px(S.row)), "рецепторы справа",
         _fmt(ex.get("orn_r"), 0), "Гц")
    W.kv(ctx, s.take(sc.px(S.row)), "вкус", _fmt(ex.get("taste"), 0), "Гц")
    # Оба канала на потолке — значит двусторонней разницы нет вовсе, и
    # рулить мухе нечем: она чует «очень сильно» с обеих сторон
    # одинаково. Молчать об этом нельзя: на экране в это время стоят два
    # честных числа, и выглядит всё работающим.
    top = float(getattr(app.st.sb, "orn_max_hz", 150.0) or 150.0)
    lo, hi = ex.get("orn_l"), ex.get("orn_r")
    if (lo is not None and hi is not None
            and float(lo) >= top - 1e-6 and float(hi) >= top - 1e-6):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Оба канала на потолке %s Гц: двусторонней разницы "
                    "нет, и поворачивать мухе не по чему. Поднимите "
                    "«насыщение» в рейке или отодвиньте источник."
                    % _fmt(top, 0), role="small", color=C.warn,
                    wrapped=True)
        s.take(h, sc.px(S.xs))

    W.section(ctx, s, "Выбранная популяция",
              right=str(app.st.sb.readout))
    W.kv(ctx, s.take(sc.px(S.row)), "выход слева",
         _fmt(ex.get("dn_l"), 3), "Гц")
    W.kv(ctx, s.take(sc.px(S.row)), "выход справа",
         _fmt(ex.get("dn_r"), 3), "Гц")
    W.kv(ctx, s.take(sc.px(S.row)), "перекос", _fmt(ex.get("asym"), 5))
    W.kv(ctx, s.take(sc.px(S.row)), "спайков за шаг",
         _fmt(ex.get("spikes"), 0), "шт")

    W.section(ctx, s, "Моторные команды")
    W.kv(ctx, s.take(sc.px(S.row)), "амплитуда слева",
         _fmt(ex.get("amp_l"), 3))
    W.kv(ctx, s.take(sc.px(S.row)), "амплитуда справа",
         _fmt(ex.get("amp_r"), 3))

    W.section(ctx, s, "Положение мухи")
    W.kv(ctx, s.take(sc.px(S.row)), "x", _fmt(ex.get("x"), 2), "мм")
    W.kv(ctx, s.take(sc.px(S.row)), "y", _fmt(ex.get("y"), 2), "мм")
    th = ex.get("theta")
    W.kv(ctx, s.take(sc.px(S.row)), "курс",
         _fmt(None if th is None else math.degrees(float(th)), 1), "°")
    d = ex.get("dist")
    W.kv(ctx, s.take(sc.px(S.row)), "до источника",
         "нет включённых" if (d is not None and float(d) < 0)
         else _fmt(d, 2), "" if (d is not None and float(d) < 0) else "мм")
    # Это косинус между курсом и направлением на источник, а не угол:
    # «курс» строкой выше — градусы, и два «курса» рядом путали бы. При
    # выключенных источниках сравнивать не с чем, и ноль там значит
    # именно это, а не «точно в сторону».
    W.kv(ctx, s.take(sc.px(S.row)), "совпадение с направлением",
         "—" if (d is not None and float(d) < 0)
         else _fmt(ex.get("grad_cos"), 3))

    W.section(ctx, s, "Подтверждено работником",
              info="Состояние модели, а не замысел окна: показано то, "
                   "что подтвердил работник. Пока правка не "
                   "подтверждена, здесь стоит прежнее.")
    shown = sb.applied_slots or []
    if not shown:
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "подтверждений ещё не было", role="small", color=C.faint)
    for i, a in enumerate(shown):
        if not s.fits(sc.px(S.row)):
            break
        on = bool(a.get("on"))
        W.kv(ctx, s.take(sc.px(S.row)), "слот %d" % (i + 1),
             ("(%s, %s) мм, сила %s"
              % (_fmt(a.get("x"), 1), _fmt(a.get("y"), 1),
                 _fmt(a.get("strength"), 2))) if on else "выключен")
    for a in (sb.applied_stim or []):
        if not s.fits(sc.px(S.row)):
            break
        W.kv(ctx, s.take(sc.px(S.row)), "стимул",
             "%s · %d нейронов · %s Гц"
             % (a.get("query", "?"), int(a.get("нейронов", 0)),
                _fmt(a.get("rate_hz"), 0)))
    W.scroll_done(ctx, "sb/met", r, s.used, token)


def _events(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    sb = app.st.sb
    content, token = W.scroll(ctx, "sb/ev", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(2))
    W.section(ctx, s, "События сеанса", first=True,
              right="%d" % len(sb.events),
              info="Отметка — модельное время применения, пришедшее от "
                   "работника. У записи «отправлено» его нет и быть не "
                   "может: в этот миг модель ещё ничего не знает.")
    colour = {"ok": C.ok, "warn": C.warn, "danger": C.danger,
              "info": C.dim}
    for e in reversed(sb.events):
        if not s.fits(f.small.gh + f.micro.gh + sc.px(4)):
            break
        t = e.get("модельное, мс")
        # Отметка — числовой гарнитурой: это столбец чисел, и ровнять их
        # пробелами в пропорциональном шрифте нельзя. Ширина столбца
        # берётся по самой отметке, а не задаётся числом в пикселях.
        stamp = (_fmt(float(t) / 1000.0, 2) + " с") if t is not None else "—"
        at = f.metric
        gut = at.width("0000,00 с") + sc.px(S.sm)
        row = s.take(max(f.small.gh, at.gh) + sc.px(2))
        ctx.dl.text(at, stamp, row.x, row.y, C.faint, 1.0, max_px=gut)
        W.label(ctx, L.Rect(row.x + gut, row.y, max(1, row.w - gut), row.h),
                str(e.get("что", "")), role="small",
                color=colour.get(e.get("уровень"), C.ink))
        det = e.get("подробно")
        if det:
            h = W.label(ctx, L.Rect(row.x + gut, s.area.y,
                                    max(1, content.w - gut), s.area.h),
                        str(det), role="micro",
                        color=colour.get(e.get("уровень"), C.faint),
                        wrapped=True)
            s.take(h, sc.px(2))
    if not sb.events:
        W.label(ctx, s.take(f.small.gh), "событий ещё не было",
                role="small", color=C.faint)
    W.scroll_done(ctx, "sb/ev", r, s.used, token)


def _commands(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    sb, ses = app.st.sb, app.ses
    content, token = W.scroll(ctx, "sb/cmd", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Команды", first=True,
              info="Три состояния, и они разные. «Отправлено» — команда "
                   "ушла в очередь. «Ожидает применения» — работник её "
                   "ещё не разобрал: правка применяется на границе шага "
                   "управления, а шаг идёт около трети секунды по стене. "
                   "«Применено» — пришло подтверждение с модельным "
                   "временем. Подтверждает только работник.")
    # Граница между «отправлено» и «ожидает применения» — время одного
    # шага управления по стене, посчитанное по ИЗМЕРЕННОЙ скорости.
    # Знать точно, добралась ли команда до работника, окно не может:
    # подтверждение приходит только о применении. Поэтому основание
    # названо прямо, а не выдано за состояние очереди.
    now = time.perf_counter()
    step_wall = (0.02 / sb.ratio) if sb.ratio > 0 else 0.5
    fresh = [r for r in sb.pending.values() if now - r["подана"] < step_wall]
    waiting = [r for r in sb.pending.values()
               if now - r["подана"] >= step_wall]
    done = sum(1 for e in sb.events
               if str(e.get("подробно", "")).startswith("применено"))
    W.kv(ctx, s.take(sc.px(S.row)), "отправлено", str(len(fresh)), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "ожидают применения",
         str(len(waiting)), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "применено", str(done), "шт")
    # Не «за сеанс»: список отправок живёт у сессии работника, то есть
    # у всего окна, и обрезается до двухсот последних.
    W.kv(ctx, s.take(sc.px(S.row)), "послано окном, последние",
         str(len(getattr(ses, "sent", ()) or ())), "шт")
    if s.fits(f.micro.gh):
        W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                ("шаг управления по стене: %s с (по измеренной скорости)"
                 % _fmt(step_wall, 2)) if sb.ratio > 0 else
                ("скорость ещё не измерена; до первого замера граница "
                 "взята как %s с" % _fmt(step_wall, 2)),
                role="micro", color=C.faint)

    if sb.pending:
        W.section(ctx, s, "Ожидают применения")
        for cid, row in list(sb.pending.items()):
            if not s.fits(sc.px(S.row)):
                break
            age = now - row["подана"]
            W.kv(ctx, s.take(sc.px(S.row)),
                 ("· " if age < step_wall else "… ")
                 + str(row.get("что", cid))[:38],
                 _fmt(age, 1), "с")
    W.scroll_done(ctx, "sb/cmd", r, s.used, token)


SPACE = {"centre": centre, "inspector": inspector, "secondary": secondary,
         "tabs": TABS, "answers": answers}
