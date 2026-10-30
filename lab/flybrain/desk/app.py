"""Оболочка приложения: зоны, рабочие пространства, полоса исполнения.

Что здесь есть и чего намеренно нет.

Есть: верхняя полоса с переключателем пространств, левая рейка, центр,
правая колонка с вкладками, нижняя полоса исполнения и строка состояния.
Зоны закреплены, тянутся разделителями, сворачиваются и разворачиваются,
и каждое пространство помнит свою раскладку.

Нет: панели нельзя перетаскивать между зонами. Это сознательно и
записано в брифе: сперва устойчивые зоны, а не разборная доска.

Разворот центра — не другой экран, а состояние раскладки: зоны прячутся,
их размеры сохраняются, камеры и выбор не трогаются.

Собрано только пространство «Тело». Остальные три показывают, что в них
будет, и честно говорят, что ещё не собраны: подменять пустоту
декорацией хуже, чем сказать прямо.
"""

import time

import numpy as np

from . import layout as L
from . import widgets as W
from .state import (BODY_VIEWS, PHASES, SANDBOX_OBJECTS, SANDBOX_Z, VISION,
                    WORKSPACES, AppState)
from .tokens import C, R, S, T

__all__ = ["App", "run"]

# Состояния работника -> фазы задачи, которые показывает интерфейс.
_PHASE_OF = {
    "idle": "idle", "queued": "prep", "setup": "prep", "warmup": "prep",
    "running": "run", "saving": "run", "done": "done", "error": "fail",
    "cancelled": "done",
}


class App:
    """Окно целиком: состояние, приём почты, кадр."""

    def __init__(self, shell, session=None):
        self.sh = shell
        self.ses = session
        self.st = AppState()
        self.st.ws.current = "body"
        # Карта коннектома строится один раз, когда работник пришлёт
        # геометрию, и живёт столько же, сколько окно: переключение
        # пространств не должно её пересобирать, иначе теряются камера и
        # выделение.
        self.brain = None
        self.region_super = {}
        self._levels = None
        self._levels_at = -1.0
        self.drag = ""
        self._cam_sent = None
        self._boot_at = time.perf_counter()
        # Учёт серии живёт здесь, а не во вкладке «Серия»: вкладку рисуют
        # не каждый кадр, и счёт эпизодов терялся при уходе в другое
        # пространство. Запись привязана к номеру задачи — чужой прогон в
        # неё не попадёт.
        self.series = None
        self._cancel_asked = False

    # ------------------------------------------------------------- приём

    def sync(self):
        """Почта работника -> состояние. Больше здесь ничего не делается."""
        s, st = self.ses, self.st
        if s is None:
            return
        s.pump(dt=getattr(self.sh, "dt", 1.0 / 60.0))
        w = st.worker
        w.alive = s.alive
        w.booting = not s.ready
        w.stage = s.boot_phase
        w.progress = s.boot_progress
        if s.hello is not None and self.brain is None:
            self._build_brain()
        if s.hello is not None and not w.hello_at:
            w.hello_at = time.perf_counter()
            w.device = s.hello.device
            w.n_neurons = int(s.hello.n_neurons)
            w.regions = tuple(s.hello.regions)
            w.readouts = tuple(s.hello.readouts)
            st.readout = st.readout or (w.readouts[0] if w.readouts else "")
            st.note("работник готов: %s, %d нейронов"
                    % (w.device, w.n_neurons))
        if s.error:
            w.last_error = str(s.error)
        if not w.alive and st.run.busy:
            # Работник умер посреди задачи. Без этой ветки окно вечно
            # показывало «Подготовка»: ошибки не будет — слать её некому.
            st.run.phase = "fail"
            st.run.stage = "работник не отвечает"
            st.run.finished = time.perf_counter()
            st.run.message = (w.last_error
                              or "процесс со счётом завершился сам")
            st.note("работник не отвечает: " + st.run.message, "error")

        r = st.run
        phase = _PHASE_OF.get(s.state, "idle")
        if r.phase == "cancel" and phase in ("prep", "run"):
            phase = "cancel"            # отмена уже послана, ждём ответа
        if not r.busy and r.phase not in ("done", "fail"):
            # Простой. Из сеанса брать нечего: его поля всё ещё помнят
            # прошлый прогон, и переносить их сюда значило бы выдавать
            # прошлые числа за состояние окна.
            while s.log:
                st.note(s.log.pop(0))
            return

        if phase != r.phase:
            if phase in ("done", "fail"):
                r.finished = time.perf_counter()
            r.phase = phase
        r.job_id = s.job_id or r.job_id
        r.stage = s.phase or r.stage
        r.progress = float(s.progress) if s.progress else (
            -1.0 if r.phase == "prep" else r.progress)
        ex = getattr(s, "extra", None) or {}
        r.t_sim = max(float(s.t_ms or 0.0),
                      float(ex.get("t_ms") or 0.0)) / 1000.0
        if "x" in ex and "y" in ex:
            t = float(ex.get("t_ms") or 0.0)
            if not st.track or st.track[-1][0] != t:
                st.track.append((t, float(ex["x"]), float(ex["y"])))
                if len(st.track) > st.track_max:
                    del st.track[:len(st.track) - st.track_max]
        if r.phase == "fail" and not r.message:
            r.message = str(s.error or s.phase or "подробности в журнале")
        if s.summary and r.phase in ("done", "fail")                 and isinstance(s.summary, dict):
            r.out_dir = str(s.summary.get("path")
                            or s.summary.get("out") or "")
            r.cancelled = bool(s.summary.get("cancelled")) or self._cancel_asked
            r.saved = bool(r.out_dir)
        self._series_pump()
        while s.log:
            st.note(s.log.pop(0))

        if r.phase in ("done", "fail"):
            # Сводку с путём к результату работник шлёт отдельным
            # сообщением, и она может прийти на кадр-другой позже
            # состояния «готово». Зафиксировать прогон раньше неё значит
            # потерять папку с результатом, поэтому ждём — но не дольше
            # полусекунды, чтобы не залипнуть, если сводки не будет
            # вовсе (так бывает при отказе).
            waited = time.perf_counter() - (r.finished or 0.0)
            if s.summary is None and r.phase == "done" and waited < 0.5:
                return
            # Прогон закончился. Переносим его целиком в «последний», а
            # текущее состояние возвращаем в «готов к запуску»: это
            # разные вещи, и показывать их надо врозь.
            import copy

            snap = copy.copy(r)
            snap.finished = snap.finished or time.perf_counter()
            if snap.phase == "done" and self._cancel_asked:
                snap.cancelled = True
            self._series_close(snap)
            st.last = snap
            st.note("прогон %s (%s): %s"
                    % (snap.job_id, snap.title or snap.kind,
                       "готово" if snap.phase == "done" else "отказ"))
            r.reset()

    def _series_pump(self):
        """Счёт выполненных эпизодов по ходу, который шлёт работник.

        Ход серии — это `done / total` из `jobs._series`, то есть доля
        именно эпизодов, а не времени. Счёт берётся только у своей задачи
        и только пока она идёт.
        """
        ser, r = self.series, self.st.run
        if not ser or ser.get("closed") or ser.get("job") != r.job_id:
            return
        if r.busy and r.progress > 0:
            n = int(round(r.progress * ser["planned"]))
            ser["done"] = max(int(ser.get("done", 0)),
                              max(0, min(ser["planned"], n)))

    def _series_close(self, snap):
        """Подвести итог серии: исходы не пересекаются.

        Запланировано = выполнено + отменено + с ошибкой, всегда. Раньше
        эти числа считались врозь и каждое по-своему, и после отмены в
        окне стояло «выполнено 4» и «отменено 4» при четырёх
        запланированных — две взаимоисключающие концовки сразу.
        """
        ser = self.series
        if not ser or ser.get("closed") or ser.get("job") != snap.job_id:
            return
        p = int(ser["planned"])
        done = max(0, min(p, int(ser.get("done", 0))))
        if snap.phase == "fail":
            # Задача отдала ошибку: эпизод, на котором она упала, не
            # выполнен и не отменён — он ошибочный, остальные не начаты.
            done = min(done, p - 1)
            ser.update(done=done, failed=1, cancelled=max(0, p - done - 1))
        elif snap.cancelled:
            ser.update(done=done, failed=0, cancelled=max(0, p - done))
        else:
            ser.update(done=p, failed=0, cancelled=0)
        ser["closed"] = True
        ser["phase"] = snap.phase
        ser["finished"] = time.time()
        ser["out_dir"] = snap.out_dir

    def _build_brain(self):
        """Собрать карту из геометрии, которую прислал работник."""
        from ..ui.brainview import BrainView

        g = self.ses.geometry()
        if g is None:
            return
        pos, code, names, rgb, _cols = g
        counts = np.bincount(code, minlength=len(names))
        self.brain = BrainView(pos, code, names, rgb, counts=counts)
        self.brain.set_mode("regions")
        self.region_super = dict(getattr(self.ses.hello, "region_super",
                                         {}) or {})
        self.st.brain.ready = True
        self.st.note("карта мозга готова: %d нейронов, %d нейропилей"
                     % (len(pos), len(names)))

    def region_levels(self):
        """Средняя активность по областям, не чаще чем раз в четверть
        секунды: на каждый кадр это лишний проход по 138 тысячам."""
        if self.brain is None or self.ses is None:
            return None
        if getattr(self.ses, "act", None) is None:
            return None
        now = time.perf_counter()
        if self._levels is None or now - self._levels_at > 0.25:
            self._levels_at = now
            self._levels = self.ses.region_levels(
                self.brain.region_code, len(self.brain.region_names),
                counts=self.brain.counts)
        return self._levels

    def region_level(self, i):
        lev = self.region_levels()
        if lev is None or not (0 <= i < len(lev)):
            return None
        return float(lev[i])

    def send_camera(self):
        """Отправить работнику настройки камеры, если они изменились."""
        if self.ses is None:
            return
        # Камера берётся того вьюпорта, который на экране. Брать всегда
        # «Тело» было ошибкой: песочница писала своё отдаление в его
        # камеру, и, вернувшись в «Тело», владелец находил чужой вид.
        cam = self.st.camera("sandbox" if self.st.ws.current == "sandbox"
                             else "body")
        base = next((v for v in BODY_VIEWS if v[0] == cam.view), BODY_VIEWS[0])
        az, el, k = base[3]
        # Расстояние всегда считается от габарита мухи, а приближение
        # владельца — множитель поверх него. Абсолютных миллиметров
        # окно не знает и знать не должно.
        pref = {"az": az + cam.yaw,
                "el": max(-89.0, min(89.0, el + cam.pitch)),
                "k": k * cam.dist, "follow": True}
        key = tuple(round(v, 3) if isinstance(v, float) else v
                    for v in pref.values())
        if key != self._cam_sent:
            self._cam_sent = key
            self.ses.ask("camera", pref)

    # ------------------------------------------------------------- кадр

    def frame(self):
        ctx = self.sh.begin()
        self.sync()
        self.send_camera()
        st = self.st
        dock = st.ws.dock()
        sc = self.sh.sc
        lay = L.compute(self.sh.w, self.sh.h, sc, dock)

        ok_run, why_run = self.run_ready()
        _k, kind_title = self.run_kind()
        st.ready_line = ("«Пуск» запускает: " + kind_title + "."
                         ) if ok_run else ("Запустить нельзя: "
                                           + why_run + ".")
        self.keyboard(ctx)
        self.top_bar(ctx, lay.topbar, lay)
        self.footer(ctx, lay.footer)

        space = SPACES.get(st.ws.current)
        # Ответы работника разбираются всегда, в каком бы пространстве мы
        # ни были: запрос мог уйти до переключения вкладки.
        for sp in SPACES.values():
            fn = sp.get("answers")
            if fn is not None:
                fn(self)
        m = sc.px(S.xs)
        if lay.has("centre"):
            space["centre"](self, ctx, lay.centre.inset(m))
        if lay.has("inspector"):
            box = lay.inspector.inset(m)
            inner = W.panel(ctx, box, "", elev=1)
            space["inspector"](self, ctx, inner)
        if lay.has("secondary"):
            self.secondary(ctx, lay.secondary.inset(m), space)
        if lay.has("timeline"):
            self.timeline(ctx, lay.timeline.inset(m))

        self.splitters(ctx, lay, dock)
        self.sh.end()

    # --------------------------------------------------------- управление

    def keyboard(self, ctx):
        dock = self.st.ws.dock()
        for ev in ctx.inp.events:
            if getattr(ev, "kind", "") != "key":
                continue
            name = getattr(ev, "name", "")
            if name == "escape":
                if ctx.focus:
                    ctx.focus = ""
                elif ctx.popup:
                    ctx.popup = ""
                elif dock.maximized:
                    dock.maximized = False
            elif name == "f11":
                dock.maximized = not dock.maximized

    def splitters(self, ctx, lay, dock):
        sc = self.sh.sc
        inp = ctx.inp
        for name, r in lay.splitters.items():
            W.splitter_handle(ctx, "sp/" + name, r,
                              axis="h" if name == "timeline" else "v",
                              dragging=(self.drag == name))
        if self.drag:
            if inp.down:
                L.drag_splitter(dock, sc, self.drag, inp.mx, inp.my)
            else:
                self.drag = ""
                dock.drag = ""
        elif inp.pressed:
            hit = L.hit_splitter(lay, inp.mx, inp.my)
            if hit:
                self.drag = hit
                L.start_drag(dock, sc, hit, inp.mx, inp.my)

    # -------------------------------------------------------------- зоны

    def top_bar(self, ctx, r, lay=None):
        dl, sc, f = ctx.dl, self.sh.sc, self.sh.fonts
        st = self.st
        dl.rect(r.x, r.y, r.x1, r.y1, C.panel, 1.0, 0.0)
        dl.rect(r.x, r.y1 - sc.hair(), r.x1, r.y1, C.border_soft, 1.0, 0.0)
        pad = sc.px(S.lg)
        at = f.workspace
        title = dict(WORKSPACES)[st.ws.current]
        dl.text(at, title, r.x + pad, r.y + (r.h - at.gh) // 2, C.ink)
        x = r.x + pad + at.width(title) + sc.px(S.xl)

        names = [n for _, n in WORKSPACES]
        keys = [k for k, _ in WORKSPACES]
        cell = max(f.body.width(n) for n in names) + sc.px(S.xl)
        wide = cell * len(names) + sc.px(S.xs) * 2
        nav = L.Rect(x, r.y + (r.h - sc.px(S.control)) // 2,
                     min(wide, max(0, r.x1 - x - sc.px(290))),
                     sc.px(S.control))
        if nav.w >= wide:
            i = W.segmented(ctx, "ws", nav, names, keys.index(st.ws.current))
            # Переключение пространства меняет только состояние вида.
            # Ни одна задача от этого не начинается и не прекращается.
            if keys[i] != st.ws.current:
                st.ws.switch(keys[i])

        # Кнопки зон: показать и спрятать рейку, колонку и нижнюю
        # полосу. Без них «свернуть панель» было только в коде.
        btn = sc.px(S.control)
        dock = st.ws.dock()
        zones = (("inspector", "left", "рейка слева"),
                 ("timeline", "bottom", "полоса исполнения"),
                 ("secondary", "right", "колонка справа"))
        dropped = set(lay.dropped if lay is not None else ())
        bx = r.x1 - sc.px(S.md) - btn * len(zones)
        for zone, side, what in zones:
            on = dock.is_open(zone) and zone not in dropped
            cell = L.Rect(bx, r.y + (r.h - btn) // 2, btn, btn)
            # Состояние показывает сам значок — закрашенная полоса
            # против пустой. Подсвечивать кнопку ещё и фоном незачем:
            # три оранжевых квадрата в углу читались как тревога.
            if W.icon_button(ctx, "zone/" + zone, cell,
                             "panel-" + side + ("" if on else ":off"),
                             kind="ghost",
                             tip=(what + ": не помещается в окно"
                                  if zone in dropped else
                                  ("скрыть " if on else "показать ") + what)):
                dock.toggle(zone)
            bx = cell.x1
        cap_x1 = r.x1 - sc.px(S.md) - btn * len(zones) - sc.px(S.md)
        run = st.run
        if lay is not None and not lay.has("timeline"):
            # Нижней полосы нет — ни спрятали, ни поместилась. Пуск и
            # остановка обязаны остаться доступными: другого органа
            # запуска в окне нет.
            w = sc.px(96)
            cell = L.Rect(cap_x1 - w, r.y + (r.h - sc.px(S.control)) // 2,
                          w, sc.px(S.control))
            if run.busy:
                if W.button(ctx, "tl/stop", cell, "Стоп", kind="danger",
                            icon="stop", enabled=(run.phase != "cancel"),
                            tip="остановить прогон"):
                    self.cancel_run()
            elif W.button(ctx, "tl/run", cell, "Пуск", kind="primary",
                          icon="play", enabled=st.worker.ready,
                          tip="запустить прогон; полоса исполнения скрыта"):
                self.start_run()
            cap_x1 = cell.x - sc.px(S.md)
        box = L.Rect(max(nav.x1 + sc.px(S.lg), cap_x1 - sc.px(300)), r.y,
                     0, r.h)
        box = L.Rect(box.x, r.y, max(0, cap_x1 - box.x), r.h)
        W.status(ctx, box, run.phase, self._run_caption(),
                 pulse=run.busy)

    # Какой вид прогона уместен в каком пространстве. «Мозг» запускает
    # стимуляцию выбранной популяции, «Тело» — эпизод в теле. Предлагать
    # в «Мозге» запуск тела было бы просто неверно.
    RUN_OF = {"body": ("body", "эпизод в теле"),
              "brain": ("stim", "стимуляция"),
              "sandbox": ("sandbox", "сеанс песочницы"),
              "experiment": ("spec", "собранный опыт")}

    def run_kind(self):
        return self.RUN_OF.get(self.st.ws.current, (None, ""))

    def sandbox_spec(self):
        """Спека сеанса песочницы: тот же вид «тело», но живой.

        Отличий от обычного эпизода три, и все заданы здесь, а не внутри
        задачи: длительность может быть нулевой (сеанс без заранее
        заданного конца), запись идёт в кольцо, а яркость насыщения
        рецепторов задана числом — сеанс начинается с выключенными
        источниками, и измеренная по началу яркость была бы нулевой.

        Исходные условия слотов уходят отдельным полем: они применяются
        до разогрева и в сохранённом прогоне стоят отдельно от правок,
        сделанных на ходу.
        """
        sb = self.st.sb
        # Зрение решается ЗДЕСЬ, до запуска, и на ходу не включается:
        # глаза либо есть в модели MuJoCo, либо нет, и слоты зрительных
        # объектов тоже создаются до компиляции. На ходу меняются
        # подача, усиление, контраст, свет и сами объекты.
        #
        # Слоты объектов просятся ТОЛЬКО при включённом зрении: без
        # глаз объект в сцене есть, но увидеть его нечем, а шесть
        # лишних геометрий сеанс оплатит. Сеанс без зрения обязан идти
        # ровно как прежде.
        live = {"sources": sb.payload()}
        body = {
            "readout": str(sb.readout or "ALPN"),
            "control_ms": 20.0,
            "keep_s": float(sb.keep_s),
            "odour_ref": float(sb.odour_ref),
            "sources": [[s.x, s.y, SANDBOX_Z] for s in sb.slots],
        }
        if sb.vision:
            body.update({
                "vision": True,
                "vision_hz": float(sb.vision_hz),
                "vision_max_hz": float(sb.vision_max_hz),
                "vision_contrast": float(sb.vision_contrast),
                "vision_light": float(sb.vision_light),
                "vis_objects": SANDBOX_OBJECTS,
            })
            # Исходные условия объектов: применяются до разогрева и в
            # сохранённом прогоне стоят отдельно от правок на ходу.
            live["objects"] = sb.obj_payload()
        body["live_initial"] = live
        return {
            "kind": "body",
            "t": float(sb.t_run),
            "seed": int(sb.seed),
            "bin_ms": 10.0,
            "open_report": False,
            "body": body,
        }

    def run_ready(self):
        """Можно ли пускать здесь и сейчас, и если нет — почему."""
        kind, title = self.run_kind()
        if kind is None:
            return False, "в этом пространстве нечего запускать"
        if not self.st.worker.ready:
            return False, "работник ещё не готов"
        if kind == "stim" and not self._stim_query():
            return False, "выберите популяцию: её и будем стимулировать"
        if kind == "spec":
            from . import ws_experiment as X
            from .. import catalog as K

            bad = K.validate(X.build_spec(self))
            if bad:
                return False, bad[0]
            return True, K.KINDS[X._state(self)["kind"]].title
        if kind == "sandbox":
            # Сеанс песочницы проверяется тем же справочником, что и
            # обычный эпизод в теле: это он и есть, только без заранее
            # заданного конца и с кольцом записи.
            from .. import catalog as K

            bad = K.validate(self.sandbox_spec())
            if bad:
                return False, bad[0]
            return True, title
        return True, title

    def _stim_query(self):
        """Кого стимулировать: выделение, иначе выбранная область."""
        b = self.st.brain
        if b.sel_idx is not None and b.sel_label:
            return b.sel_label
        if b.region >= 0 and self.brain is not None:
            return "region:" + self.brain.region_names[b.region]
        return ""

    def _run_caption(self):
        run, w = self.st.run, self.st.worker
        if w.booting:
            return "Работник поднимается: " + (w.stage or "…")
        if run.phase == "idle":
            if self.st.ws.current == "analysis":
                return "История прогонов: здесь ничего не считается"
            ok, why = self.run_ready()
            _kind, title = self.run_kind()
            return ("Готов к запуску: " + title) if ok                 else ("Не готов: " + why)
        name = {"prep": "Подготовка", "run": "Идёт", "cancel": "Останавливаю",
                "done": "Готово", "fail": "Отказ"}.get(run.phase, run.phase)
        return name + (": " + run.stage if run.stage else "")

    def footer(self, ctx, r):
        dl, sc, f = ctx.dl, self.sh.sc, self.sh.fonts
        dl.rect(r.x, r.y, r.x1, r.y1, C.panel, 1.0, 0.0)
        am = f.micro
        y = r.y + (r.h - am.gh) // 2
        w = self.st.worker
        left = "%s · %s нейронов · %s" % (
            "работник жив" if w.alive else "работника нет",
            "{:,}".format(w.n_neurons).replace(",", " "),
            w.device or "—")
        dl.text(am, left, r.x + sc.px(S.md), y, C.faint)
        right = "%d×%d · %.0f кадр/с · %.1f мс · %d запусков" % (
            self.sh.w, self.sh.h, self.sh.fps, self.sh.frame_ms,
            self.sh.launches)
        dl.text_right(am, right, r.x1 - sc.px(S.md), y, C.faint)

    def secondary(self, ctx, r, space):
        sc, f = self.sh.sc, self.sh.fonts
        st = self.st
        ctx.dl.rect(r.x, r.y, r.x1, r.y1, C.panel, 1.0, sc.px(R.panel))
        names = space["tabs"]
        head, body = r.cut_top(sc.px(S.row + 8))
        # Вкладки нужны именно потому, что показывать всё сразу нельзя:
        # четыре панели в колонке 300 px превращаются в полоски.
        cur = min(st.ws.tab(default=0), len(names) - 1)
        i = W.tabs(ctx, "sec", head, names, cur)
        if i != cur:
            st.ws.set_tab(i)
        space["secondary"](self, ctx, body.inset(sc.px(S.md)), i)

    # ------------------------------------------------- полоса исполнения

    def timeline(self, ctx, r):
        """Нижняя полоса: пуск, остановка, время, этап, прогресс.

        Время модельное, время по стене и их отношение показываются
        порознь: одно число из трёх всегда врёт.
        """
        dl, sc, f = ctx.dl, self.sh.sc, self.sh.fonts
        st, run = self.st, self.st.run
        dl.rect(r.x, r.y, r.x1, r.y1, C.panel, 1.0, sc.px(R.panel))
        area = r.inset(sc.px(S.md))
        busy = run.busy

        # Полоса разложена на четыре группы с подписями и волосяными
        # разделителями. Прежде всё полезное стояло в ней вперемешку, и
        # панель читалась как россыпь чисел.
        cap = f.micro
        caps, area = area.cut_top(cap.gh + sc.px(S.xs), 0)

        def group(width, title, last=False):
            """Отрезать группу слева, подписать, отбить разделителем."""
            nonlocal area, caps
            cell, area = area.cut_left(width, sc.px(S.lg))
            head, caps = caps.cut_left(width, sc.px(S.lg))
            dl.text(cap, title, head.x, head.y, C.faint,
                    max_px=max(0, head.w))
            if not last and area.w > 0:
                x = cell.x1 + sc.px(S.lg) // 2
                dl.rect(x, caps.y, x + sc.hair(), area.y1, C.border_soft,
                        1.0, 0.0)
            return cell

        # 1. Управление
        ok_run, why_run = self.run_ready()
        _kind, kind_title = self.run_kind()
        ctrl = group(sc.px(150), "УПРАВЛЕНИЕ")
        cst = L.Stack(ctrl, gap=sc.px(S.sm))
        if st.ws.current == "analysis" and not busy:
            # В истории запускать нечего, и отключённый «Пуск» здесь —
            # только обещание, которого никто не даёт. Вместо него —
            # дорога туда, где опыт собирают.
            if W.button(ctx, "tl/toexp", cst.take(sc.px(S.control_lg)),
                        "К опыту", icon="expand",
                        tip="перейти в «Эксперимент» и собрать прогон"):
                st.ws.switch("experiment")
            if cst.fits(sc.px(S.row)):
                W.label(ctx, cst.take(sc.px(S.row)),
                        "здесь не считают", role="small", color=C.faint)
        elif not busy:
            if W.button(ctx, "tl/run", cst.take(sc.px(S.control_lg)),
                        "Пуск", kind="primary", icon="play",
                        enabled=ok_run,
                        tip=("запустить: " + kind_title) if ok_run
                            else why_run):
                self.start_run()
        else:
            if W.button(ctx, "tl/stop", cst.take(sc.px(S.control_lg)),
                        "Стоп", kind="danger", icon="stop",
                        enabled=(run.phase != "cancel"),
                        tip="остановить между шагами управления"):
                self.cancel_run()
        if cst.fits(sc.px(S.row)):
            W.status(ctx, cst.take(sc.px(S.row)), run.phase,
                     dict(zip(PHASES,
                              ("Простой", "Подготовка", "Идёт",
                               "Остановка", "Готово", "Отказ"))
                          ).get(run.phase, run.phase), pulse=busy)

        # 2. Время. Пока идёт счёт — про текущий прогон, в простое —
        # про последний законченный, и об этом прямо сказано в подписи
        # группы: иначе прошлые числа читаются как нынешнее состояние.
        shown_of = run if busy else (st.last or run)
        rt = shown_of.realtime
        shown = [(_fmt(shown_of.t_sim, 2), "модельное, с", None),
                 (_fmt(shown_of.t_wall, 1), "по стене, с", None),
                 ("—" if rt is None else _fmt(rt, 3 if rt < 0.1 else 2),
                  "доля реального",
                  C.warn if (rt is not None and rt < 0.1) else None),
                 (str(shown_of.job_id or "—"), "номер задачи", None)]
        need = [max(f.metric.width(v), f.small.width(t)) + sc.px(S.md)
                for v, t, _c in shown]
        fit, total = 0, 0
        for wq in need:
            if total + wq > area.w * 0.52:
                break
            total += wq
            fit += 1
        fit = max(1, fit)
        if busy:
            head = "ВРЕМЯ · " + (run.title or run.kind or "прогон")
        elif st.last is not None:
            head = ("ПОСЛЕДНИЙ ПРОГОН · "
                    + (st.last.title or st.last.kind or "")
                    + {"fail": " · отказ",
                       "cancelled": " · остановлен"}.get(st.last.outcome, ""))
        else:
            head = "ВРЕМЯ"
        nums = group(sum(need[:fit]), head)
        x = nums.x
        for wq, (v, t, col) in zip(need[:fit], shown[:fit]):
            W.metric(ctx, L.Rect(x, nums.y, wq - sc.px(S.sm), nums.h),
                     v, t, color=col)
            x += wq

        # 3. Ход
        bars = group(min(max(sc.px(220), area.w // 2), sc.px(460)), "ХОД")
        bst = L.Stack(bars, gap=sc.px(S.sm))
        if bst.fits(sc.px(34)) and busy:
            stage = run.stage or "—"
            # Доля считается по модельному времени, когда оно известно:
            # это единственная честная мера хода эпизода. Пока тело ещё
            # собирается, доли нет вовсе, и рисовать «0 %» нельзя —
            # подготовка занимает около тридцати секунд, и ноль на
            # экране читался бы как «ничего не происходит».
            frac = None
            if run.t_target > 0 and run.t_sim > 0:
                frac = min(1.0, run.t_sim / run.t_target)
            elif run.progress > 0:
                frac = run.progress
            if frac is None and busy:
                # У сеанса без заранее заданного конца доли нет и быть
                # не может: его заканчивает остановка. Вместо выдуманных
                # процентов — прожитое модельное время.
                note = ("этап: " + stage)
                if run.t_target <= 0 and run.t_sim > 0:
                    note += ("  ·  %s с модельного, конец не задан"
                             % _fmt(run.t_sim, 2))
                W.progress(ctx, bst.take(sc.px(34)), 0.0, note,
                           indeterminate=True)
            else:
                W.progress(ctx, bst.take(sc.px(34)), frac or 0.0,
                           "этап: " + stage + "  ·  %s из %s с модельного"
                           % (_fmt(run.t_sim, 2), _fmt(run.t_target, 1)))
        elif bst.fits(f.small.gh):
            last = st.last
            if last is None:
                W.label(ctx, bst.take(f.small.gh), "прогонов ещё не было",
                        role="small", color=C.faint)
            else:
                # Три разных исхода, а не два: остановленный прогон
                # сохранён, но досчитан не до конца, и называть его
                # завершённым — врать.
                word, dot = {"done": ("Завершён", "done"),
                             "cancelled": ("Остановлен", "cancel"),
                             "fail": ("Отказ", "fail")}.get(
                                 last.outcome, ("Готово", "done"))
                W.status(ctx, bst.take(sc.px(S.row)), dot,
                         word + ": " + (last.title or last.kind))
                if bst.fits(f.small.gh) and last.out_dir:
                    W.label(ctx, bst.take(f.small.gh),
                            "сохранено в " + last.out_dir.split("\\")[-1],
                            role="small", color=C.faint)
                elif bst.fits(f.small.gh) and last.message:
                    W.label(ctx, bst.take(f.small.gh), last.message,
                            role="small", color=C.danger)

        # 4. Что дальше
        hint = group(area.w, "ЧТО ДАЛЬШЕ", last=True)
        if hint.w > sc.px(140):
            W.label(ctx, hint, _hint(st), role="small", color=C.dim,
                    wrapped=True)

    # ------------------------------------------------------------ задача

    def start_run(self, spec=None, title=""):
        """Запустить то, что уместно здесь, либо заданную спеку.

        Вид опыта берётся от рабочего пространства: в «Теле» это эпизод
        в теле, в «Мозге» — стимуляция выбранной популяции. Предлагать
        в «Мозге» запуск тела неправильно по смыслу.
        """
        st = self.st
        if self.ses is None or not st.worker.ready:
            return None
        if spec is None:
            kind, title = self.run_kind()
            ok, why = self.run_ready()
            if not ok:
                st.note("запустить нечего: " + why, "warn")
                return None
            if kind == "spec":
                from . import ws_experiment as X

                spec = X.build_spec(self)
                title = X.KINDS_TITLE(spec["kind"])
            elif kind == "sandbox":
                spec = self.sandbox_spec()
            elif kind == "body":
                body = {}
                if st.readout:
                    body["readout"] = st.readout
                spec = {"kind": "body", "t": float(st.t_run),
                        "seed": int(st.seed), "open_report": False,
                        "body": body}
            else:
                spec = {"kind": "stim", "t": float(st.t_run),
                        "seed": int(st.seed),
                        "bin_ms": float(st.bin_ms),
                        "open_report": False,
                        "stim": {"activate": [
                            {"query": self._stim_query(),
                             "rate_hz": float(st.rate_hz)}]}}
        kind = str(spec.get("kind", "stim"))
        self._cancel_asked = False
        job = self.ses.submit(spec)
        if job:
            st.track.clear()
            # Песочница забывает прошлый сеанс: модельное время пойдёт
            # заново, и её скорость, подтверждения и ожидания от прошлого
            # прогона описывали бы не этот.
            st.sb.new_session()
            rep = int(spec.get("repeat") or 0)
            # Серия заводится только там, где она есть. Иначе запись от
            # прошлой серии оставалась на экране и наполнялась числами
            # следующего, одиночного прогона.
            self.series = ({"job": job, "kind": kind,
                            "title": title or kind, "planned": rep * 2,
                            "done": 0, "cancelled": 0, "failed": 0,
                            "closed": False, "phase": "",
                            "started": time.time(), "out_dir": "",
                            "episode_t": float(spec.get("repeat_t")
                                               or spec.get("t") or 0.0)}
                           if rep else None)
            st.run.begin(job, kind, float(spec.get("t", st.t_run)))
            st.run.title = title or dict(
                self.RUN_OF.values()).get(kind, kind)
            st.note("пуск: %s, %s с"
                    % (st.run.title, _fmt(spec.get("t", st.t_run), 1)))
        else:
            st.note("работник занят или ещё не готов", "warn")
        return job

    def cancel_run(self):
        if self.ses is not None:
            self._cancel_asked = True
            self.ses.cancel()
            self.st.run.phase = "cancel"
            self.st.run.stage = "останавливаю"
            self.st.note("остановка по кнопке")


def _ready_line(st):
    """Что запустится по «Пуску» здесь. Заполняется оболочкой."""
    return st.ready_line or "Готов к запуску."


def _hint(st):
    """Одна фраза о том, что сейчас уместно сделать или чего ждать."""
    run, w = st.run, st.worker
    if not w.alive:
        return "Работника нет. Без него считать нечем."
    if w.booting:
        return ("Работник поднимается: " + (w.stage or "…")
                + ". Около семи секунд: коннектом на 138 639 нейронов и "
                  "матрица на 15 миллионов связей.")
    if run.phase == "prep":
        return ("Сборка тела MuJoCo, разогрев нисходящих нейронов. "
                "Около тридцати секунд, кадров до её конца не будет.")
    if run.phase == "run":
        return "Идёт счёт. «Стоп» останавливает между шагами управления."
    if run.phase == "cancel":
        return "Остановка послана, ждём конца текущего шага."
    if run.phase == "done":
        return "Прогон закончен. Результат сохранён в out/."
    if run.phase == "fail":
        return run.message or "Прогон не удался, подробности в журнале."
    last = st.last
    tail = ""
    if last is not None:
        tail = ("  Последний прогон (%s) %s."
                % (last.title or last.kind,
                   {"done": "завершён", "cancelled": "остановлен на полпути",
                    "fail": "отказал"}.get(last.outcome, last.phase)))
    if st.ws.current == "analysis":
        # В истории запускать нечего, и обещать «Пуск» здесь незачем.
        # Прошлый прогон — это запись, а не идущий счёт.
        return ("Это сохранённые прогоны, здесь ничего не считается. "
                "Выберите прогон в каталоге слева." + tail)
    return _ready_line(st) + tail


def _fmt(v, nd=2):
    s = ("{:." + str(nd) + "f}").format(float(v))
    return s.replace(".", ",")


# ------------------------------------------------- рабочие пространства

def _todo(title, lines):
    """Честная заглушка: что здесь будет и чего пока нет."""

    def centre(app, ctx, r):
        sc, f = app.sh.sc, app.sh.fonts
        inner = W.panel(ctx, r, title)
        st = L.Stack(inner, gap=sc.px(S.sm))
        W.label(ctx, st.take(f.body.gh), "Это пространство ещё не собрано.",
                role="body", color=C.dim)
        st.skip(sc.px(S.sm))
        for line in lines:
            h = W.label(ctx, L.Rect(inner.x, st.area.y, inner.w,
                                    f.small.gh * 3), "— " + line,
                        role="small", color=C.faint, wrapped=True)
            st.take(h, sc.px(S.xs))

    def inspector(app, ctx, r):
        W.label(ctx, r, "Нет настроек: пространство не собрано",
                role="small", color=C.faint, wrapped=True)

    def secondary(app, ctx, r, tab):
        W.label(ctx, r, "Пусто", role="small", color=C.faint)

    return {"centre": centre, "inspector": inspector,
            "secondary": secondary, "tabs": ("—",)}


from . import ws_analysis                                    # noqa: E402
from . import ws_body                                        # noqa: E402
from . import ws_brain                                       # noqa: E402
from . import ws_experiment                                  # noqa: E402
from . import ws_sandbox                                     # noqa: E402

SPACES = {
    "body": ws_body.SPACE,
    "brain": ws_brain.SPACE,
    "experiment": ws_experiment.SPACE,
    "sandbox": ws_sandbox.SPACE,
    "analysis": ws_analysis.SPACE,
}


# ---------------------------------------------------------------- запуск

def run(device=None, size=(1600, 900), seconds=0.0, shots=None,
        selftest=None):
    """Открыть окно. `seconds` закрывает его само, `shots` снимает кадры.

    `selftest` — путь для отчёта самопроверки. С ним окно открывается
    так же, как по двойному щелчку, но ведёт себя по короткому сценарию
    из `desk/selftest.py` и закрывается само. Нужно это ровно для одного:
    проверить собранный `FlyBrain.exe` настоящим запуском, а не словами.
    """
    from .shell import Shell
    from ..ui.session import Session

    sh = Shell("FlyBrain", size)
    ses = Session(device=device)
    app = App(sh, ses)
    script = None
    if selftest:
        from .selftest import SelfTest

        script = SelfTest(selftest)
    t0 = time.perf_counter()
    done = set()
    code = 0
    try:
        while sh.running:
            app.frame()
            now = time.perf_counter() - t0
            if shots:
                for when, path in shots:
                    if when not in done and now >= when:
                        done.add(when)
                        sh.shot(path)
            if script is not None and not script.step(app):
                break
            if seconds and now > seconds:
                break
    finally:
        if script is not None:
            code = script.finish(app, sh)
        ses.close()
        sh.close()
    if script is not None and code:
        raise SystemExit(code)
    return app
