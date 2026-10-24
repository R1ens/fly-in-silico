"""Окно приложения: главный цикл и состояние сеанса.

Здесь нет ни одного вызова рисования — всё отдано панелям, — и нет ни одного
обращения к движку: мозг считает работник за процессом, а окно только
спрашивает `Session`. Из-за этого окно не импортирует ни torch, ни mujoco и
не блокируется ни на чём.
"""

import time

import numpy as np
import taichi as ti

from . import brainview, gpu, kit, layout, panels, shell, theme
from .. import spec as S
from .query import QueryBuilder
from .session import Session

__all__ = ["AppState", "run"]


class AppState:
    """Всё, что окно помнит между кадрами."""

    def __init__(self):
        self.mode = "stim"
        self.want_run = False
        self.save_name = ""

        self.qb = QueryBuilder.parse("class:gustatory+side:left")
        self.key_i = 1
        self.val_i = -1

        self.rate = 200.0
        self.t_stim = 0.8
        self.bin_ms = 10.0

        self.readout = "ALPN"
        self.t_body = 4.0
        self.dn_tau = 0.5
        self.swap = False

        # арена
        self.t_arena = 12.0
        self.arena_readout = ""
        self.comment = False

        # зонд стороны
        self.t_lateral = 10.0
        self.base_hz = 75.0

        # серия вместо одного прогона
        self.series = False
        self.series_n = 6
        self.series_t = 4.0

        # демо
        self.demo_i = 0

        # агент
        self.goal_i = 0
        self.agent_steps = 6

        self.track = []
        self.track_t = None
        self.focus_click = None
        self.rail_open = True
        self.runs = None          # состояние обзора прогонов


def _spec(st):
    """Собрать `RunSpec` ровно так же, как это делает командная строка.

    Окно не лепит свой словарь параметров: схема одна, и всё, что здесь
    выбрано кнопками, разбирается тем же `jobs.run`, что и `fly run`.
    """
    sp = S.RunSpec(kind=st.mode, open_report=False)

    if st.mode == "stim":
        sp.t = float(st.t_stim)
        sp.bin_ms = float(st.bin_ms)
        sp.stim = S.StimSpec(activate=[
            S.StimTerm(query=st.qb.text(), rate_hz=float(st.rate))])
    elif st.mode == "arena":
        sp.t = float(st.t_arena)
        sp.bin_ms = 20.0
        sp.comment = bool(st.comment)
        sp.arena = S.ArenaSpec(readout=st.arena_readout or None,
                               dn_tau_s=float(st.dn_tau))
    elif st.mode == "body":
        sp.t = float(st.t_body)
        sp.bin_ms = 20.0
        sp.body = S.BodySpec(readout=st.readout, dn_tau_s=float(st.dn_tau),
                             swap_antennae=bool(st.swap))
    elif st.mode == "lateral":
        sp.t = float(st.t_lateral)
        sp.lateral = S.LateralSpec(base_hz=float(st.base_hz))
    elif st.mode == "agent":
        sp.agent = S.AgentSpec(goal=S.GOALS[st.goal_i % len(S.GOALS)],
                               max_steps=int(st.agent_steps))
    elif st.mode == "demos":
        sp = S.from_dict(S.DEMOS[st.demo_i % len(S.DEMOS)].to_dict())
        sp.open_report = False
        sp.out = None          # демо из окна не затирают сохранённое
        return sp

    if st.series and st.mode in ("arena", "body"):
        sp.repeat = int(st.series_n)
        sp.repeat_t = float(st.series_t)
    if st.save_name:
        sp.out = st.save_name
    return sp


class Scene:
    """Всё, что нужно нарисовать кадр, кроме самого окна.

    Вынесено из цикла нарочно: тогда кадр собирается и без окна — а в этой
    среде окно самовольно меняет размер, и проверить вид в задуманном размере
    иначе нельзя.
    """

    def __init__(self, sess, st=None):
        self.sess = sess
        self.st = st or AppState()
        self.bv = None
        self.names = self.code = self.counts = self.cols = None

    def ensure_map(self):
        if self.bv is None and self.sess.ready:
            pos, code, names, rgb, cols = self.sess.geometry()
            self.code, self.names, self.cols = code, names, cols
            self.counts = np.bincount(code, minlength=len(names))
            self.bv = brainview.BrainView(pos, code, names, rgb, self.counts)
        return self.bv

    def keys(self, inp, on_quit=None):
        bv = self.bv
        if inp.pressed("escape") and on_quit:
            on_quit()
        if bv is None:
            return
        if inp.pressed(" "):
            bv.next_view()
        if inp.pressed("l"):
            bv.show_labels = not bv.show_labels
        if inp.pressed("o"):
            bv.only_active = not bv.only_active
        if inp.pressed("up"):
            inp.scroll(1)
        if inp.pressed("down"):
            inp.scroll(-1)

    def draw(self, ui, surf, fps=0.0):
        sess, st = self.sess, self.st
        self.ensure_map()
        lay = layout.compute(surf.w, surf.h, ui.scale,
                             strip=(st.mode in ("body", "arena")),
                             rail_open=st.rail_open)
        inp = ui.input
        bv = self.bv

        if st.mode == "runs" and lay.has("brain"):
            self._runs(ui, kit.box(*lay.brain))
        elif bv is not None and lay.has("brain"):
            area = kit.box(*lay.brain)
            area = panels.brain_toolbar(ui, bv, area)
            vp = (area.x0, area.y0, area.w, area.h)
            bv.set_activity(sess.disp if sess.disp is not None else bv.act)
            over_ui = ((lay.rail and inp.x < lay.rail[2])
                       or (lay.meters and inp.x > lay.meters[0]))
            bv.interact(vp, (inp.x, inp.y), inp.down and not over_ui,
                        inp.wheel if not over_ui else 0.0)
            # Подписи нейропилей идут своим списком: в общем буфере деталей
            # глифы рисуются атласом основного кегля, а карте нужен мелкий —
            # иначе подписи просто не выкладываются.
            bdl = gpu.DrawList()
            bv.draw(surf, bdl, ui.small, vp)
            bdl.frame(vp[0] - 1, vp[1] - 1, vp[0] + vp[2] + 1,
                      vp[1] + vp[3] + 1, theme.HAIRLINE)
            surf.draw(bdl, ui.small)
            if not over_ui:
                bv.pick((inp.x, inp.y))
                panels.hover_tip(ui, sess, bv, inp, vp)

        if lay.has("rail"):
            panels.rail(ui, sess, st, kit.box(*lay.rail))
        if lay.has("meters") and bv is not None:
            panels.meters(ui, sess, st, kit.box(*lay.meters),
                          self.names, self.code, self.counts, self.cols)
        if lay.has("strip"):
            panels.strip(ui, sess, st, kit.box(*lay.strip))
        panels.foot(ui, sess, _FootInfo(surf.w, surf.h, fps),
                    kit.box(*lay.foot), lay)

        if st.focus_click is not None and bv is not None:
            bv.set_focus(None if bv.focus == st.focus_click else st.focus_click)
            st.focus_click = None
        return lay

    def _runs(self, ui, area):
        """Обзор сохранённых прогонов вместо карты.

        Панель подключается по требованию: окно должно открываться и
        работать, даже если её ещё нет.
        """
        st = self.st
        try:
            from . import runs as R
        except ImportError:
            ui.dl.rect(area.x0, area.y0, area.x1, area.y1, theme.PANEL)
            ui.dl.frame(area.x0, area.y0, area.x1, area.y1, theme.HAIRLINE)
            ui._txt_centre(ui.small, "обзор прогонов ещё не собран",
                           area.x0, area.x1, ui._mid(area, ui.small),
                           theme.FAINT)
            return
        if st.runs is None:
            st.runs = R.RunsState()
        R.panel(ui, st.runs, area,
                on_open=_open_file,
                on_rerun_spec=lambda d: self._submit_raw(d))

    def _submit_raw(self, raw):
        """Запустить прогон по готовой спеке — например из списка популяций."""
        if not self.sess.busy and self.sess.ready:
            self.st.track = []
            self.st.track_t = None
            self.sess.submit(dict(raw))

    def apply(self):
        """Поставить задачу, если нажали «Пуск»."""
        st, sess = self.st, self.sess
        if not st.want_run:
            return
        st.want_run = False
        sp = _spec(st)
        if sp.kind == "stim" and not (sp.stim.activate
                                      and sp.stim.activate[0].query):
            sess.log.append("нечего зажигать: запрос пуст")
        else:
            st.track = []
            st.track_t = None
            sess.submit(sp.to_dict())


def _open_file(path):
    """Открыть отчёт в том, чем система открывает html.

    Браузер здесь не интерфейс, а просмотрщик готового артефакта: окно его
    не встраивает и не зависит от него.
    """
    import webbrowser

    try:
        webbrowser.open(str(path))
    except Exception:                                          # noqa: BLE001
        pass


class _FootInfo:
    """Подвалу нужны только размер и частота кадров, не всё окно."""

    def __init__(self, w, h, fps):
        self.w, self.h, self.fps = w, h, fps


def run(device=None, seconds=0.0, shot=None, auto=None):
    sh = shell.Shell("Мозг мухи", (1500, 880))
    ui = kit.Ui()
    sess = Session(device)
    scene = Scene(sess)
    st = scene.st
    started = last = time.perf_counter()
    shot_done = False

    while sh.running:
        if seconds and time.perf_counter() - started > seconds:
            break

        w, h = sh.window.get_window_shape()
        sh.surf.ensure(int(w), int(h))
        ui.input.pull(sh.window, sh.surf.w, sh.surf.h, ti.ui)
        ui.begin(sh.surf.w, sh.surf.h)
        sh.surf.clear(theme.BG)

        now = time.perf_counter()
        sess.pump(now - last)
        last = now

        scene.ensure_map()
        scene.keys(ui.input, on_quit=sh.stop)
        scene.draw(ui, sh.surf, sh.fps)
        scene.apply()

        ui.flush(sh.surf)
        sh.finish()

        # Снимок для проверки вида: снимается чтением поля, потому что
        # снимок swapchain в этой среде выходит чёрным.
        if shot and not shot_done and sess.frames_seen > 25:
            sh.shot(shot)
            shot_done = True
        if auto and sess.ready and not sess.busy and sess.state != "done"                 and time.perf_counter() - started > 2.0:
            st.mode = auto
            st.want_run = True

    if seconds:
        print("состояние: " + str(sess.state)
              + " | кадров: " + str(sess.frames_seen)
              + " | потеряно: " + str(sess.frames_dropped)
              + " | активных: " + str(sess.n_active)
              + " | пик: " + format(sess.peak_hz, ".1f")
              + " | шкала: " + format(sess.scale, ".1f"))
        print("размеры окна: " + str(sh.resizes))
        print("журнал: " + " / ".join(x[:60] for x in sess.log[-4:]))
    sess.close()
    sh.close()
