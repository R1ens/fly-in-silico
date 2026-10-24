"""Нативное окно: свой OpenGL плюс Dear ImGui, без браузера и без Qt.

Устройство такое же, как у окна MuJoCo из примеров FlyGym: одно окно, свой
контекст, вся отрисовка наша. Вид взят у AmbientGlass — живой процедурный фон
и панели из матового стекла, которые показывают размытый фон под собой; см.
:mod:`flybrain.glass`.

Кадр собирается так:

  1. ``custom_background`` рисует сцену (фон + 138 639 нейронов) в текстуру,
     размывает её и выводит композит с панелями-стёклами;
  2. ``show_gui`` кладёт поверх виджеты ImGui с прозрачным фоном окон, так
     что видно стекло, а не серый прямоугольник темы;
  3. положение окон ImGui уходит в шейдер на следующий кадр — задержки в
     один кадр не видно.

Счёт идёт в рабочем потоке: окно не должно замирать на время прогона.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import data as D, glass, report, world

TITLE = "Мозг мухи"

PAD = 18.0
LEFT_W = 330.0
RIGHT_W = 300.0


# ----------------------------------------------------------------- камера


class Camera:
    VIEWS = [("фронт", 0.0, 0.0), ("сверху", 0.0, -1.44),
             ("сбоку", math.pi / 2, 0.0), ("3/4", -0.6, -0.35)]

    def __init__(self, radius):
        self.radius = radius
        self.yaw = 0.0
        self.pitch = 0.0
        self.dist = radius * 2.05
        self.view = 0

    def next_view(self):
        self.view = (self.view + 1) % len(self.VIEWS)
        _, self.yaw, self.pitch = self.VIEWS[self.view]
        return self.VIEWS[self.view][0]

    def orbit(self, dx, dy):
        self.yaw += dx * 0.006
        self.pitch = float(np.clip(self.pitch + dy * 0.006, -1.5, 1.5))

    def zoom(self, wheel):
        self.dist = float(np.clip(self.dist * (1.0 - wheel * 0.09),
                                  self.radius * 0.7, self.radius * 9.0))

    def mvp(self, aspect, x_shift=0.0):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        eye = np.array([math.sin(self.yaw) * cp * self.dist,
                        -sp * self.dist,
                        math.cos(self.yaw) * cp * self.dist], np.float32)
        return _perspective(0.82, aspect, self.radius * 0.1, self.radius * 14,
                            x_shift) @ \
            _look_at(eye, np.zeros(3, np.float32), np.array([0, 1, 0], np.float32))


def _perspective(fovy, aspect, near, far, x_shift=0.0):
    """``x_shift`` сдвигает фрустум по горизонтали в координатах отсечения:
    так мозг оказывается посередине свободного места, а не посередине окна,
    которое с боков закрыто панелями."""
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[0, 2] = x_shift
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up):
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = x, y, z
    m[:3, 3] = -np.array([x @ eye, y @ eye, z @ eye])
    return m


# ------------------------------------------------------------- состояние


@dataclass
class UiState:
    query: str = "class:gustatory+side:left"
    silence: str = ""
    rate: float = 200.0
    pulse_on: float = 100.0
    pulse_off: float = 400.0
    duration: float = 0.6
    bin_ms: float = 10.0

    readout: int = 0
    dn_tau: float = 0.5
    arena_t: float = 5.0
    swap: bool = False

    mode_regions: bool = False
    only_active: bool = False
    show_points: bool = True
    tab: int = 0

    status: str = "готов"
    hint: str = ""
    running: bool = False
    t_ms: float = 0.0
    n_active: int = 0
    peak_hz: float = 0.0
    top_region: str = "—"
    summary: list = field(default_factory=list)
    regions: list = field(default_factory=list)


class Session:
    """Коннектом, веса и текущий прогон. Живёт всё время работы окна."""

    def __init__(self, device=None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("загружаю коннектом...")
        self.neurons = D.load_neurons()
        self.weights = D.load_weights(device=self.device)
        self.n = len(self.neurons)

        coords = self.neurons[["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        good = np.isfinite(coords).all(axis=1)
        centre = np.nanmean(coords[good], axis=0)
        coords = np.where(good[:, None], coords, centre[None, :]).astype(np.float32)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        c = (lo + hi) / 2
        self.pos = np.stack([coords[:, 0] - c[0],
                             -(coords[:, 1] - c[1]),
                             -(coords[:, 2] - c[2])], axis=1).astype(np.float32)
        self.radius = float(np.max(hi - lo) * 0.62)

        self.region_names = sorted(self.neurons["region"].unique().tolist())
        code = {r: i for i, r in enumerate(self.region_names)}
        self.region_code = self.neurons["region"].map(code).to_numpy(np.float32)
        colours = report.region_colors(self.region_names)
        self.region_rgb = np.zeros((64, 3), np.float32)
        for i, r in enumerate(self.region_names[:64]):
            h = colours[r].lstrip("#")
            self.region_rgb[i] = [int(h[j:j + 2], 16) / 255 for j in (0, 2, 4)]
        self.region_sizes = np.array(
            [max(int((self.neurons["region"] == r).sum()), 1)
             for r in self.region_names], np.float32)

        self.readouts = sorted(world.probe_candidates(self.neurons))
        self.act = np.zeros(self.n, np.float32)      # сглаженная частота, Гц
        self.disp = np.zeros(self.n, np.float32)     # 0..1 для шейдера
        self.scale = 60.0
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = threading.Event()
        print(f"готово: {self.n} нейронов на {self.device}")

    # ---- приём кадров от счёта ----

    def push(self, t_ms, counts, bin_ms):
        hz_per_spike = 1000.0 / max(bin_ms, 1e-6)
        decay = math.exp(-bin_ms / 50.0)
        with self.lock:
            self.act *= decay
            nz = np.nonzero(counts)[0]
            if len(nz):
                self.act[nz] += counts[nz] * hz_per_spike * (1 - decay)
            peak = float(self.act.max()) if self.n else 0.0
            self.scale = max(self.scale * 0.99, min(400.0, max(10.0, peak)))
            np.divide(self.act, self.scale, out=self.disp)
            self.frame_t = t_ms
            self.frame_peak = peak
            self.frame_n = int(len(nz))

    def reset_activity(self):
        with self.lock:
            self.act[:] = 0
            self.disp[:] = 0
            self.scale = 60.0

    def region_levels(self):
        """Средняя активность по областям — для панели и для цвета фона."""
        with self.lock:
            a = self.act
            acc = np.bincount(self.region_code.astype(np.int64), weights=a,
                              minlength=len(self.region_names))
        return acc / self.region_sizes

    def busy(self):
        return self.thread is not None and self.thread.is_alive()


# ------------------------------------------------------------------ окно


def run(device=None):
    import moderngl
    from imgui_bundle import imgui, hello_imgui, ImVec2, ImVec4

    sess = Session(device=device)
    ui = UiState()
    cam = Camera(sess.radius)
    gl: dict = {}
    panels: list[tuple[float, float, float, float]] = []
    t_start = time.perf_counter()
    pulse = {"v": 0.0, "last": 0.0}

    # ---------------------------------------------------------- GL

    def init_gl():
        ctx = moderngl.create_context()
        gl["ctx"] = ctx
        quad = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        gl["quad_vbo"] = ctx.buffer(quad.tobytes())

        gl["bg"] = ctx.program(vertex_shader=glass.QUAD_VS,
                               fragment_shader=glass.BACKGROUND_FS)
        gl["blur"] = ctx.program(vertex_shader=glass.QUAD_VS,
                                 fragment_shader=glass.BLUR_FS)
        gl["comp"] = ctx.program(vertex_shader=glass.QUAD_VS,
                                 fragment_shader=glass.COMPOSITE_FS)
        gl["pts"] = ctx.program(vertex_shader=glass.POINTS_VS,
                                fragment_shader=glass.POINTS_FS)

        gl["bg_vao"] = ctx.vertex_array(gl["bg"], [(gl["quad_vbo"], "2f", "in_pos")])
        gl["blur_vao"] = ctx.vertex_array(gl["blur"], [(gl["quad_vbo"], "2f", "in_pos")])
        gl["comp_vao"] = ctx.vertex_array(gl["comp"], [(gl["quad_vbo"], "2f", "in_pos")])

        gl["pos_vbo"] = ctx.buffer(sess.pos.tobytes())
        gl["reg_vbo"] = ctx.buffer(sess.region_code.tobytes())
        gl["act_vbo"] = ctx.buffer(sess.disp.tobytes(), dynamic=True)
        gl["pts_vao"] = ctx.vertex_array(gl["pts"], [
            (gl["pos_vbo"], "3f", "in_pos"),
            (gl["reg_vbo"], "1f", "in_reg"),
            (gl["act_vbo"], "1f", "in_act"),
        ])
        gl["pts"]["uRegCol"].write(sess.region_rgb.tobytes())
        gl["size"] = (0, 0)

    def ensure_targets(w, h):
        if gl.get("size") == (w, h) or w < 8 or h < 8:
            return
        ctx = gl["ctx"]
        for key in ("scene_fbo", "blur_a", "blur_b"):
            if key in gl:
                gl[key].release()
        gl["scene_tex"] = ctx.texture((w, h), 3, dtype="f2")
        gl["scene_tex"].filter = (moderngl.LINEAR, moderngl.LINEAR)
        gl["scene_fbo"] = ctx.framebuffer([gl["scene_tex"]])
        hw, hh = max(w // 3, 2), max(h // 3, 2)
        gl["ta"] = ctx.texture((hw, hh), 3, dtype="f2")
        gl["tb"] = ctx.texture((hw, hh), 3, dtype="f2")
        for t in (gl["ta"], gl["tb"]):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        gl["blur_a"] = ctx.framebuffer([gl["ta"]])
        gl["blur_b"] = ctx.framebuffer([gl["tb"]])
        gl["size"] = (w, h)
        gl["half"] = (hw, hh)

    def draw_background():
        io = imgui.get_io()
        w = max(int(io.display_size.x), 8)
        h = max(int(io.display_size.y), 8)
        ctx = gl["ctx"]
        ensure_targets(w, h)

        levels = sess.region_levels()
        order = np.argsort(-levels)
        top = order[0] if len(order) else 0
        second = order[1] if len(order) > 1 else top
        lvl = float(np.clip(levels[top] / 40.0, 0.0, 1.0)) if len(order) else 0.0
        accent = sess.region_rgb[top % 64] if levels[top] > 0.05 else np.array(
            [0.32, 0.46, 0.72], np.float32)
        accent2 = sess.region_rgb[second % 64] if levels[second] > 0.05 else np.array(
            [0.55, 0.30, 0.62], np.float32)

        now = time.perf_counter()
        total = float(levels.sum())
        rise = max(0.0, total - pulse["last"])
        pulse["last"] = total
        pulse["v"] = max(pulse["v"] * 0.90, min(1.0, rise / 6.0))

        t = now - t_start

        # 1. сцена в текстуру
        gl["scene_fbo"].use()
        ctx.clear(0.0, 0.0, 0.0)
        ctx.disable(moderngl.DEPTH_TEST)
        p = gl["bg"]
        p["uSize"].value = (float(w), float(h))
        p["uTime"].value = t
        p["uAccent"].value = tuple(float(x) for x in accent)
        p["uAccent2"].value = tuple(float(x) for x in accent2)
        p["uPulse"].value = float(pulse["v"])
        p["uLevel"].value = lvl
        p["uCentre"].value = (w * 0.5, h * 0.46)
        gl["bg_vao"].render(moderngl.TRIANGLES)

        if ui.show_points:
            with sess.lock:
                gl["act_vbo"].write(sess.disp.tobytes())
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            q = gl["pts"]
            # Центр свободного места между панелями, в координатах отсечения.
            free_c = (LEFT_W + 2 * PAD + (w - RIGHT_W - PAD)) * 0.5
            shift = (2.0 * free_c / w - 1.0)
            q["uMVP"].write(cam.mvp(w / h, shift).T.astype(np.float32).tobytes())
            q["uPointScale"].value = 2.9 * max(1.0, h / 900.0)
            q["uMode"].value = 1.0 if ui.mode_regions else 0.0
            q["uFocus"].value = -1.0
            gl["pts_vao"].render(moderngl.POINTS, vertices=sess.n)
            ctx.disable(moderngl.BLEND)

        # 2. размытие в две проходки на трети разрешения
        hw, hh = gl["half"]
        gl["blur_a"].use()
        gl["scene_tex"].use(0)
        gl["blur"]["uTex"].value = 0
        gl["blur"]["uDir"].value = (2.4 / hw, 0.0)
        gl["blur_vao"].render(moderngl.TRIANGLES)
        gl["blur_b"].use()
        gl["ta"].use(0)
        gl["blur"]["uDir"].value = (0.0, 2.4 / hh)
        gl["blur_vao"].render(moderngl.TRIANGLES)

        # 3. композит на экран
        ctx.screen.use()
        gl["scene_tex"].use(0)
        gl["tb"].use(1)
        c = gl["comp"]
        c["uScene"].value = 0
        c["uBlur"].value = 1
        c["uSize"].value = (float(w), float(h))
        c["uRadius"].value = 14.0
        c["uTint"].value = 0.045
        c["uFrost"].value = 1.0
        c["uAccent"].value = tuple(float(x) for x in accent)
        c["uPanelCount"].value = min(len(panels), glass.MAX_PANELS)
        flat = np.zeros((glass.MAX_PANELS, 4), np.float32)
        for i, r in enumerate(panels[:glass.MAX_PANELS]):
            flat[i] = r
        c["uPanels"].write(flat.tobytes())
        gl["comp_vao"].render(moderngl.TRIANGLES)

    # ---------------------------------------------------------- счёт

    def start_stim():
        from . import sim

        def work():
            try:
                df = sess.neurons
                idx = D.select(df, ui.query)
                if len(idx) == 0:
                    ui.hint = "запрос ничего не нашёл"
                    return
                sil = D.select(df, ui.silence) if ui.silence.strip() else np.array([], np.int64)
                sess.reset_activity()
                stim = sim.Stimulus(idx, ui.rate, t_on_ms=ui.pulse_on,
                                    t_off_ms=ui.pulse_off or None, name=ui.query)
                cfg = sim.RunConfig(t_run_s=ui.duration, bin_ms=ui.bin_ms,
                                    device=sess.device)
                ui.status = "считаю..."
                res = sim.run([stim], silenced=sil, config=cfg, neurons=df,
                              weights=sess.weights, progress=False,
                              on_bin=lambda b, t, c: sess.push(t, c, ui.bin_ms),
                              should_stop=sess.stop_flag.is_set)
                rt = res.rate_table()
                ui.summary = [
                    ("активных", f"{len(rt)}"),
                    ("спайков", f"{int(res.counts.sum()):,}".replace(",", " ")),
                    ("счёт", f"{res.timing['sim_s']} с"),
                ]
                ui.status = "готово"
            except Exception as exc:  # noqa: BLE001
                ui.status = "ошибка"
                ui.hint = str(exc)[:160]
            finally:
                ui.running = False

        sess.stop_flag.clear()
        ui.running = True
        sess.thread = threading.Thread(target=work, daemon=True)
        sess.thread.start()

    def start_body():
        from . import body

        def work():
            try:
                sess.reset_activity()
                cfg = body.BodyConfig(
                    duration_s=ui.arena_t, device=sess.device,
                    readout=sess.readouts[ui.readout], dn_tau_s=ui.dn_tau,
                    swap_antennae=ui.swap, view=True)
                ui.status = "тело: окно MuJoCo открыто"
                run_ = body.run_episode(
                    cfg=cfg, neurons=sess.neurons, weights=sess.weights,
                    progress=False,
                    on_step=lambda s, rec, counts: sess.push(
                        rec["t_ms"], counts, cfg.control_ms),
                    should_stop=sess.stop_flag.is_set)
                m = run_.metrics
                ui.summary = [
                    ("градиент", f"{m['grad_alignment']:+.3f}"),
                    ("до еды", f"{m['dist_start_mm']:.1f} → {m['dist_end_mm']:.1f} мм"),
                    ("скорость", f"{m['speed_mm_s']} мм/с"),
                ]
                ui.status = "готово"
            except Exception as exc:  # noqa: BLE001
                ui.status = "ошибка"
                ui.hint = str(exc)[:160]
            finally:
                ui.running = False

        sess.stop_flag.clear()
        ui.running = True
        sess.thread = threading.Thread(target=work, daemon=True)
        sess.thread.start()

    # ---------------------------------------------------------- виджеты

    def style():
        s = imgui.get_style()
        s.window_rounding = 14.0
        s.frame_rounding = 7.0
        s.grab_rounding = 7.0
        s.window_padding = ImVec2(16, 14)
        s.item_spacing = ImVec2(9, 8)
        s.frame_padding = ImVec2(9, 6)
        s.window_border_size = 0.0
        C = imgui.Col_
        # Фон окон прозрачный: под ними стекло из шейдера, а не заливка темы.
        for col, v in (
            (C.window_bg, (0, 0, 0, 0)),
            (C.child_bg, (0, 0, 0, 0)),
            (C.popup_bg, (0.05, 0.07, 0.10, 0.96)),
            (C.frame_bg, (1, 1, 1, 0.06)),
            (C.frame_bg_hovered, (1, 1, 1, 0.11)),
            (C.frame_bg_active, (1, 1, 1, 0.15)),
            (C.button, (1, 1, 1, 0.09)),
            (C.button_hovered, (1, 0.70, 0.35, 0.28)),
            (C.button_active, (1, 0.70, 0.35, 0.45)),
            (C.slider_grab, (1, 0.70, 0.35, 0.85)),
            (C.slider_grab_active, (1, 0.78, 0.45, 1.0)),
            (C.check_mark, (1, 0.70, 0.35, 1.0)),
            (C.text, (0.90, 0.94, 0.99, 1.0)),
            (C.text_disabled, (0.55, 0.61, 0.69, 1.0)),
            (C.header, (1, 1, 1, 0.08)),
            (C.header_hovered, (1, 1, 1, 0.13)),
            (C.separator, (1, 1, 1, 0.10)),
            (C.title_bg, (0, 0, 0, 0)),
            (C.title_bg_active, (0, 0, 0, 0)),
            (C.scrollbar_bg, (0, 0, 0, 0)),
        ):
            s.set_color_(col.value, ImVec4(*v))

    def gui():
        nonlocal panels
        io = imgui.get_io()
        w, h = io.display_size.x, io.display_size.y
        rects = []
        W = imgui.WindowFlags_
        flags = (W.no_collapse.value | W.no_title_bar.value
                 | W.no_bring_to_front_on_focus.value | W.no_move.value
                 | W.no_resize.value | W.no_scrollbar.value
                 | W.no_saved_settings.value)

        # ---- левая панель ----
        pad, pw = PAD, LEFT_W
        imgui.set_next_window_pos(ImVec2(pad, pad), imgui.Cond_.always.value)
        imgui.set_next_window_size(ImVec2(pw, h - 2 * pad), imgui.Cond_.always.value)
        imgui.begin("##left", None, flags)
        rects.append((pad, pad, pad + pw, h - pad))

        imgui.text_colored(ImVec4(1, 0.70, 0.35, 1.0), TITLE.upper())
        imgui.text_disabled(f"{sess.n} нейронов · {sess.device}")
        imgui.dummy(ImVec2(0, 6))

        if imgui.button("Стимуляция", ImVec2(142, 30)):
            ui.tab = 0
        imgui.same_line()
        if imgui.button("Тело", ImVec2(142, 30)):
            ui.tab = 1
        imgui.dummy(ImVec2(0, 8))

        if ui.tab == 0:
            imgui.text_disabled("КОГО ЗАЖЕЧЬ")
            changed, ui.query = imgui.input_text("##q", ui.query)
            if changed:
                try:
                    k = len(D.select(sess.neurons, ui.query))
                    ui.hint = f"{k} нейронов" if k else "ничего не найдено"
                except ValueError as exc:
                    ui.hint = str(exc)[:80]
            imgui.text_disabled(ui.hint or " ")
            _, ui.rate = imgui.slider_float("Гц", ui.rate, 10.0, 400.0)
            _, ui.pulse_on = imgui.slider_float("с, мс", ui.pulse_on, 0.0, 1000.0)
            _, ui.pulse_off = imgui.slider_float("по, мс", ui.pulse_off, 0.0, 2000.0)
            imgui.dummy(ImVec2(0, 4))
            imgui.text_disabled("КОГО ВЫКЛЮЧИТЬ")
            _, ui.silence = imgui.input_text("##sil", ui.silence)
            _, ui.duration = imgui.slider_float("длит., с", ui.duration, 0.1, 5.0)
        else:
            imgui.text_disabled("ВЫХОД НА РУЛЁЖКУ")
            _, ui.readout = imgui.combo("##ro", ui.readout, sess.readouts, 12)
            _, ui.dn_tau = imgui.slider_float("tau, с", ui.dn_tau, 0.05, 1.5)
            _, ui.arena_t = imgui.slider_float("длит., с", ui.arena_t, 1.0, 20.0)
            _, ui.swap = imgui.checkbox("контроль: переставить антенны", ui.swap)
            imgui.text_disabled("тело идёт в отдельном окне MuJoCo")

        imgui.dummy(ImVec2(0, 10))
        imgui.separator()
        imgui.dummy(ImVec2(0, 6))

        imgui.begin_disabled(ui.running)
        if imgui.button("Пуск", ImVec2(142, 34)):
            start_stim() if ui.tab == 0 else start_body()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not ui.running)
        if imgui.button("Стоп", ImVec2(142, 34)):
            sess.stop_flag.set()
        imgui.end_disabled()

        imgui.dummy(ImVec2(0, 6))
        imgui.text_disabled(ui.status)
        imgui.dummy(ImVec2(0, 10))
        _, ui.mode_regions = imgui.checkbox("раскрасить по областям", ui.mode_regions)
        _, ui.show_points = imgui.checkbox("показывать нейроны", ui.show_points)
        if imgui.button(f"вид: {Camera.VIEWS[cam.view][0]}", ImVec2(142, 28)):
            cam.next_view()
        imgui.end()

        # ---- правая панель ----
        rw = RIGHT_W
        rx = w - rw - pad
        imgui.set_next_window_pos(ImVec2(rx, pad), imgui.Cond_.always.value)
        imgui.set_next_window_size(ImVec2(rw, h - 2 * pad), imgui.Cond_.always.value)
        imgui.begin("##right", None, flags)
        rects.append((rx, pad, rx + rw, h - pad))

        with sess.lock:
            t_ms = getattr(sess, "frame_t", 0.0)
            peak = getattr(sess, "frame_peak", 0.0)
            n_act = getattr(sess, "frame_n", 0)
        imgui.text_colored(ImVec4(1, 0.70, 0.35, 1.0), f"{t_ms:.0f} мс")
        imgui.text_disabled(f"активно {n_act}    пик {peak:.0f} Гц")
        imgui.dummy(ImVec2(0, 8))

        imgui.text_disabled("ОБЛАСТИ СЕЙЧАС")
        levels = sess.region_levels()
        order = np.argsort(-levels)[:14]
        top = float(levels[order[0]]) if len(order) else 1.0
        for i in order:
            v = float(levels[i])
            if v <= 0.01:
                continue
            name = sess.region_names[i]
            col = sess.region_rgb[i % 64]
            imgui.text_colored(ImVec4(*col, 1.0), f"{name:<6}")
            imgui.same_line(84)
            imgui.push_style_color(imgui.Col_.plot_histogram.value,
                                   ImVec4(*col, 0.85))
            imgui.progress_bar(min(v / max(top, 1e-6), 1.0), ImVec2(150, 11), "")
            imgui.pop_style_color()
            imgui.same_line()
            imgui.text_disabled(f"{v:.1f}")

        if ui.summary:
            imgui.dummy(ImVec2(0, 12))
            imgui.text_disabled("ИТОГ")
            for k, v in ui.summary:
                imgui.text_disabled(k)
                imgui.same_line(150)
                imgui.text(str(v))
        imgui.end()

        panels = rects

        # ---- мышь для камеры там, где нет окон ----
        if not io.want_capture_mouse:
            if imgui.is_mouse_dragging(0):
                d = imgui.get_io().mouse_delta
                cam.orbit(d.x, d.y)
            if abs(io.mouse_wheel) > 0:
                cam.zoom(io.mouse_wheel)

    # ---------------------------------------------------------- запуск

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE
    params.app_window_params.window_geometry.size = (1500, 900)
    params.imgui_window_params.show_menu_bar = False
    params.imgui_window_params.default_imgui_window_type = \
        hello_imgui.DefaultImGuiWindowType.no_default_window
    params.fps_idling.enable_idling = False
    params.callbacks.post_init = init_gl
    params.callbacks.setup_imgui_style = style
    params.callbacks.custom_background = draw_background
    params.callbacks.show_gui = gui
    hello_imgui.run(params)
