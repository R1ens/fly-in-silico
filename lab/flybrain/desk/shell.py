"""Окно: жизненный цикл, размер, масштаб, ввод, кадр.

Здесь собрано всё, что приходится знать про повадки GGUI и Windows, и ни
один компонент про это знать не должен.

Повадки, стоившие проверки:

1. Размер окна меняется сам, без события: окно, открытое как 1500×880,
   в замерах становилось 803×984, а потом 755×984. Поэтому размер
   опрашивается каждый кадр, а поля кадра пересоздаются.
2. Снимок swapchain (`save_image`, `get_image_buffer_as_numpy`) из-под
   автоматики отдаёт чёрное. Смотреть на кадр надо чтением накопителя, что
   и делает `shot`.
3. Масштаб DPI нельзя брать «как есть». Если процесс не объявил себя
   осведомлённым о DPI, Windows растягивает окно сама, и ещё один
   коэффициент сверху сделал бы интерфейс вдвое крупнее нужного. Поэтому
   `_dpi` сперва спрашивает осведомлённость процесса и только потом — DPI
   окна.
4. Курсор GGUI отсчитывается снизу, а рисование — сверху.
"""

import time

import numpy as np

from ..ui import gpu
from . import keys as keysmod
from .canvas import Canvas
from .tokens import C, Scale
from .widgets import Ctx, Fonts, Input

__all__ = ["Shell", "Offscreen", "dpi_report"]


def _dpi_report():
    """Осведомлённость процесса о DPI и DPI окна. Отдельно, чтобы врать
    было нечем: в отчёте о проверке видно, что именно измерено."""
    out = {"aware": None, "window_dpi": None, "system_dpi": None,
            "scale": 1.0, "note": ""}
    try:
        import ctypes
        from ctypes import wintypes

        shcore = ctypes.windll.shcore
        user32 = ctypes.windll.user32
        val = ctypes.c_int(0)
        if shcore.GetProcessDpiAwareness(None, ctypes.byref(val)) == 0:
            out["aware"] = {0: "unaware", 1: "system", 2: "per-monitor"}.get(
                val.value, str(val.value))
        dc = user32.GetDC(0)
        out["system_dpi"] = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)
        user32.ReleaseDC(0, dc)
        del wintypes
    except Exception as exc:                       # pragma: no cover
        out["note"] = repr(exc)
    return out


def dpi_report(hwnd=None):
    rep = _dpi_report()
    if hwnd:
        try:
            import ctypes
            rep["window_dpi"] = ctypes.windll.user32.GetDpiForWindow(hwnd)
        except Exception as exc:                   # pragma: no cover
            rep["note"] = (rep["note"] + " " + repr(exc)).strip()
    # Коэффициент берётся только там, где кадр действительно в настоящих
    # пикселях. У неосведомлённого процесса растягивает сама Windows.
    if rep["aware"] in ("system", "per-monitor"):
        d = rep["window_dpi"] or rep["system_dpi"] or 96
        rep["scale"] = round(d / 96.0, 4)
        rep["note"] = (rep["note"] + " масштаб считаем сами").strip()
    else:
        rep["note"] = (rep["note"] +
                       " окно растягивает Windows, свой масштаб = 1").strip()
    return rep


class Shell:
    """Окно, поверхность кадра, шрифты, ввод и контекст компонентов."""

    def __init__(self, title="FlyBrain", size=(1600, 900), pos=(60, 40),
                 vsync=True, native_keys=True, dpi=None):
        import taichi as ti

        self.ti = ti
        ti.init(arch=ti.vulkan, offline_cache=True)
        self.title = title
        self.window = ti.ui.Window(title, size, vsync=vsync, pos=pos)
        self.gui_canvas = self.window.get_canvas()
        self.surf = gpu.Surface()

        self.keys = keysmod.Keys(self.window, title=title,
                                 native=native_keys)
        self.dpi_info = dpi_report(self.keys.hwnd)
        self.sc = Scale(dpi=self.dpi_info["scale"] if dpi is None
                        else float(dpi), w=size[0], h=size[1])
        self.fonts = Fonts(self.sc)

        self.inp = Input()
        self.main = Canvas()
        self.overlay = Canvas()
        self.ctx = Ctx(self.main, self.overlay, self.fonts, self.sc,
                       self.inp, now=0.0)

        self.fps = 0.0
        self.dt = 1.0 / 60.0        # настоящее время между кадрами
        self.launches = 0
        self.frame_ms = 0.0
        self._frames = 0
        self._tmark = time.perf_counter()
        self._last = self._tmark
        self._t0 = self._tmark
        self._was_down = False
        self._click_at = (0, 0, 0.0)
        self.resizes = []

    # ------------------------------------------------------------------

    @property
    def w(self):
        return self.surf.w

    @property
    def h(self):
        return self.surf.h

    @property
    def running(self):
        return self.window.running

    def stop(self):
        self.window.running = False

    def close(self):
        try:
            self.keys.close()
        finally:
            self.window.destroy()

    # ------------------------------------------------------------------

    def begin(self):
        """Начало кадра. Возвращает контекст компонентов."""
        w, h = self.window.get_window_shape()
        if self.surf.ensure(int(w), int(h)):
            self.resizes.append((self.surf.w, self.surf.h))
        self.sc.set_size(self.surf.w, self.surf.h)
        self.fonts.rebuild(self.sc)

        now = time.perf_counter()
        # Шаг времени — настоящий, а не обратное к счётчику кадров в
        # секунду. Тот обновляется раз в полсекунды и в начале равен
        # нулю, из-за чего затухание следа активности получало шаг в
        # целую секунду и гасило след в ноль на каждом кадре.
        self.dt = min(0.25, max(1e-4, now - self._last))
        self._last = now

        inp = self.inp
        cx, cy = self.window.get_cursor_pos()
        down = self.window.is_pressed(self.ti.ui.LMB)
        inp.set_mouse(int(cx * self.surf.w),
                      int((1.0 - cy) * self.surf.h), down)

        events, wheel = [], 0.0
        for ev in self.keys.poll():
            if getattr(ev, "kind", "") == "wheel":
                wheel += float(getattr(ev, "wheel", 0.0))
            else:
                events.append(ev)
        inp.events = events
        inp.wheel = wheel

        if inp.pressed:
            px, py, pt = self._click_at
            inp.dclick = (now - pt < 0.35 and abs(px - inp.mx) < 6
                          and abs(py - inp.my) < 6)
            self._click_at = (inp.mx, inp.my, now)

        self.main.clear()
        self.overlay.clear()
        self.surf.clear(C.bg)
        self.ctx.begin(now - self._t0)
        return self.ctx

    def end(self):
        """Конец кадра: выливка фасадов, показ, счёт."""
        self.ctx.end()
        t0 = time.perf_counter()
        self.launches = self.main.launches() + self.overlay.launches()
        self.main.flush(self.surf)
        self.overlay.flush(self.surf)
        self.surf.present(self.gui_canvas)
        self.window.show()
        self.inp.end_frame()

        self.frame_ms = (time.perf_counter() - t0) * 1000.0
        self._frames += 1
        now = time.perf_counter()
        if now - self._tmark > 0.5:
            self.fps = self._frames / (now - self._tmark)
            self._frames = 0
            self._tmark = now

    # ------------------------------------------------------------------

    def shot(self, path):
        """Снимок кадра чтением накопителя, а не swapchain."""
        from PIL import Image

        a = np.clip(self.surf.acc.to_numpy(), 0.0, 1.0)
        img = (a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
        Image.fromarray(img).save(str(path))
        return str(path)

    def describe(self):
        return {
            "размер": (self.surf.w, self.surf.h),
            "масштаб": round(self.sc.k, 3),
            "dpi": self.dpi_info,
            "клавиатура": "нативная" if self.keys.native else
                          ("запасной путь: " + str(self.keys.failure)),
            "кадр, мс": round(self.frame_ms, 2),
            "запусков ядра": self.launches,
            "кадр/с": round(self.fps, 1),
        }


class Offscreen:
    """Та же оболочка, но без окна.

    Нужна двум вещам сразу: снимкам в заданном размере (окно в этой среде
    меняет размер само, и попросить у него 1280×720 нельзя) и проверке в
    непрерывной сборке, где окна нет вовсе. Интерфейс тот же, что у
    `Shell`, поэтому `App` не знает разницы.
    """

    def __init__(self, size=(1600, 900), dpi=1.0, title="FlyBrain"):
        import taichi as ti

        if not getattr(Offscreen, "_inited", False):
            ti.init(arch=ti.vulkan, offline_cache=True)
            Offscreen._inited = True
        self.title = title
        self.surf = gpu.Surface()
        self.surf.ensure(int(size[0]), int(size[1]))
        self.sc = Scale(dpi=float(dpi), w=self.surf.w, h=self.surf.h)
        self.fonts = Fonts(self.sc)
        self.inp = Input()
        self.inp.mx = self.inp.my = -4000
        self.main = Canvas()
        self.overlay = Canvas()
        self.ctx = Ctx(self.main, self.overlay, self.fonts, self.sc,
                       self.inp, now=0.3)
        self.keys = type("NoKeys", (), {"native": False, "hwnd": None,
                                        "failure": "окна нет",
                                        "poll": lambda self: [],
                                        "close": lambda self: None})()
        self.dpi_info = {"aware": "n/a", "scale": float(dpi),
                         "note": "без окна"}
        self.fps = 0.0
        self.dt = 1.0 / 60.0
        self.frame_ms = 0.0
        self.launches = 0
        self.resizes = []
        self.running = True
        self.t = 0.3
        self._last = time.perf_counter()

    @property
    def w(self):
        return self.surf.w

    @property
    def h(self):
        return self.surf.h

    def resize(self, w, h, dpi=None):
        self.surf.ensure(int(w), int(h))
        if dpi is not None:
            self.sc.dpi = float(dpi)
        self.sc.set_size(self.surf.w, self.surf.h)
        self.fonts.rebuild(self.sc)

    def begin(self):
        now = time.perf_counter()
        self.dt = min(0.25, max(1e-4, now - self._last))
        self._last = now
        self.main.clear()
        self.overlay.clear()
        self.surf.clear(C.bg)
        self.ctx.begin(self.t)
        return self.ctx

    def end(self):
        self.ctx.end()
        t0 = time.perf_counter()
        self.launches = self.main.launches() + self.overlay.launches()
        self.main.flush(self.surf)
        self.overlay.flush(self.surf)
        # Без окна `present` не зовётся, а синхронизация нужна та же:
        # иначе кадр, собранный в поверхность, утекает ровно так же,
        # как кадр, выведенный на экран (`gpu.Surface.finish`).
        self.surf.finish()
        self.frame_ms = (time.perf_counter() - t0) * 1000.0
        self.inp.end_frame()

    def shot(self, path):
        return Shell.shot(self, path)

    def stop(self):
        self.running = False

    def close(self):
        pass

    def describe(self):
        return {"размер": (self.w, self.h), "масштаб": round(self.sc.k, 3),
                "без окна": True}
