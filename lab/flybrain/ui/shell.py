"""Оболочка окна: жизненный цикл, размер, ввод, кадр.

Отдельно от `app.py`, потому что здесь живёт всё, что приходится знать про
повадки GGUI, и ни одна панель про это знать не должна.

Три повадки, стоившие проверки:

1. Размер окна **меняется сам**, без всякого события: в замерах окно, открытое
   как 1500×880, становилось 803×984, а потом 755×984. Поэтому размер
   опрашивается каждый кадр, а поля кадра пересоздаются.
2. Метрика поэтому не прибивается к пикселям. Масштаб считается от площади
   окна, кегль выбирается из испечённых атласов, и всё остальное идёт через
   `scale`.
3. Снимок swapchain (`save_image`, `get_image_buffer_as_numpy`) в этой среде
   отдаёт чёрное. Проверять вид надо чтением поля `Surface.acc`, а не снимком
   экрана; `shot()` ниже это и делает.
"""

import time

import numpy as np
import taichi as ti

from . import font as F
from . import gpu, theme

__all__ = ["Input", "Shell"]

# Кегли, под которые испечены атласы. Между ними масштаб выбирает ближайший.
SIZES = (12, 13, 14, 16, 18, 20)
REF_AREA = 1500.0 * 880.0


class Input:
    """Ввод за кадр.

    Taichi отдаёт события списком и положение курсора в долях окна; сюда это
    приходит уже в пикселях. Колесо в GGUI отдельного события не имеет, и его
    приходится собирать из того, что есть, — поэтому `wheel` заполняется
    оболочкой и может остаться нулём.
    """

    __slots__ = ("x", "y", "down", "clicked", "released", "wheel",
                 "keys", "shift", "ctrl", "alt", "dt")

    def __init__(self):
        self.x = self.y = 0
        self.down = False
        self.clicked = False
        self.released = False
        self.wheel = 0.0
        self.keys = []
        self.shift = self.ctrl = self.alt = False
        self.dt = 0.0

    def inside(self, rect):
        x0, y0, x1, y1 = rect
        return x0 <= self.x < x1 and y0 <= self.y < y1

    @property
    def cursor(self):
        return (self.x, self.y)


class Shell:
    """Окно, поверхность кадра, атласы и ввод."""

    def __init__(self, title="Мозг мухи", size=(1500, 880), pos=(70, 50),
                 vsync=True):
        ti.init(arch=ti.vulkan, offline_cache=True)
        self.window = ti.ui.Window(title, size, vsync=vsync, pos=pos)
        self.canvas = self.window.get_canvas()
        self.surf = gpu.Surface()
        self.input = Input()

        self.scale = 1.0
        self.atlas = None          # основной кегль
        self.small = None          # мелкий кегль, подписи и единицы
        self.title_atlas = None    # заголовки
        self._size_now = None

        self.fps = 0.0
        self._frames = 0
        self._tmark = time.perf_counter()
        self._last = self._tmark
        self._was_down = False
        self.resizes = []

    # ------------------------------------------------------------------

    @property
    def w(self):
        return self.surf.w

    @property
    def h(self):
        return self.surf.h

    def _pick_size(self, want):
        return min(SIZES, key=lambda s: abs(s - want))

    def _fit(self, w, h):
        """Масштаб от площади окна, а не от одной из сторон.

        По одной стороне считать нельзя: окно бывает портретным, и по ширине
        интерфейс тогда ужимается до нечитаемого, а по высоте раздувается.
        """
        self.scale = max(0.78, min(2.0, (w * h) ** 0.5 / REF_AREA ** 0.5))
        body = self._pick_size(round(theme.SIZE_BODY * self.scale))
        small = self._pick_size(round(theme.SIZE_SMALL * self.scale))
        title = self._pick_size(round(theme.SIZE_TITLE * self.scale))
        if self._size_now != (body, small, title):
            self.atlas = F.load(body)
            self.small = F.load(small)
            self.title_atlas = F.load(title)
            self._size_now = (body, small, title)

    def px(self, v):
        """Перевести метрику темы в пиксели текущего масштаба."""
        return int(round(v * self.scale))

    # ------------------------------------------------------------------

    def begin(self):
        """Начало кадра: размер, ввод, очистка. Возвращает `DrawList`."""
        w, h = self.window.get_window_shape()
        if self.surf.ensure(int(w), int(h)):
            self.resizes.append((int(w), int(h)))
        self._fit(self.surf.w, self.surf.h)

        inp = self.input
        now = time.perf_counter()
        inp.dt = now - self._last
        self._last = now

        cx, cy = self.window.get_cursor_pos()
        inp.x = int(cx * self.surf.w)
        # курсор GGUI отсчитывается снизу, экранные координаты — сверху
        inp.y = int((1.0 - cy) * self.surf.h)

        inp.keys = []
        inp.wheel = 0.0
        for e in self.window.get_events(ti.ui.PRESS):
            inp.keys.append(str(e.key))
        inp.shift = self.window.is_pressed(ti.ui.SHIFT)
        inp.ctrl = self.window.is_pressed(ti.ui.CTRL)
        inp.alt = self.window.is_pressed(ti.ui.ALT)

        down = self.window.is_pressed(ti.ui.LMB)
        inp.clicked = down and not self._was_down
        inp.released = (not down) and self._was_down
        inp.down = down
        self._was_down = down

        self.surf.clear(theme.BG)
        return gpu.DrawList()

    def end(self, dl):
        """Конец кадра: один проход на тип команды и показ."""
        self.surf.draw(dl, self.atlas)
        self.surf.present(self.canvas)
        self.window.show()

        self._frames += 1
        now = time.perf_counter()
        if now - self._tmark > 0.5:
            self.fps = self._frames / (now - self._tmark)
            self._frames = 0
            self._tmark = now

    def finish(self):
        """Показать уже собранный кадр.

        Нужна тем, кто рисует не через `begin`/`end`: буфер команд в таком
        случае ведёт набор виджетов, а оболочке остаётся показ и счёт кадров.
        """
        self.surf.present(self.canvas)
        self.window.show()
        self._frames += 1
        now = time.perf_counter()
        if now - self._tmark > 0.5:
            self.fps = self._frames / (now - self._tmark)
            self._frames = 0
            self._tmark = now

    @property
    def running(self):
        return self.window.running

    def stop(self):
        self.window.running = False

    def close(self):
        self.window.destroy()

    # ------------------------------------------------------------------

    def shot(self, path):
        """Снимок кадра чтением поля, а не swapchain.

        `save_image` здесь отдаёт чёрное, поэтому единственный способ
        посмотреть на кадр из-под автоматики — прочитать накопитель.
        """
        from PIL import Image

        a = np.clip(self.surf.acc.to_numpy(), 0.0, 1.0)
        img = (a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        return path
