"""Один фасад над несколькими списками команд.

`Surface.draw` принимает один атлас на вызов: номера глифов в списке
команд относятся к конкретному ремню. А интерфейсу нужно семь кеглей
одновременно, и держать по списку на атлас вручную — верный способ
получить текст без обрезки (в прежнем наборе подписи областей так и
пропали: они лежали в списке, который рисовали без атласа).

Поэтому здесь: фигуры идут в один список, текст — в список своего
атласа, обрезка применяется ко всем сразу, а `flush` отдаёт их
поверхности в правильном порядке.

Чего это стоит и почему это терпимо: внутри одного фасада все фигуры
ложатся раньше всего текста. Значит панель, нарисованная позже, текст под
собой не закроет. Для раскрытых списков и подсказок поэтому держится
второй фасад — наложение, — и он выливается после первого.
"""

from ..ui.gpu import DrawList

__all__ = ["Canvas"]

FAR = 1e9


class Canvas:
    """Фигуры, картинки и текст нескольких кеглей за один кадр."""

    def __init__(self):
        self.shapes = DrawList()
        self.texts = []             # [(atlas, DrawList)] в порядке первого
        self._by_id = {}            # id(atlas) -> DrawList
        self._clip = (-FAR, -FAR, FAR, FAR)

    def clear(self):
        self.shapes.clear()
        for _, dl in self.texts:
            dl.clear()
        self._clip = (-FAR, -FAR, FAR, FAR)

    @property
    def empty(self):
        return not self.shapes.cmds and not self.shapes.images and \
            not any(dl.gi for _, dl in self.texts)

    # -- обрезка --------------------------------------------------------

    def push_clip(self, x0, y0, x1, y1):
        prev = self._clip
        self._clip = (max(prev[0], float(x0)), max(prev[1], float(y0)),
                      min(prev[2], float(x1)), min(prev[3], float(y1)))
        self.shapes._clip = self._clip
        return prev

    def pop_clip(self, prev):
        self._clip = prev
        self.shapes._clip = prev

    def no_clip(self):
        self._clip = (-FAR, -FAR, FAR, FAR)
        self.shapes._clip = self._clip

    # -- фигуры ---------------------------------------------------------

    def rect(self, *a, **kw):
        self.shapes.rect(*a, **kw)

    def frame(self, *a, **kw):
        self.shapes.frame(*a, **kw)

    def outline(self, *a, **kw):
        self.shapes.outline(*a, **kw)

    def seg(self, *a, **kw):
        self.shapes.seg(*a, **kw)

    def polyline(self, *a, **kw):
        self.shapes.polyline(*a, **kw)

    def hline(self, *a, **kw):
        self.shapes.hline(*a, **kw)

    def vline(self, *a, **kw):
        self.shapes.vline(*a, **kw)

    def image(self, *a, **kw):
        self.shapes.image(*a, **kw)

    # -- текст ----------------------------------------------------------

    def _list_for(self, atlas):
        dl = self._by_id.get(id(atlas))
        if dl is None:
            dl = DrawList()
            self._by_id[id(atlas)] = dl
            self.texts.append((atlas, dl))
        # Обрезка ставится на каждый вызов: список мог появиться уже внутри
        # вложенной обрезки, и наследовать ему нечего.
        dl._clip = self._clip
        return dl

    def text(self, atlas, s, x, y, color, alpha=1.0, max_px=None):
        return self._list_for(atlas).text(atlas, s, x, y, color, alpha,
                                          max_px)

    def text_right(self, atlas, s, x_right, y, color, alpha=1.0):
        return self._list_for(atlas).text_right(atlas, s, x_right, y,
                                                color, alpha)

    def text_centre(self, atlas, s, x0, x1, y, color, alpha=1.0):
        return self._list_for(atlas).text_centre(atlas, s, x0, x1, y,
                                                 color, alpha)

    # -- выдача ---------------------------------------------------------

    def launches(self):
        """Сколько запусков ядра будет стоить этот фасад."""
        n = 1 if (self.shapes.cmds or self.shapes.images) else 0
        n += len(self.shapes.images)
        return n + sum(1 for _, dl in self.texts if dl.gi)

    def flush(self, surf):
        if self.shapes.cmds or self.shapes.images:
            surf.draw(self.shapes, None)
        for atlas, dl in self.texts:
            if dl.gi:
                surf.draw(dl, atlas)
