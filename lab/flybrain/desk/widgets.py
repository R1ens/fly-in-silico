"""Набор компонентов с полным набором состояний.

Режим немедленный: функция рисует деталь и тут же отвечает, что с ней
сделали. Состояние, которое должно жить между кадрами, держит либо
вызывающий (строка поля ввода, выбранная вкладка), либо контекст по
опознавателю детали (какой список раскрыт, где каретка, сколько
прокручено) — но никогда сама функция.

Каждая деталь обязана уметь шесть состояний: обычное, под курсором,
нажатое, в фокусе клавиатуры, отключённое, выбранное. Поэтому вид
собирается не по месту, а из `_surface` и `_ink`: добавить состояние —
значит поправить одно место, а не семнадцать.

Правила вида, из которых здесь ничего не выпадает:

* разделение высотой поверхности, а не рамкой вокруг каждой детали;
* акцент принадлежит действию и текущему выбору, больше ничему;
* кегль задан ролью и не зависит от того, растянули панель или нет;
* длинный текст усекается или переносится, но не выходит за границу.
"""

import math

from .layout import Rect, ScrollState, scroll_begin, scroll_end, wrap_text
from .textinput import TextInput
from .tokens import C, ELEV, R, S, T, mix

try:                                            # иконки появляются отдельно
    from . import icons as _icons
except Exception:                               # pragma: no cover
    _icons = None

__all__ = ["Input", "Ctx", "Fonts", "button", "icon_button", "toggle",
           "segmented", "text_field", "numeric", "combo", "slider",
           "panel", "section", "scroll", "scroll_done", "tabs", "status",
           "tip", "progress", "viewport_toolbar", "splitter_handle",
           "label", "kv", "metric", "rule", "chip", "JOB_STATES"]


# ---------------------------------------------------------------- ввод

class Input:
    """Состояние мыши и клавиатуры за один кадр."""

    __slots__ = ("mx", "my", "down", "pressed", "released", "wheel",
                 "events", "dclick", "mods")

    def __init__(self):
        self.mx = self.my = -1
        self.down = self.pressed = self.released = self.dclick = False
        self.wheel = 0.0
        self.events = []
        self.mods = ()

    def set_mouse(self, x, y, down):
        was = self.down
        self.mx, self.my = int(x), int(y)
        self.down = bool(down)
        self.pressed = self.down and not was
        self.released = was and not self.down

    def end_frame(self):
        self.pressed = self.released = self.dclick = False
        self.wheel = 0.0
        self.events = []

    def typed(self, ch):
        """Был ли набран этот символ: пробел приходит текстом, не именем."""
        for ev in self.events:
            if getattr(ev, "kind", "") == "text" and                     getattr(ev, "text", "") == ch:
                return True
        return False

    def key(self, name):
        """Было ли нажатие названной клавиши в этом кадре."""
        for ev in self.events:
            if getattr(ev, "kind", "") == "key" and \
                    getattr(ev, "name", "") == name:
                return True
        return False


# ---------------------------------------------------------------- шрифты

class Fonts:
    """Атлас на каждую роль кегля.

    Пересобирается только при смене масштаба: печение атласа стоит
    десятки миллисекунд, и делать это в кадре нельзя.
    """

    ROLES = ("workspace", "heading", "body", "small", "micro", "metric",
             "metric_big")

    def __init__(self, sc, loader=None):
        self._load = loader or self._default_loader
        self.k = None
        self.atlas = {}
        self.rebuild(sc)

    @staticmethod
    def _default_loader(size, role):
        from ..ui import font as fontmod
        return fontmod.load(size, role=role)

    def rebuild(self, sc):
        if self.k == sc.k:
            return False
        self.k = sc.k
        out = {}
        for name in self.ROLES:
            role = T.role.get(name, "ui")
            size = sc.size(name)
            if role == "num":
                # Цифры подбираются по высоте прописной, а не по своей
                # собственной: иначе числа рядом с текстом кажутся мельче.
                from ..ui import font as fontmod
                size = fontmod.match_size(size, role="num", against="ui")
            out[name] = self._load(size, role)
        self.atlas = out
        return True

    def __getattr__(self, name):
        try:
            return self.__dict__["atlas"][name]
        except KeyError:
            raise AttributeError(name)

    def line(self, name):
        return int(self.atlas[name].gh)


# ---------------------------------------------------------------- контекст

class Ctx:
    """Всё, что нужно деталям на один кадр."""

    def __init__(self, dl, overlay, fonts, sc, inp, now=0.0):
        self.dl = dl                # основной список команд
        self.overlay = overlay      # раскрытые списки и подсказки, поверх
        self.f = fonts
        self.sc = sc
        self.inp = inp
        self.now = float(now)

        self.hot = ""               # опознаватель под курсором
        self.active = ""            # что держат нажатым
        self.focus = ""             # фокус клавиатуры
        self.popup = ""             # какой список раскрыт
        self.enabled = True         # можно выключить целое поддерево

        self._tips = []
        self._hover_since = {}
        # Где какая деталь оказалась в этом кадре. Нужно проверкам: без
        # этого навести курсор на подсказку из автоматики можно только
        # угадыванием координат.
        self._rects = {}
        self._edits = {}            # опознаватель -> TextInput
        self._scrolls = {}          # опознаватель -> ScrollState
        self._blocked = []          # области, перекрытые раскрытым списком

    # -- кадр -----------------------------------------------------------

    def begin(self, now):
        self.now = float(now)
        self.hot = ""
        self._tips = []
        self._blocked = []
        self._rects = {}

    def end(self):
        """Дорисовать подсказки. Зовётся последней."""
        if not self.inp.down:
            self.active = ""
        for r, text in self._tips:
            self._draw_tip(r, text)
        self._tips = []

    def px(self, v):
        return self.sc.px(v)

    def layer(self, canvas):
        """Перевести рисование в другой фасад. Возвращает прежний.

        Нужно всему, что обязано лечь поверх картинки: внутри одного
        фасада все фигуры рисуются раньше всех картинок, и полоса
        инструментов над вьюпортом иначе уходит под кадр тела.
        """
        prev = self.dl
        self.dl = canvas
        return prev

    # -- попадание ------------------------------------------------------

    def over(self, r):
        """Курсор внутри и ничем не перекрыт.

        Своя перекрывающая область деталь не перекрывает: полоса
        инструментов и раскрытый список объявляют себя перекрытием,
        чтобы клик не проваливался на вьюпорт под ними, — и тем самым
        глушили собственные кнопки. Поэтому перекрытие не считается,
        когда деталь целиком лежит внутри него.
        """
        mx, my = self.inp.mx, self.inp.my
        if not r.contains(mx, my):
            return False
        for b in self._blocked:
            if not b.contains(mx, my):
                continue
            inside = (b.x <= r.x and b.y <= r.y
                      and r.x1 <= b.x1 and r.y1 <= b.y1)
            if not inside:
                return False
        return True

    def block(self, r):
        """Объявить область перекрытой: клики под ней не считаются."""
        self._blocked.append(r)

    def scroll_state(self, wid):
        st = self._scrolls.get(wid)
        if st is None:
            st = self._scrolls[wid] = ScrollState()
        return st

    def editor(self, wid, text="", allow=None, max_len=None):
        ti = self._edits.get(wid)
        if ti is None:
            ti = self._edits[wid] = TextInput(text, allow=allow,
                                              max_len=max_len)
            ti.select_all()
        return ti

    def drop_editor(self, wid):
        self._edits.pop(wid, None)

    # -- опрос состояния детали ----------------------------------------

    def probe(self, wid, r, enabled=True, focusable=False):
        """Единая точка: кто под курсором, что нажато, что щёлкнули.

        Возвращает кортеж (hover, held, clicked, focused).
        """
        enabled = bool(enabled) and self.enabled
        self._rects[wid] = r
        # Отсчёт наведения идёт и у отключённой детали: пояснение, почему
        # кнопка не нажимается, нужно как раз там, где она не нажимается.
        over = self.over(r)
        if over:
            self._hover_since.setdefault(wid, self.now)
        else:
            self._hover_since.pop(wid, None)
        hover = enabled and over
        if hover:
            self.hot = wid
        clicked = False
        if hover and self.inp.pressed:
            self.active = wid
            if focusable:
                self.focus = wid
        held = self.active == wid and self.inp.down
        if self.active == wid and self.inp.released:
            clicked = hover
            self.active = ""
        return hover, held, clicked, (self.focus == wid and enabled)

    def rect_of(self, wid):
        """Где деталь была в прошлом кадре. Для проверок наведения."""
        return self._rects.get(wid)

    def hovered_for(self, wid):
        t0 = self._hover_since.get(wid)
        return 0.0 if t0 is None else self.now - t0

    # -- подсказка ------------------------------------------------------

    def want_tip(self, wid, r, text, delay=0.45):
        if text and self.hovered_for(wid) >= delay:
            self._tips.append((r, text))

    def _draw_tip(self, r, text):
        """Подсказка. Длинный текст переносится, а не уезжает за окно.

        Многострочность здесь не для красоты: длинные пояснения ушли из
        панелей в подсказки, чтобы рейка не превращалась в стену текста.
        """
        dl, f, sc = self.overlay, self.f, self.sc
        at = f.small
        pad = sc.px(S.sm)
        wrap_w = sc.px(300)
        lines = wrap_text(at, text, wrap_w) or [text]
        step = at.gh + sc.px(3)
        w = max(at.width(s) for s in lines) + pad * 2
        h = step * len(lines) - sc.px(3) + pad
        x = min(max(sc.px(S.xs), r.cx - w // 2),
                max(0, sc.w - w - sc.px(S.xs)))
        y = r.y - h - sc.px(S.xs)
        if y < sc.px(S.xs):
            y = r.y1 + sc.px(S.xs)
        dl.rect(x, y, x + w, y + h, C.raised, 0.97, sc.px(R.control))
        dl.outline(x, y, x + w, y + h, C.border, sc.hair(), 0.9,
                   sc.px(R.control))
        ty = y + pad // 2
        for line in lines:
            dl.text(at, line, x + pad, ty, C.ink)
            ty += step


# ---------------------------------------------------------------- вид

def _surface(base, hover=False, held=False, selected=False, enabled=True):
    """Цвет поверхности по состоянию. Одно место на весь набор."""
    if not enabled:
        return mix(base, C.bg, 0.45)
    if held:
        return C.pressed
    if selected:
        return mix(C.selected, C.hover, 0.35) if hover else C.selected
    if hover:
        return C.hover
    return base


def _ink(enabled=True, dim=False, on_accent=False):
    if on_accent:
        return C.on_accent
    if not enabled:
        return C.faint
    return C.dim if dim else C.ink


def _focus_ring(ctx, r, show, pad=2, radius=None):
    if not show:
        return
    dl, sc = ctx.dl, ctx.sc
    p = sc.px(pad)
    rr = sc.px(R.control if radius is None else radius)
    dl.outline(r.x - p, r.y - p, r.x1 + p, r.y1 + p, C.focus_ring,
             max(1, sc.px(2)), 0.9, rr + p)


def _caret(dl, x, y, size, color, up=False, alpha=1.0):
    """Маленький шеврон.

    В наборе иконок есть только `chevron_down`, а здесь нужны обе стороны:
    шаг числового поля вверх и вниз, раскрытый список и закрытый.
    """
    s = float(size)
    w = max(1.0, s * 0.11)
    mx, half = x + s / 2.0, s * 0.3
    y0, y1 = ((y + s * 0.62, y + s * 0.38) if up else
              (y + s * 0.38, y + s * 0.62))
    dl.seg(mx - half, y0, mx, y1, color, w, alpha)
    dl.seg(mx, y1, mx + half, y0, color, w, alpha)


def _panel_icon(dl, x, y, size, color, side="left", on=True, alpha=1.0):
    """Значок зоны: рамка и закрашенная полоса с нужной стороны."""
    s = int(size)
    dl.outline(x, y, x + s, y + s, color, 1.0, alpha * 0.9, 2.0)
    t = max(2, s // 3)
    if side == "left":
        box = (x + 1, y + 1, x + 1 + t, y + s - 1)
    elif side == "right":
        box = (x + s - 1 - t, y + 1, x + s - 1, y + s - 1)
    else:
        box = (x + 1, y + s - 1 - t, x + s - 1, y + s - 1)
    if on:
        dl.rect(box[0], box[1], box[2], box[3], color, alpha, 1.0)
    else:
        dl.outline(box[0], box[1], box[2], box[3], color, 1.0,
                   alpha * 0.45, 1.0)


def _draw_icon(dl, name, x, y, size, color, alpha=1.0):
    """Единственный мост к набору иконок.

    Пока набор не готов, деталь не падает и не остаётся пустой: на месте
    иконки встаёт заметная заглушка, которую видно на тестовом экране.
    """
    if name in ("caret-up", "caret-down"):
        _caret(dl, x, y, size, color, up=(name == "caret-up"), alpha=alpha)
        return True
    if name.startswith("panel-"):
        side, _, state = name[6:].partition(":")
        _panel_icon(dl, x, y, size, color, side, state != "off", alpha)
        return True
    if _icons is not None and name in getattr(_icons, "NAMES", ()):
        fn = getattr(_icons, "draw", None)
        if fn is not None:
            fn(dl, name, x, y, size, color, alpha)
            return True
    s = int(size)
    dl.outline(x, y, x + s, y + s, color, 1, alpha * 0.55, 2.0)
    dl.rect(x + s // 3, y + s // 3, x + s - s // 3, y + s - s // 3,
            color, alpha * 0.55, 1.0)
    return False


# ---------------------------------------------------------------- 1. Button

def button(ctx, wid, r, text="", kind="default", icon=None, enabled=True,
           selected=False, tip=None, align="centre"):
    """Кнопка. `kind`: default, primary, danger, ghost, quiet."""
    if r.w <= 0 or r.h <= 0:
        return False
    hover, held, clicked, focused = ctx.probe(wid, r, enabled,
                                              focusable=True)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    rad = sc.px(R.control)
    at = f.body
    on_accent = False

    if kind == "primary":
        base = C.accent
        fill = (mix(C.accent, C.bg, 0.5) if not enabled else
                C.accent if not (hover or held) else
                mix(C.accent, C.accent_hi, 0.6 if hover and not held else 0.0)
                if not held else mix(C.accent, C.bg, 0.25))
        on_accent = enabled
    elif kind == "danger":
        fill = (mix(C.danger, C.bg, 0.5) if not enabled else
                C.danger if not (hover or held) else
                mix(C.danger, C.ink, 0.12) if hover and not held else
                mix(C.danger, C.bg, 0.25))
        on_accent = False
    elif kind in ("ghost", "quiet"):
        fill = None if not (hover or held or selected) else \
            _surface(C.raised, hover, held, selected, enabled)
    else:
        fill = _surface(C.raised, hover, held, selected, enabled)

    if fill is not None:
        dl.rect(r.x, r.y, r.x1, r.y1, fill, 1.0, rad)
    if selected and kind not in ("primary", "danger"):
        dl.rect(r.x, r.y1 - sc.px(2), r.x1, r.y1, C.accent, 1.0, 0.0)

    ink = (C.ink if kind == "danger" and enabled else
           _ink(enabled, dim=(kind == "quiet" and not hover),
                on_accent=on_accent))

    pad = sc.px(S.md)
    isz = sc.px(18)
    tw = at.width(text) if text else 0
    content = tw + (isz + sc.px(S.sm) if icon else 0)
    if align == "left":
        x = r.x + pad
    else:
        x = r.x + max(pad, (r.w - content) // 2)
    avail = r.x1 - pad - x
    if icon:
        _draw_icon(dl, icon, x, r.y + (r.h - isz) // 2, isz, ink)
        x += isz + sc.px(S.sm)
        avail = r.x1 - pad - x
    if text and avail > 0:
        dl.text(at, text, x, r.y + (r.h - at.gh) // 2, ink,
                max_px=avail)

    _focus_ring(ctx, r, focused and not held)
    ctx.want_tip(wid, r, tip)
    return clicked


# ------------------------------------------------------------ 2. IconButton

def icon_button(ctx, wid, r, icon, enabled=True, selected=False, tip=None,
                kind="default"):
    if r.w <= 0 or r.h <= 0:
        return False
    hover, held, clicked, focused = ctx.probe(wid, r, enabled,
                                              focusable=True)
    dl, sc = ctx.dl, ctx.sc
    rad = sc.px(R.control)
    if kind == "ghost" and not (hover or held or selected):
        pass
    else:
        dl.rect(r.x, r.y, r.x1, r.y1,
                _surface(C.raised, hover, held, selected, enabled), 1.0, rad)
    ink = C.accent if (selected and enabled) else _ink(enabled, dim=not hover)
    isz = min(r.w, r.h) - sc.px(S.sm) * 2
    isz = max(sc.px(12), isz)
    _draw_icon(dl, icon, r.x + (r.w - isz) // 2, r.y + (r.h - isz) // 2,
               isz, ink)
    _focus_ring(ctx, r, focused and not held)
    ctx.want_tip(wid, r, tip)
    return clicked


# ---------------------------------------------------------------- 3. Toggle

def toggle(ctx, wid, r, value, text="", enabled=True, tip=None):
    """Переключатель: сам ползунок слева, подпись справа."""
    if r.w <= 0 or r.h <= 0:
        return value
    hover, held, clicked, focused = ctx.probe(wid, r, enabled,
                                              focusable=True)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    if focused and (ctx.inp.key("enter") or ctx.inp.typed(" ")):
        clicked = True
    h = min(r.h, sc.px(18))
    w = int(h * 1.85)
    y = r.y + (r.h - h) // 2
    track = (mix(C.accent, C.bg, 0.3) if not enabled and value else
             C.accent if value else _surface(C.sunken, hover, held,
                                             enabled=enabled))
    dl.rect(r.x, y, r.x + w, y + h, track, 1.0, h / 2)
    knob = h - sc.px(4)
    kx = r.x + (w - knob - sc.px(2)) if value else r.x + sc.px(2)
    dl.rect(kx, y + sc.px(2), kx + knob, y + sc.px(2) + knob,
            C.ink if enabled else C.faint, 1.0, knob / 2)
    if text:
        at = f.body
        tx = r.x + w + sc.px(S.sm)
        dl.text(at, text, tx, r.y + (r.h - at.gh) // 2,
                _ink(enabled), max_px=max(0, r.x1 - tx))
    _focus_ring(ctx, r.with_w(w).moved(0, y - r.y).with_h(h), focused,
                radius=R.pill)
    ctx.want_tip(wid, r, tip)
    return (not value) if clicked else value


# ------------------------------------------------- 4. SegmentedControl

def segmented(ctx, wid, r, options, index, enabled=True, tips=None):
    """Выбор одного из нескольких. Для двух-пяти вариантов."""
    if r.w <= 0 or r.h <= 0:
        return index
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    n = max(1, len(options))
    dl.rect(r.x, r.y, r.x1, r.y1, C.sunken, 1.0, sc.px(R.control))
    pad = sc.px(2)
    inner = r.inset(pad)
    cells = inner.cols(n)
    out = index
    at = f.body
    for i, cell in enumerate(cells):
        sub = wid + "/" + str(i)
        sel = (i == index)
        hover, held, clicked, _ = ctx.probe(sub, cell, enabled)
        if sel:
            dl.rect(cell.x, cell.y, cell.x1, cell.y1,
                    C.raised if enabled else mix(C.raised, C.bg, 0.45),
                    1.0, sc.px(R.control) - pad)
            dl.rect(cell.x + sc.px(S.sm), cell.y1 - sc.px(2),
                    cell.x1 - sc.px(S.sm), cell.y1 - sc.px(1),
                    C.accent, 1.0, 0.0)
        elif hover or held:
            dl.rect(cell.x, cell.y, cell.x1, cell.y1,
                    C.hover if hover and not held else C.pressed, 0.7,
                    sc.px(R.control) - pad)
        ink = _ink(enabled, dim=not sel)
        dl.text_centre(at, at.fit(str(options[i]), cell.w - sc.px(S.sm)),
                       cell.x, cell.x1, cell.y + (cell.h - at.gh) // 2, ink)
        if clicked:
            out = i
        if tips and i < len(tips):
            ctx.want_tip(sub, cell, tips[i])
    return out


# ------------------------------------------------------------ 5. TextField

def text_field(ctx, wid, r, ti, placeholder="", enabled=True, icon=None,
               tip=None, on_enter=None):
    """Поле ввода. Строку и каретку держит переданный `TextInput`."""
    if r.w <= 0 or r.h <= 0:
        return False
    hover, held, clicked, focused = ctx.probe(wid, r, enabled,
                                              focusable=True)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.body
    rad = sc.px(R.control)
    fill = C.sunken if enabled else mix(C.sunken, C.bg, 0.5)
    dl.rect(r.x, r.y, r.x1, r.y1, fill, 1.0, rad)
    if focused:
        dl.outline(r.x, r.y, r.x1, r.y1, C.accent, max(1, sc.px(1.5)), 1.0, rad)
    elif hover and enabled:
        dl.outline(r.x, r.y, r.x1, r.y1, C.border, sc.hair(), 1.0, rad)

    pad = sc.px(S.sm)
    x = r.x + pad
    if icon:
        isz = sc.px(16)
        _draw_icon(dl, icon, x, r.y + (r.h - isz) // 2, isz, C.faint)
        x += isz + pad
    view = Rect(x, r.y, max(0, r.x1 - pad - x), r.h)

    changed = False
    if focused and enabled:
        for ev in ctx.inp.events:
            if getattr(ev, "kind", "") == "key" and \
                    getattr(ev, "name", "") == "enter":
                if on_enter:
                    on_enter(ti.text)
                continue
            if ti.apply_event(ev):
                changed = True

    # Прокрутка строки под каретку: каретка всегда видна.
    st = ctx.scroll_state(wid + "/h")
    full = at.width(ti.text)
    cx = at.width(ti.text[:ti.caret])
    st.content, st.view = full + sc.px(2), view.w
    if cx - st.offset > view.w - sc.px(6):
        st.offset = cx - view.w + sc.px(6)
    if cx - st.offset < 0:
        st.offset = cx
    st.clamp()
    ox = int(st.offset)

    prev = dl.push_clip(view.x, view.y, view.x1, view.y1)
    ty = r.y + (r.h - at.gh) // 2
    if ti.text:
        a, b = ti.selection()
        if focused and a != b:
            sx = view.x - ox + at.width(ti.text[:a])
            ex = view.x - ox + at.width(ti.text[:b])
            dl.rect(sx, ty - sc.px(2), ex, ty + at.gh + sc.px(2),
                    C.selected, 1.0, 2.0)
        dl.text(at, ti.text, view.x - ox, ty, _ink(enabled))
    elif placeholder:
        dl.text(at, placeholder, view.x, ty, C.faint, max_px=view.w)
    if focused and enabled and (ctx.now % 1.06) < 0.62:
        kx = view.x - ox + cx
        dl.rect(kx, ty - sc.px(2), kx + max(1, sc.px(1.5)),
                ty + at.gh + sc.px(2), C.accent, 1.0, 0.0)
    dl.pop_clip(prev)

    if clicked:
        # Каретка в место щелчка, а не в конец строки.
        want = ctx.inp.mx - view.x + ox
        best, acc = 0, 0
        for i, ch in enumerate(ti.text):
            w = at.width(ch)
            if acc + w / 2 > want:
                break
            acc += w
            best = i + 1
        ti.caret, ti.sel_anchor = best, None
    ctx.want_tip(wid, r, tip)
    return changed


# ----------------------------------------------------------- 6. NumericField

def numeric(ctx, wid, r, value, lo=None, hi=None, step=1.0, unit="",
            fmt="{:g}", enabled=True, label="", tip=None):
    """Числовое поле: подпись слева, значение справа, шаг щелчком.

    Правка идёт через то же поле ввода, но фильтр пропускает только
    цифры, знак и точку — в числовое поле нельзя уронить букву.
    """
    if r.w <= 0 or r.h <= 0:
        return float(value)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at, atn = f.body, f.metric
    rad = sc.px(R.control)
    editing = ctx.focus == wid

    lab_w = 0
    if label:
        lab_w = min(at.width(label) + sc.px(S.sm), r.w // 2)
    field = Rect(r.x + lab_w, r.y, max(0, r.w - lab_w), r.h)
    hover, held, clicked, focused = ctx.probe(wid, field, enabled,
                                              focusable=True)
    if label:
        dl.text(at, label, r.x, r.y + (r.h - at.gh) // 2, C.dim,
                max_px=max(0, lab_w - sc.px(S.sm)))

    dl.rect(field.x, field.y, field.x1, field.y1,
            C.sunken if enabled else mix(C.sunken, C.bg, 0.5), 1.0, rad)
    if focused:
        dl.outline(field.x, field.y, field.x1, field.y1, C.accent,
                 max(1, sc.px(1.5)), 1.0, rad)
    elif hover and enabled:
        dl.outline(field.x, field.y, field.x1, field.y1, C.border,
                 sc.hair(), 1.0, rad)

    out = float(value)
    step_w = sc.px(20) if (hover or focused) and enabled else 0
    body = Rect(field.x, field.y, field.w - step_w, field.h)

    if focused and enabled:
        ti = ctx.editor(wid, fmt.format(value), allow="0123456789+-.eE")
        for ev in ctx.inp.events:
            name = getattr(ev, "name", "")
            if getattr(ev, "kind", "") == "key" and name in ("enter",
                                                             "escape"):
                if name == "enter":
                    try:
                        out = float(ti.text.replace(",", "."))
                    except ValueError:
                        out = float(value)
                ctx.focus = ""
                ctx.drop_editor(wid)
                ti = None
                break
            ti.apply_event(ev)
        if ti is not None:
            prev = dl.push_clip(body.x, body.y, body.x1, body.y1)
            tx = body.x + sc.px(S.sm)
            ty = body.y + (body.h - atn.gh) // 2
            a, b = ti.selection()
            if a != b:
                dl.rect(tx + atn.width(ti.text[:a]), ty - sc.px(2),
                        tx + atn.width(ti.text[:b]), ty + atn.gh + sc.px(2),
                        C.selected, 1.0, 2.0)
            dl.text(atn, ti.text, tx, ty, C.ink)
            if (ctx.now % 1.06) < 0.62:
                kx = tx + atn.width(ti.text[:ti.caret])
                dl.rect(kx, ty - sc.px(2), kx + max(1, sc.px(1.5)),
                        ty + atn.gh + sc.px(2), C.accent, 1.0, 0.0)
            dl.pop_clip(prev)
    else:
        ctx.drop_editor(wid)
        txt = fmt.format(value)
        ty = body.y + (body.h - atn.gh) // 2
        x1 = body.x1 - sc.px(S.sm)
        if unit:
            au = f.small
            uw = au.width(unit)
            dl.text(au, unit, x1 - uw, body.y + (body.h - au.gh) // 2,
                    C.faint)
            x1 -= uw + sc.px(S.xs)
        dl.text_right(atn, atn.fit(txt, max(0, x1 - body.x - sc.px(S.sm))),
                      x1, ty, _ink(enabled))

    if step_w:
        up = Rect(field.x1 - step_w, field.y, step_w, field.h // 2)
        dn = Rect(field.x1 - step_w, field.y + field.h // 2, step_w,
                  field.h - field.h // 2)
        for rr, delta, ic in ((up, step, "caret-up"), (dn, -step,
                                                       "caret-down")):
            hv, hd, cl, _ = ctx.probe(wid + ic, rr, enabled)
            if hv:
                dl.rect(rr.x, rr.y, rr.x1, rr.y1,
                        C.pressed if hd else C.hover, 1.0, 2.0)
            isz = sc.px(10)
            _draw_icon(dl, ic, rr.x + (rr.w - isz) // 2,
                       rr.y + (rr.h - isz) // 2, isz,
                       C.ink if hv else C.dim)
            if cl:
                out = float(value) + delta
    if ctx.inp.wheel and hover and enabled:
        out = out + step * ctx.inp.wheel

    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    ctx.want_tip(wid, field, tip)
    return out


# ---------------------------------------------------------------- 7. ComboBox

def combo(ctx, wid, r, options, index, enabled=True, tip=None,
          placeholder="выбрать", max_rows=9):
    """Выпадающий список. Раскрытая часть рисуется в наложении."""
    if r.w <= 0 or r.h <= 0:
        return index
    hover, held, clicked, focused = ctx.probe(wid, r, enabled,
                                             focusable=True)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.body
    rad = sc.px(R.control)
    is_open = ctx.popup == wid
    dl.rect(r.x, r.y, r.x1, r.y1,
            _surface(C.raised, hover, held, enabled=enabled), 1.0, rad)
    if is_open or focused:
        dl.outline(r.x, r.y, r.x1, r.y1, C.accent, max(1, sc.px(1.5)), 1.0, rad)

    pad = sc.px(S.sm)
    isz = sc.px(12)
    cur = str(options[index]) if 0 <= index < len(options) else placeholder
    avail = max(0, r.w - pad * 3 - isz)
    dl.text(at, cur, r.x + pad, r.y + (r.h - at.gh) // 2,
            _ink(enabled, dim=not (0 <= index < len(options))),
            max_px=avail)
    if at.width(cur) > avail:
        # Подпись усечена — значит значение надо показать целиком, иначе
        # выбранное имя становится недоступно вовсе.
        tip = (cur + " · " + tip) if tip else cur
    _draw_icon(dl, "caret-up" if is_open else "caret-down",
               r.x1 - pad - isz, r.y + (r.h - isz) // 2, isz,
               _ink(enabled, dim=True))

    if clicked:
        ctx.popup = "" if is_open else wid
        is_open = ctx.popup == wid

    out = index
    if is_open and enabled and options:
        row = sc.px(S.row)
        shown = min(max_rows, len(options))
        ph = row * shown + sc.px(S.xs) * 2
        below = r.y1 + sc.px(S.xs)
        pr = Rect(r.x, below if below + ph <= ctx.sc.h else
                  max(0, r.y - ph - sc.px(S.xs)), r.w, ph)
        ov = ctx.overlay
        ov.rect(pr.x, pr.y, pr.x1, pr.y1, C.raised, 1.0, sc.px(R.popup))
        ov.outline(pr.x, pr.y, pr.x1, pr.y1, C.border, sc.hair(), 1.0,
                 sc.px(R.popup))
        ctx.block(pr)

        st = ctx.scroll_state(wid)
        area = pr.inset(sc.px(S.xs))
        if ctx.over(pr) and ctx.inp.wheel:
            st.by(-ctx.inp.wheel * row)
        st.view = area.h
        st.content = row * len(options)
        st.clamp()
        prev = ov.push_clip(area.x, area.y, area.x1, area.y1)
        y = area.y - int(st.offset)
        for i, opt in enumerate(options):
            cell = Rect(area.x, y + i * row, area.w, row)
            if cell.y1 < area.y or cell.y > area.y1:
                continue
            sub = wid + "/o" + str(i)
            inside, _held, hit, _f = ctx.probe(sub, cell, True)
            if inside:
                ov.rect(cell.x, cell.y, cell.x1, cell.y1, C.hover, 1.0,
                        sc.px(R.control))
            if i == index:
                ov.rect(cell.x, cell.y, cell.x1, cell.y1, C.selected, 1.0,
                        sc.px(R.control))
                ov.rect(cell.x + sc.px(2), cell.y + sc.px(S.xs),
                        cell.x + sc.px(4), cell.y1 - sc.px(S.xs),
                        C.accent, 1.0, 1.0)
            ov.text(at, str(opt), cell.x + sc.px(S.sm),
                    cell.y + (cell.h - at.gh) // 2,
                    C.ink if (i == index or inside) else C.dim,
                    max_px=area.w - sc.px(S.md))
            if inside and at.width(str(opt)) > area.w - sc.px(S.md):
                ctx.want_tip(sub, cell, str(opt), delay=0.3)
            # Щелчком считается и нажатие по пункту, и отпускание над ним
            # после нажатия на самом списке.
            if hit or (inside and ctx.inp.released):
                out = i
                ctx.popup = ""
        ov.pop_clip(prev)
        if st.content > st.view:
            frac = st.view / st.content
            th = max(sc.px(24), int(area.h * frac))
            ty = area.y + int((area.h - th) *
                              (st.offset / max(st.max_offset, 1)))
            ov.rect(area.x1 - sc.px(4), ty, area.x1 - sc.px(2), ty + th,
                    C.border, 1.0, 2.0)
        # Щелчок мимо списка его закрывает.
        if ctx.inp.pressed and not pr.contains(ctx.inp.mx, ctx.inp.my) \
                and not r.contains(ctx.inp.mx, ctx.inp.my):
            ctx.popup = ""
    ctx.want_tip(wid, r, tip)
    return out


# ---------------------------------------------------------------- 8. Slider

def slider(ctx, wid, r, value, lo, hi, label="", unit="", fmt="{:g}",
           step=0.0, enabled=True, tip=None):
    """Ползунок с подписью сверху и числом справа.

    Подпись и число занимают свою строку, а дорожка — свою: раньше они
    накладывались, когда панель сужалась.
    """
    if r.w <= 0 or r.h <= 0:
        return float(value)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at, atn = f.small, f.metric
    head = 0
    if (label or unit) and r.h >= max(at.gh, atn.gh) + sc.px(S.xs) + sc.px(8):
        head = max(at.gh, atn.gh) + sc.px(S.xs)
        txt = fmt.format(value) + ((" " + unit) if unit else "")
        nw = atn.width(txt)
        dl.text(at, label, r.x, r.y + (head - sc.px(S.xs) - at.gh) // 2
                + (atn.gh - at.gh) // 2, C.dim,
                max_px=max(0, r.w - nw - sc.px(S.sm)))
        dl.text_right(atn, txt, r.x1, r.y, _ink(enabled))
    # Дорожка всегда внутри своей коробки, какой бы тесной та ни была.
    track = Rect(r.x, r.y + head, r.w, max(0, r.h - head))
    if track.h <= 0:
        return float(value)
    hover, held, clicked, focused = ctx.probe(wid, track.pad(
        0, -sc.px(S.xs), 0, -sc.px(S.xs)), enabled, focusable=True)

    th = sc.px(5)
    ty = track.y + (track.h - th) // 2
    dl.rect(track.x, ty, track.x1, ty + th, C.sunken, 1.0, th / 2)
    span = max(1e-9, float(hi) - float(lo))
    frac = min(1.0, max(0.0, (float(value) - float(lo)) / span))
    fx = track.x + int(track.w * frac)
    dl.rect(track.x, ty, fx, ty + th,
            C.accent if enabled else mix(C.accent, C.bg, 0.5), 1.0, th / 2)
    knob = sc.px(14 if (hover or held) else 12)
    dl.rect(fx - knob // 2, ty + th // 2 - knob // 2,
            fx - knob // 2 + knob, ty + th // 2 - knob // 2 + knob,
            C.ink if enabled else C.faint, 1.0, knob / 2)

    out = float(value)
    if held and enabled:
        t = (ctx.inp.mx - track.x) / max(1, track.w)
        out = float(lo) + span * min(1.0, max(0.0, t))
    elif focused and enabled:
        d = (step or span / 50.0)
        if ctx.inp.key("left"):
            out -= d
        if ctx.inp.key("right"):
            out += d
    if ctx.inp.wheel and hover and enabled:
        out += (step or span / 50.0) * ctx.inp.wheel
    if step:
        out = round(out / step) * step
    out = min(float(hi), max(float(lo), out))
    _focus_ring(ctx, Rect(track.x, ty, track.w, th), focused and not held,
                radius=R.pill)
    ctx.want_tip(wid, track, tip)
    return out


# ---------------------------------------------------------------- 9. Panel

def panel(ctx, r, title="", elev=1, pad=None, right=None, radius=None):
    """Панель. Возвращает прямоугольник содержимого.

    Разделение — ступенью фона, не рамкой: рамка вокруг каждого элемента
    в брифе прямо запрещена.
    """
    if r.w <= 0 or r.h <= 0:
        return r
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    rad = sc.px(R.panel if radius is None else radius)
    dl.rect(r.x, r.y, r.x1, r.y1, ELEV.get(elev, C.panel), 1.0, rad)
    p = sc.px(S.md if pad is None else pad)
    inner = r.inset(p)
    if title:
        at = f.heading
        head, inner = inner.cut_top(at.gh, sc.px(S.sm))
        x1 = head.x1
        if right:
            ar = f.small
            dl.text_right(ar, right, head.x1,
                          head.y + (head.h - ar.gh) // 2, C.faint)
            x1 -= ar.width(right) + sc.px(S.sm)
        dl.text(at, title, head.x, head.y, C.dim,
                max_px=max(0, x1 - head.x))
        line, inner = inner.cut_top(sc.hair(), sc.px(S.sm))
        dl.rect(line.x, line.y, line.x1, line.y + sc.hair(),
                C.border_soft, 1.0, 0.0)
    return inner


# ---------------------------------------------------------------- 10. Section

def section(ctx, stack, title, right="", info=None, wid=None, first=False,
            fold=None):
    """Заголовок раздела внутри панели. Двигает курсор сам.

    Заголовок отделён не только кеглем: слева стоит короткая засечка
    акцентом, сверху — воздух. Без этого «Прогон» и «длина» читались
    одним уровнем, и панель выглядела списком, а не разделами.

    `info` — длинное пояснение: оно уходит в подсказку на значке, а не в
    абзац под заголовком.

    `fold` — если передано, раздел сворачивается: заголовок становится
    кнопкой, справа появляется галочка, и функция возвращает новое
    состояние. Редкие настройки так не исчезают из интерфейса, а просто
    не занимают место, пока не понадобятся.
    """
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.heading
    if not first:
        stack.skip(sc.px(S.sm))
    if not stack.fits(at.gh):
        return fold
    row = stack.take(at.gh, sc.px(S.xs))
    tick = sc.px(3)
    dl.rect(row.x, row.y + sc.px(2), row.x + tick, row.y + at.gh - sc.px(2),
            C.accent, 1.0, 1.0)
    tx = row.x + tick + sc.px(S.sm)
    x1 = row.x1
    if info:
        d = at.gh
        spot = Rect(row.x1 - d, row.y, d, d)
        iid = (wid or title) + "/info"
        hover, _held, _cl, _fc = ctx.probe(iid, spot, True)
        _draw_icon(dl, "info", spot.x + sc.px(2), spot.y + sc.px(2),
                   d - sc.px(4), C.ink if hover else C.faint)
        ctx.want_tip(iid, spot, info, delay=0.25)
        x1 -= d + sc.px(S.sm)
    if right:
        ar = f.small
        dl.text_right(ar, right, x1, row.y + (at.gh - ar.gh) // 2, C.faint)
        x1 -= ar.width(right) + sc.px(S.sm)
    hot = False
    if fold is not None:
        fid = (wid or title) + "/fold"
        hit = Rect(row.x, row.y, max(0, x1 - row.x), row.h)
        hot, _h, clicked, _fc = ctx.probe(fid, hit, True)
        if clicked:
            fold = not fold
        cs = sc.px(8)
        _caret(dl, x1 - cs, row.y + (at.gh - cs // 2) // 2, cs,
               C.ink if hot else C.dim, up=bool(fold))
        ctx.want_tip(fid, hit,
                     ("свернуть раздел" if fold else "развернуть раздел"))
        x1 -= cs + sc.px(S.sm)
    dl.text(at, title, tx, row.y, C.ink if not hot else C.ink,
            max_px=max(0, x1 - tx))
    if stack.fits(sc.hair()):
        line = stack.take(sc.hair(), sc.px(S.sm))
        dl.rect(line.x, line.y, line.x1, line.y + sc.hair(), C.border_soft,
                1.0, 0.0)
    return fold


def rule(ctx, stack, gap=None):
    sc = ctx.sc
    g = sc.px(S.md if gap is None else gap)
    if not stack.fits(sc.hair() + g):
        return
    stack.skip(g // 2)
    line = stack.take(sc.hair(), g // 2)
    ctx.dl.rect(line.x, line.y, line.x1, line.y + sc.hair(), C.border_soft,
                1.0, 0.0)


# -------------------------------------------------------- 11. ScrollContainer

def scroll(ctx, wid, r, step=None, gutter=True):
    """Начать прокручиваемую область. Вернуть (курсорная область, маркер).

    Место под полосу прокрутки отводится всегда, даже когда полосы нет:
    иначе появление полосы на один элемент сдвигает всю выкладку, и
    содержимое прыгает прямо под курсором.
    """
    st = ctx.scroll_state(wid)
    wheel = ctx.inp.wheel if ctx.over(r) else 0.0
    content, prev = scroll_begin(ctx.dl, r, st, wheel,
                                 ctx.sc.px(48 if step is None else step))
    if gutter:
        content = Rect(content.x, content.y,
                       max(0, content.w - ctx.sc.px(S.md)), content.h)
    return content, (st, prev)


def scroll_done(ctx, wid, r, used, token):
    st, prev = token
    scroll_end(ctx.dl, r, st, used, prev, bar_w=ctx.sc.px(4),
               track=C.sunken, thumb=C.border)


# ---------------------------------------------------------------- 12. Tabs

def tabs(ctx, wid, r, names, index, enabled=True, counts=None):
    """Вкладки правой колонки. Показывается только выбранная."""
    if r.w <= 0 or r.h <= 0:
        return index
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.body
    out = index
    x = r.x

    def widths(pad, labels):
        out = []
        for i, name in enumerate(labels):
            w = at.width(name) + pad * 2
            if counts and i < len(counts) and counts[i] is not None:
                w += f.micro.width(str(counts[i])) + sc.px(S.sm)
            out.append(w)
        return out

    # Вкладка, которая не поместилась, недостижима: при узкой колонке
    # «Журнал» просто пропадал. Поэтому сперва ужимается внутренний
    # отступ, и только потом — сами подписи.
    labels = list(names)
    pad = sc.px(S.md)
    while pad > sc.px(S.xs) and sum(widths(pad, labels)) > r.w:
        pad -= max(1, sc.px(1))
    if sum(widths(pad, labels)) > r.w:
        room = r.w - sum(w - at.width(n) for w, n
                         in zip(widths(pad, labels), labels))
        share = max(sc.px(S.lg), room // max(1, len(labels)))
        labels = [at.fit(n, share) for n in labels]
    ws = widths(pad, labels)

    for i, name in enumerate(labels):
        w = ws[i]
        cell = Rect(x, r.y, min(w, max(0, r.x1 - x)), r.h)
        if cell.w <= sc.px(S.sm):
            break
        sub = wid + "/" + str(i)
        hover, held, clicked, _ = ctx.probe(sub, cell, enabled)
        sel = i == index
        if sel:
            dl.rect(cell.x, cell.y, cell.x1, cell.y1, C.panel, 1.0, 0.0)
            dl.rect(cell.x, cell.y, cell.x1, cell.y + sc.px(2), C.accent,
                    1.0, 0.0)
        elif hover or held:
            dl.rect(cell.x, cell.y, cell.x1, cell.y1,
                    C.pressed if held else C.hover, 0.8, 0.0)
        tx = cell.x + pad
        dl.text(at, name, tx, cell.y + (cell.h - at.gh) // 2,
                _ink(enabled, dim=not sel),
                max_px=max(0, cell.x1 - pad - tx))
        if counts and i < len(counts) and counts[i] is not None:
            am = f.micro
            dl.text_right(am, str(counts[i]), cell.x1 - pad + sc.px(S.xs),
                          cell.y + (cell.h - am.gh) // 2, C.faint)
        if clicked:
            out = i
        x = cell.x1
    if x < r.x1:
        dl.rect(x, r.y1 - sc.hair(), r.x1, r.y1, C.border_soft, 1.0, 0.0)
    return out


# ------------------------------------------------------- 13. StatusIndicator

# Состояния задачи названы явно, как требует бриф: никакого «просто идёт».
JOB_STATES = {
    "idle": ("Простой", C.faint),
    "prep": ("Подготовка", C.info),
    "run": ("Идёт", C.ok),
    "cancel": ("Остановка", C.warn),
    "done": ("Готово", C.ok),
    "fail": ("Отказ", C.danger),
}


def status(ctx, r, state, text=None, pulse=False):
    """Точка состояния и подпись."""
    if r.w <= 0 or r.h <= 0:
        return r.x1
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    name, col = JOB_STATES.get(state, (str(state), C.dim))
    at = f.small
    d = sc.px(8)
    y = r.y + (r.h - d) // 2
    a = 1.0
    if pulse:
        a = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(ctx.now * 4.2))
    dl.rect(r.x, y, r.x + d, y + d, col, a, d / 2)
    tx = r.x + d + sc.px(S.sm)
    dl.text(at, text if text is not None else name, tx,
            r.y + (r.h - at.gh) // 2, C.dim, max_px=max(0, r.x1 - tx))
    return r.x1


# ---------------------------------------------------------------- 14. Tooltip

def tip(ctx, wid, r, text, delay=0.45):
    """Подсказка для чего угодно, в том числе не-кнопки."""
    if ctx.over(r):
        ctx.hot = wid
        ctx._hover_since.setdefault(wid, ctx.now)
    else:
        ctx._hover_since.pop(wid, None)
    ctx.want_tip(wid, r, text, delay)


# ------------------------------------------------------------ 15. ProgressBar

def progress(ctx, r, frac, text="", indeterminate=False, color=None):
    if r.w <= 0 or r.h <= 0:
        return None
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.small
    bar = r
    if text:
        head, bar = r.cut_top(at.gh, sc.px(S.xs))
        pct = "" if indeterminate else "{:.0f} %".format(frac * 100)
        pw = at.width(pct)
        dl.text(at, text, head.x, head.y, C.dim,
                max_px=max(0, head.w - pw - sc.px(S.sm)))
        if pct:
            dl.text_right(f.small, pct, head.x1, head.y, C.ink)
    h = max(sc.px(4), min(bar.h, sc.px(6)))
    y = bar.y + (bar.h - h) // 2
    dl.rect(bar.x, y, bar.x1, y + h, C.sunken, 1.0, h / 2)
    col = color or C.accent
    if indeterminate:
        # Неизвестная длительность показывается бегунком, а не ложной
        # долей: 27 секунд подготовки нельзя рисовать как «0 %».
        w = max(sc.px(40), bar.w // 5)
        t = (ctx.now * 0.55) % 1.0
        x = bar.x + int((bar.w + w) * t) - w
        x0, x1 = max(bar.x, x), min(bar.x1, x + w)
        if x1 > x0:
            dl.rect(x0, y, x1, y + h, col, 0.9, h / 2)
    else:
        w = int(bar.w * min(1.0, max(0.0, frac)))
        if w > 0:
            dl.rect(bar.x, y, bar.x + w, y + h, col, 1.0, h / 2)


# -------------------------------------------------------- 16. ViewportToolbar

def viewport_toolbar(ctx, wid, r, items, align="left"):
    """Плавающая полоса над вьюпортом.

    Виды деталей:

    * ``("-",)``                      — разделитель группы;
    * ``(имя, иконка, подсказка)``    — кнопка-действие;
    * ``("seg", имя, подписи, номер)`` — взаимоисключающий выбор словами.

    Выбор вида сделан подписями, а не иконками, сознательно: четыре
    одинаковых значка камеры не говорят ничего, и по снимку было не
    понять, какая кнопка что делает. Действия остаются иконками, но у
    каждой есть подсказка.

    Возвращает ``(нажатая кнопка, {имя группы: новый номер})``.
    """
    if r.w <= 0 or r.h <= 0:
        return None, {}
    back = ctx.layer(ctx.overlay)        # поверх картинки вьюпорта
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    btn = sc.px(S.control)
    pad = sc.px(S.xs)
    gap = sc.px(S.sm)

    def seg_w(labels):
        cell = max(f.body.width(str(t)) for t in labels) + sc.px(S.lg)
        return cell * len(labels) + sc.px(4)

    # Ширина каждой детали, чтобы разложить их по строкам. Кнопка, не
    # поместившаяся в строку, становится недостижимой, а последними в
    # полосе стоят как раз «сбросить» и «развернуть».
    ws = [gap if it[0] == "-" else
          (seg_w(it[2]) if it[0] == "seg" else btn) for it in items]
    room = max(btn, r.w - pad * 2)
    rows, cur, acc = [], [], 0
    for it, w in zip(items, ws):
        if cur and acc + w > room:
            while cur and cur[-1][0][0] == "-":
                acc -= cur.pop()[1]
            rows.append((cur, acc))
            cur, acc = [], 0
            if it[0] == "-":
                continue
        cur.append((it, w))
        acc += w
    if cur:
        rows.append((cur, acc))
    width = max((a for _c, a in rows), default=0) + pad * 2
    x = (r.x if align == "left" else
         r.x1 - width if align == "right" else r.cx - width // 2)
    bar = Rect(x, r.y, min(width, r.w),
               btn * len(rows) + pad * (len(rows) + 1))
    dl.rect(bar.x, bar.y, bar.x1, bar.y1, C.panel, 0.94, sc.px(R.popup))
    dl.outline(bar.x, bar.y, bar.x1, bar.y1, C.border, sc.hair(), 0.7,
               sc.px(R.popup))
    ctx.block(bar)

    hit, picks = None, {}
    ry = bar.y + pad
    for line, _acc in rows:
        cx = bar.x + pad
        for it, w in line:
            if it[0] == "-":
                dl.rect(cx + gap // 2, ry + pad,
                        cx + gap // 2 + sc.hair(), ry + btn - pad,
                        C.border, 1.0, 0.0)
                cx += gap
                continue
            if it[0] == "seg":
                name, labels, index = it[1], it[2], it[3]
                tips = it[4] if len(it) > 4 else None
                cell = Rect(cx, ry, min(w, max(0, bar.x1 - pad - cx)), btn)
                new = segmented(ctx, wid + "/" + name, cell, labels, index,
                                tips=tips)
                if new != index:
                    picks[name] = new
                cx = cell.x1
                continue
            name, ic = it[0], it[1]
            tp = it[2] if len(it) > 2 else None
            on = bool(it[3]) if len(it) > 3 else False
            cell = Rect(cx, ry, btn, btn)
            if icon_button(ctx, wid + "/" + name, cell, ic, selected=on,
                           tip=tp, kind="ghost"):
                hit = name
            cx = cell.x1
        ry += btn + pad
    ctx.layer(back)
    return hit, picks


# ---------------------------------------------------------------- 17. Splitter

def splitter_handle(ctx, wid, r, axis="v", dragging=False):
    """Полоса, за которую тянут зону. Логику хода считает `layout`."""
    if r.w <= 0 or r.h <= 0:
        return (False, False)
    dl, sc = ctx.dl, ctx.sc
    hover, held, _, _ = ctx.probe(wid, r, True)
    if dragging or held:
        col, a = C.accent, 0.85
    elif hover:
        col, a = C.border, 1.0
    else:
        col, a = C.border_soft, 1.0
    if axis == "v":
        x = r.cx
        dl.rect(x - sc.hair(), r.y, x + sc.hair(), r.y1, col, a, 0.0)
        if hover or dragging or held:
            g = sc.px(14)
            dl.rect(x - sc.px(1.5), r.cy - g, x + sc.px(1.5), r.cy + g,
                    col, a, 2.0)
    else:
        y = r.cy
        dl.rect(r.x, y - sc.hair(), r.x1, y + sc.hair(), col, a, 0.0)
        if hover or dragging or held:
            g = sc.px(14)
            dl.rect(r.cx - g, y - sc.px(1.5), r.cx + g, y + sc.px(1.5),
                    col, a, 2.0)
    return hover, held


# ---------------------------------------------------------------- мелочи

def label(ctx, r, text, role="body", color=None, align="left", wrapped=False):
    """Подпись. При `wrapped` переносится и возвращает занятую высоту."""
    if r.w <= 0 or r.h <= 0:
        return 0
    dl, f = ctx.dl, ctx.f
    at = f.atlas[role]
    col = color if color is not None else C.dim
    if not wrapped:
        if align == "right":
            dl.text_right(at, at.fit(text, r.w), r.x1, r.y, col)
        elif align == "centre":
            dl.text_centre(at, at.fit(text, r.w), r.x, r.x1, r.y, col)
        else:
            dl.text(at, text, r.x, r.y, col, max_px=r.w)
        return at.gh
    y = r.y
    step = at.gh + ctx.sc.px(2)
    for line in wrap_text(at, text, r.w):
        if y + at.gh > r.y1:
            break
        dl.text(at, line, r.x, y, col)
        y += step
    return y - r.y


def kv(ctx, r, key, value, unit="", role="body"):
    """Пара «название — значение». Значение числовой гарнитурой, справа."""
    if r.w <= 0 or r.h <= 0:
        return None
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    ak, av = f.atlas[role], f.metric
    y = r.y + (r.h - av.gh) // 2 if r.h > av.gh else r.y
    x1 = r.x1
    if unit:
        au = f.micro
        dl.text_right(au, unit, x1, y + (av.gh - au.gh), C.faint)
        x1 -= au.width(unit) + sc.px(S.xs)
    vw = av.width(str(value))
    dl.text_right(av, str(value), x1, y, C.ink)
    dl.text(ak, key, r.x, y + (av.gh - ak.gh) // 2, C.dim,
            max_px=max(0, x1 - vw - sc.px(S.sm) - r.x))


def metric(ctx, r, value, text="", unit="", big=False, color=None):
    """Одно значимое число с подписью под ним."""
    if r.w <= 0 or r.h <= 0:
        return 0
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    av = f.metric_big if big else f.metric
    x = r.x
    dl.text(av, str(value), x, r.y, color or C.ink)
    x += av.width(str(value))
    if unit:
        au = f.small
        dl.text(au, " " + unit, x, r.y + av.gh - au.gh, C.faint)
    if text:
        at = f.small
        dl.text(at, text, r.x, r.y + av.gh + sc.px(S.xs), C.dim,
                max_px=r.w)
        return av.gh + sc.px(S.xs) + at.gh
    return av.gh


def chip(ctx, wid, r, text, enabled=True, selected=False, closable=False):
    """Чип множественного выбора. Второй ответ — «закрыть»."""
    if r.w <= 0 or r.h <= 0:
        return (False, False)
    hover, held, clicked, _ = ctx.probe(wid, r, enabled)
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.small
    dl.rect(r.x, r.y, r.x1, r.y1,
            _surface(C.raised, hover, held, selected, enabled), 1.0,
            sc.px(R.pill))
    if selected:
        dl.outline(r.x, r.y, r.x1, r.y1, C.accent, sc.hair(), 1.0,
                 sc.px(R.pill))
    pad = sc.px(S.md)
    x1 = r.x1 - pad
    closed = False
    if closable:
        isz = sc.px(10)
        cr = Rect(r.x1 - pad - isz, r.y + (r.h - isz) // 2, isz, isz)
        hv, _, cl, _ = ctx.probe(wid + "/x", cr.inset(-sc.px(S.xs)), enabled)
        _draw_icon(dl, "close", cr.x, cr.y, isz, C.ink if hv else C.faint)
        closed = cl
        x1 = cr.x - sc.px(S.xs)
    dl.text(at, text, r.x + pad, r.y + (r.h - at.gh) // 2,
            _ink(enabled, dim=not (selected or hover)),
            max_px=max(0, x1 - r.x - pad))
    return clicked and not closed, closed
