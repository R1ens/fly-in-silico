"""Графики: линия, траектория, столбцы, гистограмма, точки по условиям.

Рисуются теми же примитивами, что и всё остальное окно: `polyline` и `seg`
складывают по команде на звено, а не по команде на точку, так что ряд из
шестисот отсчётов стоит столько же, сколько рамка панели.

Правило, общее для всех: **подпись величины, единицы и источник рисуются
вместе с данными, а не рядом в справке**. График без единиц — это картинка,
а не измерение, и в этом пространстве таких нет. Оговорка, если она есть у
графика, печатается прямо под ним жёлтым и не прячется в подсказку.

Прореживание. Ряд длиннее ширины области в пикселях рисуется по минимуму и
максимуму на столбец пикселей, а не каждой точкой: так видно и размах, и
форма, и при этом не теряются одиночные выбросы, которые съело бы обычное
«каждая n-я точка».
"""

import math

from . import layout as L
from . import widgets as W
from .tokens import C, R, S

__all__ = ["draw"]

# Цвета рядов. Данные живут отдельно от органов управления: акцент
# принадлежит действию, и график его не занимает.
SERIES = (C.plot, C.plot_alt, C.ok, C.warn)


def _fmt(v):
    """Число на оси: без хвоста, с запятой, как везде в окне."""
    a = abs(v)
    if a >= 1000:
        s = "%.0f" % v
    elif a >= 10:
        s = "%.1f" % v
    elif a >= 1:
        s = "%.2f" % v
    elif a == 0:
        s = "0"
    else:
        s = "%.3f" % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _nice_bounds(lo, hi):
    """Границы оси с запасом и с нулём, когда он рядом."""
    if not (lo == lo) or not (hi == hi):          # nan
        return 0.0, 1.0
    if hi - lo < 1e-12:
        d = max(abs(hi) * 0.1, 0.5)
        return lo - d, hi + d
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    if lo > 0 and lo < (hi - lo) * 0.5:
        lo = 0.0
    if hi < 0 and -hi < (hi - lo) * 0.5:
        hi = 0.0
    return lo, hi


def _head(ctx, r, data, stack):
    """Заголовок: что за величина и в чём она. Возвращает оставшееся."""
    sc, f = ctx.sc, ctx.f
    p = data["plot"]
    row = stack.take(f.body.gh + sc.px(S.xs))
    unit = ("[" + p.unit + "]") if p.unit else "[безразмерная]"
    uw = f.micro.width(unit)
    ctx.dl.text(f.body, p.title, row.x, row.y, C.ink,
                max_px=max(0, row.w - uw - sc.px(S.sm)))
    ctx.dl.text_right(f.micro, unit, row.x1,
                      row.y + (f.body.gh - f.micro.gh) // 2, C.dim)
    h = W.label(ctx, L.Rect(row.x, stack.area.y, row.w, stack.area.h),
                p.what, role="micro", color=C.faint, wrapped=True)
    stack.take(h, sc.px(S.xs))


def _foot(ctx, r, data, stack):
    """Подвал: откуда взято и чего нельзя по этим данным утверждать."""
    sc, f = ctx.sc, ctx.f
    p = data["plot"]
    src = "источник: " + p.source
    if data.get("note"):
        src += "  ·  " + str(data["note"])
    h = W.label(ctx, L.Rect(stack.area.x, stack.area.y, stack.area.w,
                            stack.area.h), src, role="micro",
                color=C.faint, wrapped=True)
    stack.take(h, sc.px(2))
    if p.caveat:
        h = W.label(ctx, L.Rect(stack.area.x, stack.area.y, stack.area.w,
                                stack.area.h), "! " + p.caveat,
                    role="micro", color=C.warn, wrapped=True)
        stack.take(h, sc.px(2))


def _plot_box(ctx, box):
    """Поле графика: утопленная поверхность и волосяная рамка."""
    dl, sc = ctx.dl, ctx.sc
    dl.rect(box.x, box.y, box.x1, box.y1, C.sunken, 1.0, sc.px(R.control))
    dl.outline(box.x, box.y, box.x1, box.y1, C.border_soft,
               width=sc.hair(), radius=sc.px(R.control))


def _grid_y(ctx, box, lo, hi, unit=""):
    """Три линии сетки с подписями. Ноль, если он внутри, — отдельно."""
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    for i in range(4):
        v = lo + (hi - lo) * i / 3.0
        y = box.y1 - int((v - lo) / (hi - lo) * box.h)
        dl.hline(box.x, box.x1, y, C.grid, sc.hair())
        dl.text(f.micro, _fmt(v), box.x + sc.px(S.xs),
                min(y + sc.px(1), box.y1 - f.micro.gh), C.faint)
    if lo < 0.0 < hi:
        y = box.y1 - int((0.0 - lo) / (hi - lo) * box.h)
        dl.hline(box.x, box.x1, y, C.border, sc.hair())


def _thin(x, y, w):
    """Прореживание по минимуму и максимуму на столбец пикселей."""
    n = len(x)
    if n <= w * 2 or w < 4:
        return list(zip(x, y))
    out = []
    per = n / float(w)
    for i in range(w):
        a, b = int(i * per), max(int(i * per) + 1, int((i + 1) * per))
        seg = y[a:b]
        if not len(seg):
            continue
        xi = x[a]
        lo_i = int(seg.argmin())
        hi_i = int(seg.argmax())
        if lo_i <= hi_i:
            out.append((xi, float(seg[lo_i])))
            out.append((xi, float(seg[hi_i])))
        else:
            out.append((xi, float(seg[hi_i])))
            out.append((xi, float(seg[lo_i])))
    return out


def _line(ctx, box, data):
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    x = data["x"]
    ser = [(n, v) for n, v in data["series"] if len(v)]
    if not len(x) or not ser:
        W.label(ctx, box, "в файле нет точек", role="small", color=C.faint)
        return
    lo = min(float(v.min()) for _n, v in ser)
    hi = max(float(v.max()) for _n, v in ser)
    lo, hi = _nice_bounds(lo, hi)
    _grid_y(ctx, box, lo, hi)
    x0, x1 = float(x[0]), float(x[-1])
    span = max(x1 - x0, 1e-9)
    prev = ctx.dl.push_clip(box.x, box.y, box.x1, box.y1)
    for i, (name, v) in enumerate(ser):
        col = SERIES[i % len(SERIES)]
        pts = [(box.x + (px - x0) / span * box.w,
                box.y1 - (pv - lo) / (hi - lo) * box.h)
               for px, pv in _thin(x, v, box.w)]
        if len(pts) > 1:
            dl.polyline(pts, col, width=sc.pxf(1.4))
    ctx.dl.pop_clip(prev)
    # Подписи рядов — у правого края, цветом ряда: легенда отдельной
    # колонкой съедала бы ширину графика.
    ly = box.y + sc.px(S.xs)
    for i, (name, _v) in enumerate(ser):
        dl.text_right(f.micro, name, box.x1 - sc.px(S.xs), ly,
                      SERIES[i % len(SERIES)])
        ly += f.micro.gh + sc.px(2)
    dl.text(f.micro, _fmt(x0), box.x + sc.px(S.xs), box.y1 - f.micro.gh
            - sc.px(2), C.faint)
    dl.text_right(f.micro, _fmt(x1) + "  " + str(data.get("xlabel", "")),
                  box.x1 - sc.px(S.xs), box.y1 - f.micro.gh - sc.px(2),
                  C.faint)


def _track(ctx, box, data):
    """Траектория: равный масштаб по осям, иначе путь врёт формой."""
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    x, y = data["x"], data["y"]
    if not len(x):
        W.label(ctx, box, "в файле нет точек", role="small", color=C.faint)
        return
    w_mm, h_mm = data.get("bounds", (0.0, 0.0))
    foods = data.get("foods") or []
    xs = [float(x.min()), float(x.max())] + [fx for fx, _fy, _r in foods]
    ys = [float(y.min()), float(y.max())] + [fy for _fx, fy, _r in foods]
    if w_mm > 0 and h_mm > 0:
        xs += [0.0, w_mm]
        ys += [0.0, h_mm]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    pad = max(hi_x - lo_x, hi_y - lo_y) * 0.06 + 1e-6
    lo_x, hi_x, lo_y, hi_y = lo_x - pad, hi_x + pad, lo_y - pad, hi_y + pad
    k = min(box.w / max(hi_x - lo_x, 1e-9), box.h / max(hi_y - lo_y, 1e-9))
    ox = box.x + (box.w - (hi_x - lo_x) * k) / 2.0
    oy = box.y + (box.h - (hi_y - lo_y) * k) / 2.0

    def to(px, py):
        # Ось Y арены смотрит вверх, экранная — вниз.
        return (ox + (px - lo_x) * k,
                oy + (hi_y - py) * k)

    prev = dl.push_clip(box.x, box.y, box.x1, box.y1)
    if w_mm > 0 and h_mm > 0:
        a, b = to(0.0, 0.0), to(w_mm, h_mm)
        dl.outline(int(min(a[0], b[0])), int(min(a[1], b[1])),
                   int(max(a[0], b[0])), int(max(a[1], b[1])),
                   C.border, width=sc.hair())
    for fx, fy, fr in foods:
        cx, cy = to(fx, fy)
        rr = max(sc.px(3), fr * k)
        dl.rect(int(cx - rr), int(cy - rr), int(cx + rr), int(cy + rr),
                C.ok, 0.55, rr)
    pts = [to(float(a), float(b)) for a, b in zip(x, y)]
    if len(pts) > 1:
        dl.polyline(pts, C.plot, width=sc.pxf(1.4))
    d = sc.px(5)
    sx, sy = pts[0]
    dl.rect(int(sx - d), int(sy - d), int(sx + d), int(sy + d),
            C.dim, 1.0, d)
    ex, ey = pts[-1]
    dl.rect(int(ex - d), int(ey - d), int(ex + d), int(ey + d),
            C.accent, 1.0, d)
    dl.pop_clip(prev)
    dl.text(f.micro, "серое — начало, янтарное — конец, зелёное — еда",
            box.x + sc.px(S.xs), box.y1 - f.micro.gh - sc.px(2), C.faint)
    dl.text_right(f.micro, "%s × %s мм" % (_fmt(hi_x - lo_x),
                                           _fmt(hi_y - lo_y)),
                  box.x1 - sc.px(S.xs), box.y + sc.px(S.xs), C.faint)


def _bars(ctx, box, data):
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    labels = data.get("labels") or []
    values = data.get("values") or []
    counts = data.get("counts") or []
    if not labels:
        W.label(ctx, box, "в файле нет строк", role="small", color=C.faint)
        return
    lo, hi = _nice_bounds(min(values + [0.0]), max(values + [0.0]))
    n = len(labels)
    # Горизонтальные столбцы: подписи популяций и областей длинные, и
    # вертикально они легли бы поперёк друг друга. Ширина колонки
    # считается по тому, что действительно будет написано, вместе с
    # числом нейронов в скобках: без него подписи обрезались в «AL (3…».
    shown = [name + ((" (%d)" % counts[i])
                     if i < len(counts) and counts[i] else "")
             for i, name in enumerate(labels)]
    lab_w = min(int(box.w * 0.38),
                max(f.micro.width(t) for t in shown) + sc.px(S.sm))
    plot = L.Rect(box.x + lab_w, box.y, box.w - lab_w, box.h)
    zero = plot.x + int((0.0 - lo) / (hi - lo) * plot.w)
    dl.vline(zero, plot.y, plot.y1, C.border, sc.hair())
    rowh = max(sc.px(10), plot.h // max(n, 1))
    prev = dl.push_clip(box.x, box.y, box.x1, box.y1)
    for i, (name, v) in enumerate(zip(labels, values)):
        y = plot.y + i * rowh
        if y + rowh > plot.y1:
            break
        xv = plot.x + int((v - lo) / (hi - lo) * plot.w)
        col = C.plot if v >= 0 else C.plot_alt
        dl.rect(min(zero, xv), y + sc.px(2), max(zero, xv) + 1,
                y + rowh - sc.px(2), col, 0.85, sc.px(2))
        dl.text(f.micro, shown[i], box.x,
                y + (rowh - f.micro.gh) // 2, C.dim, max_px=lab_w - sc.px(4))
        dl.text(f.micro, _fmt(v), (max(zero, xv) + sc.px(S.xs)),
                y + (rowh - f.micro.gh) // 2, C.faint)
    dl.pop_clip(prev)


def _hist(ctx, box, data):
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    edges, counts = data["edges"], data["counts"]
    if not len(counts):
        W.label(ctx, box, "нечего считать", role="small", color=C.faint)
        return
    hi = max(1.0, float(counts.max()))
    _grid_y(ctx, box, 0.0, hi)
    n = len(counts)
    bw = box.w / float(n)
    for i, c in enumerate(counts):
        h = int(float(c) / hi * (box.h - sc.px(4)))
        x0 = box.x + int(i * bw)
        x1 = box.x + int((i + 1) * bw) - sc.px(1)
        if h > 0:
            dl.rect(x0, box.y1 - h, max(x1, x0 + 1), box.y1, C.plot, 0.85,
                    sc.px(2))
    dl.text(f.micro, _fmt(float(edges[0])), box.x + sc.px(S.xs),
            box.y1 - f.micro.gh - sc.px(2), C.faint)
    dl.text_right(f.micro, _fmt(float(edges[-1])) + "  Гц",
                  box.x1 - sc.px(S.xs), box.y1 - f.micro.gh - sc.px(2),
                  C.faint)


def _strip(ctx, box, data):
    """Точки по условиям: видно каждый эпизод, а не только среднее."""
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    groups = data.get("groups") or []
    if not groups:
        W.label(ctx, box, "эпизодов в файле нет", role="small",
                color=C.faint)
        return
    allv = [v for _n, vals, _m, _s in groups for v in vals
            if v == v and abs(v) != float("inf")]
    if not allv:
        W.label(ctx, box, "в файле только нечисловые значения",
                role="small", color=C.faint)
        return
    lo, hi = _nice_bounds(min(allv), max(allv))
    _grid_y(ctx, box, lo, hi)
    cw = box.w / float(len(groups))
    d = sc.px(4)
    for gi, (name, vals, mean, sem) in enumerate(groups):
        cx = box.x + cw * (gi + 0.5)
        # Точки разводятся по горизонтали детерминированно: одинаковые
        # значения иначе ложатся друг на друга и читаются как одно.
        for k, v in enumerate(vals):
            jitter = ((k * 7919) % 101 / 100.0 - 0.5) * cw * 0.42
            y = box.y1 - (v - lo) / (hi - lo) * box.h
            dl.rect(int(cx + jitter - d), int(y - d), int(cx + jitter + d),
                    int(y + d), C.plot, 0.8, d)
        my = box.y1 - (mean - lo) / (hi - lo) * box.h
        dl.hline(int(cx - cw * 0.3), int(cx + cw * 0.3), int(my),
                 C.accent, sc.px(2))
        # Усы рисуются только там, где sem — число: у серии из одного
        # эпизода он записан как `nan`, и усов у неё нет, потому что
        # разброса по одному наблюдению не бывает.
        ok_sem = (sem is not None and float(sem) == float(sem)
                  and abs(float(sem)) != float("inf") and float(sem) > 0)
        if ok_sem:
            ys = []
            for k in (-1.0, 1.0):
                yy = box.y1 - (mean + k * float(sem) - lo) / (hi - lo) * box.h
                ys.append(yy)
                dl.hline(int(cx - cw * 0.16), int(cx + cw * 0.16), int(yy),
                         C.accent, sc.hair())
            dl.vline(int(cx), int(min(ys)), int(max(ys)),
                     C.accent, sc.hair())
        dl.text_centre(f.micro, "%s · n %d" % (name, len(vals)),
                       int(box.x + cw * gi), int(box.x + cw * (gi + 1)),
                       box.y1 - f.micro.gh - sc.px(2), C.dim)
    dl.text(f.micro, "янтарная черта — среднее, усы — ±sem",
            box.x + sc.px(S.xs), box.y + sc.px(S.xs), C.faint)


_DRAW = {"line": _line, "track": _track, "bar": _bars, "hist": _hist,
         "strip": _strip}


def draw(ctx, r, data):
    """Нарисовать график целиком: шапка, поле, подвал. Вернуть высоту."""
    sc = ctx.sc
    st = L.Stack(r, gap=sc.px(S.xs))
    _head(ctx, r, data, st)
    kind = data.get("kind", "line")
    left = st.area.h
    # График занимает то место, что осталось под ним, но не больше
    # разумного: растянутый на весь экран ряд читается не лучше, а
    # зажатый в сто пикселей — хуже. Снизу всегда остаётся полоса под
    # источник и оговорку.
    box_h = max(sc.px(90), min(left - sc.px(52), sc.px(560)))
    box = st.take(box_h)
    _plot_box(ctx, box)
    inner = box.inset(sc.px(S.sm))
    fn = _DRAW.get(kind)
    if fn is not None and inner.w > sc.px(30) and inner.h > sc.px(24):
        fn(ctx, inner, data)
    st.skip(sc.px(S.xs))
    _foot(ctx, r, data, st)
    return st.used
