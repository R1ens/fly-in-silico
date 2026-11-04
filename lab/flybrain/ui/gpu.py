"""Ядра Taichi и буфер команд.

Весь кадр собирается в одно поле-накопитель и отдаётся на экран одним
`canvas.set_image`. Это снимает ограничение GGUI «один холст — одна картинка»:
`canvas.scene()` нельзя ограничить прямоугольником, а своё ядро — можно, и
мозг живёт в своей области окна, а не на весь экран.

Три правила, которые стоили проверки и записаны здесь, чтобы не выяснять
заново:

1. У поля кадра **начало координат внизу слева**: `acc[x, 0]` — нижняя строка.
   Наружу этот модуль показывает привычные экранные координаты (y вниз от
   верхнего края) и переводит их на границе ядра. Ни один вызывающий модуль о
   перевороте не знает.
2. **Запуск ядра стоит около 195 микросекунд.** Поэтому рисование на деталь
   запрещено: `DrawList` копит команды, а на кадр уходит по одному запуску на
   тип команды. Замер: 400 прямоугольников — 78 мс отдельными запусками против
   0,26 мс одним ядром.
3. **Порядок команд обязан соблюдаться.** Первая сборка параллелила по
   командам, и пересекающиеся прямоугольники записывались в неопределённом
   порядке: из двадцати команд, где каждая вторая ложится поверх предыдущей,
   верхний цвет оставался лишь на половине площади, и панели выходили
   изъеденными. Теперь фигуры разбираются по плиткам экрана, поток считает
   **один пиксель** и проходит список своей плитки **по порядку**. Каждый
   пиксель пишется ровно один раз, и наложение ведёт себя как положено.

Из третьего правила следует и устройство команд: прямоугольник и отрезок —
одна и та же запись с разной проверкой попадания. Отрезку больше ничего и не
нужно, зато путь мухи и кривая на графике стоят по команде на звено, а не по
команде на точку.
"""

import math

import numpy as np
import taichi as ti

__all__ = ["TILE", "CMD", "Surface", "DrawList",
           "look_at", "projection", "orbit_mvp"]

TILE = 64            # сторона плитки экрана

# Ширина записи команды: вид, геометрия (4), цвет (3), альфа, толщина,
# прямоугольник обрезки (4), запас (2).
CMD = 16
KIND_RECT = 0.0
KIND_SEG = 1.0
FAR = 1.0e9          # «не обрезать»


# ---------------------------------------------------------------- камера

def look_at(eye, target, up):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    s = np.cross(fwd, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -fwd
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, fwd @ eye
    return m


def projection(fovy_deg, aspect, near=0.05, far=40.0):
    t = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = t / max(aspect, 1e-6)
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orbit_mvp(yaw, pitch, dist, aspect, fovy=38.0):
    eye = np.array([dist * math.cos(pitch) * math.sin(yaw),
                    dist * math.sin(pitch),
                    dist * math.cos(pitch) * math.cos(yaw)], dtype=np.float32)
    view = look_at(eye, np.zeros(3, np.float32),
                   np.array([0.0, 1.0, 0.0], np.float32))
    return np.ascontiguousarray(projection(fovy, aspect) @ view), eye


# ---------------------------------------------------------------- команды

class DrawList:
    """Накопитель команд кадра.

    Координаты экранные: x вправо, y вниз от верхнего края окна. Команды
    исполняются в том порядке, в каком добавлены, — наложение работает.
    """

    __slots__ = ("cmds", "gi", "gx", "gy", "gc", "ga", "gclip",
                 "images", "_clip")

    def __init__(self):
        self.clear()

    def clear(self):
        self.cmds = []
        self.gi, self.gx, self.gy = [], [], []
        self.gc, self.ga, self.gclip = [], [], []
        self.images = []
        self._clip = (-FAR, -FAR, FAR, FAR)

    # Несколько модулей считают команды через `len(dl.rects)`.
    @property
    def rects(self):
        return self.cmds

    # -- обрезка --------------------------------------------------------

    def push_clip(self, x0, y0, x1, y1):
        """Ограничить всё последующее прямоугольником.

        Нужно спискам и журналу: строка на краю должна показываться
        наполовину, а не пропадать целиком. Вложенные обрезки пересекаются.
        """
        prev = self._clip
        self._clip = (max(prev[0], float(x0)), max(prev[1], float(y0)),
                      min(prev[2], float(x1)), min(prev[3], float(y1)))
        return prev

    def pop_clip(self, prev):
        self._clip = prev

    def no_clip(self):
        self._clip = (-FAR, -FAR, FAR, FAR)

    # -- фигуры ---------------------------------------------------------

    def rect(self, x0, y0, x1, y1, color, alpha=1.0, radius=0.0):
        """Прямоугольник, при `radius` — со скруглением и сглаженным краем.

        Скругление считается расстоянием до фигуры прямо в ядре, поэтому
        углы выходят чистыми на любом размере и ничего не стоит сверх
        обычной заливки.
        """
        if x1 > x0 and y1 > y0 and alpha > 0.002:
            c = self._clip
            r = max(0.0, min(float(radius),
                             (x1 - x0) * 0.5, (y1 - y0) * 0.5))
            self.cmds.append((KIND_RECT, float(int(x0)), float(int(y0)),
                              float(int(x1)), float(int(y1)),
                              color[0], color[1], color[2], float(alpha),
                              0.0, c[0], c[1], c[2], c[3], r, 0.0))

    def outline(self, x0, y0, x1, y1, color, width=1.0, alpha=1.0,
                radius=0.0):
        """Обводка кольцом: середина остаётся нетронутой.

        Чем это отличается от `frame`: `frame` со скруглением кладёт
        сплошную заливку и закрашивает всё, что под ней, если не передать
        `inner`. Обводке знать цвет середины не нужно — кольцо считается
        в ядре вычитанием поджатой заливки, и край остаётся сглаженным.
        """
        if x1 > x0 and y1 > y0 and alpha > 0.002 and width > 0.0:
            c = self._clip
            r = max(0.0, min(float(radius), (x1 - x0) * 0.5,
                             (y1 - y0) * 0.5))
            self.cmds.append((KIND_RECT, float(x0), float(y0),
                              float(x1), float(y1),
                              color[0], color[1], color[2], float(alpha),
                              0.0, c[0], c[1], c[2], c[3], r,
                              float(width)))

    def seg(self, x0, y0, x1, y1, color, width=1.5, alpha=1.0):
        """Отрезок с мягким краем: соединения выходят без зазубрин."""
        if alpha > 0.002 and width > 0.0:
            c = self._clip
            self.cmds.append((KIND_SEG, float(x0), float(y0), float(x1),
                              float(y1), color[0], color[1], color[2],
                              float(alpha), float(width),
                              c[0], c[1], c[2], c[3], 0.0, 0.0))

    def polyline(self, pts, color, width=1.5, alpha=1.0, closed=False):
        """Ломаная: по команде на звено, а не по команде на точку."""
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            self.seg(pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1],
                     color, width, alpha)
        if closed:
            self.seg(pts[-1, 0], pts[-1, 1], pts[0, 0], pts[0, 1],
                     color, width, alpha)

    def frame(self, x0, y0, x1, y1, color, width=1, alpha=1.0, radius=0.0,
              inner=None):
        """Рамка. При скруглении рисуется как кольцо: внешняя заливка и
        внутренняя поверх неё, иначе угол получается рваным.

        `inner` — чем закрасить середину; без него середина остаётся как
        была, и рамка видна кольцом.
        """
        if radius <= 0.0:
            self.rect(x0, y0, x1, y0 + width, color, alpha)
            self.rect(x0, y1 - width, x1, y1, color, alpha)
            self.rect(x0, y0 + width, x0 + width, y1 - width, color, alpha)
            self.rect(x1 - width, y0 + width, x1, y1 - width, color, alpha)
            return
        self.rect(x0, y0, x1, y1, color, alpha, radius)
        if inner is not None:
            self.rect(x0 + width, y0 + width, x1 - width, y1 - width,
                      inner, 1.0, max(0.0, radius - width))

    def hline(self, x0, x1, y, color, width=1, alpha=1.0):
        self.rect(x0, y, x1, y + width, color, alpha)

    def vline(self, x, y0, y1, color, width=1, alpha=1.0):
        self.rect(x, y0, x + width, y1, color, alpha)

    # -- текст ----------------------------------------------------------

    def text(self, atlas, s, x, y, color, alpha=1.0, max_px=None):
        """Положить строку. Возвращает x после последнего глифа."""
        if max_px is not None:
            s = atlas.fit(s, max_px)
        gw, index = atlas.gw, atlas.index
        fallback = index.get("?", 0)
        clip = self._clip
        for ch in s:
            g = index.get(ch, fallback)
            if ch != " ":
                self.gi.append(g)
                self.gx.append(int(x))
                self.gy.append(int(y))
                self.gc.append(color)
                self.ga.append(float(alpha))
                self.gclip.append(clip)
            x += int(gw[g])
        return x

    def text_right(self, atlas, s, x_right, y, color, alpha=1.0):
        return self.text(atlas, s, x_right - atlas.width(s), y, color, alpha)

    def text_centre(self, atlas, s, x0, x1, y, color, alpha=1.0):
        return self.text(atlas, s, (x0 + x1 - atlas.width(s)) // 2, y,
                         color, alpha)

    # -- картинки -------------------------------------------------------

    def image(self, rgb, x0, y0, w, h, alpha=1.0):
        self.images.append((rgb, int(x0), int(y0), int(w), int(h),
                            float(alpha)))

    # -- склейка --------------------------------------------------------

    def extend(self, other):
        """Дописать команды другого списка в конец этого.

        Нужно тем, кто копит кадр несколькими списками ради удобства
        выкладки: слить их и отдать одним запуском дешевле, чем звать ядро
        на каждый, а порядок при склейке сохраняется.
        """
        self.cmds.extend(other.cmds)
        self.images.extend(other.images)
        self.gi.extend(other.gi)
        self.gx.extend(other.gx)
        self.gy.extend(other.gy)
        self.gc.extend(other.gc)
        self.ga.extend(other.ga)
        self.gclip.extend(other.gclip)

    # -- выкладка в массивы --------------------------------------------

    def _bin(self, W, H, tile=TILE):
        """Разложить фигуры по плиткам экрана.

        Внутри плитки номера идут по возрастанию, то есть в порядке
        добавления, — именно на этом держится предсказуемое наложение.
        """
        if not self.cmds:
            return None, None, None, 0, 0
        a = np.asarray(self.cmds, dtype=np.float32)
        half = np.where(a[:, 0] == KIND_SEG, a[:, 9] * 0.5 + 1.0, 0.0)
        x0 = np.maximum(np.minimum(a[:, 1], a[:, 3]) - half, a[:, 10])
        y0 = np.maximum(np.minimum(a[:, 2], a[:, 4]) - half, a[:, 11])
        x1 = np.minimum(np.maximum(a[:, 1], a[:, 3]) + half, a[:, 12])
        y1 = np.minimum(np.maximum(a[:, 2], a[:, 4]) + half, a[:, 13])

        tiles_x = (W + tile - 1) // tile
        tiles_y = (H + tile - 1) // tile
        tx = np.clip(np.floor(x0 / tile), 0, tiles_x - 1).astype(np.int64)
        ty = np.clip(np.floor(y0 / tile), 0, tiles_y - 1).astype(np.int64)
        tx2 = np.clip(np.ceil(x1 / tile), 0, tiles_x).astype(np.int64)
        ty2 = np.clip(np.ceil(y1 / tile), 0, tiles_y).astype(np.int64)
        # пустые и целиком обрезанные фигуры выпадают сами
        empty = (x1 <= x0) | (y1 <= y0)
        nx = np.where(empty, 0, np.maximum(tx2 - tx, 0))
        ny = np.where(empty, 0, np.maximum(ty2 - ty, 0))
        counts = nx * ny
        total = int(counts.sum())
        if total == 0:
            return None, None, None, 0

        owner = np.repeat(np.arange(len(a), dtype=np.int64), counts)
        # Сдвиг внутри своей фигуры без списковых включений: на полутора
        # тысячах команд `concatenate([arange(c) ...])` стоил 1,9 мс.
        within = (np.arange(total, dtype=np.int64)
                  - np.repeat(np.cumsum(counts) - counts, counts))
        nxo = np.maximum(nx[owner], 1)
        cx = tx[owner] + within % nxo
        cy = ty[owner] + within // nxo
        tile_id = cy * tiles_x + cx

        order = np.lexsort((owner, tile_id))
        ntiles = tiles_x * tiles_y
        items = owner[order].astype(np.int32)
        cnt = np.bincount(tile_id[order], minlength=ntiles)
        starts = np.zeros(ntiles + 1, dtype=np.int32)
        np.cumsum(cnt, out=starts[1:])
        # Число плиток отдаётся отдельно, хотя сейчас оно и равно
        # `len(starts) - 1`: ядро обязано получать длину числом, а не
        # вычитать её из формы массива. Так его нельзя сломать, подав
        # массив другой длины.
        return (np.ascontiguousarray(a), np.ascontiguousarray(items),
                np.ascontiguousarray(starts), tiles_x, ntiles)

    def _glyph_arrays(self):
        """Шесть массивов глифов плюс их длина.

        Длина отдаётся отдельно по той же причине, что число плиток в
        `_bin`: ядро не должно выводить её из формы массива.
        """
        n = len(self.gi)
        if not n:
            return None
        return (np.asarray(self.gi, dtype=np.int32),
                np.asarray(self.gx, dtype=np.int32),
                np.asarray(self.gy, dtype=np.int32),
                np.ascontiguousarray(np.asarray(self.gc, dtype=np.float32)),
                np.asarray(self.ga, dtype=np.float32),
                np.ascontiguousarray(np.asarray(self.gclip, dtype=np.float32)),
                n)

# ---------------------------------------------------------------- поверхность

class Surface:
    """Поля кадра и ядра над ними.

    Окно само меняет размер и бывает портретным, причём `get_window_shape()`
    отдаёт новое значение без всякого события, поэтому размер опрашивается
    каждый кадр, а поля пересоздаются через `FieldsBuilder`. Ядра при этом
    перекомпилируются: они держат ссылку на конкретные поля.
    """

    def __init__(self):
        self.w = self.h = 0
        self._tree = None
        self.acc = None
        self.out = None

    def ensure(self, w, h):
        w, h = max(64, int(w)), max(64, int(h))
        if w == self.w and h == self.h:
            return False
        if self._tree is not None:
            self._tree.destroy()
        self.acc = ti.Vector.field(3, ti.f32)
        self.out = ti.Vector.field(3, ti.u8)
        fb = ti.FieldsBuilder()
        fb.dense(ti.ij, (w, h)).place(self.acc, self.out)
        self._tree = fb.finalize()
        self.w, self.h = w, h
        self._compile()
        # Поля надо материализовать здесь же, пустым проходом очистки.
        # Если этого не сделать, а размер сменится до первого настоящего
        # кадра — а окно в этой среде меняет размер само, — то `destroy`
        # сносит ещё не материализованное дерево, и следующий запуск ядра
        # падает на «Members of the following field have different
        # shapes». Стоит это один проход по экрану.
        self._k_clear(0.0, 0.0, 0.0)
        return True

    # ------------------------------------------------------------------

    def _compile(self):
        acc, out = self.acc, self.out

        @ti.kernel
        def k_clear(r: ti.f32, g: ti.f32, b: ti.f32):
            for x, y in acc:
                acc[x, y] = ti.Vector([r, g, b])

        @ti.kernel
        def k_tone(gamma: ti.f32):
            for x, y in acc:
                v = ti.min(acc[x, y], ti.Vector([1.0, 1.0, 1.0]))
                v = ti.max(v, ti.Vector([0.0, 0.0, 0.0]))
                if gamma != 1.0:
                    v = ti.Vector([ti.pow(v[0], gamma), ti.pow(v[1], gamma),
                                   ti.pow(v[2], gamma)])
                out[x, y] = ti.Vector([ti.cast(v[0] * 255.0 + 0.5, ti.u8),
                                       ti.cast(v[1] * 255.0 + 0.5, ti.u8),
                                       ti.cast(v[2] * 255.0 + 0.5, ti.u8)])

        @ti.kernel
        def k_splat(p: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                    code: ti.types.ndarray(dtype=ti.i32, ndim=1),
                    cmap: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    rcol: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    m: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    vx: ti.i32, vy: ti.i32, vw: ti.i32, vh: ti.i32,
                    H: ti.i32, base: ti.f32, rad0: ti.f32, rad1: ti.f32,
                    gain: ti.f32, mode: ti.i32, focus: ti.i32, dim: ti.f32,
                    sr: ti.f32, sg: ti.f32, sb: ti.f32,
                    dist: ti.f32, fade: ti.f32, rfloor: ti.f32):
            """Облако нейронов. Цвет считается здесь, а не на процессоре.

            Раскраска 138 639 точек по палитре стоила бы несколько
            миллисекунд на кадр на процессоре, причём каждый кадр заново: она
            зависит от активности, которая меняется постоянно. В ядре это
            бесплатно — точка всё равно уже прочитана.
            """
            for i in range(p.shape[0]):
                x, y, z = p[i, 0], p[i, 1], p[i, 2]
                cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
                if cw > 1e-6:
                    inv = 1.0 / cw
                    ndx = (m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]) * inv
                    ndy = (m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]) * inv
                    if -1.2 < ndx < 1.2 and -1.2 < ndy < 1.2:
                        av = ti.min(1.0, ti.max(0.0, a[i]))
                        r = 0.0
                        g = 0.0
                        bl = 0.0
                        if mode == 0:
                            n = cmap.shape[0]
                            # Частоты распределены с тяжёлым хвостом, а
                            # шкала строится по пику — и это съедает почти
                            # всю картину. Замер на стимуле в AL: работают
                            # 15 100 нейронов, медиана 18 Гц, девяносто
                            # процентов ниже 32, а пик 324. При линейной
                            # подстановке медианный нейрон попадает в 0,05
                            # палитры, то есть в чёрное, и кадр выходит
                            # пустым при пятнадцати тысячах работающих
                            # клеток. Корень растягивает нижнюю половину,
                            # порядок частот при этом не меняется ничуть:
                            # кто горел ярче, горит ярче и после.
                            lav = ti.sqrt(av)
                            t = lav * ti.cast(n - 1, ti.f32)
                            i0 = ti.cast(t, ti.i32)
                            i1 = ti.min(i0 + 1, n - 1)
                            f = t - ti.cast(i0, ti.f32)
                            r = cmap[i0, 0] + (cmap[i1, 0] - cmap[i0, 0]) * f
                            g = cmap[i0, 1] + (cmap[i1, 1] - cmap[i0, 1]) * f
                            bl = cmap[i0, 2] + (cmap[i1, 2] - cmap[i0, 2]) * f
                            # Молчащий нейрон magma красит почти в чёрный, и до
                            # первого прогона карта выходила пустой. Поэтому
                            # нижний край палитры уводится в холодный силуэт:
                            # форма мозга видна всегда, а горит только работа.
                            k = ti.min(1.0, lav * 4.0)
                            r = sr + (r - sr) * k
                            g = sg + (g - sg) * k
                            bl = sb + (bl - sb) * k
                        else:
                            k = code[i]
                            # Доля цвета нейропиля, не зависящая от
                            # активности. В анатомии она равна единице:
                            # там показывают строение, а не работу, и
                            # при нулевой активности прежние 0,20
                            # оставляли от облака двадцатую долю яркости.
                            sh = rfloor + (1.0 - rfloor) * av
                            r = rcol[k, 0] * sh
                            g = rcol[k, 1] * sh
                            bl = rcol[k, 2] * sh
                        if focus >= 0 and code[i] != focus:
                            r *= dim
                            g *= dim
                            bl *= dim
                        # Затухание по дальности. Без него облако выходит
                        # плоской пылью одинаковой яркости: ближняя
                        # поверхность мозга не отличается от дальней, и
                        # формы не видно вовсе.
                        if fade > 0.0:
                            k = 1.0 - fade * (cw - dist) / ti.max(dist, 1e-3)
                            k = ti.min(1.35, ti.max(0.18, k))
                            r *= k
                            g *= k
                            bl *= k
                        sx = vx + (ndx * 0.5 + 0.5) * vw
                        sy = vy + (1.0 - (ndy * 0.5 + 0.5)) * vh
                        rad = rad0 + rad1 * av
                        r2 = rad * rad
                        ir = ti.cast(ti.ceil(rad), ti.i32)
                        x0 = ti.cast(sx, ti.i32)
                        y0 = ti.cast(sy, ti.i32)
                        amp = gain * (base + (1.0 - base) * av)
                        for dx in range(-ir, ir + 1):
                            for dy in range(-ir, ir + 1):
                                px, py = x0 + dx, y0 + dy
                                if vx <= px < vx + vw and vy <= py < vy + vh:
                                    fy = H - 1 - py
                                    d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                                    if d2 <= r2:
                                        wt = amp * ti.exp(-2.5 * d2 / r2)
                                        ti.atomic_add(acc[px, fy][0], r * wt)
                                        ti.atomic_add(acc[px, fy][1], g * wt)
                                        ti.atomic_add(acc[px, fy][2], bl * wt)

        @ti.kernel
        def k_shapes(cmd: ti.types.ndarray(dtype=ti.f32, ndim=2),
                     items: ti.types.ndarray(dtype=ti.i32, ndim=1),
                     starts: ti.types.ndarray(dtype=ti.i32, ndim=1),
                     tiles_x: ti.i32, side: ti.i32, W: ti.i32, H: ti.i32,
                     ntiles: ti.i32):
            """Поток — пиксель, список своей плитки проходится по порядку."""
            for t, j in ti.ndrange(ntiles, side * side):
                lo, hi = starts[t], starts[t + 1]
                if hi > lo:
                    x = (t % tiles_x) * side + j % side
                    y = (t // tiles_x) * side + j // side
                    if x < W and y < H:
                        fy = H - 1 - y
                        col = acc[x, fy]
                        px = ti.cast(x, ti.f32) + 0.5
                        py = ti.cast(y, ti.f32) + 0.5
                        for s in range(lo, hi):
                            k = items[s]
                            if (cmd[k, 10] <= px < cmd[k, 12]
                                    and cmd[k, 11] <= py < cmd[k, 13]):
                                cover = 0.0
                                if cmd[k, 0] == 0.0:
                                    rr = cmd[k, 14]
                                    bw = cmd[k, 15]
                                    if rr <= 0.0 and bw <= 0.0:
                                        if (cmd[k, 1] <= px < cmd[k, 3]
                                                and cmd[k, 2] <= py < cmd[k, 4]):
                                            cover = 1.0
                                    else:
                                        # расстояние до скруглённого
                                        # прямоугольника: угол выходит чистым
                                        # без отдельной геометрии
                                        hx = (cmd[k, 3] - cmd[k, 1]) * 0.5
                                        hy = (cmd[k, 4] - cmd[k, 2]) * 0.5
                                        ox = ti.abs(px - (cmd[k, 1] + hx)) - (hx - rr)
                                        oy = ti.abs(py - (cmd[k, 2] + hy)) - (hy - rr)
                                        mx = ti.max(ox, 0.0)
                                        my = ti.max(oy, 0.0)
                                        d = (ti.sqrt(mx * mx + my * my)
                                             + ti.min(ti.max(ox, oy), 0.0) - rr)
                                        cover = ti.min(1.0, ti.max(0.0, 0.5 - d))
                                        if bw > 0.0:
                                            # кольцо: из заливки вычитается
                                            # заливка, поджатая на толщину.
                                            # Так обводка не закрашивает то,
                                            # что под ней, и угол остаётся
                                            # чистым
                                            cover -= ti.min(1.0, ti.max(
                                                0.0, 0.5 - (d + bw)))
                                            cover = ti.max(cover, 0.0)
                                else:
                                    ax, ay = cmd[k, 1], cmd[k, 2]
                                    bx, by = cmd[k, 3], cmd[k, 4]
                                    ux, uy = bx - ax, by - ay
                                    ll = ux * ux + uy * uy
                                    tt = 0.0
                                    if ll > 1e-9:
                                        tt = ((px - ax) * ux + (py - ay) * uy) / ll
                                        tt = ti.min(1.0, ti.max(0.0, tt))
                                    dx = px - (ax + tt * ux)
                                    dy = py - (ay + tt * uy)
                                    d = ti.sqrt(dx * dx + dy * dy)
                                    # мягкий край в один пиксель: соединения
                                    # звеньев выходят без зазубрин
                                    cover = ti.min(1.0, ti.max(
                                        0.0, cmd[k, 9] * 0.5 + 0.5 - d))
                                if cover > 0.0:
                                    al = cmd[k, 8] * cover
                                    col = ti.Vector([
                                        col[0] + (cmd[k, 5] - col[0]) * al,
                                        col[1] + (cmd[k, 6] - col[1]) * al,
                                        col[2] + (cmd[k, 7] - col[2]) * al])
                        acc[x, fy] = col

        @ti.kernel
        def k_text(atlas: ti.types.ndarray(dtype=ti.u8, ndim=2),
                   agx: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   agw: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   gi: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   gx: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   gy: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   gc: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   ga: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   gq: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   gh: ti.i32, mw: ti.i32, W: ti.i32, H: ti.i32,
                   n: ti.i32):
            for k, j in ti.ndrange(n, mw * gh):
                u, v = j % mw, j // mw
                g = gi[k]
                if u < agw[g]:
                    x = gx[k] + u
                    y = gy[k] + v
                    fy = H - 1 - y
                    if (0 <= x < W and 0 <= fy < H
                            and gq[k, 0] <= x < gq[k, 2]
                            and gq[k, 1] <= y < gq[k, 3]):
                        al = ti.cast(atlas[agx[g] + u, v], ti.f32) / 255.0 * ga[k]
                        if al > 0.004:
                            o = acc[x, fy]
                            acc[x, fy] = ti.Vector([
                                o[0] + (gc[k, 0] - o[0]) * al,
                                o[1] + (gc[k, 1] - o[1]) * al,
                                o[2] + (gc[k, 2] - o[2]) * al])

        @ti.kernel
        def k_image(src: ti.types.ndarray(dtype=ti.u8, ndim=3),
                    x0: ti.i32, y0: ti.i32, dw: ti.i32, dh: ti.i32,
                    W: ti.i32, H: ti.i32, alpha: ti.f32):
            sh, sw = src.shape[0], src.shape[1]
            for i, j in ti.ndrange(dw, dh):
                x, y = x0 + i, y0 + j
                fy = H - 1 - y
                if 0 <= x < W and 0 <= fy < H:
                    # Выборка билинейная, а не ближайшим пикселем: кадр
                    # тела приходит 960x720, а вьюпорт бывает и шире, и
                    # уже. Ближайший пиксель давал лестницу на ногах и
                    # муар на клетчатом полу — ровно то, из-за чего сцена
                    # выглядела сырее остального окна.
                    fu = (ti.cast(i, ti.f32) + 0.5) / dw * sw - 0.5
                    fv = (ti.cast(j, ti.f32) + 0.5) / dh * sh - 0.5
                    u0 = ti.cast(ti.floor(fu), ti.i32)
                    v0 = ti.cast(ti.floor(fv), ti.i32)
                    tu = fu - ti.cast(u0, ti.f32)
                    tv = fv - ti.cast(v0, ti.f32)
                    u0 = ti.min(ti.max(u0, 0), sw - 1)
                    v0 = ti.min(ti.max(v0, 0), sh - 1)
                    u1 = ti.min(u0 + 1, sw - 1)
                    v1 = ti.min(v0 + 1, sh - 1)
                    o = acc[x, fy]
                    for c in ti.static(range(3)):
                        a00 = ti.cast(src[v0, u0, c], ti.f32)
                        a10 = ti.cast(src[v0, u1, c], ti.f32)
                        a01 = ti.cast(src[v1, u0, c], ti.f32)
                        a11 = ti.cast(src[v1, u1, c], ti.f32)
                        top = a00 + (a10 - a00) * tu
                        bot = a01 + (a11 - a01) * tu
                        v = (top + (bot - top) * tv) / 255.0
                        o[c] = o[c] + (v - o[c]) * alpha
                    acc[x, fy] = o

        self._k_clear = k_clear
        self._k_tone = k_tone
        self._k_splat = k_splat
        self._k_shapes = k_shapes
        self._k_text = k_text
        self._k_image = k_image

    # ------------------------------------------------------------------

    def clear(self, color):
        self._k_clear(color[0], color[1], color[2])

    def splat(self, pos, act, code, cmap, rcol, mvp, viewport, mode=0,
              focus=-1, dim=0.07, base=0.14, rad0=1.25, rad1=2.6, gain=1.0,
              silhouette=(0.23, 0.28, 0.35), dist=3.0, fade=0.62,
              rfloor=0.20):
        """Облако нейронов в заданный прямоугольник окна.

        Аддитивно, без теста глубины, с гауссовым спадом — ровно то, что
        делает `assets/viewer.js`, перенесённое в ядро. `mode` 0 — magma по
        активности, 1 — цвет нейропиля; `focus` гасит всё, кроме одной
        области.
        """
        vx, vy, vw, vh = (int(v) for v in viewport)
        if vw <= 0 or vh <= 0:
            return
        self._k_splat(pos, act, code, cmap, rcol, mvp, vx, vy, vw, vh, self.h,
                      float(base), float(rad0), float(rad1), float(gain),
                      int(mode), int(focus), float(dim),
                      float(silhouette[0]), float(silhouette[1]),
                      float(silhouette[2]), float(dist), float(fade),
                      float(rfloor))

    def draw(self, dl, atlas=None):
        """Фигуры, картинки и текст: по одному запуску ядра на вид."""
        cmd, items, starts, tiles_x, ntiles = dl._bin(self.w, self.h)
        if cmd is not None:
            self._k_shapes(cmd, items, starts, tiles_x, TILE, self.w, self.h,
                           ntiles)
        for img in dl.images:
            rgb, x0, y0, w, h = img[:5]
            alpha = img[5] if len(img) > 5 else 1.0
            self._k_image(np.ascontiguousarray(rgb), x0, y0, w, h,
                          self.w, self.h, float(alpha))
        g = dl._glyph_arrays()
        if g is not None and atlas is not None:
            gi, gx, gy, gc, ga, gq, n = g
            self._k_text(atlas.pixels, atlas.gx, atlas.gw,
                         gi, gx, gy, gc, ga, gq,
                         atlas.gh, atlas.max_gw, self.w, self.h, n)

    def present(self, canvas, gamma=1.0):
        self._k_tone(float(gamma))
        canvas.set_image(self.out)
        self.finish()

    @staticmethod
    def finish():
        """Дождаться видеокарты. РАЗ в кадр, и это не про скорость.

        Taichi 1.7.4 держит на каждом незавершённом запуске свою
        память и отпускает её только на синхронизации. Замерено
        отдельной пробой на одном и том же кадре: 600 кадров без
        синхронизации — **+34,32 МБ**, те же 600 с `ti.sync()` на
        кадр — **+0,06 МБ**. В окне это давало около десяти мегабайт
        в секунду: долгий сеанс намерил 541 -> 1408 МБ за 105 секунд,
        ровно и без насыщения.

        Течёт это одинаково в v2 и в v3 — проверено запуском той же
        пробы на замороженном дереве (0,059 против 0,0595 МБ на кадр),
        то есть беда не этого этапа, а всей отрисовки с самого начала.
        Просто до сих пор никто не держал окно открытым достаточно
        долго, чтобы её увидеть.

        Ждать здесь нечего: кадр всё равно показывается только после
        того, как ядра его досчитали.
        """
        ti.sync()
