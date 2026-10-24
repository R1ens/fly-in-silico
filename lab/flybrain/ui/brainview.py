"""Карта мозга: камера, подписи нейропилей, наведение, фокус.

Это перенос того, что делает `assets/viewer.js`, на своё ядро: аддитивные
сплаты с гауссовым спадом, палитра magma по активности либо цвет нейропиля,
подписи областей с уходом от перекрытий, затемнение всего, кроме выбранной
области, четыре предустановленных вида.

Чего в вебе не было и что добавлено здесь: карта живёт в своём прямоугольнике
окна, а не на весь холст. Ради этого всё и считается своим ядром — штатную
сцену Taichi вьюпортом ограничить нельзя.
"""

import math

import numpy as np

from . import gpu, theme

__all__ = ["VIEWS", "BrainView"]

# Четыре вида, как в отчёте: фронт, сверху, сбоку, три четверти.
VIEWS = (
    ("фронт", 0.0, 0.0),
    ("сверху", 0.0, 1.35),
    ("сбоку", math.pi / 2, 0.0),
    ("три четверти", 0.85, 0.45),
)

# Колесо двигает множитель к подогнанному расстоянию, а не само
# расстояние: иначе при смене вида масштаб каждый раз сбивается.
ZOOM_MIN, ZOOM_MAX = 0.25, 4.0


class BrainView:
    """Состояние карты и всё, что с ней делает мышь."""

    def __init__(self, pos, region_code, region_names, region_rgb,
                 counts=None):
        self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        self.n = self.pos.shape[0]
        self.region_code = np.ascontiguousarray(region_code, dtype=np.int32)
        self.region_names = list(region_names)
        self.region_rgb = np.ascontiguousarray(region_rgb, dtype=np.float32)
        self.counts = (counts if counts is not None
                       else np.bincount(self.region_code,
                                        minlength=len(self.region_names)))

        # Якорь области — медиана её точек: устойчивее среднего к хвостам,
        # а подпись должна стоять в гуще, а не между двумя долями.
        self.anchors = np.zeros((len(self.region_names), 3), dtype=np.float32)
        for i in range(len(self.region_names)):
            sel = self.pos[self.region_code == i]
            if len(sel):
                self.anchors[i] = np.median(sel, axis=0)

        self.act = np.zeros(self.n, dtype=np.float32)
        # Палитра уходит в ядро: цвет точки считается там, где точка и так
        # прочитана. Перекрашивать 138 639 записей на процессоре каждый кадр
        # стоило бы несколько миллисекунд, а активность меняется постоянно.
        self._cmap = np.ascontiguousarray(
            np.asarray(theme.MAGMA, dtype=np.float32))

        self.view_i = 3
        _, self.yaw, self.pitch = VIEWS[self.view_i]
        # Расстояние не задаётся числом: оно подбирается под охват облака в
        # текущем повороте, а колесо двигает только множитель. Иначе при
        # повороте «сбоку» мозг занимает треть кадра, а «сверху» вылезает.
        self.zoom = 1.0
        self.dist = 3.0
        self._fit_key = None
        # Подвыборка для подгонки: по 12 тысячам точек охват считается с той
        # же точностью, но за доли миллисекунды.
        self._fit_step = max(1, self.n // 12000)
        self._fit_pts = np.ascontiguousarray(self.pos[::self._fit_step])
        self._base = np.zeros(3, dtype=np.float32)      # центр набора
        self._centre = np.zeros(3, dtype=np.float32)    # куда смотрит камера
        self._fit_dist = 3.0
        self._fit_rev = 0
        self.mode = "activity"        # activity | regions
        self.only_active = False
        self.show_labels = True
        # Сила затухания по дальности: 0 — плоская пыль, 1 — резкая
        # передняя поверхность и почти чёрная задняя.
        self.fade = 0.62
        self.region_floor = 0.20      # сколько цвета нейропиля вне работы
        self.base_regions = 0.42      # яркость облака в анатомии
        self.base_activity = 0.13     # в активности светит сама активность
        self.rad0 = 1.25              # радиус точки при нулевой активности
        self.rad1 = 2.6               # прибавка радиуса при полной
        self.dim = 0.07               # во сколько гаснет невыделенное
        self.focus = None             # номер области или None
        self.hover = None             # номер нейрона под курсором

        self._drag = None
        self._mvp = None
        self._vp = (0, 0, 1, 1)

        # Выделение произвольного набора нейронов. Ядро умеет гасить всё,
        # кроме одной области, — этим и пользуемся: выделенные получают
        # лишний, несуществующий код области с отдельным цветом, и фокус
        # ставится на него. Своей команды в ядре для этого не нужно.
        self.sel_idx = None           # numpy-массив индексов или None
        # Как показывать выделение. Это НЕ режим цвета: анатомия и
        # активность переключаются отдельно и на выбор не влияют.
        #   context — выбранное ярко, остальной мозг приглушён;
        #   only    — только набор, остального нет вовсе;
        #   all     — выделение не влияет на картинку.
        self.sel_view = "context"
        self.sel_label = ""
        self._sel_code = None
        self._sel_rgb = None
        self._label_boxes = []        # [(x0,y0,x1,y1, номер области)]
        self._fit_idx = None          # по чему подбирать охват
        self._proj_key = None         # кэш проекции для наведения
        self._proj = None
        self._act_rev = 0             # счётчик смен активности
        self._pick_step = max(1, self.n // 40000)

    # ------------------------------------------------------------------ данные

    def set_activity(self, act):
        """Нормированная активность 0..1 на нейрон."""
        self._act_rev += 1
        if act is not None:
            self.act = np.ascontiguousarray(act, dtype=np.float32)

    def set_mode(self, mode):
        self.mode = mode

    def set_focus(self, region_i):
        self.focus = region_i

    # ------------------------------------------------------------ выделение

    def set_selection(self, idx, label="", colour=(1.0, 0.72, 0.36)):
        """Выделить набор нейронов. `idx` — индексы или None.

        Выделение живёт отдельно от режима цвета: переключение анатомии и
        активности его не трогает, как и требует бриф.
        """
        if idx is None or len(idx) == 0:
            self.sel_idx = None
            self.sel_label = ""
            self._sel_code = self._sel_rgb = None
            return 0
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self.n)]
        self.sel_idx = idx
        self.sel_label = str(label)
        extra = len(self.region_names)
        code = self.region_code.copy()
        code[idx] = extra
        self._sel_code = np.ascontiguousarray(code)
        self._sel_rgb = np.ascontiguousarray(
            np.vstack([self.region_rgb,
                       np.asarray(colour, dtype=np.float32)[None, :]]))
        return int(len(idx))

    def clear_selection(self):
        return self.set_selection(None)

    def fit_to(self, idx=None):
        """По чему подбирать охват: по всему облаку или по набору.

        «Показать выделенное» — это именно смена того, что вписывается в
        кадр, а не отдельная камера.
        """
        if idx is None or len(idx) == 0:
            self._fit_idx = None
            self._fit_pts = np.ascontiguousarray(self.pos[::self._fit_step])
            self._base = np.zeros(3, dtype=np.float32)
        else:
            p = self.pos[np.asarray(idx, dtype=np.int64)]
            if len(p) > 20000:
                p = p[:: max(1, len(p) // 20000)]
            self._fit_idx = True
            self._fit_pts = np.ascontiguousarray(p)
            self._base = p.mean(axis=0).astype(np.float32)
        self._fit_rev += 1
        self._fit_key = None

    def next_view(self):
        self.view_i = (self.view_i + 1) % len(VIEWS)
        _, self.yaw, self.pitch = VIEWS[self.view_i]

    def _fit(self, aspect, fovy=38.0, margin=1.08):
        """Расстояние и точка, на которую смотрит камера.

        Берётся не радиус сферы, а настоящий размах в плоскости экрана при
        текущем повороте: мозг мухи широкий и плоский, и по сфере он всегда
        оказывался бы вдвое меньше кадра.

        Размах считается по краям, а не по модулю от нуля. Разница не
        косметическая: центр габаритной коробки — не центр облака, и при
        симметричной подгонке мозг уезжал за правый край, оставляя пустоту
        слева. Теперь камера смотрит в середину настоящего размаха.
        """
        key = (round(self.yaw, 2), round(self.pitch, 2), round(aspect, 3),
               int(self._fit_rev))
        if key == self._fit_key:
            return self._fit_dist

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        # оси экрана в мировых координатах для камеры на орбите
        right = np.array([cy, 0.0, -sy], dtype=np.float32)
        up = np.array([-sy * sp, cp, -cy * sp], dtype=np.float32)
        pts = self._fit_pts - self._base[None, :]
        px = pts @ right
        py = pts @ up
        # края по процентилю, а не по максимуму: десяток выбросов не
        # должен отодвигать камеру от всего остального
        lo_x, hi_x = np.percentile(px, (0.3, 99.7))
        lo_y, hi_y = np.percentile(py, (0.3, 99.7))
        rx = float(hi_x - lo_x) * 0.5
        ry = float(hi_y - lo_y) * 0.5
        t = math.tan(math.radians(fovy) * 0.5)
        need = max(ry / t, rx / (t * max(aspect, 1e-6)))
        self._centre = (self._base
                        + right * float(lo_x + hi_x) * 0.5
                        + up * float(lo_y + hi_y) * 0.5).astype(np.float32)

        # Это была только первая догадка, ортографическая. Проекция
        # перспективная, и ближняя половина облака проецируется заметно
        # крупнее дальней: при такой подгонке за кадром оказывалось от
        # трёх до девяти процентов точек. Уточняем по настоящей
        # проекции — несколько шагов, каждый стоит одного прохода по
        # прорежённому облаку.
        d = max(need, 0.2)
        for _ in range(6):
            m = self._mvp_for(d, aspect, fovy)
            w = self._fit_pts @ m[3, :3] + m[3, 3]
            good = w > 1e-6
            if not good.any():
                break
            ww = w[good]
            ndx = (self._fit_pts[good] @ m[0, :3] + m[0, 3]) / ww
            ndy = (self._fit_pts[good] @ m[1, :3] + m[1, 3]) / ww
            lx, hx = np.percentile(ndx, (0.2, 99.8))
            ly, hy = np.percentile(ndy, (0.2, 99.8))
            # Сперва ставим середину размаха в центр кадра: середина
            # ортографическая и перспективная — не одно и то же, и без
            # этого облако прижималось к правому краю.
            cx, cy = float(lx + hx) * 0.5, float(ly + hy) * 0.5
            if abs(cx) > 0.004 or abs(cy) > 0.004:
                self._centre = (self._centre
                                + right * (cx * d * t * max(aspect, 1e-6))
                                + up * (cy * d * t)).astype(np.float32)
            half = max(float(hx - lx), 0.0) * 0.5
            halfy = max(float(hy - ly), 0.0) * 0.5
            k = max(half, halfy) * margin
            if abs(k - 1.0) < 0.015 and abs(cx) <= 0.004                     and abs(cy) <= 0.004:
                break
            d = max(0.2, d * k)
        self._fit_key = key
        self._fit_dist = d
        return self._fit_dist

    def _mvp_for(self, dist, aspect, fovy=38.0):
        """Матрица для пробного расстояния, уже со сдвигом на центр."""
        m, _ = gpu.orbit_mvp(self.yaw, self.pitch, dist, aspect, fovy=fovy)
        if float(np.abs(self._centre).max()) > 1e-6:
            m = m.copy()
            m[:, 3] = m[:, 3] - m[:, :3] @ self._centre
        return m

    def refit(self):
        self._fit_key = None

    @property
    def view_name(self):
        return VIEWS[self.view_i][0]

    # ------------------------------------------------------------------ мышь

    def interact(self, viewport, cursor, pressed, wheel=0.0):
        """Вращение перетаскиванием и приближение колесом внутри вьюпорта."""
        vx, vy, vw, vh = viewport
        cx, cy = cursor
        inside = vx <= cx < vx + vw and vy <= cy < vy + vh

        if pressed and (inside or self._drag is not None):
            if self._drag is not None:
                self.yaw -= (cx - self._drag[0]) / max(vw, 1) * 4.0
                self.pitch = max(-1.45, min(
                    1.45, self.pitch + (cy - self._drag[1]) / max(vh, 1) * 4.0))
                self.view_i = -1
            self._drag = (cx, cy)
        else:
            self._drag = None

        if wheel and inside:
            self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * (0.9 ** wheel)))
        return inside

    def pick(self, cursor, radius=14.0, active_only=True):
        """Ближайший нейрон под курсором, или None.

        Проецируется на CPU и только когда курсор в карте: для 9–40 тысяч
        активных это около миллисекунды, а каждый кадр это и не нужно.
        """
        if self._mvp is None:
            return None
        vx, vy, vw, vh = self._vp
        cx, cy = cursor
        if not (vx <= cx < vx + vw and vy <= cy < vy + vh):
            self.hover = None
            return None

        key = (self._mvp.tobytes(), self._vp, bool(active_only),
               int(self._act_rev))
        cached = self._proj if self._proj_key == key else None
        if cached is None:
            if active_only:
                idx = np.nonzero(self.act > 0.01)[0]
            else:
                # Всё облако проецировать незачем: наведение отвечает на
                # вопрос «куда я тычу», и прорежённого хватает. Полный
                # проход по 138 тысячам стоил 24 мс на каждом движении
                # мыши, прорежённый — около трёх.
                idx = np.arange(0, self.n, self._pick_step)
            if not len(idx):
                self._proj_key, self._proj = key, None
                self.hover = None
                return None
            p = self.pos[idx]
            m = self._mvp
            w = p @ m[3, :3] + m[3, 3]
            ok = w > 1e-6
            if not ok.any():
                self._proj_key, self._proj = key, None
                self.hover = None
                return None
            idx, p, w = idx[ok], p[ok], w[ok]
            sx = vx + ((p @ m[0, :3] + m[0, 3]) / w * 0.5 + 0.5) * vw
            sy = vy + (1.0 - ((p @ m[1, :3] + m[1, 3]) / w * 0.5 + 0.5)) * vh
            self._proj_key, self._proj = key, (idx, sx, sy)
        else:
            idx, sx, sy = cached
        d2 = (sx - cx) ** 2 + (sy - cy) ** 2
        j = int(np.argmin(d2))
        self.hover = int(idx[j]) if d2[j] <= radius * radius else None
        return self.hover

    # ------------------------------------------------------------------ вывод

    def draw(self, surf, dl, atlas, viewport, gain=1.0):
        vx, vy, vw, vh = (int(v) for v in viewport)
        self._vp = (vx, vy, vw, vh)
        if vw < 8 or vh < 8:
            return
        aspect = vw / max(vh, 1)
        self.dist = self._fit(aspect) * self.zoom
        # Камера на орбите всегда смотрит в ноль. Чтобы навести её на
        # выделенный участок, мир сдвигается под неё: матрица
        # домножается справа на перенос. Так проекция, подписи и
        # наведение остаются согласованными — они все берут `_mvp`.
        self._mvp = self._mvp_for(self.dist, aspect)

        mode = 1 if self.mode == "regions" else 0
        focus = -1 if self.focus is None else int(self.focus)
        code, rgb = self.region_code, self.region_rgb
        if self.sel_idx is not None and self._sel_code is not None                 and self.sel_view != "all":
            # Выделение гасит всё, кроме выбранного, но цвет не отбирает:
            # приглушение в ядре считается по коду и работает в обоих
            # режимах. В активности выделение оставляет магму, иначе
            # режим активности с выделением показывал бы анатомию.
            code, rgb = self._sel_code, self._sel_rgb
            focus = len(self.region_names)
            if mode == 1:
                pass
        act = self.act
        if self.sel_idx is not None and self.sel_view == "only":
            # Только набор: рисуем подмножество целиком ярким. Фокус тут
            # не нужен — гасить уже нечего.
            sel = np.asarray(self.sel_idx, dtype=np.int64)
            surf.splat(np.ascontiguousarray(self.pos[sel]),
                       np.ascontiguousarray(act[sel]),
                       np.ascontiguousarray(code[sel]),
                       self._cmap, rgb, self._mvp, viewport,
                       mode=mode, focus=-1,
                       base=self.base_regions if mode == 1 else 0.30,
                       gain=gain, dist=self.dist, fade=self.fade,
                       rad0=self.rad0, rad1=self.rad1, dim=self.dim,
                       rfloor=self.region_floor)
        elif self.only_active:
            # Фоновое облако скрыто: остаются только сработавшие, и им можно
            # поднять яркость, иначе карта выглядит пустой.
            sel = np.nonzero(act > 0.01)[0]
            if len(sel):
                surf.splat(np.ascontiguousarray(self.pos[sel]),
                           np.ascontiguousarray(act[sel]),
                           np.ascontiguousarray(code[sel]),
                           self._cmap, rgb, self._mvp, viewport,
                           mode=mode, focus=focus, base=0.30, gain=gain,
                           dist=self.dist, fade=self.fade,
                           rad0=self.rad0, rad1=self.rad1, dim=self.dim,
                           rfloor=self.region_floor)
        else:
            base = (self.base_regions if mode == 1
                    else self.base_activity)
            surf.splat(self.pos, act, code, self._cmap,
                       rgb, self._mvp, viewport,
                       mode=mode, focus=focus, base=base, gain=gain,
                       dist=self.dist, fade=self.fade,
                       rad0=self.rad0, rad1=self.rad1, dim=self.dim,
                       rfloor=self.region_floor)

        if self.show_labels:
            self._labels(dl, atlas, viewport)

    def _labels(self, dl, atlas, viewport):
        """Подписи нейропилей, перекрывающиеся пропускаются.

        Порядок — по числу нейронов: крупные области получают подпись первыми,
        мелкие уступают. Так же устроено в `viewer.js`, и это единственное,
        что спасает карту от каши из тридцати шести подписей.
        """
        vx, vy, vw, vh = viewport
        m = self._mvp
        a = self.anchors
        w = a @ m[3, :3] + m[3, 3]
        ok = w > 1e-6
        sx = np.full(len(a), -1e9, dtype=np.float32)
        sy = np.full(len(a), -1e9, dtype=np.float32)
        sx[ok] = vx + ((a[ok] @ m[0, :3] + m[0, 3]) / w[ok] * 0.5 + 0.5) * vw
        sy[ok] = vy + (1.0 - ((a[ok] @ m[1, :3] + m[1, 3]) / w[ok] * 0.5 + 0.5)) * vh

        order = np.argsort(-np.asarray(self.counts))
        taken = []
        self._label_boxes = []
        pad = 4
        for i in order:
            if not ok[i]:
                continue
            name = self.region_names[i]
            if name == "unknown":
                continue
            tw = atlas.width(name)
            x0 = int(sx[i]) - tw // 2
            y0 = int(sy[i]) - atlas.gh // 2
            x1, y1 = x0 + tw, y0 + atlas.gh
            if x0 < vx + 2 or x1 > vx + vw - 2 or y0 < vy + 2 or y1 > vy + vh - 2:
                continue
            clash = False
            for bx0, by0, bx1, by1 in taken:
                if (x0 - pad < bx1 and x1 + pad > bx0
                        and y0 - pad < by1 and y1 + pad > by0):
                    clash = True
                    break
            if clash:
                continue
            taken.append((x0, y0, x1, y1))
            self._label_boxes.append((x0 - 3, y0 - 1, x1 + 3, y1 + 1, int(i)))

            focused = self.focus == i
            col = self.region_rgb[i] if focused else theme.DIM
            alpha = 1.0 if (self.focus is None or focused) else 0.25
            dl.rect(x0 - 4, y0 - 2, x1 + 4, y1 + 2, theme.BG,
                    0.82 * alpha, 3.0)
            dl.text(atlas, name, x0, y0, col, alpha)

    def label_hits(self, cursor):
        """Номер области, на подпись которой показывает курсор, или None."""
        cx, cy = cursor
        for x0, y0, x1, y1, i in self._label_boxes:
            if x0 <= cx < x1 and y0 <= cy < y1:
                return i
        return None

    def region_at(self, cursor, radius=18.0):
        """Нейропиль под курсором: по ближайшему нейрону, а не по подписи.

        Наведение на область — это вопрос «в какой нейропиль я тычу», и
        честный ответ даёт ближайшая точка облака, спроецированная на
        экран. Подпись отвечает только за свою рамку.
        """
        i = self.label_hits(cursor)
        if i is not None:
            return i
        j = self.pick(cursor, radius=radius, active_only=False)
        return None if j is None else int(self.region_code[j])
