"""Пул интерактивных объектов в сцене: создаётся заранее, живёт всегда.

MuJoCo не умеет добавлять тела в уже собранную модель. Значит объекты,
которыми можно двигать во время сеанса, обязаны быть созданы **до
компиляции**, а «появление» и «исчезновение» — это перемещение в
сторону и обратно. Тот же приём уже работает в проекте для источников
запаха: `OdorArena` создаёт mocap-тела `odor_source_marker_{i}`, а
`live.BodyLive` двигает их на ходу.

Что даёт пул:

* **движущиеся зрительные объекты** — mocap-положение меняется на
  границе шага управления;
* **нависание** — объект приближается по курсу, и угловой размер растёт
  сам, по геометрии, а не умножением на коэффициент. Предсказание
  2·arctan(r/d) проверяется замером по сетчатке;
* **свет и контраст** — `geom_rgba` модели записывается на месте, как
  уже делает работник с материалом пола;
* **траектории** — последовательность положений;
* **препятствия** — mocap-тело кинематическое и участвует в контактах.

Чего пул НЕ делает: не меняет физику, когда объекты убраны. Траектория
тела совпадает **побитово** с прогоном вовсе без пула, и это проверено
(`probe/objects.py`) — иначе пул испортил бы все прежние эпизоды, а их
28. Сетчатка побитового равенства дать не может: шесть лишних
геометрий меняют порядок растеризации, и последние разряды расходятся
при том же `qpos`. Остаток замерен и равен 6,8e-05, то есть **0,017
разрядности** 8-битного кадра (один уровень яркости — 1/255 ≈ 0,0039).
Для сравнения, включённый объект даёт затемнение 0,939.

**Объекты убраны по умолчанию.** Сеанс без них обязан считаться ровно
как раньше, и «убран» значит именно убран: тело уехало за пределы арены
и его геометрия прозрачна.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POOL = 6                     # сколько объектов создаётся заранее
NAME = "vis_object_%d"       # имя mocap-тела слота

# Куда уезжает убранный объект. Число выбрано не «подальше, чтобы не
# мешало», а по замеру, и первая попытка была ошибкой: парковка на
# −10 000 мм раздувала `model.stat.extent` с 60 до 10 031, а эта
# величина задаёт в MuJoCo ближнюю и дальнюю плоскости отсечения
# камеры. Физика оставалась побитово той же (`qpos` совпадал), а
# СЕТЧАТКА менялась во всей сцене: 409 омматидиев из 1442 с разницей
# больше 0,01 и максимумом 0,826 — причём одинаково при любом
# положении, размере и цвете объекта, потому что дело было вовсе не в
# видимости.
#
# Поэтому убранный объект стоит в 5 мм ПОД полом: это внутри прежних
# границ сцены, `stat.extent` не меняется, объект прозрачен
# (alpha = 0) и не участвует в контактах (`contype = 0`). Проверено:
# `stat.extent` остаётся 60,000, траектория совпадает побитово, а
# расхождение сетчатки падает до 6,8e-05 — ниже разрядности кадра
# (`probe/objects.py`).
PARK = (0.0, 0.0, -5.0)   # мм, как и остальные координаты арены

# Размер по умолчанию: шар радиусом 1 мм. Взят не произвольно — таким
# пользуется опубликованный зрительный таксис NeuroMechFly (чёрный шар
# радиусом 1 мм, скорость 10 мм/с), и это даёт сравнимость с его
# результатами (`RESEARCH.md` §7).
RADIUS = 1.0
RGBA = (0.1, 0.1, 0.1, 1.0)   # почти чёрный: объект должен быть виден


@dataclass
class Slot:
    """Один объект пула и всё, что про него известно.

    ``pos`` — положение ЗАКАЗАННОЕ, а не то, что сейчас в модели. У
    выключенного слота в модели стоит :data:`PARK`, но показывать и
    отдавать наружу надо заказанное: иначе включение выглядело бы как
    появление объекта из-под пола, а снимок состояния не прошёл бы
    собственную проверку (``z`` парковки отрицателен). Точно так же
    устроена сила выключенного источника запаха в `live.BodyLive`.
    """

    i: int
    on: bool = False
    pos: tuple = (0.0, 0.0, 1.5)
    radius: float = RADIUS
    rgba: tuple = RGBA
    # Что с этим слотом делали. Нужно журналу сеанса: по этому списку
    # событие можно привязать к модельному времени.
    история: list = field(default_factory=list)

    def name(self) -> str:
        return NAME % self.i


def inject(arena, n=POOL, radius=RADIUS, rgba=RGBA):
    """Вписать пул объектов в MJCF арены ДО компиляции.

    Зовётся на собранной `OdorArena` (или любой другой, у которой есть
    `root_element`), но обязательно прежде чем модель уйдёт в
    `Simulation`: после компиляции тела не добавляются.

    Возвращает список слотов. Геометрия создаётся сразу прозрачной и
    уехавшей в `PARK`: сцена без включённых объектов обязана быть
    неотличима от сцены без пула.
    """
    slots = []
    for i in range(int(n)):
        body = arena.root_element.worldbody.add(
            "body", name=NAME % i, pos=PARK, mocap=True)
        body.add("geom", type="sphere", name=NAME % i + "_geom",
                 size=(float(radius),), rgba=(0.0, 0.0, 0.0, 0.0),
                 contype=0, conaffinity=0)
        slots.append(Slot(i=i, radius=float(radius), rgba=tuple(rgba)))
    return slots


def discover(body, n=POOL):
    """Найти пул в уже собранной модели по именам тел.

    Нужно, чтобы живые условия (`live.BodyLive`) могли управлять пулом,
    не меняя подписи своей привязки и не завязываясь на то, кто именно
    собирал арену. Тот же приём там уже применён к mocap-строкам
    маркеров запаха: номера ищутся по имени, а не предполагаются.

    Возвращает привязанный :class:`Pool`, если в модели есть хотя бы
    один слот, и ``None``, если пул не вписывали вовсе — тогда эпизод
    считается ровно как раньше.
    """
    import mujoco

    m = body.physics.model.ptr
    slots = []
    for i in range(int(n)):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, NAME % i) < 0:
            break
        slots.append(Slot(i=i))
    if not slots:
        return None
    return Pool(slots).attach(body)


class Pool:
    """Живое управление пулом: привязка к модели и правки на ходу.

    Контракт ровно такой же, как у слотов источника запаха в
    `live.BodyLive`: привязка один раз, когда тело собрано, дальше
    правки на месте, и каждая правка применяется на границе шага
    управления, а не посреди него.
    """

    def __init__(self, slots):
        self.slots = list(slots)
        self.model = None
        self.data = None
        self._mocap = {}      # i -> номер mocap-тела
        self._geom = {}       # i -> номер геометрии

    def attach(self, body):
        """Связать с настоящей моделью MuJoCo.

        Номера тел и геометрий ищутся по ИМЕНИ, а не предполагаются:
        порядок тел в модели — не обещание MuJoCo, и в этом проекте уже
        один раз расходились столбцы сенсоров, найденные числом.
        """
        import mujoco

        self.model = body.physics.model.ptr
        self.data = body.physics.data.ptr
        m = self.model
        for s in self.slots:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s.name())
            if bid < 0:
                raise ValueError("в модели нет тела %r — пул не вписан "
                                 "до компиляции" % s.name())
            mid = int(m.body_mocapid[bid])
            if mid < 0:
                raise ValueError("тело %r не mocap — двигать его на ходу "
                                 "нельзя" % s.name())
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                    s.name() + "_geom")
            if gid < 0:
                raise ValueError("в модели нет геометрии %r"
                                 % (s.name() + "_geom"))
            self._mocap[s.i] = mid
            self._geom[s.i] = gid
        self._push()
        return self

    def _push(self):
        """Записать состояние всех слотов в модель и данные."""
        if self.model is None:
            return
        for s in self.slots:
            mid, gid = self._mocap[s.i], self._geom[s.i]
            # Единица длины модели — МИЛЛИМЕТР, как и во всей арене:
            # маркер источника, заданный как (25; 0; 1,5), лежит в
            # `mocap_pos` ровно как (25; 0; 1,5). Первая редакция делила
            # на 1000, приняв метры, и объекты оказывались внутри головы
            # мухи — отсюда провалились сразу три проверки пробы.
            pos = np.asarray(s.pos if s.on else PARK, dtype=float)
            self.data.mocap_pos[mid] = pos
            rgba = s.rgba if s.on else (0.0, 0.0, 0.0, 0.0)
            self.model.geom_rgba[gid] = np.asarray(rgba, dtype=float)
            self.model.geom_size[gid][0] = float(s.radius)

    def set(self, i, on=None, pos=None, radius=None, rgba=None,
            t_ms=None, cmd=None):
        """Правка одного слота. Возвращает, что именно стало.

        Запись в `история` несёт модельное время и номер команды: по ним
        событие попадает в журнал сеанса и в таймлайн. Без них правка
        была бы видна в сцене и не видна в записи.
        """
        s = next(x for x in self.slots if x.i == int(i))
        было = {"включён": s.on, "положение": s.pos,
                "радиус": s.radius, "цвет": s.rgba}
        if on is not None:
            s.on = bool(on)
        if pos is not None:
            s.pos = tuple(float(v) for v in pos)
        if radius is not None:
            s.radius = float(radius)
        if rgba is not None:
            s.rgba = tuple(float(v) for v in rgba)
        стало = {"включён": s.on, "положение": s.pos,
                 "радиус": s.radius, "цвет": s.rgba}
        s.история.append({"модельное, мс": t_ms, "команда": cmd,
                          "было": было, "стало": стало})
        self._push()
        return стало

    def angular_size(self, i, eye_pos):
        """Угловой размер объекта с точки зрения глаза, в градусах.

        Это предсказание геометрии, а не замер по сетчатке: ``2·arctan(r/d)``.
        Сравнение предсказания с замером и есть проверка того, что
        нависание в сцене настоящее.
        """
        s = next(x for x in self.slots if x.i == int(i))
        if not s.on:
            return 0.0
        d = float(np.linalg.norm(np.asarray(s.pos, dtype=float)
                                 - np.asarray(eye_pos, dtype=float)))
        if d <= s.radius:
            return 180.0
        return float(np.degrees(2.0 * np.arctan(s.radius / d)))

    def state(self):
        """Состояние пула для записи сеанса."""
        return [{"слот": s.i, "включён": s.on,
                 "положение, мм": [round(v, 3) for v in s.pos],
                 "радиус, мм": s.radius,
                 "цвет": [round(v, 3) for v in s.rgba],
                 "правок": len(s.история)} for s in self.slots]


# --------------------------------------------------- заданные траектории

# Пределы отрезка траектории. Те же, что у живой ручки `objects` в
# `live.BODY_KNOBS` (`_body_object`), и это не совпадение: заданная
# траектория и правка из окна обязаны принимать одно и то же, иначе
# «подвинуть мышью» и «записать в спеку» значили бы разное.
XY_LIM = 150.0          # мм, половина пола OdorArena 300×300
Z_LIM = (0.0, 60.0)     # мм: ниже пола не видно, выше поля зрения нет
R_LIM = (0.05, 20.0)    # мм


def _leg(tag, d):
    """Один отрезок траектории: проверка целиком до всякой записи."""
    known = {"slot", "t0_ms", "t1_ms", "from", "to", "radius", "grey", "on"}
    extra = set(d) - known
    if extra:
        raise ValueError("%s: лишние поля %s; есть %s"
                         % (tag, ", ".join(sorted(extra)),
                            ", ".join(sorted(known))))

    def num(name, v, lo, hi):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("%s.%s: нужно число, пришло %r"
                             % (tag, name, v)) from None
        if not (lo <= v <= hi):
            raise ValueError("%s.%s: %g вне предела %g…%g"
                             % (tag, name, v, lo, hi))
        return v

    def point(name, v):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError("%s.%s: нужны три числа (x, y, z), пришло %r"
                             % (tag, name, v))
        return (num(name + "[0]", v[0], -XY_LIM, XY_LIM),
                num(name + "[1]", v[1], -XY_LIM, XY_LIM),
                num(name + "[2]", v[2], *Z_LIM))

    on = d.get("on", True)
    if not isinstance(on, bool):
        raise ValueError("%s.on: нужно да или нет, пришло %r" % (tag, on))
    if "from" not in d:
        raise ValueError("%s: нет поля from — откуда объект идёт" % tag)
    a = point("from", d["from"])
    # Неподвижный объект — это отрезок, у которого `to` не задано: так
    # «стоит» и «едет» описываются одним и тем же полем, а не двумя
    # разными ветками, которые могут разойтись.
    b = point("to", d["to"]) if d.get("to") is not None else a
    t0 = num("t0_ms", d.get("t0_ms", 0.0), 0.0, 1e9)
    t1 = num("t1_ms", d.get("t1_ms", t0), 0.0, 1e9)
    if t1 < t0:
        raise ValueError("%s: t1_ms %g раньше t0_ms %g" % (tag, t1, t0))
    return {"slot": int(num("slot", d.get("slot", 0), 0, POOL - 1)),
            "t0_ms": t0, "t1_ms": t1, "from": a, "to": b,
            "radius": num("radius", d.get("radius", RADIUS), *R_LIM),
            "grey": num("grey", d.get("grey", RGBA[0]), 0.0, 1.0),
            "on": on}


class Track:
    """Заданная траектория объектов: воспроизводимая и сохранённая.

    Зачем она нужна, когда есть живая ручка `objects`. Живая ручка — это
    **человек мышью**, и повторить такой сеанс нельзя: правки легли в
    журнал с модельным временем, но заново их никто не наберёт. Трек,
    наоборот, целиком лежит в спеке прогона: тот же файл даёт то же
    движение объекта до миллисекунды модельного времени.

    Правило одно и объявлено: между ``t0_ms`` и ``t1_ms`` положение
    линейно по МОДЕЛЬНОМУ времени, до ``t0_ms`` объект убран, после
    ``t1_ms`` стоит в конечной точке. Применяется на границе шага
    управления — там же, где живые правки, и по той же причине: внутри
    шага состояние модели не целостно.

    Нависание отсюда получается само, геометрией: отрезок «издалека по
    курсу к мухе» даёт угловой размер 2·arctan(r/d), и предсказание
    сходится с замером по сетчатке с корреляцией 0,9990
    (`probe/objects.py`). Тёмное и светлое нависание — это один и тот же
    отрезок с ``grey`` 0,1 и 0,9; ничего кроме цвета в них не
    различается, и поэтому их можно сравнивать.
    """

    def __init__(self, legs):
        if not isinstance(legs, (list, tuple)):
            raise ValueError("траектория: нужен список отрезков, пришло %r"
                             % type(legs).__name__)
        self.legs = [_leg("траектория[%d]" % i,
                          d if isinstance(d, dict) else {})
                     for i, d in enumerate(legs)]
        if not self.legs:
            raise ValueError("траектория: ни одного отрезка")
        self.slots_used = sorted({w["slot"] for w in self.legs})
        self._last = {}

    def need_slots(self):
        """Сколько слотов пула обязано быть в модели."""
        return max(self.slots_used) + 1

    def at(self, t_ms):
        """Чего трек хочет от каждого слота на это модельное время."""
        want = {}
        for w in self.legs:
            if t_ms < w["t0_ms"]:
                continue
            span = w["t1_ms"] - w["t0_ms"]
            u = 1.0 if span <= 0 else min(
                1.0, (float(t_ms) - w["t0_ms"]) / span)
            a, b = np.asarray(w["from"]), np.asarray(w["to"])
            pos = tuple(float(v) for v in (a + (b - a) * u))
            g = w["grey"]
            # Позже заданный отрезок перекрывает прежний: так один слот
            # можно провести по ломаной, перечислив звенья по порядку.
            want[w["slot"]] = {"on": w["on"], "pos": pos,
                               "radius": w["radius"],
                               "rgba": (g, g, g, 1.0)}
        return want

    def apply(self, pool, t_ms, cmd="трек"):
        """Поставить слоты по треку. Возвращает номера тронутых слотов.

        Правка в модель идёт ТОЛЬКО когда заказанное изменилось: иначе
        журнал слота распухал бы одинаковыми записями по одной на шаг
        управления, а `Pool._push` переписывал бы всю модель впустую.
        """
        touched = []
        for i, w in self.at(t_ms).items():
            if i >= len(pool.slots):
                raise ValueError(
                    "трек просит слот %d, а в пуле их %d: число слотов "
                    "задаётся при сборке модели (`objects.inject`), на "
                    "ходу тела в MuJoCo не добавляются"
                    % (i, len(pool.slots)))
            key = (w["on"], tuple(round(v, 6) for v in w["pos"]),
                   round(w["radius"], 6), tuple(round(v, 6)
                                                for v in w["rgba"]))
            if self._last.get(i) == key:
                continue
            self._last[i] = key
            pool.set(i, on=w["on"], pos=w["pos"], radius=w["radius"],
                     rgba=w["rgba"], t_ms=t_ms, cmd=cmd)
            touched.append(i)
        return touched

    def state(self):
        """Трек для записи прогона: ровно то, что было задано."""
        return [{"слот": w["slot"],
                 "с, мс": w["t0_ms"], "по, мс": w["t1_ms"],
                 "из, мм": [round(v, 3) for v in w["from"]],
                 "в, мм": [round(v, 3) for v in w["to"]],
                 "радиус, мм": w["radius"], "серость": w["grey"],
                 "включён": w["on"],
                 "неподвижен": w["from"] == w["to"]} for w in self.legs]


# Готовые треки. Это не украшение: нависание, движение и неподвижность
# должны задаваться ОДНИМ способом, иначе каждый опыт пересобирает
# геометрию заново и сравнивать их нельзя.
#
# Числа взяты из опубликованного зрительного таксиса NeuroMechFly (шар
# радиусом 1 мм) и из замеров нависания в `probe/objects.py`: муха
# стоит около (0; 0), объект идёт по курсу +x с 30 мм до 3 мм за
# секунду модельного времени — это 27 мм/с, и угловой размер за это
# время растёт с 3,8° до 38,9°.
PRESETS = {
    "loom_dark": [
        {"slot": 0, "t0_ms": 0.0, "t1_ms": 1000.0,
         "from": (30.0, 0.0, 1.5), "to": (3.0, 0.0, 1.5),
         "radius": 1.0, "grey": 0.1},
    ],
    "loom_light": [
        {"slot": 0, "t0_ms": 0.0, "t1_ms": 1000.0,
         "from": (30.0, 0.0, 1.5), "to": (3.0, 0.0, 1.5),
         "radius": 1.0, "grey": 0.9},
    ],
    "sweep": [
        # Движущийся объект, который НЕ нависает: идёт поперёк курса на
        # постоянном удалении, то есть угловой размер держится, а место
        # в поле зрения едет. Нужен затем, чтобы отличить ответ на
        # движение от ответа на расширение.
        {"slot": 0, "t0_ms": 0.0, "t1_ms": 1000.0,
         "from": (10.0, -10.0, 1.5), "to": (10.0, 10.0, 1.5),
         "radius": 1.0, "grey": 0.1},
    ],
    "still": [
        # Неподвижный объект: встречный замер к обоим движущимся. Стоит
        # там, где нависающий оказывается на середине пути.
        {"slot": 0, "t0_ms": 0.0, "t1_ms": 0.0,
         "from": (16.5, 0.0, 1.5), "to": None,
         "radius": 1.0, "grey": 0.1},
    ],
}


def preset(name):
    """Готовый трек по имени. Неизвестное имя — отказ со списком."""
    name = str(name)
    if name not in PRESETS:
        raise ValueError("неизвестный трек %r; есть: %s"
                         % (name, ", ".join(sorted(PRESETS))))
    return [dict(d) for d in PRESETS[name]]
