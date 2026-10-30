"""Живые условия: что разрешено менять в идущем прогоне.

Песочнице нужно, чтобы правка из окна доходила до работающей модели, а
не до её копии. Поэтому здесь нет второго состояния: :class:`ArenaLive`
держит ссылки на тот самый объект арены и тот самый ``BodyParams``,
которые читает :func:`flybrain.loop.run_episode` на каждом шаге, и
меняет их на месте, на границе шага управления (см.
``jobs.Ctx.control_point``).

Три правила, на которых это держится.

**Разрешённое перечислено, а не выведено.** :data:`KNOBS` — единственный
список того, что можно менять, с типом, пределами и единицей. Чего в нём
нет, то отклоняется с причиной. Иначе окно (или чужой код) смогло бы
записать в модель что угодно, в том числе то, от чего зависит уже
снятая калибровка.

**Либо всё, либо ничего.** Сначала проверяется весь словарь, потом
пишется. Наполовину применённая правка — это состояние, которого нет ни
в одном описании опыта, и воспроизвести такой прогон нельзя.

**Правка — часть результата.** Каждая возвращает применённые значения,
работник подтверждает её модельным временем, а сохранённый прогон
несёт список правок. Эпизод, у которого посреди счёта поменяли усиление
рецепторов, не должен выглядеть как чистый эпизод.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Сколько источников и кормушек разрешено. Предел не из вкуса: поле
# запаха и его градиент считаются перебором всех источников на каждом
# шаге (`world.Arena.concentration`), и сотня источников тихо съела бы
# шаг управления.
MAX_SOURCES = 8


@dataclass(frozen=True)
class Knob:
    """Одно разрешённое к изменению свойство.

    ``where`` — куда писать: ``"params"`` (придуманная половина, объект
    ``world.BodyParams``) или ``"arena"`` (среда). ``marked`` помечает
    те, что меняют научный смысл прогона, а не только условия: такую
    правку отчёт обязан показать отдельно.
    """

    name: str
    where: str
    kind: str              # "float" | "int" | "bool"
    lo: float | None
    hi: float | None
    unit: str
    what: str
    marked: bool = False


KNOBS = {k.name: k for k in (
    # --- запах: вход обонятельных рецепторов ---
    Knob("orn_max_hz", "params", "float", 0.0, 400.0, "Гц",
         "частота рецепторов при насыщении"),
    Knob("orn_gain", "params", "float", 0.0, 20.0, "",
         "во сколько концентрация переводится в долю насыщения"),
    Knob("grn_hz", "params", "float", 0.0, 400.0, "Гц",
         "частота вкусовых рецепторов на кормушке"),
    # --- приближение: вход LPLC2 ---
    Knob("loom_max_hz", "params", "float", 0.0, 400.0, "Гц",
         "частота LPLC2 при стене в упор"),
    Knob("loom_range_mm", "params", "float", 0.1, 60.0, "мм",
         "с какого расстояния стена начинает надвигаться"),
    # --- рулёжка: как выход мозга становится движением ---
    Knob("speed_base_mm_s", "params", "float", 0.0, 60.0, "мм/с",
         "скорость при нулевом нисходящем выходе"),
    Knob("speed_max_mm_s", "params", "float", 0.0, 120.0, "мм/с",
         "предел скорости"),
    Knob("turn_gain_rad_s", "params", "float", 0.0, 200.0, "рад/с",
         "усиление перекоса в поворот"),
    Knob("turn_max_rad_s", "params", "float", 0.0, 30.0, "рад/с",
         "предел угловой скорости"),
    Knob("turn_sign", "params", "float", -1.0, 1.0, "",
         "знак связи полушарий с поворотом"),
    Knob("dn_tau_s", "params", "float", 0.0, 5.0, "с",
         "сглаживание нисходящего выхода"),
    Knob("smooth", "params", "float", 0.0, 1.0, "",
         "сглаживание скорости и поворота на шаг"),
    Knob("feed_pause_s", "params", "float", 0.0, 10.0, "с",
         "сколько муха стоит на кормушке"),
    # --- вмешательства, меняющие смысл прогона ---
    Knob("swap_antennae", "params", "bool", None, None, "",
         "поменять антенны местами: контроль на двусторонний путь",
         marked=True),
    Knob("asym_bias", "params", "float", -1.0, 1.0, "",
         "вычитаемый собственный перекос популяции", marked=True),
    Knob("dn_ref_hz", "params", "float", 0.01, 1000.0, "Гц",
         "какая частота считается полным выходом", marked=True),
    # --- среда ---
    Knob("odours", "arena", "sources", None, None, "",
         "источники запаха: x, y, сила, сигма"),
    Knob("foods", "arena", "foods", None, None, "",
         "кормушки: x, y, радиус"),
)}


def _num(name, value, lo, hi, unit):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: нужно число, пришло %r" % (name, value))
    if not math.isfinite(v):
        raise ValueError("%s: число должно быть конечным, пришло %r"
                         % (name, value))
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError("%s: допустимо от %g до %g%s, пришло %g"
                         % (name, lo, hi, (" " + unit) if unit else "", v))
    return v


def _bool(name, value):
    if isinstance(value, bool):
        return value
    if value in (0, 1, "0", "1", "да", "нет", "true", "false"):
        return value in (1, "1", "да", "true")
    raise ValueError("%s: нужно да или нет, пришло %r" % (name, value))


def _sources(name, value, maker):
    """Разобрать список источников или кормушек целиком.

    Список заменяется целиком, а не правится по индексу: иначе окно и
    работник разошлись бы в нумерации после добавления или удаления, и
    ползунок стал бы двигать не тот источник.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s: нужен список, пришло %r" % (name, type(value).__name__))
    if len(value) > MAX_SOURCES:
        raise ValueError("%s: не больше %d, пришло %d"
                         % (name, MAX_SOURCES, len(value)))
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("%s[%d]: нужен словарь, пришло %r"
                             % (name, i, type(item).__name__))
        out.append(maker("%s[%d]" % (name, i), item))
    return out


def _odour(tag, d):
    from . import world

    extra = set(d) - {"x", "y", "strength", "sigma"}
    if extra:
        raise ValueError("%s: лишние поля %s" % (tag, ", ".join(sorted(extra))))
    return world.Odour(
        x=_num(tag + ".x", d.get("x", 0.0), -200.0, 200.0, "мм"),
        y=_num(tag + ".y", d.get("y", 0.0), -200.0, 200.0, "мм"),
        strength=_num(tag + ".strength", d.get("strength", 1.0),
                      0.0, 100.0, ""),
        # Сигма нулём обращает поле запаха в деление на ноль, а не в
        # точечный источник, поэтому снизу стоит не ноль.
        sigma=_num(tag + ".sigma", d.get("sigma", 12.0), 0.1, 200.0, "мм"),
    )


def _food(tag, d):
    from . import world

    extra = set(d) - {"x", "y", "radius"}
    if extra:
        raise ValueError("%s: лишние поля %s" % (tag, ", ".join(sorted(extra))))
    return world.Food(
        x=_num(tag + ".x", d.get("x", 0.0), -200.0, 200.0, "мм"),
        y=_num(tag + ".y", d.get("y", 0.0), -200.0, 200.0, "мм"),
        radius=_num(tag + ".radius", d.get("radius", 2.5), 0.0, 60.0, "мм"),
    )


class ArenaLive:
    """Живые условия эпизода на арене.

    Держит ссылки, а не копии: ``params`` — тот же объект, который
    ``world.Body`` читает как ``self.p`` на каждом шаге, ``arena`` — та
    же среда, у которой ``loop.run_episode`` спрашивает концентрацию.
    Поэтому записанное здесь видно модели со следующего шага и не
    требует ни перезапуска, ни пересборки движка.

    Чего здесь сознательно нет: всего, что движок читает один раз при
    сборке. Набор сенсорных нейронов, ``no_refractory``, семя и
    устройство менять на ходу нельзя — это не «условие», а другой
    прогон, и подменять их молча значит выдать один опыт за другой.
    """

    def __init__(self, arena, params, warm=None):
        self.arena = arena
        self.params = params
        # Что откалибровано на разогреве. Правка такого поля стирает
        # калибровку, и прогон перестаёт быть сравнимым с эталонным —
        # отсюда `marked` у `dn_ref_hz` и `asym_bias`.
        self.warm = dict(warm or {})
        self.log = []

    # -- что можно ------------------------------------------------------

    def schema(self):
        """Описание разрешённого — для окна и для отчёта."""
        return [{"имя": k.name, "тип": k.kind, "от": k.lo, "до": k.hi,
                 "единица": k.unit, "что": k.what, "помечено": k.marked,
                 "сейчас": self._read(k)}
                for k in KNOBS.values()]

    def snapshot(self):
        return {k.name: self._read(k) for k in KNOBS.values()}

    def _read(self, k):
        if k.where == "params":
            return getattr(self.params, k.name, None)
        if k.name == "odours":
            return [{"x": o.x, "y": o.y, "strength": o.strength,
                     "sigma": o.sigma} for o in self.arena.odours]
        if k.name == "foods":
            return [{"x": f.x, "y": f.y, "radius": f.radius}
                    for f in self.arena.foods]
        return None

    # -- применение -----------------------------------------------------

    def apply(self, payload, initial=False, cmd_id=None, t_ms=None):
        """Проверить всё, потом записать. Возвращает применённое.

        Порядок именно такой: при отказе модель остаётся в том
        состоянии, в каком была, и подтверждение несёт причину. Иначе
        половина правки уже уехала бы в идущий расчёт.

        ``cmd_id`` и ``t_ms`` идут в журнал прогона: без них по
        сохранённому файлу не сказать, ни какая это была команда, ни на
        какой миллисекунде она сработала.
        """
        _ = initial
        if not isinstance(payload, dict):
            raise TypeError("правка должна быть словарём, пришло %r"
                            % type(payload).__name__)
        if not payload:
            raise ValueError("пустая правка: нечего менять")
        unknown = [k for k in payload if k not in KNOBS]
        if unknown:
            raise KeyError("менять нельзя: %s; можно: %s"
                           % (", ".join(sorted(unknown)),
                              ", ".join(sorted(KNOBS))))

        ready = {}
        for name, value in payload.items():
            k = KNOBS[name]
            if k.kind == "float":
                ready[name] = _num(name, value, k.lo, k.hi, k.unit)
            elif k.kind == "int":
                ready[name] = int(_num(name, value, k.lo, k.hi, k.unit))
            elif k.kind == "bool":
                ready[name] = _bool(name, value)
            elif k.kind == "sources":
                ready[name] = _sources(name, value, _odour)
            elif k.kind == "foods":
                ready[name] = _sources(name, value, _food)
            else:                                     # pragma: no cover
                raise ValueError("%s: неизвестный тип %r" % (name, k.kind))

        was, now = {}, {}
        for name, value in ready.items():
            k = KNOBS[name]
            was[name] = self._read(k)
            if k.where == "params":
                setattr(self.params, name, value)
            elif name == "odours":
                self.arena.odours = value
            elif name == "foods":
                self.arena.foods = value
            now[name] = self._read(k)

        marked = sorted(n for n in ready if KNOBS[n].marked)
        rec = {"модельное, мс": (None if t_ms is None
                                 else round(float(t_ms), 3)),
               "команда": cmd_id, "было": was, "стало": now}
        if marked:
            rec["помечено"] = marked
            rec["оговорка"] = (
                "изменено на ходу: %s — прогон перестал быть сравнимым с "
                "эталонным" % ", ".join(marked))
        self.log.append(rec)
        return now


# ------------------------------------------------- живые условия в теле

# Предел прямой стимуляции. Не из вкуса: `sim.Engine._poisson` считает
# `torch.bernoulli(rates * DT/1000)`, и при DT = 0,1 мс вероятность
# доходит до единицы на 10 000 Гц. Выше — не стимул, а постоянная
# единица на каждом тике, то есть не пуассоновский вход вовсе. Тысяча
# герц оставляет до этого предела десятикратный запас.
MAX_STIM_HZ = 1000.0

# Сколько прямых стимулов держать одновременно. Каждый — это выборка по
# DSL и запись в тензор на каждом шаге; предел нужен, чтобы шаг
# управления не съела сама подача.
MAX_STIMS = 6


BODY_KNOBS = {k.name: k for k in (
    # --- источники запаха: настоящие mocap-тела модели MuJoCo ---
    Knob("sources", "arena", "body_sources", None, None, "",
         "слоты источника запаха: x, y, сила, включён"),
    # --- зрительные объекты: тот же приём, что и у источников ---
    # Слоты созданы до компиляции модели (`objects.inject`), потому что
    # MuJoCo не добавляет тела в собранную модель. «Появление» объекта —
    # это включение свободного слота, а не создание тела.
    Knob("objects", "arena", "body_objects", None, None, "",
         "слоты зрительного объекта: x, y, z, радиус, серость, включён"),
    # --- зрение: подача, усиление, свет и контраст ---
    #
    # Чего здесь НЕТ и быть не может: самого `vision` и частоты
    # обновления. Оба решаются при сборке тела — глаза либо есть в
    # модели MuJoCo, либо нет, а частота зашита в
    # `Fly._eff_visual_render_interval` при создании. Менять их на ходу
    # значило бы пересобрать модель, то есть начать другой сеанс.
    #
    # `vision_feed` помечено: сеанс, у которого посреди счёта выключили
    # подачу зрения, не сравним с эталонным, и отчёт обязан это
    # показать.
    Knob("vision_feed", "params", "bool", None, None, "",
         "подавать ли кадр глаз в сеть (глаза продолжают рисовать "
         "кадр в любом случае)", marked=True),
    Knob("vision_max_hz", "params", "float", 0.0, MAX_STIM_HZ, "Гц",
         "частота фоторецепторов при насыщении"),
    Knob("vision_ref", "params", "float", 1e-6, 1.0, "",
         "яркость кадра, считающаяся насыщением", marked=True),
    Knob("vision_contrast", "params", "float", 0.0, 8.0, "×",
         "контраст кадра: clip(0,5 + (I − 0,5)·k). Объявленное "
         "преобразование кадра, а не свойство сцены"),
    # Свет — единственная зрительная ручка, которая пишет не в
    # настройку, а В МОДЕЛЬ: это настоящие источники MuJoCo, и правка
    # меняет сам рендер кадра глаза.
    Knob("vision_light", "arena", "body_light", 0.0, 4.0, "×",
         "множитель к свету арены: настоящие источники MuJoCo "
         "light_top и light_bottom"),
    # --- вход рецепторов ---
    Knob("orn_max_hz", "params", "float", 0.0, MAX_STIM_HZ, "Гц",
         "частота обонятельных рецепторов при насыщении"),
    Knob("grn_hz", "params", "float", 0.0, MAX_STIM_HZ, "Гц",
         "частота вкусовых рецепторов на источнике"),
    Knob("food_radius_mm", "params", "float", 0.0, 60.0, "мм",
         "с какого расстояния источник считается достигнутым"),
    Knob("odour_ref", "params", "float", 1e-9, 100.0, "",
         "яркость, которая считается насыщением рецепторов",
         marked=True),
    # --- как выход мозга становится шагом ---
    Knob("amp_base", "params", "float", 0.0, 2.0, "",
         "амплитуда шага при нулевой асимметрии"),
    Knob("amp_span", "params", "float", 0.0, 2.0, "",
         "на сколько её меняет одна сигма асимметрии"),
    Knob("amp_min", "params", "float", 0.0, 2.0, "",
         "нижний предел амплитуды"),
    Knob("amp_max", "params", "float", 0.0, 2.0, "",
         "верхний предел амплитуды"),
    Knob("dn_tau_s", "params", "float", 0.0, 5.0, "с",
         "сглаживание выхода популяции"),
    # --- вмешательства, меняющие смысл сеанса ---
    Knob("swap_antennae", "params", "bool", None, None, "",
         "поменять антенны местами: контроль на двусторонний путь",
         marked=True),
    Knob("cross", "params", "bool", None, None, "",
         "перекрёстная связь полушарий с ногами", marked=True),
    # --- прямая стимуляция по DSL ---
    Knob("stim", "params", "stims", None, None, "",
         "прямые нейронные стимулы: запрос DSL и частота"),
    # --- виртуальные вмешательства ---
    #
    # Помечено: вмешательство меняет не условия, а саму сеть. Прогон с
    # поражением не сравнивается с эталонным, и оговорка должна ехать
    # вместе с результатом.
    Knob("lesion", "params", "lesions", None, None, "",
         "вмешательства: запрос DSL, вид и длительность", marked=True),
)}


def _body_source(tag, d):
    extra = set(d) - {"x", "y", "strength", "on"}
    if extra:
        raise ValueError("%s: лишние поля %s" % (tag, ", ".join(sorted(extra))))
    on = d.get("on", True)
    if not isinstance(on, bool):
        raise ValueError("%s.on: нужно да или нет, пришло %r" % (tag, on))
    return {
        # Пределы — половина арены OdorArena (300×300 мм): за краем пола
        # источник висел бы в воздухе, и сцена показывала бы ерунду.
        "x": _num(tag + ".x", d.get("x", 0.0), -150.0, 150.0, "мм"),
        "y": _num(tag + ".y", d.get("y", 0.0), -150.0, 150.0, "мм"),
        "strength": _num(tag + ".strength", d.get("strength", 1.0),
                         0.0, 100.0, ""),
        "on": on,
    }


def _body_object(tag, d):
    """Один слот зрительного объекта: проверка до записи.

    Пределы те же, что у источника запаха: половина арены `OdorArena`
    (300×300 мм). Высота ограничена отдельно — объект ниже пола не
    виден, а выше 60 мм выходит из поля зрения мухи и смысла не имеет.
    """
    extra = set(d) - {"x", "y", "z", "radius", "grey", "on"}
    if extra:
        raise ValueError("%s: лишние поля %s" % (tag, ", ".join(sorted(extra))))
    on = d.get("on", False)
    if not isinstance(on, bool):
        raise ValueError("%s.on: нужно да или нет, пришло %r" % (tag, on))
    return {
        "x": _num(tag + ".x", d.get("x", 0.0), -150.0, 150.0, "мм"),
        "y": _num(tag + ".y", d.get("y", 0.0), -150.0, 150.0, "мм"),
        "z": _num(tag + ".z", d.get("z", 1.5), 0.0, 60.0, "мм"),
        "radius": _num(tag + ".radius", d.get("radius", 1.0),
                       0.05, 20.0, "мм"),
        # Серость, а не цвет: объект нужен как зрительный раздражитель, и
        # различать его по оттенку сетчатка всё равно не может — два
        # канала фоторецептора сводятся в один (`retina.intensity`).
        "grey": _num(tag + ".grey", d.get("grey", 0.1), 0.0, 1.0, ""),
        "on": on,
    }


def _stims(name, value):
    """Разобрать список прямых стимулов. Запросы DSL здесь не проверяются.

    Разбор запроса требует таблицы нейронов, а она есть только у
    привязанного состояния, поэтому запрос проверяется в
    :meth:`BodyLive._resolve_stims` — там же, где становится известно,
    сколько нейронов он нашёл.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s: нужен список, пришло %r"
                         % (name, type(value).__name__))
    if len(value) > MAX_STIMS:
        raise ValueError("%s: не больше %d стимулов, пришло %d"
                         % (name, MAX_STIMS, len(value)))
    out = []
    for i, item in enumerate(value):
        tag = "%s[%d]" % (name, i)
        if not isinstance(item, dict):
            raise ValueError("%s: нужен словарь, пришло %r"
                             % (tag, type(item).__name__))
        extra = set(item) - {"query", "rate_hz"}
        if extra:
            raise ValueError("%s: лишние поля %s"
                             % (tag, ", ".join(sorted(extra))))
        q = str(item.get("query", "")).strip()
        if not q:
            raise ValueError("%s.query: пустой запрос" % tag)
        out.append({"query": q,
                    "rate_hz": _num(tag + ".rate_hz",
                                    item.get("rate_hz", 0.0),
                                    0.0, MAX_STIM_HZ, "Гц")})
    return out


# ------------------------------------------------- виртуальные вмешательства

# Сколько вмешательств держать одновременно. Предел не от скупости: на
# каждое приходится выборка по DSL и объединение индексов при каждой
# правке, а главное — список должен помещаться в панель и в журнал
# целиком, иначе «какие вмешательства сейчас действуют» перестаёт быть
# ответом и становится поиском.
MAX_LESIONS = 8

# Три операции над нейроном и то, что они значат в цепочке вычислений.
# Четвёртая операция из задания — «восстановление нормальной работы» —
# это пустой список: отдельного вида для неё нет нарочно, иначе
# «снято» было бы состоянием, которое можно забыть снять.
LESION_KINDS = {
    "fire": "не выдают спайк (мембрана копит)",
    "hold": "мембрана удержана на vReset",
    "out": "исходящие отключены",
    "in": "входящие отключены",
}

# Какие виды задевают одну и ту же величину. Две операции над одной
# популяцией сами по себе совместимы — маски стоят в разных местах
# цепочки и перемножаются независимо, — но `fire` и `hold` спорят за
# одно: что делать с мембраной, пока нейрон молчит. Одновременно их
# ставить бессмысленно, и работник отказывает до применения, а не
# применяет наполовину.
LESION_CONFLICT = {("fire", "hold")}

# Предел длительности вмешательства в модельном времени. Час модельного
# времени при измеренной скорости около 0,06 — это больше полусуток по
# стене; всё, что дольше, практичнее ставить «до ручной отмены».
MAX_LESION_MS = 3_600_000.0


def _lesions(name, value):
    """Разобрать список вмешательств. Запросы DSL здесь не проверяются.

    Как и у стимулов, разбор запроса требует таблицы нейронов и потому
    делается в привязанном состоянии — но ТОЖЕ на этапе проверки, а не
    записи: отказ после того, как часть масок уже поставлена, оставил бы
    модель в состоянии, которого нет ни в одном описании опыта.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s: нужен список, пришло %r"
                         % (name, type(value).__name__))
    if len(value) > MAX_LESIONS:
        raise ValueError("%s: не больше %d вмешательств, пришло %d"
                         % (name, MAX_LESIONS, len(value)))
    out = []
    for i, item in enumerate(value):
        tag = "%s[%d]" % (name, i)
        if not isinstance(item, dict):
            raise ValueError("%s: нужен словарь, пришло %r"
                             % (tag, type(item).__name__))
        extra = set(item) - {"query", "kind", "ms"}
        if extra:
            raise ValueError("%s: лишние поля %s"
                             % (tag, ", ".join(sorted(extra))))
        q = str(item.get("query", "")).strip()
        if not q:
            raise ValueError("%s.query: пустой запрос" % tag)
        kind = str(item.get("kind", "")).strip()
        if kind not in LESION_KINDS:
            raise ValueError("%s.kind: %r не из списка (%s)"
                             % (tag, kind, ", ".join(sorted(LESION_KINDS))))
        ms = item.get("ms")
        if ms in (None, "", 0, 0.0):
            ms = None
        else:
            ms = _num(tag + ".ms", ms, 0.1, MAX_LESION_MS, "мс")
        out.append({"query": q, "kind": kind, "ms": ms})
    return out


class BodyLive:
    """Живые условия сеанса в теле: настоящая модель MuJoCo.

    Отличие от :class:`ArenaLive` принципиальное. Там средой была наша
    же плоская арена, и менять в ней можно было что угодно. Здесь среда —
    ``OdorArena`` из FlyGym внутри MuJoCo, и у неё два следствия.

    **Слоты источников создаются при сборке модели.** Маркер источника —
    это ``mocap``-тело MJCF (``odor_source_marker_<i>``), и добавить тело
    в собранную модель нельзя, не пересобрав её. Поэтому «добавить
    источник» — это включить свободный слот, а не создать объект. Сколько
    слотов, столько и задаётся при запуске сеанса.

    **Яркость считается по кэшам, а не по массиву положений.**
    ``OdorArena.get_olfaction`` читает ``_odor_source_repeated`` и
    ``_peak_intensity_repeated``, собранные в конструкторе. Записать
    новое положение в ``odor_source`` и успокоиться нельзя: кэш надо
    пересобрать, иначе муха будет чуять источник там, где его уже нет.

    Что меняется на каждое перемещение слота, и всё это — состояние
    модели, а не картинка:

    * ``arena.odor_source[i]`` и кэш ``_odor_source_repeated`` — то, что
      считают сенсоры антенн;
    * ``physics.data.mocap_pos[i]`` — то, где маркер стоит в сцене;
    * ``sources[i]`` — массив, по которому эпизод считает расстояние до
      источника, вкус и совпадение курса с направлением на него;
    * прозрачность маркера (``model.geom_rgba``) — выключенный слот не
      должен выглядеть работающим.

    Первые три обязаны совпадать, иначе окно показывает одно, а модель
    чует другое. Это и проверяет ``probe/sandbox_link.py``.
    """

    # Прозрачность маркера выключенного слота. Совсем прятать его нельзя:
    # слот остаётся в модели, и видеть, где он, пользователю надо.
    OFF_ALPHA = 0.18

    def __init__(self, initial=None):
        self.initial = dict(initial or {})
        self.body = None
        self.sources = None
        self.params = None
        self.neurons = None
        self.device = "cpu"
        self.log = []
        self.start = {}
        self._stims = []          # [{"query","rate_hz","n","idx"}]
        self._drive = None        # тензор (1, n) с прямыми стимулами
        self._idx_cache = {}
        self._kept = {}           # заказанная сила слота, как её просили
        # Включённость хранится отдельно от силы. Выводить её из
        # `strength > 0` нельзя: слот, включённый с нулевой силой, читался
        # бы как выключенный, тумблер в окне стоял бы «включён», а панель
        # показывала бы «выключен» — и ни одного признака расхождения.
        self._on = {}
        self._mocap = {}
        self._geom = {}
        # Пул зрительных объектов. Находится по именам тел при привязке
        # (`objects.discover`), а не передаётся снаружи: иначе подпись
        # привязки пришлось бы менять, а её зовёт `body.run_episode`.
        # `None` значит, что пул в арену не вписывали — тогда эпизод
        # считается ровно как раньше.
        self.pool = None
        self.engine = None
        self.lesions = []

    # -- привязка к собранной модели ------------------------------------

    def attach(self, body, sources, cfg, neurons, device, engine=None):
        self.body = body
        self.sources = sources
        self.params = cfg
        self.neurons = neurons
        self.device = device
        # Движок нужен вмешательствам: маски живут в нём, в той же
        # памяти, которую читает шаг и захваченный CUDA-граф.
        prev = list(self.lesions)
        self.engine = engine
        if prev:
            # У нового движка маски пусты, а список вмешательств остался
            # бы от прежнего: окно показывало бы действующими те, чего
            # модель не считает. Восстановить их нельзя — срок считался
            # от прежнего модельного времени, которое началось заново, —
            # поэтому они завершаются явно и с записью причины.
            self.lesions = []
            self.log.append({
                "что": "вмешательства сняты при перепривязке",
                "модельное, мс": 0.0, "команда": None,
                "снято": [{"id": w["id"], "запрос": w["query"],
                           "вид": w["kind"], "нейронов": w["n"]}
                          for w in prev],
                "почему": "движок пересобран, модельное время началось "
                          "заново: срок прежних вмешательств считать не "
                          "от чего, и маски нового движка пусты"})
        if engine is not None:
            # Маски приводятся в согласие со списком всегда, даже когда
            # список пуст: это и есть единое состояние работника и окна.
            self._push_masks()
        self._marker_rows()
        if self.initial:
            # Это исходные условия сеанса, а не правка на ходу: зовётся
            # до разогрева и в журнале стоит отдельной записью.
            self.start = dict(self.apply(self.initial, initial=True))
        self.start.setdefault("sources", self._read_sources())
        self.start.setdefault("odour_ref", getattr(cfg, "odour_ref", None))
        # Исходные условия двигают маркеры источника прямо в `mjData`, мимо
        # биндинга dm_control, то есть не помечая состояние грязным.
        # Быстрый контроллер походки (`flybrain.gait`) переиспользует
        # наблюдение прошлого шага, и после правки состояния его надо
        # забыть, а не полагаться на то, что правка «вроде бы не про ноги».
        inval = getattr(body, "invalidate_observation", None)
        if inval is not None:
            inval()

    def _marker_rows(self):
        """Строки `mocap_pos` и геометрия маркеров — по имени, не по счёту.

        Отказ здесь громкий и сразу. Если сборка FlyGym переименует
        маркеры, молчаливый пропуск дал бы правку наполовину: яркость и
        массив эпизода переехали бы, а маркер в сцене остался бы на
        прежнем месте — окно показывало бы одно, а муха чуяла другое.
        Проверяется и `mocapid`: у немокапного тела он −1, и запись в
        `mocap_pos[-1]` молча сдвинула бы чужое тело.
        """
        # Пул зрительных объектов ищется тем же способом и здесь же: по
        # именам тел в уже собранной модели. Нет пула — нет и ручки.
        from . import objects as _objects

        self.pool = _objects.discover(self.body)

        m = self.body.physics.model
        self._mocap, self._geom = {}, {}
        for i in range(int(m.nbody)):
            try:
                name = m.body(i).name
            except Exception:                         # noqa: BLE001
                name = ""
            if not name or "odor_source_marker" not in str(name):
                continue
            try:
                slot = int(str(name).rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            row = int(m.body_mocapid[i])
            if row < 0:
                raise RuntimeError(
                    "маркер источника %r не мокапное тело (mocapid %d): "
                    "положение источника задать нечем" % (str(name), row))
            self._mocap[slot] = row
            adr, num = int(m.body_geomadr[i]), int(m.body_geomnum[i])
            if num > 0:
                self._geom[slot] = adr
        want = int(self.body.arena.odor_source.shape[0])
        missing = [i for i in range(want) if i not in self._mocap]
        if missing:
            raise RuntimeError(
                "в модели %d источников, а маркеров нашлось %d: нет слотов "
                "%s. Их имена задаёт `OdorArena` (`odor_source_marker_<i>`); "
                "без них перенос источника не дойдёт до сцены"
                % (want, len(self._mocap), missing))

    # -- что модель считает прямо сейчас --------------------------------

    def slots(self):
        if self.body is None:
            return 0
        return int(self.body.arena.odor_source.shape[0])

    # -- зрительные объекты ---------------------------------------------

    def object_slots(self):
        return 0 if self.pool is None else len(self.pool.slots)

    def _read_objects(self):
        """Что в пуле сейчас. Пустой список, если пул не вписан."""
        if self.pool is None:
            return []
        out = []
        for s in self.pool.slots:
            out.append({"x": round(float(s.pos[0]), 4),
                        "y": round(float(s.pos[1]), 4),
                        "z": round(float(s.pos[2]), 4),
                        "radius": round(float(s.radius), 4),
                        "grey": round(float(s.rgba[0]), 4),
                        "on": bool(s.on)})
        return out

    def _write_objects(self, want, t_ms=None, cmd_id=None):
        """Слоты объектов — в модель, через тот же `Pool.set`.

        Модельное время и номер команды уходят в историю слота: правка,
        которой нет в журнале, не считается сделанной, и по этим полям
        событие попадает в таймлайн сеанса.
        """
        for i, d in enumerate(want):
            g = float(d["grey"])
            self.pool.set(i, on=d["on"],
                          pos=(d["x"], d["y"], d["z"]),
                          radius=d["radius"], rgba=(g, g, g, 1.0),
                          t_ms=t_ms, cmd=cmd_id)

    def _read_sources(self):
        if self.body is None:
            return []
        ar = self.body.arena
        out = []
        for i in range(self.slots()):
            # Сила показывается заказанная, а не та, что сейчас в модели:
            # у выключенного слота в модели ноль, и иначе включение
            # выглядело бы как появление силы из ниоткуда.
            out.append({"x": round(float(ar.odor_source[i, 0]), 4),
                        "y": round(float(ar.odor_source[i, 1]), 4),
                        "strength": round(float(self._kept.get(
                            i, float(ar.peak_odor_intensity[i, 0]))), 6),
                        "on": bool(self._on.get(i, True))})
        return out

    def active_sources(self):
        """Маска включённых слотов — для расстояния, вкуса и курса.

        Выключенный слот остаётся в модели, но едой не является:
        выбирать его как ближайший источник значит кормить муху тем,
        чего нет. Поэтому эпизод спрашивает маску, а не считает по всем.
        """
        if self.body is None:
            return None
        if not self._on:
            # Ничего ещё не задавали: ни исходных условий, ни правок.
            # `None` значит «маски нет», и эпизод считает по всем
            # источникам — то есть ровно так, как считался без живых
            # условий вообще. Возвращать список из одних `True` было
            # хуже: равенство с прежним расчётом держалось бы на
            # арифметическом совпадении `np.where`, а не на том, что
            # этот код не исполняется.
            return None
        # Включённый слот с нулевой силой — всё ещё включённый: его не
        # чует ни одна антенна, но едой он считается, и расстояние до
        # него осмысленно. Поэтому маска берётся из признака, а не из
        # силы.
        return [bool(self._on.get(i, True)) for i in range(self.slots())]

    def snapshot(self):
        out = {}
        for k in BODY_KNOBS.values():
            if k.name == "sources":
                out["sources"] = self._read_sources()
            elif k.name == "objects":
                out["objects"] = self._read_objects()
            elif k.name == "stim":
                out["stim"] = [{"query": s["query"], "rate_hz": s["rate_hz"],
                                "нейронов": s["n"]} for s in self._stims]
            elif k.name == "lesion":
                out["lesion"] = self.lesions_shown()
            else:
                out[k.name] = getattr(self.params, k.name, None)
        return out

    def schema(self):
        cur = self.snapshot()
        return [{"имя": k.name, "тип": k.kind, "от": k.lo, "до": k.hi,
                 "единица": k.unit, "что": k.what, "помечено": k.marked,
                 "сейчас": cur.get(k.name)}
                for k in BODY_KNOBS.values()]

    # -- прямые стимулы -------------------------------------------------

    def add_drive(self, rates):
        """Поднять частоты шага до заданных прямыми стимулами.

        Про это надо знать одно, и это не очевидно: сенсорные каналы
        собраны в `no_refractory` при сборке движка, а нейроны, выбранные
        стимулом на ходу, — нет. Значит у них остаётся рефрактерный
        период, и та же частота даст меньше спайков, чем на антеннах.
        Менять `no_refractory` на ходу нельзя: это пересборка движка.
        """
        import torch

        if self._drive is not None:
            # Не сложение, а максимум. Сложение нарушало объявленный
            # предел: стимул на 1000 Гц, наведённый на те же нейроны, что
            # сенсорный канал, давал бы 1000 + 150 Гц — выше того, что
            # названо пределом, и нигде это не было бы видно. Максимум
            # держит предел и читается однозначно: стимул задаёт нижнюю
            # границу частоты, а не добавку к ней.
            torch.maximum(rates, self._drive, out=rates)

    def _resolve_stims(self, want):
        """Разобрать запросы DSL. Только проверка, ничего не меняет.

        Делается это на этапе проверки, а не записи, и именно поэтому.
        Разбор запроса — единственное в правке, что может отказать уже
        после того, как слоты источника переписаны в модель: тогда
        подтверждение сказало бы «отклонено», а модель была бы изменена
        наполовину, и воспроизвести такой сеанс по журналу стало бы
        нельзя. Теперь отказать может только проверка.
        """
        from . import data as D

        ready = []
        for i, s in enumerate(want):
            idx = self._idx_cache.get(s["query"])
            if idx is None:
                try:
                    idx = D.select(self.neurons, s["query"])
                except ValueError as exc:
                    raise ValueError("стимул %d: запрос %r не разбирается: %s"
                                     % (i + 1, s["query"], exc))
                self._idx_cache[s["query"]] = idx
            if not len(idx):
                raise ValueError("стимул %d: запрос %r не нашёл ни одного "
                                 "нейрона" % (i + 1, s["query"]))
            ready.append(dict(s, n=int(len(idx)), idx=idx))
        return ready

    def _install_stims(self, ready):
        """Записать разобранные стимулы. Отказать здесь уже нечему."""
        import torch

        if not ready:
            self._stims, self._drive = [], None
            return []
        n = len(self.neurons)
        drive = torch.zeros(1, n, device=self.device)
        for s in ready:
            t = torch.as_tensor(s["idx"], device=self.device)
            # Два стимула на одни нейроны не складываются, а побеждает
            # последний: сложение тихо вышло бы за предел Пуассона.
            drive[0, t] = float(s["rate_hz"])
        self._stims, self._drive = ready, drive
        return [{"query": s["query"], "rate_hz": s["rate_hz"],
                 "нейронов": s["n"]} for s in ready]

    # -- вмешательства --------------------------------------------------

    def _resolve_lesions(self, want, t_ms=None, cmd_id=None):
        """Разобрать запросы и посчитать, сколько нейронов и связей задето.

        Только проверка: ничего не ставится. Связи считаются по
        `Engine.fan()` — он берёт длины строк и счётчик по столбцам
        матрицы один раз за прогон, а не пересобирает 15 млн связей.
        """
        from . import data as D

        if self.engine is None:
            raise ValueError("движок не привязан: вмешиваться не во что")
        n_in, n_out = self.engine.fan()
        ready = []
        for i, w in enumerate(want):
            idx = self._idx_cache.get(w["query"])
            if idx is None:
                try:
                    idx = D.select(self.neurons, w["query"])
                except ValueError as exc:
                    raise ValueError(
                        "вмешательство %d: запрос %r не разбирается: %s"
                        % (i + 1, w["query"], exc))
                self._idx_cache[w["query"]] = idx
            if not len(idx):
                raise ValueError("вмешательство %d: запрос %r не нашёл ни "
                                 "одного нейрона" % (i + 1, w["query"]))
            import torch

            t = torch.as_tensor(idx, device=self.engine.device)
            # Какие связи задеты, зависит от вида: отключая исходящие,
            # задеваем исходящие выбранных, отключая входящие — входящие.
            # Подавление спайка задевает исходящие: получатели перестают
            # получать, хотя сами связи целы.
            syn = int((n_in[t] if w["kind"] == "in" else n_out[t]).sum())

            # Срок — целым числом шагов управления, и это правило, а не
            # округление «как получится». Снять вмешательство можно
            # только на границе шага (`tick`), поэтому срок в 1 мс при
            # шаге 20 мс длился бы 20 мс, а подтверждение называло бы 1.
            # Теперь фактический срок считается ДО подтверждения и
            # уходит в него вместе с заказанным.
            step = float(getattr(self.params, "control_ms", 20.0) or 20.0)
            if w["ms"] is None:
                steps = None
                real_ms = None
            else:
                steps = max(1, int(math.ceil(float(w["ms"]) / step - 1e-9)))
                real_ms = steps * step
            start = float(t_ms) if t_ms is not None else 0.0
            ready.append(dict(w, idx=idx, n=int(len(idx)), synapses=syn,
                              # Свой номер у каждого вмешательства, а не
                              # общий номер команды: иначе снятие одного
                              # по сроку убирало бы из списка все, что
                              # пришли той же командой.
                              id="%s.%d" % (cmd_id or "—", i + 1),
                              cmd=cmd_id,
                              шагов=steps, срок_мс=real_ms,
                              from_ms=start,
                              until_ms=(None if real_ms is None
                                        else start + real_ms)))

        # Несовместимые сочетания отклоняются ДО применения, а не
        # применяются наполовину.
        seen = {}
        for w in ready:
            key = (w["query"], w["kind"])
            if key in seen:
                raise ValueError(
                    "вмешательство %r вида %r задано дважды: какое из них "
                    "снимать по сроку — неразрешимо"
                    % (w["query"], w["kind"]))
            seen[key] = w
        for w in ready:
            for other in ready:
                if other is w:
                    continue
                pair = tuple(sorted((w["kind"], other["kind"])))
                if (w["query"] == other["query"]
                        and tuple(sorted(pair)) in
                        {tuple(sorted(c)) for c in LESION_CONFLICT}):
                    raise ValueError(
                        "на %r заданы сразу %r и %r: оба решают, что "
                        "делать с мембраной молчащего нейрона, и вместе "
                        "они противоречивы. Выберите одно"
                        % (w["query"], w["kind"], other["kind"]))
        return ready

    def _push_masks(self):
        """Собрать три списка индексов из действующих вмешательств."""
        import numpy as np

        by = {k: [] for k in LESION_KINDS}
        for w in self.lesions:
            by[w["kind"]].append(w["idx"])
        out = {}
        for k, parts in by.items():
            out[k] = (np.unique(np.concatenate(parts)) if parts
                      else np.array([], dtype=np.int64))
        self.engine.set_masks(fire=out["fire"], out=out["out"],
                              inc=out["in"], hold=out["hold"])
        return out

    def _install_lesions(self, ready):
        self.lesions = list(ready)
        self._push_masks()
        return self.lesions_shown()

    def lesions_shown(self):
        """Действующие вмешательства — для подтверждения, окна и журнала."""
        return [{"id": w["id"], "запрос": w["query"], "вид": w["kind"],
                 "что": LESION_KINDS[w["kind"]], "нейронов": w["n"],
                 "связей": w["synapses"],
                 "с, мс": round(w["from_ms"], 3),
                 "до, мс": (None if w["until_ms"] is None
                            else round(w["until_ms"], 3)),
                 # Заказанный срок и фактический — врозь. Фактический
                 # известен до подтверждения, и показывать «применено на
                 # 1 мс» там, где действовало 20, нельзя.
                 "просили, мс": w.get("ms"),
                 "срок, мс": w.get("срок_мс"),
                 "шагов управления": w.get("шагов")}
                for w in self.lesions]

    def tick(self, t_ms):
        """Снять вмешательства, у которых вышел срок. Зовётся на границе шага.

        Срок считается в МОДЕЛЬНОМ времени, а не по стене: при скорости
        около 0,06 секунда по стене — это 60 мс модели, и вмешательство
        «на полсекунды» по стене длилось бы восемь секунд модели.

        Точность снятия — один шаг управления (20 мс): именно там
        состояние модели целостно, и только там вмешательство можно
        снять, не разрывая шаг физики посреди вычисления.
        """
        if not self.lesions:
            return []
        t = float(t_ms)
        done = [w for w in self.lesions
                if w["until_ms"] is not None and t >= w["until_ms"]]
        if not done:
            return []
        self.lesions = [w for w in self.lesions if w not in done]
        self._push_masks()
        rec = {"что": "срок вмешательства вышел",
               "модельное, мс": round(t, 3),
               "снято": [{"id": w["id"], "запрос": w["query"],
                          "вид": w["kind"], "нейронов": w["n"],
                          "срок, мс": w.get("срок_мс")}
                         for w in done],
               "осталось": self.lesions_shown()}
        self.log.append(rec)
        return done

    # -- применение -----------------------------------------------------

    def apply(self, payload, initial=False, cmd_id=None, t_ms=None):
        """Проверить всё, потом записать. Возвращает применённое.

        Порядок именно такой: при отказе модель остаётся в том
        состоянии, в каком была, и подтверждение несёт причину.

        ``cmd_id`` и ``t_ms`` — номер команды и модельное время границы
        шага, на которой её применили. Они идут в журнал прогона: без
        них по сохранённому файлу нельзя сказать, ни какая команда это
        была, ни на какой миллисекунде она сработала.
        """
        if self.body is None:
            raise ValueError("сеанс ещё не собран: менять нечего")
        if not isinstance(payload, dict):
            raise TypeError("правка должна быть словарём, пришло %r"
                            % type(payload).__name__)
        if not payload:
            raise ValueError("пустая правка: нечего менять")
        unknown = [k for k in payload if k not in BODY_KNOBS]
        if unknown:
            raise KeyError("менять нельзя: %s; можно: %s"
                           % (", ".join(sorted(unknown)),
                              ", ".join(sorted(BODY_KNOBS))))

        slots = self.slots()
        ready = {}
        for name, value in payload.items():
            k = BODY_KNOBS[name]
            if k.kind == "float":
                ready[name] = _num(name, value, k.lo, k.hi, k.unit)
            elif k.kind == "bool":
                ready[name] = _bool(name, value)
            elif k.kind == "body_sources":
                if not isinstance(value, (list, tuple)):
                    raise ValueError("sources: нужен список из %d слотов"
                                     % slots)
                if len(value) != slots:
                    raise ValueError(
                        "sources: слотов в модели %d, а пришло %d. Слот — "
                        "это тело MJCF, их число задаётся при сборке модели "
                        "и на ходу не меняется: «добавить источник» — это "
                        "включить свободный слот" % (slots, len(value)))
                ready[name] = [
                    _body_source("sources[%d]" % i,
                                 d if isinstance(d, dict) else {})
                    for i, d in enumerate(value)]
            elif k.kind == "body_objects":
                n = self.object_slots()
                if not isinstance(value, (list, tuple)):
                    raise ValueError("objects: нужен список из %d слотов"
                                     % n)
                if not n:
                    raise ValueError(
                        "objects: в модели нет слотов зрительных объектов. "
                        "Их создаёт `objects.inject` ДО компиляции модели; "
                        "добавить тело в собранную модель MuJoCo нельзя")
                if len(value) != n:
                    raise ValueError(
                        "objects: слотов в модели %d, а пришло %d. Слот — "
                        "это тело MJCF, их число задаётся при сборке модели "
                        "и на ходу не меняется: «добавить объект» — это "
                        "включить свободный слот" % (n, len(value)))
                ready[name] = [
                    _body_object("objects[%d]" % i,
                                 d if isinstance(d, dict) else {})
                    for i, d in enumerate(value)]
            elif k.kind == "body_light":
                # Свет проверяется здесь, а до модели доходит ниже: у
                # сеанса без зрения глаз нет, но свет в сцене есть, и
                # отказывать в нём было бы выдумкой — кадр работника
                # освещён теми же источниками.
                ready[name] = _num(name, value, k.lo, k.hi, k.unit)
            elif k.kind == "stims":
                # Разбор запросов — здесь же, в проверке: иначе отказ
                # пришёлся бы на этап записи, когда слоты уже переписаны.
                ready[name] = self._resolve_stims(_stims(name, value))
            elif k.kind == "lesions":
                ready[name] = self._resolve_lesions(
                    _lesions(name, value), t_ms=t_ms, cmd_id=cmd_id)
            else:                                     # pragma: no cover
                raise ValueError("%s: неизвестный тип %r" % (name, k.kind))

        was, now = {}, {}
        snap = self.snapshot()
        for name, value in ready.items():
            was[name] = snap.get(name)
            if name == "sources":
                self._write_sources(value)
                now[name] = self._read_sources()
            elif name == "objects":
                self._write_objects(value, t_ms=t_ms, cmd_id=cmd_id)
                now[name] = self._read_objects()
            elif name == "vision_light":
                # Пишется и в модель, и в настройку. В модель — потому
                # что это настоящий свет; в настройку — потому что по
                # сохранённому прогону должно быть видно, с каким
                # множителем он считался.
                from . import vision as _V

                _V.scene_light(self.body, value)
                setattr(self.params, name, value)
                # Отдаётся тот же множитель, что показывает `snapshot`:
                # «было» и «стало» в журнале обязаны быть одной
                # величиной, иначе правку не прочитать. Что получилось в
                # самой модели, уезжает в `timing["зрение"]["свет сцены"]`.
                now[name] = getattr(self.params, name)
            elif name == "stim":
                now[name] = self._install_stims(value)
            elif name == "lesion":
                now[name] = self._install_lesions(value)
            else:
                setattr(self.params, name, value)
                now[name] = getattr(self.params, name)

        marked = sorted(n for n in ready if BODY_KNOBS[n].marked)
        rec = {"что": "исходные условия" if initial else "правка",
               # У исходных условий модельное время нулевое по смыслу:
               # они применяются до разогрева.
               "модельное, мс": (0.0 if initial else
                                 (None if t_ms is None
                                  else round(float(t_ms), 3))),
               "команда": cmd_id,
               "было": was, "стало": now}
        if marked and not initial:
            rec["помечено"] = marked
            rec["оговорка"] = (
                "изменено на ходу: %s — сеанс перестал быть сравнимым с "
                "эталонным" % ", ".join(marked))
        self.log.append(rec)
        return now

    def _write_sources(self, want):
        """Положение и сила слотов — в модель, в кэш, в сцену и в массив."""
        import numpy as np

        ar = self.body.arena
        for i, d in enumerate(want):
            ar.odor_source[i, 0] = d["x"]
            ar.odor_source[i, 1] = d["y"]
            # Сила выключенного слота — ноль: так его не чует ни одна
            # антенна, где бы он ни стоял. Заказанную силу помним, чтобы
            # включение вернуло её, а не единицу по умолчанию.
            self._kept[i] = d["strength"]
            self._on[i] = bool(d["on"])
            ar.peak_odor_intensity[i, 0] = d["strength"] if d["on"] else 0.0
            if self.sources is not None and i < len(self.sources):
                self.sources[i, 0] = d["x"]
                self.sources[i, 1] = d["y"]
            row = self._mocap.get(i)
            if row is not None:
                # Маркер — mocap-тело: его положение задаётся данными, а
                # не моделью, и следующий шаг физики поставит его туда.
                self.body.physics.data.mocap_pos[row, 0] = d["x"]
                self.body.physics.data.mocap_pos[row, 1] = d["y"]
            g = self._geom.get(i)
            if g is not None:
                self.body.physics.model.geom_rgba[g, 3] = (
                    1.0 if d["on"] else self.OFF_ALPHA)

        # Кэши `OdorArena` собраны в её конструкторе: без пересборки муха
        # будет чуять источник там, где его уже нет.
        rep = ar.odor_source[:, np.newaxis, np.newaxis, :]
        rep = np.repeat(rep, ar.odor_dimensions, axis=1)
        ar._odor_source_repeated = np.repeat(rep, ar.num_sensors, axis=2)
        pk = ar.peak_odor_intensity[:, :, np.newaxis]
        ar._peak_intensity_repeated = np.repeat(pk, ar.num_sensors, axis=2)
        # Маркеры и кэши арены правлены мимо биндинга dm_control, то есть
        # состояние не помечено грязным. Наблюдение прошлого шага,
        # которое держит быстрый контроллер походки (`flybrain.gait`),
        # после этого не переиспользуется.
        inval = getattr(self.body, "invalidate_observation", None)
        if inval is not None:
            inval()
