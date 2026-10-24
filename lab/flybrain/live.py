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

    def apply(self, payload):
        """Проверить всё, потом записать. Возвращает применённое.

        Порядок именно такой: при отказе модель остаётся в том
        состоянии, в каком была, и подтверждение несёт причину. Иначе
        половина правки уже уехала бы в идущий расчёт.
        """
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
        rec = {"было": was, "стало": now}
        if marked:
            rec["помечено"] = marked
            rec["оговорка"] = (
                "изменено на ходу: %s — прогон перестал быть сравнимым с "
                "эталонным" % ", ".join(marked))
        self.log.append(rec)
        return now
