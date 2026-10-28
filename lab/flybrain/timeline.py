"""Единый таймлайн модельного времени: что записано, чего нет и где предел.

Ось здесь одна — **модельное время**, и только оно. Настенное время идёт
рядом как замер (сколько стоил шаг), а не как ось: при измеренной
скорости 0,055–0,078 от настоящего времени одна и та же секунда по стене
покрывает разное число шагов модели, и строить по ней ряд значит мерить
не модель, а загрузку видеокарты.

Единица записи — **шаг управления**, 20 мс модельного времени (100 шагов
физики и 200 шагов движка LIF). Между шагами данных нет и не будет
никогда: внутри шага состояние модели не целостно, снимать его оттуда
нечем. Отсюда главное свойство этого модуля: **на вопрос о моменте,
которого в записи нет, он отвечает «нет данных», а не числом.**
Интерполяция соседей на шаге 20 мс отличается от правды на уровне шума,
то есть выглядела бы правдоподобно и была бы выдумкой. См.
:class:`Hole` и :meth:`Timeline.value`.

Что покрывает таймлайн — перечислено в :data:`CHANNELS`, по строке на
сигнал: сенсорные каналы, выбранная нейронная активность, нисходящие,
моторные команды, траектория тела. Изменения среды и вмешательства идут
не рядами, а событиями (:meth:`Timeline.event`): они случаются в один
момент, и растягивать их на ряд значило бы придумывать им длительность.
**Новый сигнал добавляется строкой в** :data:`CHANNELS` **и ключом в
записи шага — механизм при этом не правится.** Зрение добавится так же:
его каналы уже объявлены и помечены как неподключённые, поэтому спросить
их можно уже сейчас, а ответ будет «канал не подключён», а не ноль.

Предел памяти объявлен и проверяем (:meth:`Timeline.budget`). Арифметика,
из-за которой он обязателен: одна строка плотной записи активности — это
138 639 × 2 байта = 277 КБ, при шаге 20 мс 13,9 МБ на секунду модельного
времени. Поэтому здесь нет ни одного растущего без предела хранилища:

* ряды каналов — кольцо на ``keep_steps`` шагов, ``float32``;
* растр зондов — кольцо тех же ``keep_steps`` шагов на не больше
  :data:`MAX_PROBE_NEURONS` нейронов, ``int16``;
* события — кольцо на ``max_events`` записей.

Кольцо проворачивается по времени один раз, при закрытии записи
(:meth:`Timeline.close`), ровно как кольцо ``bins`` в ``loop.py`` и
``body.py``: до этого строка шага ``i`` лежит в слоте ``i % keep_steps``,
и расставить строки по порядку можно только зная окончательное число
шагов.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Chan", "CHANNELS", "GROUPS", "FORMS", "Hole", "Timeline",
    "MAX_PROBE_NEURONS", "MAX_EVENTS", "DEFAULT_KEEP_S", "load",
    "ROW_BYTES", "channels_of",
]


# Сколько нейронов разрешено держать в растре зондов. Предел не из вкуса:
# это и есть граница между «записью выбранной активности» и плотным
# растром всего мозга, который стоит 277 КБ на шаг. 4096 нейронов при
# шаге 20 мс — это 410 КБ на секунду модельного времени, в 34 раза
# меньше плотного.
#
# Умещается в такой растр НЕ любая популяция съёма, и это замер, а не
# оценка: кандидатов в `world.probe_candidates` 63, и семь из них предел
# превышают — `region:ME` 66 839 нейронов (33 888 слева и 32 951
# справа), `region:LA` 12 153, `region:LO` 10 456, `region:LOP` 6827,
# `region:MB` 5836, `KC` 5177, `region:AL` 4103. Такой зонд
# отсекается с объяснением
# (`ValueError` ниже и `jobs._tape`), а не обрезается молча: по обрезку
# нельзя сказать, чья это активность. Умещаются все нисходящие
# (`super:descending`, 645 + 646 = 1291) и всё, что мельче.
#
# Прежняя редакция этого примечания называла «самую крупную популяцию —
# 6400 нейронов T4». Числа неверны дважды, и оба раза из-за ловушки
# селектора (`RESEARCH.md` §9): T4 среди кандидатов нет вовсе, а 6400 —
# это свободный `type:T4`, который затягивает двадцать посторонних
# типов; точный `type=T4` даёт в v783 **ноль** нейронов. Проверено
# запросом.
MAX_PROBE_NEURONS = 4096

# Сколько событий держать. События — это правки условий, вмешательства,
# изменения среды, паузы и шаги: их на сеанс десятки, а не тысячи.
# Предел нужен, чтобы восьмичасовой сеанс не копил их без конца.
MAX_EVENTS = 4000

# Окно записи по умолчанию, с. То же число, что стоит в песочнице
# (`desk/state.py`, `SandboxState.keep_s`).
DEFAULT_KEEP_S = 20.0


GROUPS = (
    ("сенсор", "что модель получает на входе"),
    ("нейронная активность", "выбранные популяции: частота и растр"),
    ("нисходящие", "выход популяции съёма и его асимметрия"),
    ("моторные команды", "что ушло в ноги или в рулёжку"),
    ("траектория", "где муха и куда смотрит"),
    ("среда", "источники, стены, условия — событиями"),
    ("вмешательства", "подавления и прямые стимулы — событиями"),
)


@dataclass(frozen=True)
class Chan:
    """Один сигнал таймлайна.

    ``key`` — ключ в записи шага (`rec`), который пишут ``loop.py`` и
    ``body.py``; ``kinds`` — у каких видов прогона этот сигнал есть.
    ``absent`` непусто у объявленного, но неподключённого канала: такой
    канал в массив не пишется, а на вопрос о нём таймлайн отвечает «канал
    не подключён» с этой самой причиной.
    """

    key: str
    group: str
    title: str
    unit: str
    what: str
    kinds: tuple
    source: str = ""
    absent: str = ""


_BOTH = ("arena", "body")


CHANNELS = tuple([
    # --- сенсор ------------------------------------------------------
    Chan("odour_l", "сенсор", "Запах, левая антенна", "",
         "интенсивность запаха на левой антенне", _BOTH,
         "arena: world.Body.sense; body: obs[odor_intensity] по имени канала"),
    Chan("odour_r", "сенсор", "Запах, правая антенна", "",
         "интенсивность запаха на правой антенне", _BOTH,
         "то же, правый канал"),
    Chan("orn_l", "сенсор", "Рецепторы слева", "Гц",
         "частота, поданная в обонятельные рецепторы левой стороны",
         ("body",), "body.run_episode: drive_brain"),
    Chan("orn_r", "сенсор", "Рецепторы справа", "Гц",
         "частота, поданная в обонятельные рецепторы правой стороны",
         ("body",), "body.run_episode: drive_brain"),
    Chan("taste", "сенсор", "Вкус", "Гц",
         "частота вкусовых рецепторов: ноль, если муха не на источнике",
         _BOTH, "grn_hz при расстоянии меньше food_radius_mm"),
    Chan("loom_l", "сенсор", "Надвигание слева", "Гц",
         "частота LPLC2 левой стороны от близкой стены", ("arena",),
         "world.Body.sense: wall_distance_ahead"),
    Chan("loom_r", "сенсор", "Надвигание справа", "Гц",
         "частота LPLC2 правой стороны от близкой стены", ("arena",),
         "world.Body.sense: wall_distance_ahead"),
    # Зрение объявлено заранее и помечено как неподключённое. Это и есть
    # «добавление сигнала — запись, а не правка механизма»: когда вход в
    # оптические доли появится, у этих строк снимется `absent`, и больше
    # ничего менять не придётся.
    Chan("retina_l", "сенсор", "Омматидии, левый глаз", "",
         "сигнал 721 омматидия левого глаза", _BOTH,
         "flygym: obs[vision], считается",
         absent="сигнал омматидиев в сеть не подаётся: входа в оптические "
                "доли не существует, 96 495 нейронов оптических долей "
                "молчат весь прогон"),
    Chan("retina_r", "сенсор", "Омматидии, правый глаз", "",
         "сигнал 721 омматидия правого глаза", _BOTH,
         "flygym: obs[vision], считается",
         absent="сигнал омматидиев в сеть не подаётся: входа в оптические "
                "доли не существует"),
    # --- нисходящие ---------------------------------------------------
    Chan("dn_l", "нисходящие", "Съём слева", "Гц",
         "средняя частота нейронов популяции съёма левого полушария",
         _BOTH, "сумма спайков популяции за шаг / (нейронов × длина шага)"),
    Chan("dn_r", "нисходящие", "Съём справа", "Гц",
         "средняя частота нейронов популяции съёма правого полушария",
         _BOTH, "то же, правое полушарие"),
    Chan("asym", "нисходящие", "Асимметрия съёма", "",
         "разность сторон, делённая на сумму, за вычетом собственного "
         "перекоса, измеренного на разогреве", _BOTH,
         "arena: world.Body.act; body: body.run_episode"),
    Chan("z", "нисходящие", "Асимметрия в сигмах шума", "",
         "асимметрия, нормированная на собственный шум разогрева и "
         "зажатая в ±3", ("body",), "body.run_episode"),
    Chan("spikes", "нейронная активность", "Спайков за шаг", "шт",
         "сумма спайков по всему мозгу за один шаг управления", _BOTH,
         "acc.sum() того же шага"),
    # --- моторные команды ---------------------------------------------
    Chan("amp_l", "моторные команды", "Амплитуда левых ног", "",
         "множитель качания левых трёх ног, уходит в "
         "HybridTurningController", ("body",), "body.run_episode: action[0]"),
    Chan("amp_r", "моторные команды", "Амплитуда правых ног", "",
         "множитель качания правых трёх ног", ("body",),
         "body.run_episode: action[1]"),
    Chan("v", "моторные команды", "Скорость", "мм/с",
         "заданная скорость центра тела (придуманная рулёжка арены)",
         ("arena",), "world.Body.act"),
    Chan("omega", "моторные команды", "Поворот", "рад/с",
         "заданная угловая скорость курса", ("arena",), "world.Body.act"),
    Chan("drive", "моторные команды", "Полнота выхода", "",
         "выход популяции, делённый на опорную частоту, не больше единицы",
         ("arena",), "world.Body.act"),
    # --- траектория ---------------------------------------------------
    Chan("x", "траектория", "Положение, x", "мм", "координата центра тела",
         _BOTH, "arena: FlyState; body: obs[fly][0]"),
    Chan("y", "траектория", "Положение, y", "мм", "координата центра тела",
         _BOTH, "arena: FlyState; body: obs[fly][0]"),
    Chan("theta", "траектория", "Курс", "рад", "направление оси тела",
         _BOTH, "arena: FlyState.theta; body: atan2 из fly_orientation"),
    Chan("dist", "траектория", "До источника", "мм",
         "расстояние до ближайшего ВКЛЮЧЁННОГО источника; −1 значит "
         "«ни одного включённого», а не ноль миллиметров", ("body",),
         "body.run_episode"),
    Chan("food_dist", "траектория", "До еды", "мм",
         "расстояние до ближайшей кормушки", ("arena",),
         "Arena.nearest_food_distance"),
    Chan("grad_cos", "траектория", "Совпадение с градиентом", "",
         "косинус между курсом и направлением на источник, от −1 до +1; "
         "ноль на шагах без включённого источника — это «не с чем "
         "сравнивать»", _BOTH, "arena и body считают его по-своему"),
    Chan("source", "траектория", "Ближайший слот", "",
         "номер включённого слота источника; −1 — ни одного", ("body",),
         "body.run_episode: nearest"),
    # --- подпись стоимости --------------------------------------------
    Chan("вызовов_движка", "нейронная активность",
         "Вызовов движка LIF за шаг", "шт",
         "сколько раз за этот шаг управления звался `sim.Engine.step`. "
         "Это подпись стоимости режима: у просмотра записанного она "
         "нулевая, у пересчёта — 200 при шаге 20 мс", _BOTH,
         "sim.Engine.calls, разность на границах шага"),
    Chan("по_стене_мс", "нейронная активность", "Шаг по стене", "мс",
         "сколько настенного времени занял этот шаг управления", _BOTH,
         "time.perf_counter на границах шага"),
])


# Три формы дырки в записи плюс четвёртая, объявленная. Перечислены
# здесь, а не строками по месту: на каждую есть проверка
# (`probe/timeline_trap.py`), и добавить форму молча нельзя.
FORMS = {
    "между кадрами": "в этот момент модельного времени запись не велась: "
                     "единица записи — шаг управления, между шагами "
                     "состояние модели не целостно",
    "раньше начала кольца": "этот момент старше окна записи: кольцо "
                            "держит последние keep_s секунд, остальное "
                            "вытеснено",
    "позже конца записи": "этот момент позже последнего записанного шага",
    "нейрон вне зондов": "активность этого нейрона не записывалась: в "
                         "растр попадают только нейроны выбранных зондов",
    "канал не подключён": "канал объявлен, но сигнала в нём нет",
    "канала нет у этого вида": "у этого вида прогона такого сигнала не "
                               "бывает",
}


@dataclass(frozen=True)
class Hole:
    """Ответ «нет данных» — и почему именно нет.

    Логического значения у дырки нарочно нет: ``if value:`` и
    ``value or 0.0`` превратили бы отсутствие данных в ноль, то есть
    ровно в ту выдумку, против которой этот класс и сделан. Отличать
    дырку от числа надо проверкой типа, а :meth:`Timeline.value` и вовсе
    отдаёт пару ``(значение, дырка)``, где непусто ровно одно.
    """

    why: str
    t_ms: float
    what: str = ""
    detail: str = ""

    @property
    def text(self):
        return "нет данных: %s" % (self.detail or FORMS.get(self.why, self.why))

    def as_dict(self):
        return {"нет данных": True, "почему": self.why, "модельное, мс":
                self.t_ms, "что": self.what,
                "подробно": self.detail or FORMS.get(self.why, "")}


def channels_of(kind):
    """Каналы, которые у этого вида прогона вообще бывают."""
    return tuple(c for c in CHANNELS if kind in c.kinds)


def _written(kind):
    """Каналы, которые реально пишутся: объявленные без `absent`."""
    return tuple(c for c in channels_of(kind) if not c.absent)


# Байт на строку рядов: по каналу на `float32`. Считается по виду
# прогона, потому что набор каналов у вида свой.
def ROW_BYTES(kind):                                          # noqa: N802
    return len(_written(kind)) * 4


class Timeline:
    """Запись сеанса по модельному времени. Кольцо, предел объявлен.

    Создаётся на один прогон, заполняется на границе шага управления
    (:meth:`put`) и закрывается один раз (:meth:`close`). Читается
    только по модельному времени: :meth:`value`, :meth:`row`,
    :meth:`neuron`.
    """

    VERSION = 1

    def __init__(self, kind="arena", control_ms=20.0, keep_s=DEFAULT_KEEP_S,
                 probes=(), probe_idx=None, max_events=MAX_EVENTS,
                 meta=None):
        if keep_s is None or keep_s <= 0:
            raise ValueError(
                "таймлайн без окна записи не бывает: ряды и растр растут "
                "по строке на шаг управления, и у сеанса без конца это "
                "рост без предела. Укажите keep_s в секундах модельного "
                "времени")
        self.kind = str(kind)
        self.control_ms = float(control_ms)
        self.keep_s = float(keep_s)
        self.keep_steps = max(1, int(round(self.keep_s * 1000.0
                                           / self.control_ms)))
        self.chans = _written(self.kind)
        self.col = {c.key: i for i, c in enumerate(self.chans)}
        self.rows = np.zeros((self.keep_steps, len(self.chans)), np.float32)
        self.t_ms = np.full(self.keep_steps, np.nan, np.float64)
        self.n_put = 0
        self.closed = False
        self.t0_ms = 0.0

        # Зонды: имя популяции -> (срез столбцов растра, число нейронов).
        # Индексы нейронов лежат рядом, чтобы «нейрон вне зондов» был
        # ответом, а не догадкой.
        self.probes = {}
        idx_parts = []
        at = 0
        for name, idx in (probe_idx or {}).items():
            a = np.asarray(idx, np.int64)
            if at + len(a) > MAX_PROBE_NEURONS:
                raise ValueError(
                    "зонд %r не умещается в растр: нейронов уже %d, в нём "
                    "%d, предел %d. Это граница между записью выбранной "
                    "активности и плотным растром всего мозга (277 КБ на "
                    "шаг): сузьте запрос, а не поднимайте предел"
                    % (name, at, len(a), MAX_PROBE_NEURONS))
            self.probes[name] = (at, len(a))
            idx_parts.append(a)
            at += len(a)
        self.probe_names = tuple(self.probes)
        self.probe_idx = (np.concatenate(idx_parts) if idx_parts
                          else np.zeros(0, np.int64))
        # Куда какой нейрон лёг. Нужно ровно для одного: отличить «нейрон
        # молчал» от «нейрон не записывался».
        self.where = {int(v): i for i, v in enumerate(self.probe_idx)}
        self.raster = np.zeros((self.keep_steps, len(self.probe_idx)),
                               np.int16)
        self.probe_hz = np.zeros((self.keep_steps, max(len(self.probes), 1)),
                                 np.float32)
        self.probe_list = tuple(probes) or tuple(self.probes)

        self.max_events = int(max_events)
        self.events = []
        self.meta = dict(meta or {})

    # ---------------------------------------------------------- запись

    def put(self, t_ms, rec, counts=None, lif_calls=0, wall_ms=0.0):
        """Записать один шаг управления.

        ``rec`` — запись шага, как её собирают ``loop.run_episode`` и
        ``body.run_episode``; берутся из неё только объявленные каналы, и
        ничего больше. ``counts`` — спайки за шаг по всем нейронам
        (``int16[N]``), из них в растр уходят только нейроны зондов.
        """
        if self.closed:
            raise ValueError("таймлайн закрыт: запись в него уже не идёт")
        slot = self.n_put % self.keep_steps
        self.t_ms[slot] = float(t_ms)
        row = self.rows[slot]
        row.fill(np.nan)
        for c in self.chans:
            if c.key == "вызовов_движка":
                row[self.col[c.key]] = float(lif_calls)
            elif c.key == "по_стене_мс":
                row[self.col[c.key]] = float(wall_ms)
            else:
                v = rec.get(c.key)
                if v is not None:
                    row[self.col[c.key]] = float(v)
        if counts is not None and len(self.probe_idx):
            took = np.asarray(counts)[self.probe_idx]
            self.raster[slot] = took.astype(np.int16)
            step_s = self.control_ms / 1000.0
            for i, name in enumerate(self.probe_names):
                a, n = self.probes[name]
                self.probe_hz[slot, i] = (float(took[a:a + n].sum())
                                          / (n * step_s)) if n else 0.0
        self.n_put += 1
        return slot

    def event(self, t_ms, group, what, detail="", payload=None):
        """Записать событие: правку среды, вмешательство, паузу, шаг.

        События — кольцо: при переполнении вытесняются самые старые, и
        это говорится в :meth:`budget`, а не замалчивается.
        """
        self.events.append({
            "модельное, мс": (None if t_ms is None else round(float(t_ms), 3)),
            "группа": str(group), "что": str(what),
            "подробно": str(detail), "значения": payload})
        if len(self.events) > self.max_events:
            del self.events[:len(self.events) - self.max_events]
        return self.events[-1]

    def close(self, n_steps=None):
        """Провернуть кольцо по времени и подписать начало записи.

        До этого строка шага ``i`` лежит в слоте ``i % keep_steps``, то
        есть при провернувшемся кольце самая старая строка стоит не
        первой. Расставить их по порядку можно только зная окончательное
        число шагов — поэтому это делается один раз и здесь.
        """
        if self.closed:
            return self
        total = int(self.n_put if n_steps is None else n_steps)
        kept = min(total, self.keep_steps)
        if total > self.keep_steps:
            shift = -(total % self.keep_steps)
            self.rows = np.roll(self.rows, shift, axis=0)
            self.t_ms = np.roll(self.t_ms, shift, axis=0)
            self.raster = np.roll(self.raster, shift, axis=0)
            self.probe_hz = np.roll(self.probe_hz, shift, axis=0)
        else:
            self.rows = self.rows[:kept]
            self.t_ms = self.t_ms[:kept]
            self.raster = self.raster[:kept]
            self.probe_hz = self.probe_hz[:kept]
        self.n_rows = kept
        self.t0_ms = float(self.t_ms[0]) if kept else 0.0
        self.closed = True
        self.steps_total = total
        return self

    # ---------------------------------------------------------- чтение

    @property
    def window(self):
        """(первое, последнее) записанное модельное время, мс."""
        good = self.t_ms[np.isfinite(self.t_ms)]
        if not len(good):
            return (None, None)
        return (float(good.min()), float(good.max()))

    def times(self):
        return self.t_ms[np.isfinite(self.t_ms)].copy()

    def _find(self, t_ms, what=""):
        """Номер строки для момента или :class:`Hole`.

        Порядок проверок важен и определён: сперва окно, потом попадание
        на шаг. Иначе момент, вытесненный кольцом, выглядел бы как
        «между кадрами», и причина отказа была бы названа неверно.
        """
        t = float(t_ms)
        lo, hi = self.window
        if lo is None:
            return Hole("позже конца записи", t, what,
                        "запись пуста: ни одного шага не записано")
        tol = 1e-6
        if t < lo - tol:
            return Hole("раньше начала кольца", t, what,
                        "окно записи начинается с %.3f мс, спрошено "
                        "%.3f мс: это модельное время вытеснено кольцом "
                        "на %g с" % (lo, t, self.keep_s))
        if t > hi + tol:
            return Hole("позже конца записи", t, what,
                        "последний записанный шаг — %.3f мс, спрошено "
                        "%.3f мс" % (hi, t))
        hit = np.nonzero(np.abs(self.t_ms - t) <= tol)[0]
        if not len(hit):
            near = self.t_ms[np.isfinite(self.t_ms)]
            before = near[near < t]
            after = near[near > t]
            return Hole(
                "между кадрами", t, what,
                "ближайшие записанные шаги — %s и %s мс; шаг управления "
                "%g мс, и между шагами записи нет: состояние модели "
                "целостно только на границе шага"
                % ("%.3f" % before.max() if len(before) else "—",
                   "%.3f" % after.min() if len(after) else "—",
                   self.control_ms))
        return int(hit[0])

    def row(self, t_ms):
        """Все каналы в этот момент или :class:`Hole`."""
        i = self._find(t_ms)
        if isinstance(i, Hole):
            return i
        return {c.key: float(self.rows[i, self.col[c.key]])
                for c in self.chans}

    def value(self, t_ms, channel):
        """Пара ``(значение, дырка)``: непусто ровно одно.

        Числом отвечается только то, что записано. Ни интерполяции, ни
        «ближайшего» здесь нет и быть не может: на шаге 20 мс
        интерполяция гладкой траектории отличается от правды на уровне
        шума, то есть выглядит правдоподобно и является выдумкой.
        """
        chan = next((c for c in CHANNELS if c.key == channel), None)
        if chan is None:
            return None, Hole("канала нет у этого вида", float(t_ms), channel,
                              "канал %r не объявлен вовсе; объявленные: %s"
                              % (channel, ", ".join(c.key for c in CHANNELS)))
        if self.kind not in chan.kinds:
            return None, Hole("канала нет у этого вида", float(t_ms), channel,
                              "у вида %r канала %r не бывает: он есть у %s"
                              % (self.kind, channel, ", ".join(chan.kinds)))
        if chan.absent:
            return None, Hole("канал не подключён", float(t_ms), channel,
                              chan.absent)
        i = self._find(t_ms, channel)
        if isinstance(i, Hole):
            return None, i
        v = float(self.rows[i, self.col[channel]])
        if not math.isfinite(v):
            return None, Hole("канал не подключён", float(t_ms), channel,
                              "канал объявлен и записывается, но на этом "
                              "шаге значения в записи шага не было")
        return v, None

    def neuron(self, t_ms, neuron):
        """Спайки одного нейрона за шаг или :class:`Hole`.

        Растр целочислен по построению: это счёт спайков за шаг
        управления. При рефрактерном периоде 2,2 мс и шаге 20 мс нейрон
        не может выдать больше ``⌊20/2,2⌋ + 1 = 10`` спайков, и дробное
        значение в растре — признак выдумки, а не округления.
        """
        j = self.where.get(int(neuron))
        if j is None:
            return None, Hole(
                "нейрон вне зондов", float(t_ms), "нейрон %d" % int(neuron),
                "в растр попали только нейроны зондов (%d нейронов: %s); "
                "нейрона %d среди них нет, и его активность не "
                "записывалась — это не «молчал»"
                % (len(self.probe_idx), ", ".join(self.probe_names) or "нет",
                   int(neuron)))
        i = self._find(t_ms, "нейрон %d" % int(neuron))
        if isinstance(i, Hole):
            return None, i
        return int(self.raster[i, j]), None

    def population(self, t_ms, name):
        """Частота популяции зонда в этот момент, Гц, или :class:`Hole`."""
        if name not in self.probes:
            return None, Hole(
                "нейрон вне зондов", float(t_ms), name,
                "зонда %r в этой записи нет; есть: %s"
                % (name, ", ".join(self.probe_names) or "ни одного"))
        i = self._find(t_ms, name)
        if isinstance(i, Hole):
            return None, i
        k = self.probe_names.index(name)
        return float(self.probe_hz[i, k]), None

    def events_between(self, t_from=None, t_to=None, group=None):
        out = []
        for e in self.events:
            t = e.get("модельное, мс")
            if group is not None and e.get("группа") != group:
                continue
            if t is not None:
                if t_from is not None and t < t_from:
                    continue
                if t_to is not None and t > t_to:
                    continue
            out.append(e)
        return out

    # ----------------------------------------------------- предел памяти

    MAX_RASTER_SPIKES = 10   # ⌊20 / 2,2⌋ + 1 при шаге 20 мс

    def raster_ceiling(self):
        """Сколько спайков за шаг нейрон может выдать физически.

        Рефрактерный период 2,2 мс (`sim.MODEL_PARAMS['tRefrac']`), шаг
        управления ``control_ms``: не больше ``⌊control_ms/2,2⌋ + 1``.
        Сенсорные каналы движок освобождает от рефрактерности
        (``no_refractory``), и для них предел другой — он назван
        отдельно, чтобы проверка не ловила на этом ложный провал.
        """
        return int(self.control_ms // 2.2) + 1

    def budget(self):
        """Объявленный предел памяти, байты, по частям.

        Ни одна часть не растёт со временем сеанса: у всех трёх предел
        задан числом шагов, нейронов или записей.
        """
        rows = self.keep_steps * len(self.chans) * 4
        times = self.keep_steps * 8
        raster = self.keep_steps * len(self.probe_idx) * 2
        hz = self.keep_steps * max(len(self.probes), 1) * 4
        ev = self.max_events * 400     # замер: запись события ≈ 400 Б
        return {
            "окно записи, с": self.keep_s,
            "шагов в окне": self.keep_steps,
            "каналов": len(self.chans),
            "нейронов в растре": int(len(self.probe_idx)),
            "ряды, Б": rows,
            "ось времени, Б": times,
            "растр зондов, Б": raster,
            "частоты популяций, Б": hz,
            "события, Б": ev,
            "всего, Б": rows + times + raster + hz + ev,
            "всего, МБ": round((rows + times + raster + hz + ev) / 2 ** 20, 3),
            "плотный растр был бы, МБ": round(
                self.keep_steps * 138639 * 2 / 2 ** 20, 1),
        }

    # ---------------------------------------------------------- на диск

    def save(self, out_dir):
        """Два файла рядом с прогоном: массивы и схема с журналом.

        Схема пишется вместе с данными нарочно: без неё по ``timeline.npz``
        не сказать ни что за столбцы в нём лежат, ни какие нейроны попали
        в растр, ни где кончается окно записи.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.closed:
            self.close()
        np.savez_compressed(
            out_dir / "timeline.npz",
            rows=self.rows, t_ms=self.t_ms, raster=self.raster,
            probe_hz=self.probe_hz,
            probe_idx=self.probe_idx.astype(np.int32),
        )
        (out_dir / "timeline.json").write_text(
            json.dumps(self.schema(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        return out_dir

    def schema(self):
        lo, hi = self.window
        return {
            "version": self.VERSION,
            "вид": self.kind,
            "шаг управления, мс": self.control_ms,
            "окно записи, с": self.keep_s,
            "шагов в окне": self.keep_steps,
            "шагов записано": int(getattr(self, "n_rows", self.n_put)),
            "шагов в сеансе": int(getattr(self, "steps_total", self.n_put)),
            "запись с, мс": lo,
            "запись до, мс": hi,
            "каналы": [{"ключ": c.key, "группа": c.group, "подпись": c.title,
                        "единица": c.unit, "что": c.what,
                        "источник": c.source} for c in self.chans],
            "объявлено, но не подключено": [
                {"ключ": c.key, "подпись": c.title, "почему": c.absent}
                for c in channels_of(self.kind) if c.absent],
            "зонды": {name: {"нейронов": n, "с столбца": a}
                      for name, (a, n) in self.probes.items()},
            "предел памяти": self.budget(),
            "предел растра, спайков на нейрон за шаг": self.raster_ceiling(),
            "события": self.events,
            "предел событий": self.max_events,
            "вытеснено событий": max(0, len(self.events) - self.max_events),
            "журнал": self.meta,
        }


def load(out_dir):
    """Собрать таймлайн обратно с диска. Ничего не пересчитывает."""
    out_dir = Path(out_dir)
    sch = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    z = np.load(out_dir / "timeline.npz", allow_pickle=False)
    tl = Timeline.__new__(Timeline)
    tl.kind = sch["вид"]
    tl.control_ms = float(sch["шаг управления, мс"])
    tl.keep_s = float(sch["окно записи, с"])
    tl.keep_steps = int(sch["шагов в окне"])
    keys = [c["ключ"] for c in sch["каналы"]]
    tl.chans = tuple(c for c in CHANNELS if c.key in keys)
    # Порядок столбцов берётся из файла, а не из нынешнего объявления:
    # иначе запись, сделанная до добавления канала, читалась бы сдвинутой.
    tl.col = {k: i for i, k in enumerate(keys)}
    tl.chans = tuple(sorted(tl.chans, key=lambda c: tl.col[c.key]))
    tl.rows = z["rows"]
    tl.t_ms = z["t_ms"]
    tl.raster = z["raster"]
    tl.probe_hz = z["probe_hz"]
    tl.probe_idx = z["probe_idx"].astype(np.int64)
    tl.probes = {name: (int(v["с столбца"]), int(v["нейронов"]))
                 for name, v in (sch.get("зонды") or {}).items()}
    tl.probe_names = tuple(tl.probes)
    tl.probe_list = tl.probe_names
    tl.where = {int(v): i for i, v in enumerate(tl.probe_idx)}
    tl.events = list(sch.get("события") or [])
    tl.max_events = int(sch.get("предел событий") or MAX_EVENTS)
    tl.meta = dict(sch.get("журнал") or {})
    tl.n_rows = int(sch.get("шагов записано") or len(tl.t_ms))
    tl.steps_total = int(sch.get("шагов в сеансе") or tl.n_rows)
    tl.n_put = tl.steps_total
    tl.t0_ms = float(sch.get("запись с, мс") or 0.0)
    tl.closed = True
    return tl
