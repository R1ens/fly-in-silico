"""Справочник научных сценариев запуска: что можно настроить и что получишь.

Это не второй `spec.py`, а описание первого **для человека и для интерфейса**.
`spec.py` знает, какие поля есть и как из них собираются конфиги; он не знает,
как поле назвать по-русски, в чём оно измеряется, какие значения осмысленны и
что появится в `out/<прогон>/` после счёта. Всё это живёт здесь — одним
списком, чтобы окно настройки опыта строилось из данных, а не из шестнадцати
мест в `desk/**`.

Правила, на которых держится файл:

* **Ничего не выдумано.** Значение по умолчанию каждого поля — это значение из
  соответствующего dataclass'а в `spec.py`, слово в слово; `probe/catalog_test.py`
  сверяет их через `dataclasses.fields` и падает на расхождении. Подпись к флагу
  командной строки и рекомендованное значение для этого вида взяты из argparse
  в `cli.py` (поле `cli` и `cli_default`).
* **Пределы помечены источником.** В argparse нет ни одного `choices` и ни одного
  числового предела, поэтому почти все минимумы и максимумы — мои, выведенные из
  смысла; у каждого стоит `limits_from="мой"`. Там, где предел действительно
  следует из кода (шаг бина не меньше `sim.DT`, частота Пуассона не больше
  10 000 Гц, глубина качания 0..1 из подсказки argparse, шаг физики не крупнее
  шага управления), стоит `limits_from="код"` и в `hint` сказано, откуда.
* **`None` — это поведение, а не ноль.** У полей с непустым `none_means` пустое
  значение несёт смысл: «измерить на разогреве», «до конца прогона», «взять
  значение из `world.BodyParams`», «имя по отметке времени». Интерфейс обязан
  уметь оставить такое поле пустым, иначе арена поедет молча: см. `ArenaSpec` и
  `RunSpec.to_body_params`.

Разбор видов: настоящих прогонов пять — `stim`, `arena`, `body`, `lateral`,
`verify`. `agent` описан шестым и честно помечен `spawns_runs=True`: сам он
ничего не считает, а порождает подпрогоны `exp-NN` через `agent.run_loop`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "VERSION", "N_NEURONS", "LIF_DT_MS", "ENGINE_STEPS_PER_S", "BODY_MEASURED",
    "Field", "Measure", "Kind", "KINDS", "MEASURES", "STIM_TERM_FIELDS",
    "NOT_IN_UI", "fields_of", "field_map", "measures_of", "series_of",
    "defaults", "bins_bytes", "n_bins_of", "estimate_wall_s",
    "validate", "limits",
]

VERSION = 1

# Нейронов в модели: `2025_Completeness_783.csv`, он же `len(D.load_neurons())`.
# Число нужно только для оценок памяти и времени, поэтому держится константой —
# считать его значит грузить таблицу, а справочник должен читаться мгновенно.
# Все функции оценки принимают `n_neurons=` и подменяют его настоящим.
N_NEURONS = 138639

# Шаг движка, мс. `sim.DT`.
LIF_DT_MS = 0.1

# Шагов движка в секунду на этой машине. Основание — `jobs.estimate`, там эта
# же константа и та же оговорка «замеры на этой машине».
ENGINE_STEPS_PER_S = 1167.0

# Замер прогона в теле: `probe/bodyfinal_timing.json`, 20 шагов управления при
# control_ms = 20, physics_dt = 2e-4, dn_tau_s = 0.5, readout ALPN, cuda.
# Ничего здесь не выведено «из головы»: все числа из того файла.
BODY_MEASURED = {
    "файл": "probe/bodyfinal_timing.json",
    "шаг_управления_мс": 393.34,       # «на шаг управления всего, мс»
    "мозг_на_шаг_мс": 130.87,          # 200 шагов LIF + синхронизация + учёт
    "физика_на_шаг_мс": 261.84,        # 100 шагов MuJoCo
    "шагов_lif_на_шаг": 200,
    "шагов_физики_на_шаг": 100,
    "подготовка_без_загрузки_с": 27.95,
    "перебор_популяций_с": 14.07,      # world.probe_candidates внутри эпизода
    "сборка_тела_с": 2.33,             # timing[setup_s] MuJoCo
    "рендерер_с": 0.79,
    "разогрев_с": 11.51,               # 75 шагов разогрева при tau 0.5
    "шагов_разогрева": 75,
    "загрузка_матрицы_с": 3.97,        # разово на работника, не на прогон
}

# Из замера: сколько стоит один шаг LIF и один шаг физики в теле.
_BODY_LIF_MS = BODY_MEASURED["мозг_на_шаг_мс"] / BODY_MEASURED["шагов_lif_на_шаг"]
_BODY_PHYS_MS = (BODY_MEASURED["физика_на_шаг_мс"]
                 / BODY_MEASURED["шагов_физики_на_шаг"])
# Подготовка тела без загрузки данных и без разогрева.
_BODY_PREP_S = (BODY_MEASURED["перебор_популяций_с"]
                + BODY_MEASURED["сборка_тела_с"] + BODY_MEASURED["рендерер_с"])

# Предел частоты пуассоновского входа следует из кода: `sim.Engine.step` зовёт
# `torch.bernoulli(rates * DT / 1000)`, а вероятность не бывает больше единицы.
MAX_POISSON_HZ = 1000.0 / (LIF_DT_MS / 1000.0) / 1000.0  # = 10000.0 Гц

# Длительности, которые принимает апстримный бенчмарк: `T_RUN_VALUES_SEC` в
# `upstream/code/benchmark.py`. Это не наше ограничение и не вкусовое: при
# любом другом значении `upstream/main.py` печатает «not in allowed values»
# и ВОЗВРАЩАЕТ НОЛЬ, ничего не посчитав. Молчаливый успех с пустыми руками —
# худший вид отказа, поэтому список продублирован здесь и проверяется до
# запуска, а `jobs._job_verify` сверяет его с настоящим при каждом прогоне.
VERIFY_T_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)

_MISSING = object()


# ------------------------------------------------------------------ записи


@dataclass(frozen=True)
class Field:
    """Одно настраиваемое поле прогона.

    `block` — где поле лежит в спеке: `None` значит корень `RunSpec`, иначе имя
    вложенного блока (`"stim"`, `"arena"`, `"body"`, `"lateral"`, `"verify"`,
    `"agent"`). `default` — значение из dataclass'а, а `cli_default` — то, что
    подставляет argparse для этого вида; они расходятся намеренно (у `RunSpec.t`
    умолчание 1 с, а `fly arena --t` даёт 12 с), и интерфейсу показывать стоит
    `cli_default`, когда он задан.
    """

    key: str
    block: str | None
    label: str
    type: str                      # float|int|bool|str|choice|query|list
    default: object = None
    min: object = None
    max: object = None
    unit: str = ""
    hint: str = ""
    choices: tuple = ()
    choices_from: str = ""         # откуда варианты берутся в рантайме
    cli: str = ""                  # флаг командной строки
    cli_default: object = _MISSING
    none_means: str = ""           # непустое: None здесь — поведение, не ноль
    limits_from: str = "нет"       # "код" | "мой" | "нет"
    required: bool = False
    item_fields: tuple = ()        # для type == "list"
    note: str = ""                 # оговорка по коду
    advanced: bool = False         # редкое: интерфейс прячет под раскрытие

    @property
    def has_cli_default(self):
        return self.cli_default is not _MISSING

    @property
    def shown_default(self):
        """Что показать в интерфейсе: умолчание вида, иначе умолчание спеки."""
        return self.cli_default if self.has_cli_default else self.default


@dataclass(frozen=True)
class Measure:
    """Что опыт даёт на выходе."""

    files: tuple = ()        # (имя, когда появляется, что внутри)
    summary: tuple = ()      # (ключ в JobResult.summary, что это)
    series: tuple = ()       # имена рядов, уходящих в окно по ходу прогона
    activity_frames: bool = False   # кадры активности мозга (ctx.frame)
    picture_frames: bool = False    # кадры картинки (ctx.step -> рендерер)
    index_kind: str = ""     # по какому ключу прогон опознаёт index.scan


@dataclass(frozen=True)
class Kind:
    key: str
    title: str
    about: str
    fields: tuple
    spawns_runs: bool = False
    notes: tuple = ()


def F(**kw):
    return Field(**kw)


# -------------------------------------------------------- общие поля RunSpec
#
# Собираются функциями, потому что у каждого вида своё умолчание argparse на
# те же самые поля `RunSpec`, а умолчание самой спеки одно.


def _f_out():
    return F(key="out", block=None, label="Имя папки в out/", type="str",
             default=None, cli="--out", none_means="отметка времени плюс слаг "
             "по первому стимулу или по виду (RunSpec.name_for)",
             hint="куда лягут файлы прогона")


def _f_title():
    return F(key="title", block=None, label="Заголовок отчёта", type="str",
             default=None, cli="--title", advanced=True,
             none_means="заголовок собирается сам из параметров прогона "
                        "(report._title, cli.cmd_arena, cli.cmd_body)",
             hint="подпись в report.html и в указателе")


def _f_t(cli_default, lo, hi, label="Длительность", hint=""):
    return F(key="t", block=None, label=label, type="float", default=1.0,
             min=lo, max=hi, unit="с", cli="--t", cli_default=cli_default,
             limits_from="мой", required=True,
             hint=hint or "модельное время прогона")


def _f_seed():
    return F(key="seed", block=None, label="Сид", type="int", default=0,
             min=0, max=2 ** 31 - 1, cli="--seed", limits_from="мой",
             hint="сид генератора движка и расстановки арены; один и тот же "
                  "сид даёт тот же прогон")


def _f_device():
    return F(key="device", block=None, label="Устройство", type="choice",
             default=None, choices=("cuda", "cpu"), cli="--device",
             none_means="cuda, если доступна, иначе cpu "
                        "(sim.run, loop.run_episode, body.run_episode)",
             hint="на процессоре счёт идёт в десятки раз медленнее")


def _f_model():
    return F(key="model", block=None, label="Модель Ollama", type="str",
             default=None, cli="--model", advanced=True,
             none_means="llm.DEFAULT_MODEL",
             hint="локальная модель для разбора, комментария или агента")


def _f_bin_ms(cli_default, label, hint, hi=200.0):
    return F(key="bin_ms", block=None, label=label, type="float", default=10.0,
             min=LIF_DT_MS, max=hi, unit="мс", cli="--bin",
             cli_default=cli_default, limits_from="код",
             hint=hint + f"; меньше {LIF_DT_MS:g} мс смысла не имеет — "
                         "`bin_steps = max(1, round(bin_ms / sim.DT))`")


def _f_repeat(hint):
    return F(key="repeat", block=None, label="Эпизодов в серии", type="int",
             default=0, min=0, max=100, cli="--bench", limits_from="мой",
             hint=hint)


def _f_repeat_t(cli_default=_MISSING):
    kw = {}
    if cli_default is not _MISSING:
        kw["cli_default"] = cli_default
    return F(key="repeat_t", block=None, label="Длительность эпизода серии",
             type="float", default=None, min=0.05, max=600.0, unit="с",
             cli="--bench-t", limits_from="мой",
             none_means="длительность эпизода равна `t` (RunSpec.episode_t)",
             hint="серия считает эпизоды этой длительности, а не `t`", **kw)


def _f_repeat_seed0():
    return F(key="repeat_seed0", block=None, label="Первый сид серии",
             type="int", default=100, min=0, max=2 ** 31 - 1,
             cli="--bench-seed", limits_from="мой", advanced=True,
             hint="смените, чтобы набрать независимые прогоны и объединить "
                  "с прежними (подсказка argparse)")


# --------------------------------------------------------------- stim: терм


STIM_TERM_FIELDS = (
    F(key="query", block="stim.activate", label="Кого зажечь", type="query",
      default="", required=True, cli="--activate/-a",
      hint="селектор `data.select`: `class:gustatory+side:left`, `type=DNp09`, "
           "`type:LPLC2`; `+` пересекает термы, запятая объединяет. Пустой "
           "запрос и запрос, не нашедший нейронов, — ошибка "
           "(RunSpec.to_stimuli)"),
    F(key="rate_hz", block="stim.activate", label="Частота стимуляции",
      type="float", default=200.0, min=0.0, max=MAX_POISSON_HZ, unit="Гц",
      cli="--rate/-r", limits_from="код",
      hint="пуассоновский вход прямо в мембрану; выше "
           f"{MAX_POISSON_HZ:g} Гц `torch.bernoulli` получит вероятность "
           "больше единицы (sim.Engine.step)"),
    F(key="t_on_ms", block="stim.activate", label="Включить на", type="float",
      default=0.0, min=0.0, max=600000.0, unit="мс", cli="--pulse ON_MS",
      limits_from="мой",
      hint="момент включения стимула внутри прогона"),
    F(key="t_off_ms", block="stim.activate", label="Выключить на", type="float",
      default=None, min=0.0, max=600000.0, unit="мс", cli="--pulse OFF_MS",
      limits_from="мой",
      none_means="стимул горит до конца прогона "
                 "(`off = steps if s.t_off_ms is None`, sim.run)",
      hint="вместе с `t_on_ms` задаёт окно импульса"),
    F(key="name", block="stim.activate", label="Имя стимула", type="str",
      default="", cli="--name",
      hint="пустое имя подменяется самим запросом (RunSpec.to_stimuli); "
           "по первому имени строится имя папки"),
)


# ------------------------------------------------------------------- виды


_STIM = Kind(
    key="stim",
    title="Стимуляция",
    about="Зажечь выбранные нейроны пуассоновским входом и записать, как "
          "активность разошлась по всему мозгу за заданное время.",
    fields=(
        _f_t(1.0, 0.01, 600.0, hint="сколько модельного времени считать"),
        _f_bin_ms(10.0, "Шаг бина записи",
                  "в бин складываются спайки каждого нейрона; он же шаг "
                  "анимации в отчёте", hi=1000.0),
        F(key="activate", block="stim", label="Стимулы", type="list",
          default=[], required=True, item_fields=STIM_TERM_FIELDS,
          min=1, max=8, limits_from="мой",
          hint="хотя бы один стимул: без них считать нечего"),
        F(key="silence", block="stim", label="Кого заглушить", type="query",
          default=None, cli="--silence/-s",
          none_means="никого не глушить",
          hint="селектор `data.select`; у этих нейронов обнуляются все входящие "
               "и исходящие синапсы (sim._silence). Контроль к опыту: "
               "`type:KC` убирает грибовидное тело"),
        F(key="trials", block="stim", label="Повторов", type="int", default=1,
          min=1, max=16, cli="--trials", limits_from="мой",
          hint="прогоны считаются одной партией (batch движка); спайки "
               "суммируются, частоты делятся на `t * trials`. Память движка "
               "растёт линейно"),
        _f_seed(),
        _f_device(),
        F(key="explain", block=None, label="Разбор локальной моделью",
          type="bool", default=False, cli="--explain", advanced=True,
          hint="после счёта Ollama пишет narrative.md"),
        _f_model(),
        _f_out(),
        _f_title(),
    ),
    notes=(
        "`max_bins` в `RunSpec` — предохранитель окна: при ненулевом значении "
        "`jobs._job_stim` отказывается считать, если `n_bins * N > max_bins`. "
        "У командной строки он выключен (0).",
    ),
)


_ARENA = Kind(
    key="arena",
    title="Арена (замкнутый цикл)",
    about="Мозг водит муху по плоской арене 60×60 мм: запах с двух антенн идёт "
          "в обонятельные рецепторы, активность выбранной популяции по "
          "полушариям превращается в скорость и поворот.",
    fields=(
        _f_t(12.0, 0.1, 7200.0, label="Длительность эпизода"),
        F(key="keep_s", block="arena", label="Окно записи", type="float",
          default=0.0, min=0.0, max=600.0, unit="с", limits_from="мой",
          hint="0 — писать активность за всю сессию. Иначе `bins` держит "
               "последние столько секунд, а счётчики и частоты остаются за "
               "всю сессию. Нужно длинной сессии: одна строка записи это "
               "138 639 × 2 байта = 277 КБ, при шаге 20 мс — 13,9 МБ на "
               "секунду модельного времени"),
        _f_bin_ms(20.0, "Шаг управления",
                  "за один шаг мозг считается `round(control_ms / sim.DT)` "
                  "раз, затем читается выход и двигается муха"),
        F(key="readout", block="arena", label="Популяция для рулёжки",
          type="choice", default=None, cli="--motor",
          choices_from="world.probe_candidates(neurons) — области и семейства "
                       "клеток, у которых не меньше восьми нейронов на сторону",
          none_means="все нисходящие нейроны (world.MOTOR: "
                     "`super:descending+side:left/right`)",
          hint="имена берутся из вывода `fly lateral`; неизвестное имя — "
               "ошибка с подсказкой похожих (RunSpec.to_body_params)",
          note="ВАЖНО: как только популяция выбрана, `dn_ref_hz` и "
               "`turn_gain_rad_s` уезжают в `BodyParams` как есть, в том числе "
               "`None`. Без выбранной популяции `None` там означает «оставить "
               "умолчание BodyParams», то есть 6 Гц и 5 рад/с без всякого "
               "разогрева. Смысл пустого поля зависит от этого — см. "
               "`RunSpec.to_body_params`"),
        F(key="dn_ref_hz", block="arena", label="Опорная частота рулёжки",
          type="float", default=None, min=0.1, max=500.0, unit="Гц",
          cli="--dn-ref", limits_from="мой",
          none_means="при выбранной популяции — ИЗМЕРИТЬ НА РАЗОГРЕВЕ "
                     "(`loop.run_episode`: `dn_ref_hz = max(tot/warm*1.3, 0.1)`); "
                     "без выбранной популяции — умолчание `BodyParams` 6.0 Гц",
          hint="частота, которая считается полным ходом: `drive = min(1, "
               "total / dn_ref_hz)` (world.Body.act)"),
        F(key="turn_gain_rad_s", block="arena", label="Усиление поворота",
          type="float", default=None, min=0.0, max=1000.0, unit="рад/с",
          cli="--turn-gain", limits_from="мой",
          none_means="при выбранной популяции — КАЛИБРОВАТЬ НА РАЗОГРЕВЕ "
                     "(`turn_gain = turn_target / сигма сглаженной асимметрии`); "
                     "без выбранной популяции — умолчание `BodyParams` 5.0",
          hint="во что превращается асимметрия полушарий; поворот всё равно "
               "обрезан `BodyParams.turn_max_rad_s = 4.5` рад/с"),
        F(key="turn_target_rad_s", block="arena",
          label="Поворот на одну сигму шума", type="float", default=None,
          min=0.0, max=4.5, unit="рад/с", cli="--turn-target",
          limits_from="код",
          none_means="умолчание `BodyParams.turn_target_rad_s` = 0.6 рад/с",
          hint="цель калибровки усиления; работает только когда "
               "`turn_gain_rad_s` пустое. Верхний предел — `turn_max_rad_s`, "
               "выше которого поворот всё равно обрезается"),
        F(key="dn_tau_s", block="arena", label="Постоянная интегрирования",
          type="float", default=None, min=0.001, max=5.0, unit="с",
          cli="--dn-tau", limits_from="мой",
          none_means="умолчание `BodyParams.dn_tau_s` = 0.12 с",
          hint="выход сглаживается, иначе пуассоновский шум 645 нисходящих "
               "нейронов за 20 мс перебивает асимметрию от запаха. Длина "
               "разогрева считается от неё: "
               "`warm = max(5, round(3000 * dn_tau_s / control_ms))`"),
        F(key="turn_sign", block="arena", label="Знак поворота", type="choice",
          default=1.0, choices=(1.0, -1.0), cli="--turn-sign",
          limits_from="код",
          hint="+1: больше активности слева — поворот налево (подсказка "
               "argparse). Любое другое число — тоже множитель, но смысла "
               "кроме ±1 у него нет"),
        F(key="swap_antennae", block="arena", label="Контроль: переставить "
          "антенны", type="bool", default=False, cli="--control",
          hint="левая концентрация идёт в правые рецепторы и наоборот. Если "
               "поворот действительно идёт через двусторонний путь, совпадение "
               "с градиентом должно упасть ниже случайного"),
        F(key="probes", block="arena", label="Записывать популяции по сторонам",
          type="bool", default=False, cli="--probe",
          hint="пишет ORN/ALPN/LH/MBON/DN/DNa01/DNa02/DNp09 по сторонам "
               "(world.PROBES) и считает, где теряется сторона запаха; "
               "кладёт probes.json"),
        _f_seed(),
        _f_device(),
        F(key="comment", block=None, label="Комментарий локальной модели",
          type="bool", default=False, cli="--no-comment", cli_default=True,
          advanced=True,
          hint="Ollama на отдельном потоке описывает, что муха делает; счёт "
               "её не ждёт. У командной строки включено, у окна выключено"),
        F(key="comment_every_ms", block=None, label="Как часто просить "
          "комментарий", type="float", default=1000.0, min=0.0, max=60000.0,
          unit="мс", cli="--comment-every", limits_from="мой", advanced=True,
          hint="работает только при включённом комментарии"),
        _f_model(),
        _f_repeat("серия эпизодов: норма против перестановки антенн. Считает "
                  "вдвое больше эпизодов, чем указано: оба условия"),
        _f_repeat_t(4.0),
        _f_repeat_seed0(),
        _f_out(),
        _f_title(),
    ),
    notes=(
        "Всё, кроме самого мозга, здесь придумано: и «запах → частота "
        "рецепторов», и «нисходящие → скорость и поворот». Так и написано в "
        "шапке `world.py`; эпизод — это «что будет, если прогнать настоящую "
        "сеть через выдуманное тело», а не поведение мухи.",
        "Серия (`repeat`) не сохраняет ни эпизоды, ни отчёт — только "
        "`bench.json` со средним, sd и sem по эпизодам.",
    ),
)


_BODY = Kind(
    key="body",
    title="В теле (NeuroMechFly v2)",
    about="Тот же мозг, но тело и походка — опубликованные: запах берётся с "
          "антенн мухи в MuJoCo, а асимметрия выхода уходит в "
          "`HybridTurningController`, который и считает ноги.",
    fields=(
        _f_t(4.0, 0.0, 7200.0, label="Длительность эпизода",
             hint="модельное время эпизода. Ноль — сеанс песочницы без "
                  "заранее заданного конца: его заканчивает только "
                  "остановка, и тогда обязательно окно записи"),
        F(key="keep_s", block="body", label="Окно записи", type="float",
          default=0.0, min=0.0, max=600.0, unit="с", limits_from="мой",
          hint="0 — писать активность за весь эпизод. Иначе `bins` держит "
               "последние столько секунд, а счётчики и частоты остаются за "
               "весь сеанс. Одна строка записи — 138 639 × 2 байта = "
               "277 КБ, при шаге 20 мс это 13,9 МБ на секунду модельного "
               "времени"),
        F(key="odour_ref", block="body", label="Яркость насыщения",
          type="float", default=None, min=1e-9, max=100.0, unit="",
          limits_from="мой",
          # Без `none_means` пустое значение считается незаполненным
          # обязательным полем, и спека, где ключ стоит со значением
          # `null` (а так его пишет `RunSpec.to_dict`), не проходит
          # проверку с фразой «значение не задано».
          none_means="измерить на разогреве по началу эпизода",
          hint="пусто — измерить на разогреве по тому, что муха чует в "
               "начале (так считались все прежние эпизоды). Песочнице "
               "нужно число: сеанс может начинаться с выключенными "
               "источниками, и измеренная яркость была бы нулевой"),
        F(key="readout", block="body", label="Популяция для рулёжки",
          type="choice", default="ALPN", required=True, cli="--readout",
          cli_default="ALPN",
          choices_from="world.probe_candidates(neurons)",
          hint="обязательное поле: неизвестное имя — `SystemExit` "
               "(body.run_episode). Пустым быть не может, в отличие от арены"),
        F(key="motor", block="body", label="Моторный путь",
          type="choice", default="legacy",
          choices=("legacy", "dn_linear", "dn_trained"),
          cli="--motor-decoder", cli_default="legacy",
          hint="что именно ведёт моторную команду. «legacy» — исторический "
               "съём с выбранной популяции (по умолчанию ALPN); "
               "«dn_linear» и «dn_trained» — прямые зонды нисходящих "
               "нейронов через декодер `flybrain.motor`",
          note="ЭТО НЕ РАВНОЗНАЧНЫЕ РЕЖИМЫ, и путать их нельзя. "
               "«legacy» с ALPN — съём с СЕНСОРНОЙ популяции, а не с "
               "нисходящего моторного пути: это измерено "
               "(`motor.readout_provenance`) и записано в сам прогон. "
               "Декодер нисходящих — ИНЖЕНЕРНЫЙ переходник, а не "
               "воспроизведённая биология. Прогон с не `legacy` с "
               "эталонными не сравнивается, и в `timing[\"оговорки\"]` об "
               "этом пишется прямо. «dn_trained» без файла весов не "
               "работает, а годного файла в дереве нет"),
        F(key="dn_tau_s", block="body", label="Постоянная интегрирования",
          type="float", default=0.5, min=0.01, max=5.0, unit="с",
          cli="--dn-tau", cli_default=0.5, limits_from="мой",
          hint="и сглаживание выхода, и длина разогрева: "
               "`warm = max(5, round(3000 * dn_tau_s / control_ms))`. При "
               "0,5 с и шаге 20 мс это 75 шагов разогрева, то есть около "
               f"{BODY_MEASURED['разогрев_с']:g} с по замеру"),
        F(key="control_ms", block="body", label="Шаг управления", type="float",
          default=20.0, min=LIF_DT_MS, max=200.0, unit="мс", cli="--control-ms",
          cli_default=20.0, limits_from="код",
          hint="за шаг считается `round(control_ms / sim.DT)` шагов мозга и "
               "`round(control_ms / 1000 / physics_dt)` шагов физики",
          note="командная строка кладёт это же число ещё и в `RunSpec.bin_ms` "
               "(`cli._spec_from_args`), а счёт берёт его из `BodySpec`. Два "
               "места на одно число: интерфейс должен менять их вместе"),
        F(key="physics_dt", block="body", label="Шаг физики", type="float",
          default=2e-4, min=1e-5, max=0.2, unit="с", cli="--physics-dt",
          cli_default=2e-4, limits_from="код",
          hint="шаг MuJoCo; крупнее шага управления смысла не имеет — "
               "`body_steps = max(1, round(control_ms/1000 / physics_dt))` "
               "станет единицей. Верхний предел проверяется против шага "
               "управления, а не этим числом"),
        F(key="cross", block="body", label="Перекрёстная связь с ногами",
          type="bool", default=True, cli="--no-cross", cli_default=True,
          hint="левое полушарие управляет правыми ногами. Замер показал, что "
               "более сильные левые ноги уводят муху вправо (шапка body.py); "
               "снять галочку — прямая связь, то есть контроль"),
        F(key="swap_antennae", block="body", label="Контроль: переставить "
          "антенны", type="bool", default=False, cli="--control",
          hint="тот же контроль, что у арены: интенсивности антенн меняются "
               "местами перед входом в рецепторы"),
        F(key="video", block="body", label="Записать видео", type="bool",
          default=False, cli="--video",
          hint="walk.mp4 рядом с прогоном; камера идёт за курсом (YawOnlyCamera)",
          note="в серии каждый эпизод пишет свой файл "
               "`walk-norm-NN.mp4` / `walk-swap-NN.mp4` в папке прогона; "
               "без заданной папки `body.run_episode` теперь отказывается "
               "писать видео, а не кладёт его мимо прогона"),
        F(key="play_speed", block="body", label="Замедление видео",
          type="float", default=0.2, min=0.01, max=1.0, cli="--play-speed",
          cli_default=0.2, limits_from="мой",
          hint="0,2 — в пять раз медленнее реального времени; муха быстрая, "
               "иначе ничего не видно (подсказка argparse)"),
        F(key="view", block="body", label="Живое окно MuJoCo", type="bool",
          default=False, cli="--view",
          hint="пассивный просмотр: видно, как муха идёт, мышь вертит камеру. "
               "Закрытие окна останавливает эпизод (body.run_episode)"),
        # ----------------------------------------------------- зрение
        F(key="vision", block="body", label="Зрение", type="bool",
          default=False, cli="--vision",
          hint="подать кадр ОБОИХ сложных глаз (721 омматидий каждый, "
               "настоящее наблюдение сцены MuJoCo) в 10 582 "
               "фоторецептора коннектома. Выключено по умолчанию: все "
               "сохранённые прогоны посчитаны без зрения, и прежний "
               "опыт обязан считаться как раньше",
          note="ОГРАНИЧЕНИЕ НЕ ПРЯЧЕТСЯ. Ранняя зрительная активность в "
               "настоящей сети установлена замером (L1 12,04 Гц, L2 "
               "12,64, L3 2,27, Tm1 2,05 против 0,000 при выключенном "
               "зрении), а ГЛУБОКАЯ ЗАВИСЯЩАЯ ОТ СТИМУЛА ПЕРЕДАЧА НЕТ: "
               "T4, T5, LPLC2, LC4 и DNp01 дают ровно 0,000 Гц "
               "(VISION.md §3а, §5). Поведение мухи на зрение не "
               "опирается: контур зрение→мотор не замкнут. Сверх того "
               "привязка омматидиев ИНЖЕНЕРНАЯ и опровергнута внешним "
               "свидетелем по взаимному совмещению двух популяций — 0 "
               "пар R7-R8 из 430 в одном омматидии против 2 из 430 у "
               "перемешанной (VISUAL_COLUMNS.md). Обеспеченное "
               "свойство каналов одно: соседство внутри популяции"),
        F(key="vision_hz", block="body", label="Обновление зрения",
          type="float", default=100.0, min=1.0, max=5000.0, unit="Гц",
          cli="--vision-hz", cli_default=100.0, limits_from="код",
          hint="как часто перерисовывается кадр глаза. FlyGym держит её "
               "целым числом герц, выше шага физики смысла нет",
          note="РАБОЧАЯ ТОЧКА, А НЕ БИОЛОГИЧЕСКАЯ ПОСТОЯННАЯ. 100 Гц "
               "выбраны по замеру цены шага: без зрения 497,5 мс, при "
               "50 Гц ×1,11, при 100 Гц ×1,16, при 250 Гц ×1,36, при "
               "500 Гц ×1,66 (RESEARCH.md §6). Быстрые переходные "
               "процессы смотрят на более высоких частотах, и для этого "
               "поле и есть. Действительная частота измеряется по "
               "отметкам FlyGym и пишется в прогон рядом с заказанной"),
        F(key="vision_eyes", block="body", label="Глаза", type="choice",
          default="both", choices=("both", "left", "right"),
          cli="--vision-eyes", cli_default="both",
          hint="«both» — оба сложных глаза, рабочая точка. Один глаз — "
               "контроль на двусторонний зрительный путь, а не режим"),
        F(key="vision_max_hz", block="body", label="Частота рецепторов",
          type="float", default=150.0, min=0.0, max=MAX_POISSON_HZ,
          unit="Гц", cli="--vision-max-hz", cli_default=150.0,
          limits_from="код",
          hint="сколько герц дают фоторецепторы при насыщении. Правило "
               "придумано и объявлено ровно одно: "
               "hz = max_hz · min(1, яркость / насыщение) — то же, что "
               "у обоняния (orn_max_hz)"),
        F(key="vision_ref", block="body", label="Яркость насыщения глаза",
          type="float", default=0.5, min=1e-6, max=1.0,
          cli="--vision-ref", cli_default=0.5, limits_from="код",
          hint="яркость кадра, считающаяся насыщением. Кадр нормирован "
               "в 0…1, поэтому и предел такой; при 0,5 в живой сцене "
               "насыщено 26,2 % омматидиев (замер probe/vision_live.py)"),
        F(key="vision_light", block="body", label="Свет сцены",
          type="float", default=1.0, min=0.0, max=4.0, unit="×",
          cli="--vision-light", cli_default=1.0, limits_from="мой",
          hint="множитель к ambient и diffuse НАСТОЯЩИХ источников "
               "MuJoCo арены FlyGym (light_top, light_bottom): меняется "
               "сам рендер кадра глаза, а не его пересчёт. 1 — модель "
               "не трогается вовсе"),
        F(key="vision_contrast", block="body", label="Контраст кадра",
          type="float", default=1.0, min=0.0, max=8.0, unit="×",
          cli="--vision-contrast", cli_default=1.0, limits_from="мой",
          hint="clip(0,5 + (I − 0,5)·k, 0, 1). При 1 преобразование не "
               "исполняется вовсе, и частоты совпадают с retina.rates "
               "до бита",
          note="это ОБЪЯВЛЕННОЕ ПРЕОБРАЗОВАНИЕ КАДРА, а не свойство "
               "сцены, и так записано в происхождении прогона. Свет, в "
               "отличие от него, настоящий"),
        F(key="vision_shuffle", block="body",
          label="Встречный замер: перемешать привязку", type="bool",
          default=False, cli="--vision-shuffle",
          hint="назначение перемешивается с сохранением числа нейронов "
               "на омматидий: суммарный вход тот же до последнего "
               "разряда, разрушена только пространственная структура",
          note="без такого прогона ни одно утверждение про "
               "пространственное зрение не стоит ничего. Замер: у L1 "
               "расхождение настоящей и перемешанной привязки "
               "+0,1483 Гц на живом кадре (VISION.md §3а)"),
        F(key="vision_adapter", block="body", label="Зрительный переходник",
          type="choice", default="retina", cli="--vision-adapter",
          cli_default="retina", choices_from="vision.names()",
          advanced=True,
          hint="чем кадр превращается в частоты. Переходник выбирается "
               "ПО ИМЕНИ, и настоящая ретинотопия или физиологическая "
               "динамика раннего зрения встанут на место сменой имени, "
               "а не переделкой обмена с работником",
          note="в этой сборке зарегистрирован один — «retina», "
               "инженерная привязка 721 омматидия. Какой переходник и "
               "какая привязка считали, видно в сохранённом прогоне: "
               "timing[\"зрение\"][\"привязка\"], включая отпечаток "
               "каждой из шести привязок и итог внешней проверки"),
        F(key="vision_channel", block="body", label="Канал фоторецептора",
          type="choice", default="max", choices=("max", "pale", "yellow"),
          cli="--vision-channel", cli_default="max", advanced=True,
          hint="у кадра NeuroMechFly два канала на омматидий — бледный "
               "и жёлтый. «max» берёт больший, как и официальная "
               "интеграция; хроматика при этом теряется, и выдумывать "
               "разделение по R7/R8 нельзя — в кадре его нет"),
        F(key="vision_mode", block="body", label="Совмещение привязки",
          type="choice", default="whiten", choices=("whiten", "rigid"),
          cli="--vision-mode", cli_default="whiten", advanced=True,
          hint="«whiten» приводит каждую ось к единичной дисперсии "
               "(сохраняет соседство, искажает пропорции), «rigid» "
               "масштабирует обе оси одним числом"),
        F(key="vision_assign", block="body", label="Назначение привязки",
          type="choice", default="capacity",
          choices=("capacity", "nearest"), cli="--vision-assign",
          cli_default="capacity", advanced=True,
          hint="«capacity» вносит анатомическое число мест на омматидий "
               "как ограничение (6 для R1-6, по 1 для R7 и R8), "
               "«nearest» — просто ближайший, и тогда 668 R7 ложатся на "
               "372 омматидия по 1,8 на каждый"),
        F(key="vis_objects", block="body", label="Слотов объектов в сцене",
          type="int", default=0, min=0, max=6, cli="--objects",
          cli_default=0, limits_from="код",
          hint="сколько зрительных объектов вписать в сцену. 0 — пула "
               "нет вовсе, и сцена неотличима от прежней (траектория "
               "тела совпадает побитово). Предел 6 — `objects.POOL`",
          note="тела MuJoCo создаются ДО компиляции модели и на ходу не "
               "добавляются, поэтому «появление» объекта — это "
               "включение свободного слота. Убранный слот стоит в 5 мм "
               "под полом: парковка на −10 000 мм раздувала "
               "model.stat.extent с 60 до 10 031 и портила камеры по "
               "всей сцене (objects.py)"),
        _f_seed(),
        _f_device(),
        _f_repeat("серия эпизодов: норма против перестановки антенн, оба "
                  "условия"),
        _f_repeat_t(),
        _f_repeat_seed0(),
        _f_out(),
        _f_title(),
    ),
    notes=(
        "Придуманными остаются ровно два места, и оба помечены в коде: "
        "«интенсивность запаха → частота рецепторов» и «асимметрия выхода → "
        "амплитуда шага». Заместителем грудного ганглия работает контроллер "
        "FlyGym.",
        "`BodySpec.sources` в `to_body_config` не попадает: источники "
        "запаха `body.run_episode` берёт отдельным параметром, и его "
        "передаёт `jobs._job_body`, а при пустом поле берётся "
        "`body.DEFAULT_SOURCES`. Поле задаёт число слотов источника, и "
        "песочница им пользуется; в «Эксперименте» оно не вынесено.",
        "`RunSpec.repeat_t` для тела работает только через окно: "
        "`jobs._series` передаёт `spec.episode_t`, а `cli.cmd_body` в серии "
        "берёт `args.t`.",
    ),
)


_LATERAL = Kind(
    key="lateral",
    title="Перебор популяций (открытый контур)",
    about="Муха стоит на месте, а обонятельный вход медленно качается синусом "
          "слева направо. Для каждой двусторонней популяции считается "
          "корреляция её асимметрии со стороной стимула — кто вообще кодирует "
          "сторону запаха.",
    fields=(
        _f_t(10.0, 0.5, 600.0, label="Длительность замера",
             hint="чем дольше, тем больше периодов качания попадёт в "
                  "корреляцию"),
        F(key="base_hz", block="lateral", label="Средняя частота рецепторов",
          type="float", default=75.0, min=0.0, max=MAX_POISSON_HZ / 2.0,
          unit="Гц", cli="--base", limits_from="код",
          hint="вход качается как `base_hz * (1 ± depth * sin)`, поэтому "
               "`base_hz * (1 + depth)` не должно превышать "
               f"{MAX_POISSON_HZ:g} Гц (предел `torch.bernoulli`)"),
        F(key="depth", block="lateral", label="Глубина качания", type="float",
          default=0.85, min=0.0, max=1.0, cli="--depth", limits_from="код",
          hint="0..1 по подсказке argparse; при 1 одна сторона на пике "
               "замолкает совсем"),
        F(key="period_s", block="lateral", label="Период качания", type="float",
          default=2.5, min=0.1, max=60.0, unit="с", cli="--period",
          limits_from="мой",
          hint="первый период отбрасывается при счёте корреляции: сети нужно "
               "осесть (`warm = min(n_steps // 3, период в шагах)`)"),
        F(key="control_ms", block="lateral", label="Шаг замера", type="float",
          default=20.0, min=LIF_DT_MS, max=200.0, unit="мс", limits_from="код",
          hint="шаг, на котором накапливается активность и качается вход",
          note="у команды `fly lateral` этого флага нет: "
               "`cli._spec_from_args` не кладёт `control_ms` в `LateralSpec`, "
               "так что из командной строки всегда 20 мс. Через окно поле "
               "работает — `jobs._job_lateral` читает `lat.control_ms`"),
        F(key="max_lag", block="lateral", label="Макс. лаг поиска", type="int",
          default=12, min=0, max=100, unit="шагов", limits_from="мой",
          hint="сигналу нужно время дойти, поэтому берётся лучшая корреляция "
               "по лагам 0..max_lag; в мс это `max_lag * control_ms`",
          note="флага в argparse тоже нет — только через окно"),
        F(key="top", block="lateral", label="Сколько строк показать",
          type="int", default=25, min=1, max=200, cli="--top",
          limits_from="мой",
          hint="только на вывод и на `summary['top']`; в lateral.json "
               "сохраняются все строки"),
        _f_seed(),
        _f_device(),
        _f_out(),
    ),
    notes=(
        "Замер нарочно вынесен из замкнутого цикла: в движущемся эпизоде выход "
        "рулит мухой, муха меняет стимул, и корреляция больше ничего не "
        "говорит о том, кто что кодирует.",
        "Это единственный вид, который ничего не кладёт в отчёт: ни "
        "report.html, ни кадров активности — только lateral.json.",
    ),
)


_VERIFY = Kind(
    key="verify",
    title="Сверка с апстримом",
    about="Прогнать опыт апстримного бенчмарка своим движком и тем же "
          "опытом — апстримным, и сравнить активные нейроны и частоты.",
    fields=(
        F(key="t", block=None, label="Длительность", type="choice",
          default=1.0, choices=VERIFY_T_VALUES, unit="с", cli="--t",
          cli_default=1.0, required=True, limits_from="код",
          choices_from="benchmark.T_RUN_VALUES_SEC из upstream/code",
          hint="одна и та же длительность уходит и своему движку, и "
               "апстримному `main.py --pytorch`. Список закрытый: апстрим "
               "принимает только эти значения, а на любом другом печатает "
               "ошибку и выходит с нулём, не посчитав ничего"),
        F(key="experiment", block="verify", label="Опыт апстрима",
          type="choice", default="sugar", choices=("sugar", "p9"),
          choices_from="benchmark.EXPERIMENTS из upstream/code/benchmark.py",
          hint="sugar — 21 вкусовой рецептор на 200 Гц, p9 — пара нисходящих "
               "DNp09 на 100 Гц"),
        _f_device(),
        _f_out(),
    ),
    notes=(
        "Шаг бина здесь не настраивается: `jobs._job_verify` и `cli.cmd_verify` "
        "жёстко ставят 50 мс, `n_trials = 1` и `seed = 0`.",
        "Частоты апстрима считаются делением его спайков на НАШУ "
        "длительность, поэтому сравнение идёт только с файлом "
        "`pytorch_t<t>s_n1.parquet` той же длительности. Файла другой "
        "длины не берут: частоты разошлись бы ровно в отношение "
        "длительностей.",
    ),
)


_AGENT = Kind(
    key="agent",
    title="Агент: модель сама ставит опыты",
    about="Локальная модель получает цель и сама выбирает, кого стимулировать "
          "и кого заглушить, порождая подпрогоны exp-NN; в конце пишет вывод "
          "по фактически посчитанному.",
    fields=(
        F(key="goal", block="agent", label="Цель", type="choice", default="",
          required=True, choices_from="spec.GOALS — четыре готовые "
          "формулировки; свободный ввод код принимает, но в окне его нет: "
          "про знаки препинания в событиях клавиатуры Taichi ничего не "
          "известно (комментарий в AgentSpec)",
          hint="пустая цель — ошибка «цель не задана» (jobs._job_agent)"),
        F(key="max_steps", block="agent", label="Шагов агента", type="int",
          default=6, min=1, max=50, cli="--steps", cli_default=6,
          limits_from="мой",
          hint="один шаг — ответ модели плюс, возможно, короткий прогон"),
        F(key="t_run_s", block="agent", label="Длительность опыта агента",
          type="float", default=0.5, min=0.05, max=10.0, unit="с", cli="--t",
          cli_default=0.5, limits_from="мой",
          hint="каждый порождённый опыт считается столько; длинные опыты "
               "умножаются на число шагов"),
        _f_model(),
        _f_out(),
    ),
    spawns_runs=True,
    notes=(
        "Вид порождает ПОДПРОГОНЫ, а не считает сам: `jobs._job_agent` зовёт "
        "`agent.run_loop`, а тот сохраняет каждый опыт папкой `exp-NN` со "
        "своим отчётом. Вложенных задач нарочно нет — задачу, которая "
        "порождает задачи, нечем было бы отменять.",
        "Ход прогона неизвестен: `jobs._job_agent` шлёт `progress = -1.0`, то "
        "есть «считаю, но сколько осталось — не знаю». Интерфейс должен уметь "
        "показать неопределённый ход.",
        "Длительность, сид, устройство и шаг бина агенту не задаются: он "
        "выбирает их сам внутри `agent.run_loop`.",
    ),
)


KINDS = {k.key: k for k in (_STIM, _ARENA, _BODY, _LATERAL, _VERIFY, _AGENT)}

# Настоящие прогоны: те пять, про которые и заведена схема (шапка spec.py).
REAL_KINDS = ("stim", "arena", "body", "lateral", "verify")


# ----------------------------------------------------------------- выход
#
# Файлы перечислены так, как их пишет код: `sim.save`, `loop.save`, `body.save`,
# ветка `worker._save` и `report.render`. Третий элемент — что внутри.


_SIM_FILES = (
    ("meta.json", "всегда", "настройки, стимулы, время счёта, число активных "
                            "и суммарные спайки (sim.save)"),
    ("rates.parquet", "всегда", "таблица активных нейронов (от 0,5 Гц): "
                                "root_id, метка, класс, область, частота"),
    ("bins.npz", "всегда", "растр активных нейронов по бинам, int16, плюс "
                           "bin_ms"),
)
_UI_FILES = (
    ("spec.json", "запуск из окна", "сама спека: по ней прогон повторяется "
                                    "одной кнопкой (worker._save)"),
    ("run.json", "запуск из окна", "исход: вид, номер задачи, отменён ли, "
                                   "заказанная длительность, когда кончился. "
                                   "Без него отменённый прогон на диске "
                                   "неотличим от законченного"),
)
_REPORT_FILE = ("report.html", "командная строка и агент; окно отчёт не пишет",
                "одностраничный отчёт с анимацией активности, областями и "
                "таблицей нейронов (report.render)")


MEASURES = {
    "stim": Measure(
        files=_SIM_FILES + (
            _REPORT_FILE,
            ("narrative.md", "при включённом разборе", "разбор прогона "
                                                       "локальной моделью"),
        ) + _UI_FILES,
        summary=(
            ("kind", "«stim»"),
            ("n_active", "нейронов с частотой от 0,5 Гц"),
            ("total_spikes", "сумма спайков по всем нейронам"),
            ("timing", "setup_s, sim_s, steps, device, realtime_ratio"),
            ("cancelled", "остановлен ли прогон досрочно"),
            ("elapsed_s", "время задачи по стене (jobs.run)"),
        ),
        series=(),
        activity_frames=True,
        picture_frames=False,
        index_kind="run",
    ),
    "arena": Measure(
        files=(
            ("episode.json", "одиночный эпизод", "арена, параметры тела, "
                                                 "настройки, метрики, "
                                                 "комментарии (loop.save)"),
            ("track.parquet", "одиночный эпизод", "запись по шагам: позиция, "
                                                  "курс, скорость, поворот, "
                                                  "запах, выход, спайки"),
            ("probes.json", "при записи популяций по сторонам",
             "корреляция асимметрии входа с асимметрией каждой популяции"),
            ("bench.json", "серия", "по условию: n_runs, среднее, sd, sem "
                                    "совпадения с градиентом, прогоны "
                                    "поимённо, смещение до еды, визиты"),
        ) + _SIM_FILES + (_REPORT_FILE,) + _UI_FILES,
        summary=(
            ("kind", "«arena», в серии «arena-bench»"),
            ("metrics", "steps, grad_alignment, turn_to_odour, "
                        "turn_steps_counted, food_dist_first_half_mm, "
                        "food_dist_second_half_mm, food_visits, food_steps, "
                        "mean_speed_mm_s, mean_dn_hz, wall_bounces"),
            ("timing", "setup_s, sim_s, steps, device, realtime_ratio"),
            ("bench", "только в серии: оба условия целиком"),
            ("cancelled", "остановлен ли эпизод досрочно"),
            ("elapsed_s", "время задачи по стене"),
        ),
        series=("dn_l", "dn_r", "odour_l", "odour_r", "v", "omega",
                "grad_cos", "food_dist"),
        activity_frames=True,
        picture_frames=False,
        index_kind="episode, в серии bench",
    ),
    "body": Measure(
        files=(
            ("embodied.json", "одиночный эпизод", "настройки, источники "
                                                  "запаха, метрики, "
                                                  "калибровка (body.save)"),
            ("track.parquet", "одиночный эпизод", "запись по шагам: позиция, "
                                                  "курс, запах, выход, "
                                                  "амплитуды ног, расстояние"),
            ("walk.mp4", "при записи видео", "ходьба сверху, камера за курсом"),
            ("bench.json", "серия", "по условию: n_runs, среднее, sd, sem "
                                    "совпадения с градиентом, смещение до еды, "
                                    "шаги на еде"),
        ) + _SIM_FILES + (_REPORT_FILE,) + _UI_FILES,
        summary=(
            ("kind", "«body», в серии «body-bench»"),
            ("metrics", "steps, grad_alignment, dist_first_half_mm, "
                        "dist_second_half_mm, dist_start_mm, dist_end_mm, "
                        "path_mm, speed_mm_s, food_steps, mean_dn_hz"),
            ("calibration", "odour_ref, dn_ref_hz, asym_bias, asym_noise, "
                            "warm_steps — что намерено на разогреве"),
            ("timing", "setup_s, sim_s, device, realtime_ratio"),
            ("bench", "только в серии"),
            ("cancelled", "остановлен ли эпизод досрочно"),
            ("picture_frames", "кадров картинки отдано окну (worker)"),
            ("render_ms", "среднее время кадра картинки, мс"),
            ("elapsed_s", "время задачи по стене"),
        ),
        series=("dn_l", "dn_r", "odour_l", "odour_r", "amp_l", "amp_r",
                "dist", "grad_cos"),
        activity_frames=True,
        picture_frames=True,
        index_kind="embodied, в серии bench",
    ),
    "lateral": Measure(
        files=(
            ("lateral.json", "всегда", "по строке на популяцию: имя, "
                                       "корреляция r, лаг в мс, сигма "
                                       "асимметрии, средняя частота, размеры "
                                       "сторон, селекторы"),
        ) + _UI_FILES,
        summary=(
            ("kind", "«lateral»"),
            ("n", "сколько популяций просчитано"),
            ("top", "первые `top` строк, отсортированных по |r|"),
            ("cancelled", "остановлен ли замер досрочно"),
            ("elapsed_s", "время задачи по стене"),
        ),
        series=(),
        activity_frames=False,
        picture_frames=False,
        index_kind="lateral",
    ),
    "verify": Measure(
        files=(("verify.json", "всегда", "паспорт сверки: версия данных, "
                                         "параметры, Jaccard, корреляция, "
                                         "активные с обеих сторон, исход и "
                                         "ограничения"),) + _SIM_FILES + _UI_FILES,
        summary=(
            ("kind", "«verify»"),
            ("rc", "код возврата апстримного процесса"),
            ("n_active", "активных нейронов в своём прогоне"),
            ("total_spikes", "спайков в своём прогоне"),
            ("elapsed_s", "время задачи по стене"),
        ),
        series=(),
        activity_frames=False,
        picture_frames=False,
        index_kind="run",
    ),
    "agent": Measure(
        files=(
            ("journal.md", "всегда", "журнал: цель, шаги, мысли, сводки "
                                     "опытов, вывод"),
            ("report.html", "если был хотя бы один опыт", "отчёт по последнему "
                                                          "опыту плюс список "
                                                          "фактически "
                                                          "посчитанного"),
            ("exp-NN/", "по опыту", "каждый опыт — отдельный прогон стимуляции "
                                    "со своими meta.json, rates.parquet, "
                                    "bins.npz и report.html"),
        ),
        summary=(
            ("kind", "«agent»"),
            ("goal", "цель, как она была задана"),
            ("answer", "вывод модели по фактическим измерениям"),
            ("experiments", "сколько опытов провёл"),
            ("dir", "папка прогона"),
            ("elapsed_s", "время задачи по стене"),
        ),
        series=(),
        activity_frames=False,
        picture_frames=False,
        index_kind="agent",
    ),
}


# Поля dataclass'ов, которые в интерфейс вынесены сознательно НЕ были.
# Перечислены здесь, а не умолчаны, чтобы расхождение со `dataclasses.fields`
# было видно глазами, а не всплывало проверкой.
NOT_IN_UI = {
    ("RunSpec", "kind"): "задаётся выбором самого вида, а не полем",
    ("RunSpec", "version"): "версия схемы, служебное",
    ("RunSpec", "about"): "строка-пояснение для человека; её пишут готовые "
                          "опыты (spec.DEMOS), а не настройщик",
    ("RunSpec", "open_report"): "открывать ли отчёт в браузере. Окно отчётов "
                                "не открывает и ставит False "
                                "(комментарий в RunSpec)",
    ("RunSpec", "max_bins"): "предохранитель окна: при ненулевом значении "
                             "`jobs._job_stim` отказывается считать слишком "
                             "длинный прогон. Ставит его интерфейс, а не "
                             "человек; `validate` эту проверку повторяет",
    ("RunSpec", "trace"): "отладочный след, не научный параметр",
    ("RunSpec", "stim"): "сам блок, а не поле",
    ("RunSpec", "arena"): "сам блок, а не поле",
    ("RunSpec", "body"): "сам блок, а не поле",
    ("RunSpec", "lateral"): "сам блок, а не поле",
    ("RunSpec", "verify"): "сам блок, а не поле",
    ("RunSpec", "agent"): "сам блок, а не поле",
    ("BodySpec", "sources"): "задаёт слоты источника запаха; идёт не через "
                             "`to_body_config`, а параметром "
                             "`body.run_episode` из `jobs._job_body`. В "
                             "«Эксперименте» не вынесено: число слотов "
                             "выбирают в «Песочнице», а научному эпизоду "
                             "хватает `body.DEFAULT_SOURCES`",
    ("BodySpec", "live_initial"): "исходные условия сеанса песочницы: тот же "
                                  "словарь, что принимает "
                                  "`live.BodyLive.apply`. В «Эксперименте» не "
                                  "вынесено — его набирают в «Песочнице» "
                                  "слотами источника, и форма опыта не должна "
                                  "повторять её рейку",
    ("BodySpec", "keep_s"): "вынесено: «Окно записи»",
    ("BodySpec", "odour_ref"): "вынесено: «Яркость насыщения»",
    # Пять ручек моторного пути. Сам `motor` ВЫНЕСЕН — выбор
    # моторного режима есть требование к продукту, а не отладочная
    # мелочь. А остальные четыре следуют за ним или правят МОДЕЛЬ,
    # а не условие опыта.
    ("BodySpec", "motor_weights"): "путь к весам для `motor=\"dn_trained\"`. В "
                                   "форме опыта не вынесено по двум "
                                   "причинам: это путь к файлу, а не "
                                   "величина, и годного файла весов в "
                                   "дереве нет вовсе — выбирать не из "
                                   "чего. Пока файл не появится, "
                                   "`dn_trained` задаётся спекой или "
                                   "командной строкой",
    ("BodySpec", "gf_bridge"): "мост через обрыв Giant Fiber. Это правка "
                               "модели, а не условие опыта: DNp01 в v783 "
                               "не даёт ни одной связи на 110 мотонейронов, "
                               "и мост достраивает то, чего в данных нет. "
                               "Доказанным путём реакции на нависание "
                               "он не является, и в форме опыта ему места "
                               "нет: задаётся спекой",
    ("BodySpec", "head_stab"): "стабилизация головы. Не режим, а ДРУГАЯ "
                               "МЕХАНИЧЕСКАЯ МОДЕЛЬ: `Fly` добавляет два "
                               "привода на шейные суставы, которых без "
                               "неё в модели нет вовсе. Два таких эпизода "
                               "несравнимы между собой, и форма опыта "
                               "не должна предлагать это как галочку",
    ("BodySpec", "neck_kp"): "жёсткость шейных приводов. Смысл есть "
                             "только при `head_stab`, и выносить его "
                             "отдельно от неё значило бы предлагать "
                             "настройку для выключенной части модели",
    ("BodySpec", "spmv"): "каким кодом считать умножение на матрицу связей: "
                          "«событийный» (по умолчанию) или "
                          "«эталонный». В форме опыта не вынесено "
                          "нарочно: дороги ЧИСЛЕННО РАВНОЗНАЧНЫ "
                          "побитово (все четыре сочетания "
                          "дорога×графы дают один отпечаток, "
                          "`probe/lif_event.py`), то есть на результат "
                          "выбор не влияет и условием опыта не "
                          "является. Выбирать его надо не под опыт, а "
                          "под машину, поэтому задаётся окружением "
                          "`FLYBRAIN_SPMV` сразу на все вызовы, флагом "
                          "`--spmv` или полем спеки. `nvcc` и MSVC для "
                          "этого НЕ нужны: ядро едет в дереве готовым "
                          "PTX и догружается JIT драйвера "
                          "(`KERNEL_SHIP.md`). Если ядро загрузить "
                          "нечем, движок говорит об этом ГРОМКО "
                          "(`evspmv.EventKernelUnavailable`), а в окне это "
                          "видно как ошибка задачи",
    ("BodySpec", "vis_track"): (
        "заданная траектория зрительных объектов: список отрезков "
        "«слот, с какого по какое модельное время, из какой точки в "
        "какую, радиус, серость» (`objects.Track`). В форме опыта не "
        "вынесено по той же причине, по которой не вынесены слоты "
        "источника запаха: это рейка, а не величина. Задаётся тремя "
        "способами, и у каждого своё место. ГОТОВЫМ ПРЕСЕТОМ — "
        "`spec.VISION_PRESETS` и `fly body --vision-preset loom-dark`: "
        "так получаются тёмное и светлое нависание, движение поперёк "
        "курса и неподвижный объект, причём парами, отличающимися ровно "
        "одним. РУКАМИ В «ПЕСОЧНИЦЕ» — ручка `objects` живых условий, с "
        "подтверждением по модельному времени. ФАЙЛОМ — "
        "`fly body --track путь.json`. В сохранённый прогон уезжает "
        "развёрнутый список отрезков, а не имя пресета: имя завтра может "
        "значить другое, а прогон обязан говорить, что считалось"),
    ("BodySpec", "probes"): "снимать ли частоты всех зондов по ходу "
                            "эпизода. Это запись, а не условие опыта: на "
                            "счёт она не влияет, а выключается только "
                            "затем, чтобы долгий сеанс не платил за "
                            "десяток ненужных рядов",
}

# Поля таймлайна (`flybrain.timeline`) в «Эксперимент» не вынесены, и
# причина у всех трёх одна: таймлайн — это запись прогона, а не условие
# опыта. Менять его настройки из формы опыта значило бы предлагать
# человеку выбор, который на результат не влияет; смотрят таймлайн в
# «Анализе», а окно его записи выбирают там же, где и окно записи
# активности, — в «Песочнице».
for _block in ("ArenaSpec", "BodySpec"):
    NOT_IN_UI[(_block, "timeline")] = (
        "писать ли единый таймлайн модельного времени. Это запись "
        "прогона, а не условие опыта: по умолчанию да, и выключается "
        "только затем, чтобы прогон лёг в точности теми же файлами, что "
        "прежние")
    NOT_IN_UI[(_block, "timeline_keep_s")] = (
        "окно таймлайна, с. Ноль значит «взять окно записи активности, а "
        "если и его нет — `timeline.DEFAULT_KEEP_S`». Отдельным полем в "
        "форме опыта не стоит: строка таймлайна стоит сотню байт против "
        "277 КБ у плотной записи, и выбирать между ними человеку не о чем")
    NOT_IN_UI[(_block, "capture_at_ms")] = (
        "модельное время, на котором снять состояние для продолжения, мс. "
        "Ставит его не форма опыта, а то место, откуда продолжают: "
        "`probe/resume_state.py` и «Песочница». Пусто — не снимать")
del _block


# --------------------------------------------------------------- доступ


def fields_of(kind):
    """Поля вида, в порядке показа."""
    k = KINDS.get(kind)
    return k.fields if k else ()


def field_map(kind):
    """`{(блок, ключ): Field}` для вида."""
    return {(f.block, f.key): f for f in fields_of(kind)}


def measures_of(kind):
    return MEASURES.get(kind)


def series_of(kind):
    """Имена рядов, приходящих в окно по ходу прогона.

    Взяты из колбэков `on_step` в `jobs.py`, где они перечислены явно; ряд
    попадает в окно только если такой ключ есть в записи шага.
    """
    m = MEASURES.get(kind)
    return tuple(m.series) if m else ()


def defaults(kind, source="spec"):
    """Словарь спеки, собранный по справочнику.

    `source="spec"` берёт значения из dataclass'ов (их и сверяет проверка),
    `source="cli"` — рекомендованные для этого вида значения из argparse.
    """
    out = {"kind": kind, "version": VERSION}
    for f in fields_of(kind):
        val = f.shown_default if source == "cli" else f.default
        if f.type == "list":
            val = list(val or [])
        if f.block is None:
            out[f.key] = val
        else:
            out.setdefault(f.block, {})[f.key] = val
    return out


def demo_spec_dict(kind, source="cli"):
    """То же, но с одним осмысленным стимулом: чтобы спеку можно было пустить."""
    d = defaults(kind, source=source)
    if kind == "stim":
        term = {f.key: f.default for f in STIM_TERM_FIELDS}
        term["query"] = "class:gustatory+side:left"
        term["name"] = "сахар слева"
        d["stim"]["activate"] = [term]
    elif kind == "agent":
        from . import spec as S

        d["agent"]["goal"] = S.GOALS[0]
    return d


# -------------------------------------------------------------- арифметика


# Предел на один тензор `bins`. Взят не из вкуса: на этой машине 24 ГБ
# видеопамяти, из них матрица связей и состояние движка занимают около
# 350 МБ, а всё остальное — запас под сам расчёт. Четыре гигабайта под
# запись проходят, десять уже упираются.
MAX_BINS_GB = 4.0

# У сверки с апстримом предел другой, и это не поблажка. Длительность
# там диктует апстрим (`benchmark.T_RUN_VALUES_SEC`), а бин жёстко
# поставлен в 50 мс (`jobs._job_verify`) — то есть ни сократить
# длительность, ни увеличить бин нельзя, выхода нет вовсе. Прежде
# сверка на 1000 с (5,5 ГБ) пускалась, и отказывать ей, советуя
# невозможное, значит потерять научный прогон. Двенадцать гигабайт —
# это то, что на этой машине ещё умещается рядом с матрицей связей.
MAX_BINS_GB_FIXED = 12.0


def n_bins_of(spec_dict):
    """Сколько строк будет в плотном `bins`.

    Формулы разные и обе из кода: у стимуляции `n_bins = ceil(steps/bin_steps)`
    с `steps = round(t*1000/DT)` и `bin_steps = max(1, round(bin_ms/DT))`
    (`sim.run`), у арены и тела — `n_steps = round(t / (control_ms/1000))`
    (`loop.run_episode`, `body.run_episode`).
    """
    kind = spec_dict.get("kind", "stim")
    t = float(_get(spec_dict, None, "t", 1.0) or 0.0)
    if kind == "stim":
        bin_ms = float(_get(spec_dict, None, "bin_ms", 10.0) or 10.0)
        steps = int(round(t * 1000.0 / LIF_DT_MS))
        bin_steps = max(1, int(round(bin_ms / LIF_DT_MS)))
        return int(math.ceil(steps / bin_steps)) if steps else 0
    if kind == "verify":
        # `jobs._job_verify` ставит бин 50 мс жёстко.
        steps = int(round(t * 1000.0 / LIF_DT_MS))
        bin_steps = max(1, int(round(50.0 / LIF_DT_MS)))
        return int(math.ceil(steps / bin_steps)) if steps else 0
    if kind == "arena":
        control_ms = float(_get(spec_dict, None, "bin_ms", 20.0) or 20.0)
        rows = max(0, int(round(t * 1000.0 / control_ms)))
        # Кольцо держит только последние `keep_s` секунд, значит и памяти
        # под `bins` нужно столько же, а не на всю сессию.
        keep_s = float(_get(spec_dict, "arena", "keep_s", 0.0) or 0.0)
        if keep_s > 0:
            rows = min(rows, max(1, int(round(keep_s * 1000.0 / control_ms))))
        return rows
    if kind == "body":
        control_ms = float(_get(spec_dict, "body", "control_ms", 20.0) or 20.0)
        keep_s = float(_get(spec_dict, "body", "keep_s", 0.0) or 0.0)
        if t <= 0:
            # Сеанс без конца: памяти нужно ровно под кольцо, а не под
            # предел шагов — иначе оценка показывала бы терабайты.
            if keep_s <= 0:
                return 0
            return max(1, int(round(keep_s * 1000.0 / control_ms)))
        rows = max(0, int(round(t * 1000.0 / control_ms)))
        if keep_s > 0:
            rows = min(rows, max(1, int(round(keep_s * 1000.0 / control_ms))))
        return rows
    return 0


def bins_bytes(spec_dict, n_neurons=N_NEURONS):
    """Память под плотный `bins`, байты.

    В коде это `torch.zeros(n_bins, N, dtype=torch.int16)` — два байта на
    нейрона на бин, и тензор живёт на том устройстве, на котором считают.
    Ни `lateral`, ни `agent` такого тензора не держат.
    """
    return n_bins_of(spec_dict) * int(n_neurons) * 2


def estimate_wall_s(spec_dict, n_neurons=N_NEURONS):
    """Сколько прогон займёт по стене, секунды (оценка).

    Для стимуляции, арены, сверки и перебора основание — та же константа, что
    в `jobs.estimate`: 1167 шагов движка в секунду. Для тела — замер из
    `probe/bodyfinal_timing.json`, где шаг управления разложен на мозг и физику.
    Загрузка данных сюда не входит: работник грузит коннектом один раз.
    """
    kind = spec_dict.get("kind", "stim")
    t = float(_get(spec_dict, None, "t", 1.0) or 0.0)
    if kind == "body" and t <= 0.0:
        # Сеанс без заранее заданного конца: его заканчивает остановка,
        # и назвать время нечем. Ноль шагов давал оценку «сборка плюс
        # один шаг», то есть конечное число там, где честный ответ —
        # «предела нет».
        return None
    lif_s = 1.0 / ENGINE_STEPS_PER_S

    if kind in ("stim", "verify"):
        trials = int(_get(spec_dict, "stim", "trials", 1) or 1)
        steps = t * 1000.0 / LIF_DT_MS
        # Партия идёт одним тензором, но время всё равно растёт с ней; берём
        # линейно — это оценка, а не замер.
        base = steps * lif_s * max(1, trials)
        return base * (2.0 if kind == "verify" else 1.0)

    if kind == "arena":
        control_ms = float(_get(spec_dict, None, "bin_ms", 20.0) or 20.0)
        ep_t = float(_get(spec_dict, None, "repeat_t", None) or t)
        repeat = int(_get(spec_dict, None, "repeat", 0) or 0)
        n_ep = (repeat * 2) if repeat else 1
        warm_s = 0.0
        a = spec_dict.get("arena") or {}
        if a.get("dn_ref_hz") is None or a.get("turn_gain_rad_s") is None:
            tau = a.get("dn_tau_s")
            tau = 0.12 if tau is None else float(tau)
            warm = max(5, int(round(3000.0 * tau / control_ms)))
            warm_s = warm * (control_ms / LIF_DT_MS) * lif_s
        per_ep = ep_t * 1000.0 / LIF_DT_MS * lif_s + warm_s
        return n_ep * per_ep

    if kind == "body":
        b = spec_dict.get("body") or {}
        control_ms = float(b.get("control_ms") or 20.0)
        physics_dt = float(b.get("physics_dt") or 2e-4)
        tau = float(b.get("dn_tau_s") or 0.5)
        ep_t = float(_get(spec_dict, None, "repeat_t", None) or t)
        repeat = int(_get(spec_dict, None, "repeat", 0) or 0)
        n_ep = (repeat * 2) if repeat else 1
        lif_steps = max(1, int(round(control_ms / LIF_DT_MS)))
        phys_steps = max(1, int(round(control_ms / 1000.0 / physics_dt)))
        step_ms = lif_steps * _BODY_LIF_MS + phys_steps * _BODY_PHYS_MS
        n_steps = max(1, int(round(ep_t * 1000.0 / control_ms)))
        warm = max(5, int(round(3000.0 * tau / control_ms)))
        warm_s = warm * lif_steps * _BODY_LIF_MS / 1000.0
        return n_ep * (_BODY_PREP_S + warm_s + n_steps * step_ms / 1000.0)

    if kind == "lateral":
        lat = spec_dict.get("lateral") or {}
        control_ms = float(lat.get("control_ms") or 20.0)
        steps = t * 1000.0 / control_ms
        lif = steps * (control_ms / LIF_DT_MS) * lif_s
        # Перебор популяций — те же 14 с, что в замере тела: это один и тот же
        # `world.probe_candidates` на 138 639 строк.
        return lif + BODY_MEASURED["перебор_популяций_с"]

    if kind == "agent":
        a = spec_dict.get("agent") or {}
        steps = int(a.get("max_steps") or 6)
        t_run = float(a.get("t_run_s") or 0.5)
        # Как в `jobs.estimate`: шаг — это ответ модели (12 с) плюс короткий
        # прогон.
        return steps * (12.0 + t_run * 1000.0 / LIF_DT_MS * lif_s)

    return 0.0


def _get(d, block, key, fallback=None):
    if block is None:
        return d.get(key, fallback)
    sub = d.get(block) or {}
    return sub.get(key, fallback)


def _fmt_s(sec):
    if sec < 60:
        return f"{sec:.0f} с"
    if sec < 3600:
        return f"{sec / 60:.1f} мин"
    return f"{sec / 3600:.1f} ч"


# ------------------------------------------------------------- проверка


_EMPTY_COLUMNS = ("cell_type", "hemibrain_type", "label", "cell_class",
                  "cell_sub_class", "super_class", "region", "super_region",
                  "top_nt", "side", "nerve", "ito_lee_hemilineage")


def _empty_neurons():
    """Пустая таблица нейронов нужной формы.

    Нужна, чтобы разобрать селектор, не загружая коннектом: `data.select` на
    ней честно падает на неизвестном ключе, на битом регулярном выражении и на
    нечисловом `idx:`, а на правильном селекторе отдаёт пустой массив.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({c: pd.Series([], dtype=object) for c in _EMPTY_COLUMNS})
    df["root_id"] = pd.Series([], dtype=np.int64)
    df["idx"] = pd.Series([], dtype=np.int64)
    return df


def check_selector(query):
    """Разобрать селектор. Возвращает None, если он синтаксически годен."""
    from . import data as D

    text = "" if query is None else str(query).strip()
    if not text:
        return "пустой селектор"
    try:
        D.select(_empty_neurons(), text)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:                                  # noqa: BLE001
        # Битое регулярное выражение приходит как `re.error`, а не ValueError.
        return f"{exc.__class__.__name__}: {exc}"
    return None


def _num_problem(f, val):
    """Число вне типа или вне пределов. Возвращает фразу или None."""
    if val is None:
        if f.none_means:
            return None
        return f"{f.label}: значение не задано"
    if f.type == "int":
        if isinstance(val, bool) or not isinstance(val, int):
            return f"{f.label}: ожидалось целое, пришло {val!r}"
    elif f.type == "float":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return f"{f.label}: ожидалось число, пришло {val!r}"
    unit = f" {f.unit}" if f.unit else ""
    if f.min is not None and val < f.min:
        return (f"{f.label}: {val:g}{unit} меньше минимума "
                f"{f.min:g}{unit} (предел — {f.limits_from})")
    if f.max is not None and val > f.max:
        return (f"{f.label}: {val:g}{unit} больше максимума "
                f"{f.max:g}{unit} (предел — {f.limits_from})")
    return None


def validate(spec_dict):
    """Проверить параметры перед запуском. Пустой список — можно пускать.

    Проверяется то, что проверяемо без работника и без коннектома: вид известен,
    обязательные поля на месте, числа в пределах, селекторы разбираются,
    сочетания не противоречат друг другу. Сколько нейронов найдёт запрос и есть
    ли такая популяция в `world.probe_candidates` — здесь не узнать, это
    проверит сам прогон.
    """
    errs = []
    d = dict(spec_dict or {})
    kind = d.get("kind")
    if kind not in KINDS:
        known = ", ".join(KINDS)
        errs.append(f"неизвестный вид прогона {kind!r}; известны: {known}")
        return errs

    # --- числа и обязательность по справочнику ---
    for f in fields_of(kind):
        present = (f.key in d) if f.block is None else (
            f.key in (d.get(f.block) or {}))
        val = _get(d, f.block, f.key, None)
        if f.type in ("int", "float"):
            if present or f.required:
                p = _num_problem(f, val)
                if p:
                    errs.append(p)
        elif f.type == "bool":
            if present and not isinstance(val, bool):
                errs.append(f"{f.label}: ожидалось да/нет, пришло {val!r}")
        elif f.type == "choice":
            if present and val is not None and f.choices and val not in f.choices:
                ok = ", ".join(str(c) for c in f.choices)
                errs.append(f"{f.label}: {val!r} не из списка ({ok})")
            if f.required and not val and val != 0:
                errs.append(f"{f.label}: обязательное поле не заполнено")
        elif f.type == "query":
            if f.required or (present and val):
                p = check_selector(val)
                if p:
                    errs.append(f"{f.label}: {p}")
        elif f.type == "list":
            items = val or []
            if f.required and not items:
                errs.append(f"{f.label}: нужен хотя бы один")
            if f.max is not None and len(items) > f.max:
                errs.append(f"{f.label}: {len(items)} штук, больше {f.max} "
                            f"не предусмотрено (предел — {f.limits_from})")

    # --- сеанс без заданного конца ---
    if kind == "body":
        t_v = float(d.get("t") or 0.0)
        keep = float(_get(d, "body", "keep_s", 0.0) or 0.0)
        if t_v == 0.0 and keep <= 0.0:
            errs.append(
                "длительность 0 — это сеанс без заранее заданного конца, и "
                "он требует окна записи: иначе `bins` растёт без предела "
                "(277 КБ на шаг управления, 13,9 МБ на секунду модельного "
                "времени). Укажите окно записи")
        elif 0.0 < t_v < 0.1:
            errs.append(
                "длительность %g с короче одного разогрева: калибровка "
                "занимает несколько шагов управления. Либо 0 (сеанс без "
                "конца), либо не меньше 0,1 с" % t_v)

    # --- плотный `bins` должен уместиться в памяти ---
    #
    # Пока у арены был предел 600 с, оценка памяти была предупреждением:
    # худшее, что можно было заказать, — 4,2 ГБ. Длинная сессия подняла
    # предел до двух часов, и там получается под сотню гигабайт: это уже
    # не предупреждение, а отказ. Выход назван в самом отказе, потому что
    # он есть — окно записи.
    if kind in ("stim", "arena", "body", "verify"):
        gb = bins_bytes(d) / 1e9
        cap = MAX_BINS_GB_FIXED if kind == "verify" else MAX_BINS_GB
        if gb > cap:
            # Шаг берётся у того вида, о котором речь: у тела это
            # `body.control_ms`, а не общий `bin_ms`, и прежний текст
            # отказа называл чужое число.
            step = (float(_get(d, "body", "control_ms", 20.0) or 20.0)
                    if kind == "body"
                    else (50.0 if kind == "verify"
                          else float(_get(d, None, "bin_ms", 20.0) or 20.0)))
            why = ("длительность %g с при шаге %g мс"
                   % (float(d.get("t") or 0.0), step))
            if kind == "arena":
                how = (" Укажите окно записи: тогда `bins` держит только "
                       "последние секунды, а счётчики и частоты остаются "
                       "за всю сессию.")
            elif kind == "body":
                how = (" Укажите окно записи, увеличьте шаг управления "
                       "или сократите длительность.")
            elif kind == "verify":
                # Советовать тут нечего, и делать вид, что есть, нельзя.
                how = (" Ни длительность, ни бин у сверки не наши: "
                       "длительность задаёт апстрим, бин 50 мс поставлен "
                       "жёстко. Выхода, кроме более короткой сверки из "
                       "списка апстрима, нет.")
            else:
                how = " Сократите длительность или увеличьте шаг бина."
            errs.append(
                "запись активности не уместится: %s даёт %.1f ГБ в одном "
                "тензоре `bins`, предел — %g ГБ.%s" % (why, gb, cap, how))

    # --- сверка: длительность не наша, а апстрима ---
    if kind == "verify":
        t_v = d.get("t")
        if t_v is None or float(t_v) not in VERIFY_T_VALUES:
            errs.append(
                "длительность %s апстрим не примет: `benchmark."
                "T_RUN_VALUES_SEC` разрешает только %s. На другом значении "
                "`upstream/main.py` печатает ошибку и выходит с НУЛЁМ, не "
                "посчитав ничего, — и сравнивать было бы не с чем"
                % (("%g с" % float(t_v)) if t_v is not None else "не задана",
                   ", ".join("%g" % v for v in VERIFY_T_VALUES)))

    # --- стимулы: каждый терм отдельно ---
    if kind == "stim":
        t_ms = float(d.get("t") or 0.0) * 1000.0
        for i, raw in enumerate(((d.get("stim") or {}).get("activate") or []), 1):
            term = dict(raw) if not isinstance(raw, dict) else raw
            for tf in STIM_TERM_FIELDS:
                tval = term.get(tf.key, tf.default)
                if tf.type == "query":
                    p = check_selector(tval)
                    if p:
                        errs.append(f"стимул {i}: {p}")
                elif tf.type in ("int", "float"):
                    p = _num_problem(tf, tval)
                    if p:
                        errs.append(f"стимул {i}: {p}")
            on = float(term.get("t_on_ms") or 0.0)
            off = term.get("t_off_ms")
            if off is not None and float(off) <= on:
                errs.append(f"стимул {i}: выключение {float(off):g} мс не позже "
                            f"включения {on:g} мс — стимул не загорится")
            if t_ms and on >= t_ms:
                errs.append(f"стимул {i}: включение на {on:g} мс выходит за "
                            f"прогон длиной {t_ms:g} мс")
        max_bins = int(d.get("max_bins") or 0)
        if max_bins:
            n_bins = n_bins_of(d)
            if n_bins * N_NEURONS > max_bins:
                errs.append(
                    f"слишком длинный прогон: {n_bins} бинов на {N_NEURONS} "
                    "нейронов — увеличьте бин или сократите длительность "
                    "(та же проверка в jobs._job_stim)")

    # --- тело: шаг физики против шага управления ---
    if kind == "body":
        b = d.get("body") or {}
        control_ms = float(b.get("control_ms") or 20.0)
        physics_dt = float(b.get("physics_dt") or 2e-4)
        if physics_dt > control_ms / 1000.0:
            errs.append(
                f"шаг физики {physics_dt:g} с крупнее шага управления "
                f"{control_ms:g} мс: на шаг придётся один шаг MuJoCo "
                "(`body_steps = max(1, round(...))`)")

    # --- перебор популяций: вход не должен уходить за предел Пуассона ---
    if kind == "lateral":
        lat = d.get("lateral") or {}
        base = float(lat.get("base_hz") or 0.0)
        depth = float(lat.get("depth") if lat.get("depth") is not None else 0.85)
        peak = base * (1.0 + depth)
        if peak > MAX_POISSON_HZ:
            errs.append(
                f"на пике качания вход {peak:g} Гц, а `torch.bernoulli` "
                f"принимает не больше {MAX_POISSON_HZ:g} Гц: уменьшите "
                "среднюю частоту или глубину")

    # --- агент ---
    if kind == "agent":
        if not str((d.get("agent") or {}).get("goal") or "").strip():
            errs.append("цель не задана (так же падает jobs._job_agent)")

    return errs


# ---------------------------------------------------------- предупреждения


def limits(spec_dict, n_neurons=N_NEURONS):
    """Предупреждения, не запрещающие пуск: во что обойдётся такой прогон.

    Числа считаются по формулам из кода и по замерам, на которые есть ссылка;
    ничего не придумано. Первые строки всегда про память и время — интерфейс
    должен их показывать до кнопки «пуск», а не после.
    """
    out = []
    d = dict(spec_dict or {})
    kind = d.get("kind")
    if kind not in KINDS:
        return [f"неизвестный вид прогона {kind!r}: оценить нечего"]

    repeat = int(d.get("repeat") or 0)
    n_ep = (repeat * 2) if repeat else 1

    # --- память под плотный bins ---
    #
    # Три разных числа, которые легко смешать и которые раньше показывались
    # одной строкой: размер ОДНОГО массива, пик памяти процесса со счётом и
    # размер того, что ляжет на диск. Известно из них только первое.
    if kind in ("stim", "arena", "body", "verify"):
        n_bins = n_bins_of(d)
        mb = bins_bytes(d, n_neurons) / 1e6
        out.append(f"оценка массива `bins`: {n_bins} × {n_neurons} × int16 = "
                   f"{mb:.1f} МБ. Это размер одного тензора на устройстве "
                   "счёта, не пик памяти работника и не размер файла")
        if mb > 2000:
            out.append(f"{mb / 1000:.1f} ГБ под один тензор — на видеокарте это "
                       "почти наверняка нехватка памяти; увеличьте шаг бина "
                       "или сократите длительность")
        elif mb > 500:
            out.append(f"{mb:.0f} МБ под один тензор: на видеокарте с 8 ГБ это "
                       "уже заметно рядом с матрицей связей")
        out.append("пик памяти работника больше этой оценки и окном не "
                   "измеряется: рядом с `bins` живут матрица связей, "
                   "состояние движка и, у тела, модель MuJoCo")
        out.append("размер сохранённого файла меньше и заранее неизвестен: "
                   "`bins.npz` пишется сжатым, и чем реже спайки, тем он "
                   "меньше — назвать число до счёта нечестно")
    else:
        out.append("плотного `bins` этот вид не держит: растр активности не "
                   "записывается, оценивать под него память нечего")

    # --- время по стене ---
    sec = estimate_wall_s(d, n_neurons)
    if sec is None:
        out.append("оценку времени счёта дать нечем: сеанс без заранее "
                   "заданного конца идёт до остановки")
        return out
    out.append(f"оценка времени счёта: {_fmt_s(sec)}"
               + (f" на {n_ep} эпизодов серии" if repeat else "")
               + " (без загрузки коннектома: работник грузит её один раз)")
    if sec > 1800:
        out.append("больше получаса: стоит сократить длительность или число "
                   "эпизодов")

    if repeat:
        ep_t = float(d.get("repeat_t") or d.get("t") or 0.0)
        out.append(f"серия считает оба условия — норму и перестановку антенн, "
                   f"{repeat} × 2 = {n_ep} эпизодов по {ep_t:g} с; отчёт и "
                   "эпизоды не сохраняются, только bench.json")
    elif d.get("repeat_t") is not None:
        out.append("задана длительность эпизода серии, но серия выключена "
                   "(`repeat = 0`): это число никуда не пойдёт")

    # --- арена ---
    if kind == "arena":
        a = d.get("arena") or {}
        control_ms = float(d.get("bin_ms") or 20.0)
        tau = a.get("dn_tau_s")
        tau_v = 0.12 if tau is None else float(tau)
        measures = a.get("dn_ref_hz") is None or a.get("turn_gain_rad_s") is None
        if measures:
            warm = max(5, int(round(3000.0 * tau_v / control_ms)))
            warm_s = warm * (control_ms / LIF_DT_MS) / ENGINE_STEPS_PER_S
            out.append(f"будет разогрев: {warm} шагов, около {warm_s:.1f} с — "
                       "на нём мерятся опорная частота, собственный перекос "
                       "сети и шум асимметрии")
        if a.get("readout") and (a.get("dn_ref_hz") is not None
                                 and a.get("turn_gain_rad_s") is not None):
            out.append("популяция выбрана, но опорная частота и усиление "
                       "заданы числами: разогрева не будет, и числа от "
                       "нисходящих по умолчанию другой популяции, скорее "
                       "всего, не подойдут")
        if not a.get("readout") and (a.get("dn_ref_hz") is None
                                     and a.get("turn_gain_rad_s") is None):
            out.append("популяция не выбрана: пустые опорная частота и усиление "
                       "НЕ означают «измерить» — берутся умолчания "
                       "`BodyParams` 6 Гц и 5 рад/с (RunSpec.to_body_params)")
        if (a.get("turn_target_rad_s") is not None
                and a.get("turn_gain_rad_s") is not None):
            out.append("поворот на сигму шума задан, но усиление тоже: цель "
                       "калибровки не используется")
        if a.get("probes"):
            out.append("запись популяций по сторонам добавляет 16 чисел на шаг "
                       "в track.parquet и кладёт probes.json")
        if d.get("comment"):
            out.append("комментарий локальной модели требует запущенной Ollama; "
                       "счёт её не ждёт, но без неё в журнал уйдёт ошибка")

    # --- тело ---
    if kind == "body":
        b = d.get("body") or {}
        control_ms = float(b.get("control_ms") or 20.0)
        physics_dt = float(b.get("physics_dt") or 2e-4)
        tau = float(b.get("dn_tau_s") or 0.5)
        lif_steps = max(1, int(round(control_ms / LIF_DT_MS)))
        phys_steps = max(1, int(round(control_ms / 1000.0 / physics_dt)))
        step_ms = lif_steps * _BODY_LIF_MS + phys_steps * _BODY_PHYS_MS
        warm = max(5, int(round(3000.0 * tau / control_ms)))
        out.append(
            f"замер (probe/bodyfinal_timing.json): шаг управления стоит "
            f"{step_ms:.0f} мс — {lif_steps} шагов мозга по "
            f"{_BODY_LIF_MS:.3f} мс и {phys_steps} шагов физики по "
            f"{_BODY_PHYS_MS:.3f} мс; при замеренных настройках это было "
            f"{BODY_MEASURED['шаг_управления_мс']:g} мс")
        out.append(
            f"подготовка около {_BODY_PREP_S + warm * lif_steps * _BODY_LIF_MS / 1000.0:.0f} с "
            f"на эпизод: перебор популяций {BODY_MEASURED['перебор_популяций_с']:g} с, "
            f"сборка тела {BODY_MEASURED['сборка_тела_с']:g} с, рендерер "
            f"{BODY_MEASURED['рендерер_с']:g} с и {warm} шагов разогрева "
            f"(замерено {BODY_MEASURED['подготовка_без_загрузки_с']:g} с при "
            "tau 0,5 с и шаге 20 мс)")
        if repeat:
            out.append(f"подготовка идёт на КАЖДЫЙ эпизод серии: × {n_ep}")
        if b.get("video") and repeat:
            out.append("видео в серии: по файлу на эпизод в папке прогона "
                       f"(`walk-norm-NN.mp4` и `walk-swap-NN.mp4`), всего "
                       f"{n_ep} штук — место под них считайте заранее")
        if b.get("view"):
            out.append("живое окно MuJoCo: закрытие окна останавливает эпизод")
        if b.get("video"):
            out.append("запись видео добавляет `body.render()` на каждый шаг "
                       "физики — это заметно дороже счёта без видео")

    # --- перебор популяций ---
    if kind == "lateral":
        lat = d.get("lateral") or {}
        control_ms = float(lat.get("control_ms") or 20.0)
        period = float(lat.get("period_s") or 2.5)
        t = float(d.get("t") or 0.0)
        out.append(f"перебор кандидатов (`world.probe_candidates`) — около "
                   f"{BODY_MEASURED['перебор_популяций_с']:g} с на 138 639 "
                   "строк, он же кэшируется на весь процесс")
        if t < 2 * period:
            out.append(f"длительность {t:g} с меньше двух периодов качания "
                       f"({period:g} с): первый период отбрасывается, и на "
                       "корреляцию почти ничего не останется")
        if control_ms != 20.0:
            out.append("шаг замера отличается от 20 мс: из командной строки "
                       "это невозможно (флага нет), только через окно")

    # --- сверка ---
    if kind == "verify":
        t_v = d.get("t")
        if t_v is not None and float(t_v) not in VERIFY_T_VALUES:
            out.append("длительность %g с апстрим не примет: он берёт "
                       "только %s. На другом значении он печатает ошибку "
                       "и выходит с нулём, не посчитав ничего — сравнивать "
                       "будет не с чем"
                       % (float(t_v),
                          ", ".join("%g" % v for v in VERIFY_T_VALUES)))
        out.append("шаг бина здесь не настраивается: жёстко 50 мс, один повтор, "
                   "сид 0")
        out.append("вид запускает апстримный `main.py --pytorch` подпроцессом: "
                   "его время в оценку не входит и обычно сравнимо со своим")
        out.append("нужен каталог `upstream/` с `code/benchmark.py` и "
                   "`main.py`: вид запускает их тем же способом, что "
                   "`fly verify`, и сравнивает частоты с последним "
                   "`pytorch_*.parquet` в `upstream/data/results`")
        out.append("отмена посреди своего прогона отменяет и сверку: "
                   "обрезанные частоты сравнивать с полным прогоном "
                   "апстрима нельзя, и окно этого не делает")

    # --- агент ---
    if kind == "agent":
        a = d.get("agent") or {}
        steps = int(a.get("max_steps") or 6)
        out.append(f"агент порождает подпрогоны: до {steps} шагов, каждый "
                   "может оказаться опытом с отдельной папкой `exp-NN`")
        out.append("ход прогона неизвестен: задача шлёт `progress = -1.0`")
        out.append("нужна запущенная Ollama: без неё агент не сделает ни шага")

    return out
