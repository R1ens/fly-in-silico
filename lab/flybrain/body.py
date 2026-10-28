"""Мозг в теле: коннектом FlyWire управляет ногами NeuroMechFly v2.

Отличие от :mod:`flybrain.loop` — там арена и физика были написаны нами, а
«нисходящие нейроны → скорость и поворот» я придумывал сам. Здесь обе
половины заменены на опубликованные:

* запах берётся с антенн мухи в MuJoCo, а не считается из гауссова шлейфа;
* нисходящий выход уходит в ``HybridTurningController`` — тот самый
  контроллер походки из NeuroMechFly v2, который ожидает ровно два числа,
  «нисходящий сигнал поворота». Ноги, их фазы, прилипание и спотыкание
  считает он.

Придуманными остаются два места: пересчёт интенсивности запаха в частоту
рецепторов — он здесь, и помечен, — и пересчёт активности в команду телу.
Второй с этапа 2 живёт не здесь, а в :mod:`flybrain.motor`: там у каждого
нефизиологичного преобразования есть имя, версия и список параметров, и
оттуда они уезжают в блок ``engineered`` снимка прогона. Заместителем
грудного ганглия работает контроллер FlyGym.

Семантика команды (прочитана из turning_controller.py): ``action[0]`` задаёт
амплитуду шага левых трёх ног, ``action[1]`` — правых, знак задаёт
направление. Замер показал, что более сильные левые ноги уводят муху вправо,
поэтому связь мозга с ногами перекрёстная: левое полушарие — правые ноги.

Разведённые имена в записи по шагам
-----------------------------------

* ``readout_<зонд>_l/r`` — ИЗМЕРЕНИЕ: частота прямого зонда в герцах,
  сумма спайков на число нейронов и длительность блока. Шесть названных
  нисходящих (DNp01, DNp04, DNa01, DNa02, DNp09, MDN) плюс теневые
  зонды — см. :mod:`flybrain.motor`;
* ``cmd_turn``, ``cmd_drive``, ``cmd_escape`` — ИНЖЕНЕРНАЯ команда;
* ``dn_l``, ``dn_r``, ``asym``, ``z``, ``amp_l``, ``amp_r`` — прежние
  поля, оставлены как есть: на них держатся ряды, графики, отчёт,
  песочница и отпечатки эталонных прогонов.

Значения по умолчанию (``motor = "legacy"``, моста через обрыв Giant
Fiber нет, стабилизации головы нет) воспроизводят прежний эпизод
ПОБИТОВО — проверено `probe/motor_equal.py` на двух деревьях кода.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

import numpy as np

from . import data as D, motor, sim, world


@dataclass
class BodyConfig:
    duration_s: float = 4.0
    control_ms: float = 20.0
    physics_dt: float = 2e-4
    seed: int = 0
    device: str | None = None

    readout: str = "ALPN"
    dn_tau_s: float = 0.5
    swap_antennae: bool = False
    cross: bool = True  # левое полушарие управляет правыми ногами

    # Откуда берутся курс и положение для метрики `grad_cos`. Это
    # ИЗМЕРЕНИЕ, а не управление: на поведение не влияет никак (метрика
    # никуда не возвращается), но на `grad_alignment` всех прогонов
    # влияет прямо.
    #
    # `"thorax"` — как было с самого начала и как посчитаны все прежние
    # прогоны: курс из `obs["fly_orientation"]` (ось x КОРНЕВОГО тела),
    # положение — грудь. Остаётся значением по умолчанию, иначе прежние
    # числа станут несравнимыми.
    #
    # `"antennae"` — явно помеченная альтернатива: положение берётся как
    # середина между обонятельными сенсорами, курс — из их собственной
    # оси. Основание: антенны висят на `LFuniculus`/`RFuniculus`, за
    # шейными и антеннальными суставами, и на 1,03 мм впереди груди
    # (`ANTENNA.md` §4), поэтому курс корневого тела — не тот курс, с
    # которого муха нюхает. Какой из двух вариантов правильнее, здесь не
    # решается: сравнение их на одном зерне — `probe/grad_ref.py`.
    #
    # В `spec.py` и `catalog.py` эта ручка НЕ объявлена нарочно: прогоны
    # из окна и из командной строки должны считаться исторической
    # метрикой, а альтернативу ставит только проба.
    grad_ref: str = "thorax"

    # Откуда берутся курс и положение для метрики `grad_cos`. Это
    # измерение, а не управление: на поведение не влияет никак, но на
    # `grad_alignment` всех прогонов влияет прямо.
    #
    # `"thorax"` — как было с самого начала и как посчитаны все прежние
    # прогоны: курс из `obs["fly_orientation"]` (ось x КОРНЕВОГО тела),
    # положение — грудь. Остаётся значением по умолчанию, иначе прежние
    # числа станут несравнимыми.
    #
    # `"antennae"` — явно помеченная альтернатива: положение берётся как
    # середина между обонятельными сенсорами, курс — из их собственной
    # оси. Основание: антенны висят на `LFuniculus`/`RFuniculus`, за
    # шейными и антеннальными суставами, и на 1,03 мм впереди груди
    # (`ANTENNA.md` §4), поэтому курс корневого тела — не тот курс, с
    # которого муха нюхает. Какой из двух вариантов правильнее, решается
    # не здесь: сравнение их на одном зерне — `probe/grad_ref.py`.
    grad_ref: str = "thorax"

    # --- придуманная часть: запах -> частота рецепторов ---
    orn_max_hz: float = 150.0
    odour_ref_factor: float = 2.5  # во сколько раз выше стартовой яркости насыщение
    grn_hz: float = 200.0
    food_radius_mm: float = 3.0

    # --- придуманная часть: асимметрия -> амплитуда шага ---
    amp_base: float = 0.9   # амплитуда при нулевой асимметрии
    amp_span: float = 0.8   # на сколько её меняет одна сигма асимметрии
    amp_min: float = 0.1
    amp_max: float = 1.4
    turn_sign: float = 1.0  # +1: как считались все прежние эпизоды

    # --- инженерный моторный путь (`flybrain.motor`) ---
    #
    # `motor` — какой декодер превращает активность в команду телу:
    #
    # * `"legacy"` — то, что было: среднее по ОДНОЙ популяции на
    #   полушарие (`readout`), асимметрия, нормировка на собственный шум.
    #   Значение по умолчанию, и менять его нельзя: иначе эталонные
    #   прогоны станут несравнимыми молча.
    # * `"dn_linear"` — вариант А: линейная карта по шести ПРЯМЫМ зондам
    #   (DNp01, DNp04, DNa01, DNa02, DNp09, MDN) с объявленными
    #   коэффициентами.
    # * `"dn_trained"` — вариант Б: те же зонды, веса из файла
    #   (`motor_weights`), обучены гребневой регрессией.
    #
    # `gf_bridge` — инженерный мост через обрыв «Giant Fiber →
    # мотонейроны». По умолчанию `"none"`: биологического пути в v783 нет
    # вовсе, и подменять его молча хуже, чем не иметь реакции.
    #
    # `head_stab` — опубликованная ОБУЧЕННАЯ стабилизация головы. По
    # умолчанию выключена: при включении в модель тела добавляются
    # приводы на шею, то есть модель становится ДРУГОЙ.
    #
    # Прямые зонды пишутся в запись ВСЕГДА, при любом декодере: это
    # измерение, оно ничего не меняет и стоит доли процента шага.
    motor: str = "legacy"
    motor_weights: str | None = None
    gf_bridge: str = "none"
    gf_ref_hz: float = 100.0
    escape_gain: float = 0.5
    head_stab: bool = False
    neck_kp: float | None = None
    probes: bool = True  # писать readout_* прямых зондов в запись

    # --- запись видео ---
    video: bool = False
    fps: int = 30
    play_speed: float = 0.2  # замедление: муха быстрая, иначе ничего не видно

    # --- живое окно MuJoCo ---
    view: bool = False

    # Пропуск лишнего пересчёта прямой динамики. Выключено по умолчанию,
    # пока равенство не доказано на этой модели: см. `_fly_class`.
    fast_physics: bool = False

    # --- сеанс песочницы ---
    #
    # `duration_s <= 0` — сеанс без заранее заданного конца: его
    # заканчивает только остановка. Такой сеанс ОБЯЗАН держать кольцо
    # записи (`keep_s > 0`), иначе плотный `bins` растёт без предела:
    # 138 639 × 2 байта = 277 КБ на шаг управления, при шаге 20 мс это
    # 13,9 МБ на секунду модельного времени.
    keep_s: float = 0.0

    # Яркость, которая считается насыщением рецепторов. `None` — мерить
    # на разогреве по тому, что муха чует в начале (так считались все
    # прежние эпизоды). Песочнице нужно число: если сеанс начинается с
    # выключенными источниками, измеренная яркость будет нулевой, и
    # первый же включённый источник насытит рецепторы до потолка.
    odour_ref: float | None = None

    view_every: int = 1  # обновлять окно раз в столько шагов управления


# Предел шагов у сеанса «без конца». Это не длительность, а то, на чём
# арифметика ещё не теряет смысл: 10 млн шагов управления по 20 мс — это
# 55 часов модельного времени, то есть при измеренной скорости около
# 0,08 от настенной порядка месяца счёта. До него не дойдёт ни один
# сеанс, а `range` и оценки времени остаются конечными.
ENDLESS_STEPS = 10_000_000


DEFAULT_SOURCES = np.array([
    [25.0, 12.0, 1.5],
    [38.0, -16.0, 1.5],
])


class EmbodiedRun:
    # Блок нефизиологичных деталей моторного пути. Отдельным полем, а не
    # внутри `timing`: `timing` — про время, а это про модель.
    engineered = None

    def __init__(self, records, counts, bins, neurons, config, sources, timing,
                 metrics, calib, sensor_idx):
        self.records = records
        self.counts = counts
        self.bins = bins
        self.neurons = neurons
        self.config = config
        self.sources = sources
        self.timing = timing
        self.metrics = metrics
        self.calib = calib
        self.sensor_idx = sensor_idx

    def as_result(self):
        # Оборванный эпизод короче заказанного, а rates = counts / t_run_s:
        # с заказанной длительностью мозг выглядел бы вялым. Фактическую
        # длительность кладёт в timing сам ``run_episode``.
        cfg = sim.RunConfig(t_run_s=self.timing.get("duration_s",
                                                    self.config.duration_s),
                            bin_ms=self.config.control_ms, n_trials=1,
                            seed=self.config.seed,
                            device=self.timing.get("device", "cpu"))
        stimuli = [sim.Stimulus(idx, 0.0, name=name)
                   for name, idx in self.sensor_idx.items()]
        return sim.Result(self.counts, self.bins, cfg, stimuli,
                          np.array([], np.int64), self.neurons, self.timing)

    def meta(self):
        out = {
            "config": asdict(self.config),
            "sources": self.sources.tolist(),
            "timing": self.timing,
            "metrics": self.metrics,
            "calibration": self.calib,
            "n_steps": len(self.records),
        }
        if self.engineered is not None:
            out["engineered"] = self.engineered
        return out


def _fly_class(fast):
    """Муха FlyGym, при `fast` — без лишнего пересчёта прямой динамики.

    Откуда берётся лишний пересчёт. `Fly.pre_step` пишет управляющие
    сигналы через `physics.bind(...).ctrl`, а dm_control считает `ctrl`
    полем, которое «пачкает» состояние (`MJDATA_TRIGGERS_DIRTY`). Первое
    же чтение наблюдений после этого видит флаг и зовёт полный
    `physics.forward()`. Замер: 2000 вызовов `mj_forward` на 2000 шагов
    физики, ровно по одному на шаг, 1,03 с из 5,70 с профиля — пятая
    часть шага управления уходит на повторный расчёт того же состояния.

    Почему это вообще может быть безопасно: `ctrl` — вход следующего
    шага, а не производная величина. Почему это НЕ очевидно безопасно: в
    модели 179 сенсоров, и если хоть один зависит от `ctrl`, пропуск
    пересчёта даст устаревшее наблюдение и изменит траекторию.

    Поэтому здесь ничего не доказывается рассуждением. Режим включается
    отдельно, а равенство проверяется `probe/body_equal.py`: траектория,
    моторные команды и активность сравниваются побайтово с прежними.

    **Проверено, равенства нет.** `probe/body_equal.py` на двух эпизодах
    по 2 с с одним семенем: тело ускоряется в 1,73 раза (56,0 с против
    32,4 с), но результат другой. Первое расхождение — на втором шаге
    управления и крошечное (x: 0,561 против 0,560), то есть устаревшим
    оказывается не всё наблюдение, а малая его часть; дальше ходьба
    раскачивает его сама — к сотому шагу x разошёлся на 4,6 мм, курс на
    4,7 рад, суммарные спайки на 1529, а выравнивание по градиенту упало
    с 0,301 до 0,094. Значит хотя бы один из 179 сенсоров всё-таки
    зависит от `ctrl`, и пропуск пересчёта даёт другую траекторию.

    Поэтому это не ускорение, а правка модели, и по правилу брифа
    («первой целью должны быть ускорения, не меняющие математические
    шаги») в научные прогоны она не идёт: `fast_physics` остаётся
    выключенным, а включённый помечает прогон оговоркой в `timing`, чтобы
    такой эпизод нельзя было сравнить с эталонным молча.
    """
    from flygym_gymnasium import Fly

    if not fast:
        return Fly

    class _FlyNoReforward(Fly):
        """То же, но без пересчёта ради снятия флага."""

        def pre_step(self, action, sim):
            was = sim.physics.is_dirty
            super().pre_step(action, sim)
            if not was:
                # Мы написали только `ctrl`. Ничего производного от него
                # до следующего `mj_step` не читается, а `mj_step` всё
                # пересчитает сам.
                sim.physics._dirty = False

    return _FlyNoReforward


def _build_body(cfg: BodyConfig, sources, video_path=None, head_stab=None):
    """Тело FlyGym. ``head_stab`` — переходник :class:`motor.HeadStabilizer`.

    Передавать его — не настройка, а ПРАВКА МОДЕЛИ: при включённой
    стабилизации `Fly` добавляет два привода на шейные суставы
    (`joint_Head_yaw`, `joint_Head`), которых без неё в модели нет
    вовсе. Поэтому эпизод со стабилизацией и без неё — это две разные
    механические модели, а не два режима одной.
    """
    from flygym_gymnasium import YawOnlyCamera
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import default_leg_sensor_placements
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    Fly = _fly_class(getattr(cfg, "fast_physics", False))

    arena = OdorArena(
        odor_source=sources,
        peak_odor_intensity=np.ones((len(sources), 1)),
    )
    extra = {}
    if head_stab is not None:
        extra = {"head_stabilization_model": head_stab,
                 "neck_kp": head_stab.neck_kp}
    fly = Fly(
        enable_adhesion=True,
        enable_olfaction=True,
        contact_sensor_placements=default_leg_sensor_placements,
        init_pose="stretch",
        control="position",
        spawn_pos=(0.0, 0.0, 0.2),
        **extra,
    )
    cams = []
    if video_path is not None:
        # Камера следит за курсом, но не за креном: иначе картинка качается
        # вместе с телом и смотреть невозможно.
        cams = [YawOnlyCamera(
            attachment_point=fly.model.worldbody,
            camera_name="camera_top",
            targeted_fly_names=[fly.name],
            play_speed=cfg.play_speed,
            fps=cfg.fps,
            output_path=str(video_path),
        )]
    sim = HybridTurningController(fly=fly, cameras=cams, arena=arena,
                                  timestep=cfg.physics_dt)
    return sim, cams


def _odour_channels(fly):
    """Какие столбцы `odor_intensity` — антенны, а какие щупики.

    Порядок задаётся словарём `config["olfaction"]["sensor_positions"]`
    и равен `LMaxillaryPalp, RMaxillaryPalp, LAntenna, RAntenna`. Раньше
    здесь стояли жёсткие индексы 0 и 1 — то есть читались **максиллярные
    щупики**, хотя в подписи значились антенны. Лево и право при этом
    были верны, но разнос щупиков 0,60 мм против 0,17 мм у антенн, и
    билатеральная разница выходила в 3,4 раза больше настоящей.

    Ищем по имени: порядок в словаре — не обещание FlyGym, и молча
    разъехаться он не должен второй раз.
    """
    names = list(fly.config["olfaction"]["sensor_positions"])
    left = right = None
    for i, name in enumerate(names):
        low = name.lower()
        if "antenna" not in low:
            continue
        if low.startswith("l") and left is None:
            left = i
        elif low.startswith("r") and right is None:
            right = i
    if left is None or right is None:
        raise ValueError(
            "в olfaction.sensor_positions не нашлось пары антенн: "
            + ", ".join(names))
    return left, right


def _pose(obs):
    """Положение в мм и курс в радианах из наблюдения FlyGym."""
    pos = np.asarray(obs["fly"])[0]
    fwd = np.asarray(obs["fly_orientation"])
    return float(pos[0]), float(pos[1]), math.atan2(float(fwd[1]), float(fwd[0]))


def run_episode(cfg: BodyConfig | None = None, neurons=None, weights=None,
                sources=None, progress=True, on_step=None, should_stop=None,
                video_path=None, on_body=None, live=None):
    """Эпизод в теле. ``on_body(body)`` зовётся один раз, когда тело собрано
    и сброшено: живому окну нужен доступ к физической модели, чтобы снимать
    кадры своим ``mujoco.Renderer`` — камера FlyGym отмеряет кадры по
    модельному времени и для живого вида не годится.

    ``live`` — состояние живых условий (:class:`flybrain.live.BodyLive`) или
    ``None``. Контракт ровно из двух вызовов, и оба на виду:

    * ``live.attach(body, sources, cfg, neurons, device, engine)`` — раз,
      когда тело собрано: состояние получает ссылки на настоящую модель
      MuJoCo, на массив положений источников и на настройку, и дальше
      меняет их на месте;
    * ``live.add_drive(rates)`` — на каждом шаге, после того как в
      ``rates`` записаны сенсорные каналы: так в расчёт попадают прямые
      нейронные стимулы;
    * ``live.active_sources()`` — на каждом шаге: какие слоты источника
      включены. Выключенный слот остаётся в модели, но едой не является.

    Без ``live`` ни один из вызовов не делается, и эпизод считается в
    точности как раньше.
    """
    import torch

    cfg = cfg or BodyConfig()
    # Копия обязательна в обеих ветках. `BodyLive` пишет в этот массив
    # на месте, когда пользователь переносит слот, и без копии правка
    # уехала бы в сам `DEFAULT_SOURCES` — то есть в КАЖДЫЙ следующий
    # эпизод этого процесса работника, молча и невидимо по файлам.
    sources = (DEFAULT_SOURCES.copy() if sources is None
               else np.array(sources, dtype=float))
    if neurons is None:
        neurons = D.load_neurons()
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(neurons)

    # --- кого слушаем и кого зажигаем ---
    cands = world.probe_candidates(neurons)
    # Прямые зонды доступны и как съём для прежнего декодера — под
    # именами с приставкой `зонд:`. В общий список они не входят: там
    # порог восемь нейронов на сторону (см. `world.dn_probe_candidates`).
    cands = {**cands, **world.dn_probe_candidates()}
    if cfg.readout not in cands:
        raise SystemExit(f"нет популяции {cfg.readout!r}; см. вывод `fly lateral`")
    ql, qr = cands[cfg.readout]
    motor_l, motor_r = D.select(neurons, ql), D.select(neurons, qr)

    sensor_idx = {
        "odour_left": D.select(neurons, world.SENSORS["odour_left"]),
        "odour_right": D.select(neurons, world.SENSORS["odour_right"]),
        "taste": D.select(neurons, world.SENSORS["taste"]),
    }
    for name, idx in list(sensor_idx.items()) + [("выход L", motor_l), ("выход R", motor_r)]:
        print(f"  {name:12s} {len(idx):6d} нейронов")

    driven = np.unique(np.concatenate(list(sensor_idx.values())))
    engine = sim.build_engine(weights=weights, neurons=neurons, device=device,
                              seed=cfg.seed, no_refractory=driven)

    t_sens = {k: torch.as_tensor(v, device=device) for k, v in sensor_idx.items()}
    t_ml = torch.as_tensor(motor_l, device=device)
    t_mr = torch.as_tensor(motor_r, device=device)
    rates = torch.zeros(1, n, device=device)

    # --- прямые зонды и инженерный декодер (`flybrain.motor`) ---
    #
    # Измеритель строится ВСЕГДА, при любом декодере: прямые зонды — это
    # измерение, они ничего не меняют, а без них спайк Giant Fiber
    # остаётся невидимым (2 нейрона из 1299 в усреднённом съёме). Прежний
    # съём по полушарию подключён к тому же измерителю под именем
    # `__readout__`, чтобы у старого и нового декодера был один вход.
    readout = motor.DnReadout(
        neurons, device, pairs={"__readout__": (ql, qr)}) \
        if cfg.probes or cfg.motor != "legacy" else None
    shape = motor.DecoderShape(
        amp_base=cfg.amp_base, amp_span=cfg.amp_span, amp_min=cfg.amp_min,
        amp_max=cfg.amp_max, cross=cfg.cross, turn_sign=cfg.turn_sign,
        dn_tau_s=cfg.dn_tau_s)
    dec_kw = {}
    if cfg.motor == "dn_trained":
        dec_kw["weights_path"] = cfg.motor_weights
    decoder = motor.make_decoder(cfg.motor, shape=shape, **dec_kw)
    bridge = motor.GiantFiberBridge(kind=cfg.gf_bridge,
                                    gf_ref_hz=cfg.gf_ref_hz,
                                    escape_gain=cfg.escape_gain)
    if readout is None:
        # Декодер `legacy` без зондов: измеритель всё равно нужен, но
        # только под один прежний съём. Так выключение зондов ничего не
        # меняет в числах, а не «почти ничего».
        readout = motor.DnReadout(neurons, device, probes={}, shadow=False,
                                  pairs={"__readout__": (ql, qr)})
    if cfg.motor != "legacy":
        missing = [k for k in getattr(decoder, "turn_weights", {})
                   if k not in readout.names]
        if missing:
            raise ValueError("декодер %r просит зонды, которых нет: %s"
                             % (cfg.motor, ", ".join(missing)))

    dt_s = cfg.control_ms / 1000.0
    brain_steps = max(1, int(round(cfg.control_ms / sim.DT)))
    body_steps = max(1, int(round(dt_s / cfg.physics_dt)))
    block_s = brain_steps * sim.DT / 1000.0
    endless = cfg.duration_s is not None and cfg.duration_s <= 0
    keep_steps = 0
    if cfg.keep_s and cfg.keep_s > 0:
        keep_steps = max(1, int(round(cfg.keep_s / dt_s)))
    if endless:
        if not keep_steps:
            raise ValueError(
                "сеанс без заданного конца требует кольца записи: укажите "
                "keep_s, иначе `bins` растёт без предела (277 КБ на шаг "
                "управления, 13,9 МБ на секунду модельного времени)")
        # Предела у сеанса нет, но у счёта шагов он должен быть: иначе
        # `range` не построить, а бесконечность в арифметике времени
        # превращает каждую оценку в nan.
        n_steps = ENDLESS_STEPS
    else:
        n_steps = int(round(cfg.duration_s / dt_s))
    requested_steps = n_steps

    print("собираю тело...")
    t0 = time.perf_counter()
    if cfg.video and video_path is None:
        # Раньше здесь подставлялся `walk.mp4` в текущем каталоге: файл
        # уходил мимо папки прогона, а в серии каждый эпизод затирал
        # предыдущий. Молча писать не туда хуже, чем отказаться.
        raise ValueError("включено видео, но путь не задан: укажите "
                         "video_path — иначе файл уйдёт мимо прогона")
    head_stab = motor.HeadStabilizer(neck_kp=cfg.neck_kp) \
        if cfg.head_stab else None
    body, cams = _build_body(cfg, sources, video_path if cfg.video else None,
                             head_stab=head_stab)
    obs, _ = body.reset(seed=cfg.seed)
    t_setup = time.perf_counter() - t0
    if live is not None:
        # До разогрева: то, что состояние применит здесь, — исходные
        # условия сеанса, а не правка на ходу.
        live.attach(body, sources, cfg, neurons, device, engine=engine)
        # Наблюдение надо перечитать. `reset` снял его ДО применения
        # условий, и дальше по нему считается и калибровка разогрева, и
        # нулевой шаг: при выключенных источниках муха «чуяла» бы
        # 0,005 и получала 150 Гц в рецепторы вместо нуля, а опора,
        # перекос и шум мерились бы на возбуждённом мозге.
        obs = body.fly.get_observation(body)
    if on_body is not None:
        on_body(body)

    viewer = None
    if cfg.view:
        import mujoco.viewer

        # Пассивный просмотр: окно не крутит физику само, мы сами зовём sync()
        # после каждого шага. Мышью можно вертеть камеру и ставить на паузу.
        viewer = mujoco.viewer.launch_passive(
            body.physics.model.ptr, body.physics.data.ptr,
            show_left_ui=False, show_right_ui=False)
        print("  открыто окно MuJoCo: мышь вращает камеру, "
              "закрытие окна останавливает эпизод")

    ant_l, ant_r = _odour_channels(body.fly)

    def pose_antennae():
        """Положение и курс ОТ ОБОНЯТЕЛЬНЫХ СЕНСОРОВ, а не от груди.

        Читается то же, что читает `Fly.get_observation`: `framepos`
        датчиков, привязанных к `LFuniculus`/`RFuniculus`. Положение —
        середина между антеннами, курс — из их собственной оси,
        повёрнутой на 90°: ось антенн поперечна курсу, и `ANTENNA.md`
        §3 показал замером, что отходит она от «курс + 90°» всего на
        −1,31…+1,02° за эпизод.

        Отличие от `_pose` не только в точке отсчёта. `_pose` зовётся в
        начале шага управления, то есть курс в метрике взят ДО 100
        шагов физики, а положение — после них; здесь и то и другое
        читается из текущего состояния модели. Это тоже часть
        альтернативы, и она объявлена, а не сделана молча.
        """
        raw = np.asarray(
            body.physics.bind(body.fly._antennae_sensors).sensordata,
            dtype=float).reshape(-1, 3)
        p_l, p_r = raw[ant_l], raw[ant_r]
        mid = 0.5 * (p_l + p_r)
        ax, ay = float(p_l[0] - p_r[0]), float(p_l[1] - p_r[1])
        # Курс — ось антенн минус 90°: при курсе 0 левая антенна лежит
        # севернее правой, то есть ось смотрит на +y, и поворот на −90°
        # даёт +x. Знак проверен пробой `probe/grad_ref.py`.
        return (float(mid[0]), float(mid[1]),
                math.atan2(ay, ax) - 0.5 * math.pi)

    def read_odour(obs):
        """Интенсивность на левой и правой АНТЕННЕ.

        Столбцы найдены по имени (`_odour_channels`), а не заданы числом.
        """
        oi = np.asarray(obs["odor_intensity"])[0]
        return float(oi[ant_l]), float(oi[ant_r])

    def drive_brain(left_i, right_i, taste_hz, ref):
        """Придуманная часть: интенсивность -> частота рецепторов."""
        if cfg.swap_antennae:
            left_i, right_i = right_i, left_i
        rl = cfg.orn_max_hz * min(1.0, left_i / ref)
        rr = cfg.orn_max_hz * min(1.0, right_i / ref)
        rates.zero_()
        rates[0, t_sens["odour_left"]] = rl
        rates[0, t_sens["odour_right"]] = rr
        if taste_hz > 0:
            rates[0, t_sens["taste"]] = taste_hz
        if live is not None:
            # Прямые нейронные стимулы — после сенсорных каналов и после
            # `rates.zero_()`: иначе их стирало бы каждым шагом.
            live.add_drive(rates)
        return rl, rr

    def step_brain():
        acc = torch.zeros(n, device=device)
        for _ in range(brain_steps):
            acc += engine.step(rates)[0]
        dl = float(acc[t_ml].sum()) / (len(motor_l) * block_s)
        dr = float(acc[t_mr].sum()) / (len(motor_r) * block_s)
        return acc, dl, dr

    def step_measure():
        """Один блок управления: мозг + ИЗМЕРЕНИЕ всех зондов.

        `readout_*` здесь и остаются измерением: ни одного
        преобразования, только сумма спайков на число нейронов и
        длительность блока.
        """
        acc, dl, dr = step_brain()
        return acc, dl, dr, readout.read(acc, block_s)

    # --- разогрев: симметричный вход, меряем опору, перекос и шум ---
    li, ri = read_odour(obs)
    if cfg.odour_ref is not None:
        # Песочница задаёт яркость насыщения числом: сеанс может
        # начинаться с выключенными источниками, и измеренная по началу
        # яркость была бы нулевой — первый же включённый источник
        # насытил бы рецепторы до потолка и градиент стал бы не виден.
        odour_ref = max(float(cfg.odour_ref), 1e-9)
    else:
        odour_ref = max((li + ri) / 2.0 * cfg.odour_ref_factor, 1e-9)
    warm = max(5, int(round(3000.0 * cfg.dn_tau_s / cfg.control_ms)))
    k = math.exp(-dt_s / max(cfg.dn_tau_s, 1e-6))
    mean_i = (li + ri) / 2.0
    drive_brain(mean_i, mean_i, 0.0, odour_ref)
    tot = 0.0
    stopped = False
    for _ in range(warm):
        _, dl, dr, rd = step_measure()
        tot += (dl + dr) / 2
        # Разогрев идёт ЧЕРЕЗ декодер: перекос и шум считаются для
        # каждого зонда сразу, одним и тем же кодом. Прежняя арифметика
        # при этом та же — просто у популяции `__readout__` она теперь
        # не отдельной веткой.
        decoder.observe(rd, dt_s)
        if should_stop is not None and should_stop():
            # Разогрев при большой tau длиннее самого эпизода, и ждать его
            # конца значит не отвечать на «стоп» десяток секунд.
            stopped = True
            break
    per_probe = decoder.calibrate()
    bias = decoder.bias.get("__readout__", 0.0)
    noise = decoder.noise.get("__readout__", 1e-4)
    decoder.reset_state()
    calib = {
        "odour_ref": round(odour_ref, 8),
        "dn_ref_hz": round(tot / warm, 4),
        "asym_bias": round(bias, 6),
        "asym_noise": round(noise, 6),
        "warm_steps": warm,
        "зонды": per_probe,
    }
    print(f"  опора {calib['dn_ref_hz']} Гц, перекос {bias:+.5f}, "
          f"шум {noise:.5f}, яркость насыщения {odour_ref:.5f}")

    # --- эпизод ---
    if stopped:
        # Один шаг всё же делаем: на пустой записи не считаются ни метрики,
        # ни итог, и вызывающему пришлось бы разбирать особый случай.
        n_steps = min(n_steps, 1)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    if not keep_steps:
        keep_steps = n_steps
    keep_steps = min(keep_steps, n_steps)
    bins = torch.zeros(keep_steps, n, dtype=torch.int16, device=device)
    records = []
    el = er = None

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=n_steps, unit="шаг", ncols=78, desc="тело")
        except ImportError:
            bar = None

    t1 = time.perf_counter()
    for step in range(n_steps):
        x, y, theta = _pose(obs)
        d = np.linalg.norm(sources[:, :2] - np.array([x, y]), axis=1)
        # Выключенный слот источника остаётся в модели, но едой не
        # является: выбрать его ближайшим значило бы кормить муху тем,
        # чего нет. Маску даёт состояние живых условий; без него, как и
        # раньше, считаются все источники.
        act = live.active_sources() if live is not None else None
        if act is not None and len(act) == len(d):
            if not any(act):
                nearest, d_near = -1, None
            else:
                dd = np.where(np.asarray(act, bool), d, np.inf)
                nearest = int(np.argmin(dd))
                d_near = float(dd[nearest])
        else:
            nearest = int(np.argmin(d))
            d_near = float(d[nearest])
        li, ri = read_odour(obs)
        taste = (cfg.grn_hz if (d_near is not None
                                and d_near < cfg.food_radius_mm) else 0.0)
        # Обе величины перечитываются из настройки на каждом шаге, а не
        # берутся один раз до цикла: иначе живая правка `odour_ref` или
        # `dn_tau_s` подтверждалась бы как применённая и попадала в
        # журнал прогона, ничего при этом не меняя. При неизменной
        # настройке значения те же, что и прежде.
        ref_now = (max(float(cfg.odour_ref), 1e-9)
                   if cfg.odour_ref is not None else odour_ref)
        k = math.exp(-dt_s / max(cfg.dn_tau_s, 1e-6))
        rl, rr = drive_brain(li, ri, taste, ref_now)

        acc, dl, dr, rd = step_measure()
        # Счётчики копятся за весь сеанс, запись — только за окно:
        # иначе частоты (`counts / t_run_s`) считались бы по обрезанной
        # записи и врали бы во столько раз, во сколько окно короче.
        bins[step % keep_steps] = acc.to(torch.int16)
        counts += acc.to(torch.int64)

        # Инженерная часть: измеренные частоты -> команда телу. Вся
        # арифметика живёт в `flybrain.motor`; здесь только вызов, и
        # `dn_tau_s` подхватывается с настройки на каждом шаге, как и
        # раньше, чтобы живая правка применялась, а не подтверждалась.
        decoder.shape.dn_tau_s = cfg.dn_tau_s
        escape = bridge.escape(rd)
        cmd = decoder.command(rd, dt_s, escape=escape)
        # Прежние поля записи. Берутся из самого декодера, а не
        # пересчитываются рядом: пересчёт «эквивалентным» выражением уже
        # один раз развёл траектории на последнем бите мантиссы.
        asym = cmd.detail.get("asym")
        if asym is None:
            asym = (decoder._asym(*decoder._sm["__readout__"])
                    - decoder.bias.get("__readout__", 0.0))
        z = cmd.detail.get("z_raw", cmd.u_turn * 3.0)
        action = cmd.as_array()

        for _ in range(body_steps):
            obs, _, term, trunc, _ = body.step(action)
            if cfg.video:
                body.render()
            if term or trunc:
                break
        closed = False
        if viewer is not None and step % cfg.view_every == 0:
            if not viewer.is_running():
                print("  окно закрыто, останавливаюсь")
                # Выйти здесь нельзя: строка `bins` этого шага уже
                # записана, а запись шага ещё нет — вышло бы на одну
                # строку записи меньше, чем шагов, и `metrics["steps"]`
                # разошёлся бы с `timing["steps_done"]`. Поэтому шаг
                # доводится до конца, а выход — после записи.
                closed = True
            else:
                viewer.sync()

        nx, ny, _ = _pose(obs)
        # Точка отсчёта метрики. По умолчанию — как было всегда: курс
        # корневого тела, снятый в начале шага управления, и положение
        # груди после него.
        gx, gy, gtheta = nx, ny, theta
        if cfg.grad_ref == "antennae":
            gx, gy, gtheta = pose_antennae()
        if nearest < 0:
            # Ни одного включённого источника: курс сравнивать не с чем,
            # и расстояние до «ближайшего» было бы выдумкой. Минус
            # единица в записи значит именно это, а не ноль миллиметров.
            grad_cos, dist_mm = 0.0, -1.0
        else:
            to_src = sources[nearest, :2] - np.array([gx, gy])
            norm = float(np.linalg.norm(to_src))
            grad_cos = float((to_src @ np.array([math.cos(gtheta),
                                                 math.sin(gtheta)])) / norm) \
                if norm > 1e-6 else 0.0
            dist_mm = float(np.linalg.norm(sources[nearest, :2] - [gx, gy]))

        rec = {
            "t_ms": step * cfg.control_ms,
            "x": round(nx, 3), "y": round(ny, 3), "theta": round(theta, 4),
            "odour_l": round(li, 7), "odour_r": round(ri, 7),
            "orn_l": round(rl, 1), "orn_r": round(rr, 1),
            "taste": round(taste, 1),
            "dn_l": round(dl, 3), "dn_r": round(dr, 3),
            "asym": round(asym, 5), "z": round(z, 3),
            "amp_l": round(float(action[0]), 3), "amp_r": round(float(action[1]), 3),
            "dist": round(dist_mm, 2),
            "grad_cos": round(grad_cos, 4),
            "source": int(nearest),
            "spikes": int(acc.sum()),
        }
        # Разведённые имена. `readout_*` — ИЗМЕРЕНИЕ: частота зонда в
        # герцах, без единого преобразования. `cmd_*` — ИНЖЕНЕРНАЯ
        # команда: поворот, тяга и вклад моста через обрыв GF. Старые
        # ключи (`dn_l`, `asym`, `z`, `amp_l`, `amp_r`) остаются на
        # месте — на них держатся ряды, графики, отчёт и песочница.
        if cfg.probes:
            for name in readout.names:
                if name == "__readout__":
                    continue
                l, r = rd[name]
                rec["readout_%s_l" % name] = round(l, 3)
                rec["readout_%s_r" % name] = round(r, 3)
        rec["cmd_turn"] = round(cmd.u_turn, 5)
        rec["cmd_drive"] = round(cmd.u_drive, 5)
        rec["cmd_escape"] = round(cmd.escape, 5)
        records.append(rec)
        if on_step is not None:
            on_step(step, rec, bins[step % keep_steps].cpu().numpy())
        if closed:
            n_steps = step + 1
            break
        if should_stop is not None and should_stop():
            n_steps = step + 1
            break
        if bar is not None:
            bar.update(1)

    if device == "cuda":
        torch.cuda.synchronize()
    t_sim = time.perf_counter() - t1
    if bar is not None:
        bar.close()
    if cfg.video and cams:
        cams[0].save_video(video_path)
        print(f"  видео: {video_path}")
    if viewer is not None:
        viewer.close()
    body.close()

    # Фактическую длительность считаем по пройденным шагам и только если
    # эпизод оборвали: n_steps * dt_s не обязано совпадать с заказанной
    # длительностью при неделимом control_ms, и подменять её зря нельзя.
    cut = n_steps < requested_steps
    dur_s = n_steps * dt_s if (cut or endless) else cfg.duration_s

    # Кольцо — по времени. До этого места строка шага `i` лежит в слоте
    # `i % keep_steps`: при провернувшемся кольце самая старая строка
    # стоит не первой, и расставить их можно только зная окончательное
    # число шагов.
    kept = min(n_steps, keep_steps)
    if n_steps > keep_steps:
        bins = torch.roll(bins, -(n_steps % keep_steps), dims=0)
    else:
        bins = bins[:kept]
    bins_t0_ms = (n_steps - kept) * cfg.control_ms
    timing = {
        "setup_s": round(t_setup, 2),
        "sim_s": round(t_sim, 2),
        "device": device,
        "realtime_ratio": round(dur_s / t_sim, 4) if t_sim else None,
        # Шаги пишутся всегда, а не только у оборванного эпизода: по ним
        # видно, дошёл ли прогон до конца, даже когда `duration_s`
        # совпадает с заказанной. Заказанная остаётся в `config` и не
        # подменяется.
        "steps_done": int(n_steps),
        "steps_requested": int(requested_steps),
        "duration_s": round(dur_s, 6),
    }
    if endless:
        timing["steps_requested"] = None   # предела не было
        timing.setdefault("оговорки", []).append(
            "сеанс без заранее заданного конца: его закончила остановка, "
            "а не исчерпание заказанного времени. «Отменён» в карточке "
            "для такого сеанса — норма, а не обрыв расчёта.")
    if kept < n_steps:
        # Запись короче сеанса. Молчать нельзя: по `bins` считают растр
        # и ход активности, и без подписи они выглядели бы как целый
        # прогон, начинающийся с нуля.
        timing["окно записи, с"] = round(kept * dt_s, 6)
        timing["запись с, мс"] = round(bins_t0_ms, 3)
        timing.setdefault("оговорки", []).append(
            "запись активности — последнее окно %.3g с из %.3g с сеанса "
            "(с %.0f мс). Счётчики и частоты посчитаны по всему сеансу, "
            "растр и ход активности — только по окну."
            % (kept * dt_s, dur_s, bins_t0_ms))
    if getattr(cfg, "fast_physics", False):
        # Прогон в быстрой физике сравнивать с эталонным нельзя, и
        # молчать об этом тоже: оговорка едет вместе с результатом в
        # `meta.json`, а оттуда в отчёт и в сверку.
        timing.setdefault("оговорки", []).append(
            "включена быстрая физика (`fast_physics`): пропущен повторный "
            "расчёт прямой динамики после записи управляющих сигналов. "
            "Замер `probe/body_equal.py`: ускорение 1,73 раза, но "
            "траектория другая — расхождение начинается на втором шаге "
            "управления и накапливается. С эталонными прогонами такой "
            "эпизод не сравнивается.")
    # Блок `engineered`: всё нефизиологичное моторного пути с именем,
    # версией, параметрами и, где есть файл весов, его sha256. Он едет в
    # `embodied.json`, а оттуда в отчёт и в сверку — то есть отличить
    # инженерную часть от измеренной можно машинно, а не на глаз.
    engineered = motor.engineered_block(
        decoder=decoder, bridge=bridge, head_stab=head_stab,
        readout_rows=readout.rows if cfg.probes else None)
    for obj in (bridge, head_stab):
        c = obj.caveat() if obj is not None else None
        if c:
            timing.setdefault("оговорки", []).append(c)
    if cfg.motor != "legacy":
        timing.setdefault("оговорки", []).append(
            "моторная команда посчитана ИНЖЕНЕРНЫМ декодером "
            "`%s` v%d по прямым зондам нисходящих, а не прежним средним "
            "по полушарию. Коэффициенты (или веса) этого декодера — не "
            "условия прогона, а его модель: см. блок `engineered`. С "
            "эталонными прогонами, посчитанными `legacy`, такой эпизод "
            "не сравнивается."
            % (decoder.name, decoder.version))
    metrics = _metrics(records, dt_s)
    metrics.update(_motor_metrics(records))
    run = EmbodiedRun(records, counts.cpu().numpy(), bins.cpu().numpy(),
                      neurons, cfg, sources, timing, metrics, calib,
                      sensor_idx)
    run.engineered = engineered
    return run


def _metrics(records, dt_s):
    if not records:
        return {}
    # При одном шаге (отмена на разогреве) вторая половина выходила
    # пустой, и `np.mean` отдавал nan с предупреждением. Тогда обе
    # половины — это сам единственный шаг.
    # Шаги, на которых не было ни одного включённого источника, пишут
    # `dist = -1` и `grad_cos = 0` — это «не с чем сравнивать», а не
    # «ноль миллиметров». Считать их наравне значило бы подмешать в
    # среднее расстояние минус единицу, а в выравнивание — нули, и обе
    # величины поехали бы тем сильнее, чем дольше источники были выключены.
    aimed = [r for r in records if r.get("dist", 0.0) >= 0.0]
    skipped = len(records) - len(aimed)
    if not aimed:
        return {"steps": len(records), "шагов без источника": skipped,
                "примечание": "ни на одном шаге не было включённого "
                              "источника: расстояние и выравнивание по "
                              "градиенту не определены"}
    half = len(aimed) // 2
    d = [r["dist"] for r in aimed]
    d_first = d[:half] if half else d
    d_second = d[half:] if half < len(d) else d
    path = sum(math.dist((records[i]["x"], records[i]["y"]),
                         (records[i - 1]["x"], records[i - 1]["y"]))
               for i in range(1, len(records)))
    out = {} if not skipped else {"шагов без источника": skipped}
    out.update({
        "steps": len(records),
        "grad_alignment": round(float(np.mean([r["grad_cos"]
                                               for r in aimed])), 4),
        "dist_first_half_mm": round(float(np.mean(d_first)), 2),
        "dist_second_half_mm": round(float(np.mean(d_second)), 2),
        "dist_start_mm": round(d[0], 2),
        "dist_end_mm": round(d[-1], 2),
        "path_mm": round(path, 2),
        "speed_mm_s": round(path / max(len(records) * dt_s, 1e-9), 2),
        "food_steps": int(sum(1 for r in records if r["taste"] > 0)),
        "mean_dn_hz": round(float(np.mean([(r["dn_l"] + r["dn_r"]) / 2
                                           for r in records])), 3),
    })
    return out


def _motor_metrics(records):
    """Насколько команда вообще ШЕВЕЛИЛАСЬ и следовала ли за раздражителем.

    Две величины, и обе нужны вместе. Разброс команды отвечает на «а
    зависит ли выход хоть от чего-нибудь»: при заглушённом съёме он
    обязан уйти в ноль. Корреляция с асимметрией раздражителя отвечает
    на «от ЧЕГО именно»: при заглушённом съёме она обязана перестать
    существовать. По одной из них ничего не доказывается: постоянная
    команда даёт нулевую корреляцию честно, а шумящая ни с чем не
    связанная команда даёт большой разброс так же честно.
    """
    if len(records) < 3:
        return {}
    turn = np.array([r.get("cmd_turn", 0.0) for r in records], float)
    amp = np.array([r["amp_l"] - r["amp_r"] for r in records], float)
    stim = np.array([(r["orn_l"] - r["orn_r"])
                     / (r["orn_l"] + r["orn_r"] + 1e-9) for r in records],
                    float)
    out = {
        "cmd_turn_sd": round(float(np.std(turn)), 6),
        "cmd_amp_diff_sd": round(float(np.std(amp)), 6),
        "stim_asym_sd": round(float(np.std(stim)), 6),
    }
    for name, a in (("r_cmd_stim", turn), ("r_amp_stim", amp)):
        if np.std(a) < 1e-12 or np.std(stim) < 1e-12:
            # Постоянный ряд: корреляции НЕ существует, и ноль здесь был
            # бы выдумкой. None читается как «не определена».
            out[name] = None
        else:
            out[name] = round(float(np.corrcoef(a, stim)[0, 1]), 4)
    return out


def bench(n_runs=6, duration_s=3.0, neurons=None, weights=None, device=None,
          base: BodyConfig | None = None, seed0=100, progress=True,
          on_episode=None, should_stop=None, out_dir=None):
    """Норма против переставленных антенн, один эпизод — одно наблюдение.

    Та же схема, что и для плоской арены: внутри эпизода курс и градиент
    меняются медленно, поэтому посчитать значимость по шагам нельзя.

    ``on_episode(cond, i, n, metrics)`` зовётся после каждого эпизода, где
    ``cond`` — название условия, ``i`` — номер эпизода с единицы.
    ``should_stop()`` проверяется между эпизодами и уходит внутрь эпизода:
    эпизод в теле идёт секунды, и ждать его конца значит не отвечать на
    «стоп». У оборванной серии в ``n_runs`` стоит фактическое число эпизодов,
    а не заказанное — по нему же считаются sd и sem.
    """
    import torch

    if neurons is None:
        neurons = D.load_neurons()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if weights is None:
        weights = D.load_weights(device=device)
    base = base or BodyConfig()

    out = {}
    stopped = False
    # Видео эпизодов: каждому своё имя внутри папки прогона. Без папки
    # `run_episode` писал `walk.mp4` в текущий каталог, и каждый
    # следующий эпизод затирал предыдущий.
    out_dir = Path(out_dir) if out_dir else None
    if base.video and out_dir is None:
        raise ValueError("видео в серии просят, а папку прогона не задали: "
                         "эпизоды затёрли бы друг друга")
    for ci, (label, swap) in enumerate((("норма", False),
                                        ("перестановка антенн", True))):
        vals, dists, visits, eps = [], [], [], []
        for i in range(n_runs):
            cfg = BodyConfig(**{**asdict(base), "swap_antennae": swap,
                                "duration_s": duration_s, "seed": seed0 + i,
                                "device": device})
            video = None
            if cfg.video:
                out_dir.mkdir(parents=True, exist_ok=True)
                video = out_dir / ("walk-%s-%02d.mp4"
                                   % ("norm" if ci == 0 else "swap", i + 1))
            r = run_episode(cfg=cfg, neurons=neurons, weights=weights,
                            progress=False, should_stop=should_stop,
                            video_path=video)
            m = r.metrics
            if "grad_alignment" not in m:
                # Эпизод, прошедший целиком без включённых источников:
                # ни выравнивания по градиенту, ни расстояния у него нет
                # (см. `_metrics`). Включить его в серию наравне с
                # остальными значило бы подмешать выдуманные нули;
                # молча пропустить — тоже нельзя, поэтому отказ громкий.
                raise ValueError(
                    "эпизод %d условия %r прошёл без единого включённого "
                    "источника: выравнивания по градиенту и расстояния у "
                    "него нет, и в серию он не идёт. Серии нужны условия, "
                    "в которых источник включён." % (i + 1, label))
            vals.append(m["grad_alignment"])
            dists.append(m["dist_end_mm"] - m["dist_start_mm"])
            visits.append(m["food_steps"])
            eps.append({"i": i + 1, "seed": int(cfg.seed),
                        "grad_alignment": round(m["grad_alignment"], 4),
                        "dist_change_mm": round(m["dist_end_mm"]
                                                - m["dist_start_mm"], 2),
                        "food_steps": int(m["food_steps"]),
                        "video": str(video.name) if video else None})
            if progress:
                print(f"  {label:20s} [{i + 1}/{n_runs}] "
                      f"градиент {m['grad_alignment']:+.3f}, "
                      f"до еды {m['dist_start_mm']:.1f} -> {m['dist_end_mm']:.1f} мм")
            if on_episode is not None:
                on_episode(label, i + 1, n_runs, m)
            if should_stop is not None and should_stop():
                stopped = True
                break
        n_done = len(vals)
        out[label] = {
            "n_runs": n_done,
            "duration_s": duration_s,
            "readout": base.readout,
            "dn_tau_s": base.dn_tau_s,
            "grad_alignment_mean": round(float(np.mean(vals)), 4),
            "grad_alignment_sd": round(float(np.std(vals, ddof=1)), 4) if n_done > 1 else None,
            "grad_alignment_sem": round(
                float(np.std(vals, ddof=1) / math.sqrt(n_done)), 4) if n_done > 1 else None,
            "runs": [round(v, 4) for v in vals],
            "seeds": [int(e["seed"]) for e in eps],
            "seed0": int(seed0),
            "episodes": eps,
            "dist_change_mm": round(float(np.mean(dists)), 2),
            "food_steps_total": int(sum(visits)),
        }
        if stopped:
            # Второе условие не считаем: сравнивать его с оборванным первым
            # нечестно, а делать вид, что серия целая, — тем более.
            break
    return out


def load(out_dir, neurons=None) -> EmbodiedRun:
    """Собрать сохранённый эпизод в теле обратно из файлов.

    Физика при этом не считается и MuJoCo не поднимается: траектория
    лежит в `track.parquet`, растр и счётчики — в `bins.npz` и
    `rates.parquet`, настройки и метрики — в `embodied.json`. Это
    зеркало `save`, которого до сих пор не было: `cli.cmd_report` на
    папке с телом молча уходил в `sim.load` и терял и траекторию, и
    метрики, и калибровку разогрева.

    Чего в файлах нет, то и не восстанавливается, и об этом говорится
    прямо: отсутствие `track.parquet` — не повод вернуть пустой эпизод
    молча.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    meta_path = out_dir / "embodied.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{out_dir}: нет embodied.json — это не прогон в теле")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    res = sim.load(out_dir, neurons=neurons)

    track = out_dir / "track.parquet"
    if not track.exists():
        raise FileNotFoundError(
            f"{out_dir}: есть embodied.json, но нет track.parquet — "
            "записи по шагам в этом прогоне не сохранено, восстановить "
            "траекторию, амплитуды ног и сигналы неоткуда")
    records = pd.read_parquet(track).to_dict("records")

    known = {f.name for f in fields(BodyConfig)}
    raw_cfg = meta.get("config") or {}
    cfg = BodyConfig(**{k: v for k, v in raw_cfg.items() if k in known})
    unknown = sorted(set(raw_cfg) - known)

    sources = np.asarray(meta.get("sources") or DEFAULT_SOURCES, float)

    # Сенсорные каналы: сперва из `meta.json` (их туда положил `sim.save`
    # вместе с индексами), иначе пересобираются из `world.SENSORS` — это
    # чистый отбор по таблице нейронов, без всякого счёта.
    sensor_idx = {st.name: np.asarray(st.indices, np.int64)
                  for st in res.stimuli
                  if st.name in ("odour_left", "odour_right", "taste")
                  and len(st.indices)}
    if len(sensor_idx) < 3:
        nn = res.neurons
        sensor_idx = {name: D.select(nn, world.SENSORS[name])
                      for name in ("odour_left", "odour_right", "taste")}

    timing = dict(meta.get("timing") or res.timing or {})
    if unknown:
        timing["unknown_config_keys"] = unknown
    run = EmbodiedRun(records, res.counts, res.bins, res.neurons, cfg,
                      sources, timing, meta.get("metrics") or {},
                      meta.get("calibration") or {}, sensor_idx)
    # Блок нефизиологичных деталей восстанавливается как есть. Его
    # отсутствие — не ошибка: прогоны до этого этапа его не писали, и у
    # них он равен None, а не пустому словарю. Разница существенная:
    # пустой словарь читался бы как «инженерных деталей не было».
    run.engineered = meta.get("engineered")
    return run


def save(run: EmbodiedRun, out_dir):
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embodied.json").write_text(
        json.dumps(run.meta(), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(run.records).to_parquet(out_dir / "track.parquet")
    sim.save(run.as_result(), out_dir)
    return out_dir
