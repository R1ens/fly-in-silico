"""Схема запуска: одна и та же для командной строки и для окна.

Нужна затем, чтобы кнопка в окне и команда в терминале собирали **одно и то
же** и исполнялись одним путём. Иначе у каждой новой возможности появляется
два места, где она заводится, и они неизбежно расходятся.

Гнать через схему все шестнадцать команд было бы ошибкой: `select`,
`regions`, `doctor` и `index` — это мгновенные запросы, а `report` и
`explain` пересобирают уже сохранённое. Настоящих прогонов пять, и схема
ровно про них: `stim`, `arena`, `body`, `lateral`, `verify`.

Правило, на котором всё держится: **схема несёт только то, что уже несёт
argparse**, а выражения, которые из неё строят конфиги, перенесены из
`cli.py` дословно. Особенно это касается ветки выбора съёма на рулёжку: там
`None` у `dn_ref_hz` и `turn_gain_rad_s` означает «измерить на разогреве», и
подменить это нулём — значит молча сломать всю арену.
"""

from dataclasses import asdict, dataclass, field, fields

import numpy as np

__all__ = ["VERSION", "KINDS", "StimTerm", "StimSpec", "ArenaSpec", "BodySpec",
           "LateralSpec", "VerifySpec", "AgentSpec", "RunSpec", "DEMOS",
           "GOALS", "from_dict"]

VERSION = 1

KINDS = ("stim", "arena", "body", "lateral", "verify", "agent")


# ---------------------------------------------------------------- блоки

@dataclass
class StimTerm:
    """Один стимул: кого зажечь, как сильно и когда."""

    query: str
    rate_hz: float = 200.0
    t_on_ms: float = 0.0
    t_off_ms: float = None
    name: str = ""


@dataclass
class StimSpec:
    activate: list = field(default_factory=list)   # список StimTerm
    silence: str = None
    trials: int = 1


@dataclass
class ArenaSpec:
    readout: str = None          # имя популяции из world.probe_candidates
    swap_antennae: bool = False
    turn_sign: float = 1.0
    dn_ref_hz: float = None      # None значит «измерить на разогреве»
    turn_gain_rad_s: float = None
    dn_tau_s: float = None
    turn_target_rad_s: float = None
    probes: bool = False
    # Окно записи активности, с. 0 — писать всю сессию, как у обычного
    # эпизода. Положительное включает кольцо: `bins` держит последние
    # `keep_s` секунд, счётчики — всю сессию. Нужно длинной сессии, где
    # полная запись это гигабайты (см. `loop.EpisodeConfig.keep_s`).
    keep_s: float = 0.0


@dataclass
class BodySpec:
    readout: str = "ALPN"
    dn_tau_s: float = 0.5
    control_ms: float = 20.0
    physics_dt: float = 2e-4
    swap_antennae: bool = False
    cross: bool = True
    video: bool = False
    play_speed: float = 0.2
    view: bool = False
    sources: list = None
    # --- сеанс песочницы ---
    #
    # `keep_s` — окно записи активности, с. 0 — писать весь сеанс, как у
    # обычного эпизода. Сеанс без заданного конца (`t <= 0`) без окна не
    # запускается: `bins` растёт по 277 КБ на шаг управления.
    keep_s: float = 0.0
    # Яркость насыщения рецепторов числом. None — мерить на разогреве,
    # как считались все прежние эпизоды.
    odour_ref: float = None
    # Исходные условия сеанса: тот же словарь, что принимает
    # `live.BodyLive.apply`. Применяется один раз, до разогрева, и
    # записывается в результат отдельно от правок на ходу.
    live_initial: dict = None


@dataclass
class LateralSpec:
    base_hz: float = 75.0
    depth: float = 0.85
    period_s: float = 2.5
    control_ms: float = 20.0
    top: int = 25
    max_lag: int = 12


@dataclass
class VerifySpec:
    experiment: str = "sugar"


@dataclass
class AgentSpec:
    """Локальная модель сама ставит опыты на коннектоме.

    Цель — готовая формулировка из списка: свободный ввод упирается в то,
    что про знаки препинания в событиях клавиатуры Taichi ничего не
    известно, а цель без запятых формулировать неудобно.
    """

    goal: str = ""
    max_steps: int = 6
    t_run_s: float = 0.5


_BLOCKS = {"stim": StimSpec, "arena": ArenaSpec, "body": BodySpec,
           "lateral": LateralSpec, "verify": VerifySpec,
           "agent": AgentSpec}


# ---------------------------------------------------------------- спека

@dataclass
class RunSpec:
    """Один прогон целиком."""

    kind: str = "stim"
    version: int = VERSION

    # куда и как показывать
    out: str = None              # имя папки в out/; None — отметка времени
    title: str = None
    about: str = None            # строка для человека: что это за опыт
    open_report: bool = True     # обратное к --no-open; окно ставит False

    # общая арифметика
    t: float = 1.0               # длительность, с
    seed: int = 0
    device: str = None
    bin_ms: float = 10.0         # stim: бин записи; прочие: шаг управления

    # локальная модель
    explain: bool = False
    comment: bool = False
    comment_every_ms: float = 1000.0
    model: str = None

    # серия вместо одного прогона
    repeat: int = 0              # 0 — один прогон
    repeat_t: float = None       # длительность эпизода серии, если другая
    repeat_seed0: int = 100

    # предохранители окна; у командной строки выключены
    max_bins: int = 0            # 0 — без предела
    trace: bool = False

    # ровно один блок непустой
    stim: StimSpec = None
    arena: ArenaSpec = None
    body: BodySpec = None
    lateral: LateralSpec = None
    verify: VerifySpec = None
    agent: AgentSpec = None

    # ------------------------------------------------------------- доступ

    def block(self):
        return getattr(self, self.kind, None)

    def ensure_block(self):
        if self.block() is None:
            setattr(self, self.kind, _BLOCKS[self.kind]())
        return self.block()

    @property
    def episode_t(self):
        """Длительность одного эпизода серии."""
        return self.t if self.repeat_t is None else self.repeat_t

    # ------------------------------------------------------------- сборка

    def to_run_config(self):
        from . import sim

        return sim.RunConfig(
            t_run_s=self.t,
            bin_ms=self.bin_ms,
            n_trials=(self.stim.trials if self.stim else 1),
            seed=self.seed,
            device=self.device,
        )

    def to_stimuli(self, neurons):
        """Стимулы из блока. Пустой запрос — ошибка, как и в командной строке."""
        from . import data as D
        from .sim import Stimulus

        out = []
        for term in (self.stim.activate if self.stim else []):
            idx = D.select(neurons, term.query)
            if len(idx) == 0:
                raise ValueError(f"запрос {term.query!r} не нашёл ни одного нейрона")
            out.append(Stimulus(
                indices=idx,
                rate_hz=term.rate_hz,
                t_on_ms=term.t_on_ms or 0.0,
                t_off_ms=term.t_off_ms,
                name=term.name or term.query,
            ))
        return out

    def silenced(self, neurons):
        from . import data as D

        q = self.stim.silence if self.stim else None
        return D.select(neurons, q) if q else np.array([], np.int64)

    def to_body_params(self, neurons):
        """`world.BodyParams` для арены.

        Перенесено из `cli.cmd_arena` слово в слово. Главное здесь — что при
        выбранном съёме `dn_ref_hz` и `turn_gain_rad_s` ставятся **в том
        числе в None**: None на обоих означает «измерить на разогреве», и
        без этого любой съём, кроме нисходящих по умолчанию, поедет.
        """
        from . import world

        a = self.arena or ArenaSpec()
        kw = {"swap_antennae": a.swap_antennae, "turn_sign": a.turn_sign}
        if a.readout:
            cands = world.probe_candidates(neurons)
            if a.readout not in cands:
                similar = [k for k in cands if a.readout.lower() in k.lower()]
                raise ValueError(
                    f"популяция {a.readout!r} не найдена. "
                    + (f"Похожие: {', '.join(similar[:8])}" if similar
                       else "Список — в выводе `fly lateral`."))
            kw["motor_left"], kw["motor_right"] = cands[a.readout]
            kw["dn_ref_hz"] = a.dn_ref_hz
            kw["turn_gain_rad_s"] = a.turn_gain_rad_s
        else:
            if a.dn_ref_hz is not None:
                kw["dn_ref_hz"] = a.dn_ref_hz
            if a.turn_gain_rad_s is not None:
                kw["turn_gain_rad_s"] = a.turn_gain_rad_s
        if a.dn_tau_s is not None:
            kw["dn_tau_s"] = a.dn_tau_s
        if a.turn_target_rad_s is not None:
            kw["turn_target_rad_s"] = a.turn_target_rad_s
        return world.BodyParams(**kw)

    def to_episode_config(self):
        from . import loop

        return loop.EpisodeConfig(
            duration_s=self.t,
            control_ms=self.bin_ms,
            seed=self.seed,
            device=self.device,
            comment=self.comment,
            comment_every_ms=self.comment_every_ms,
            model=self.model,
            keep_s=(self.arena.keep_s if self.arena else 0.0) or 0.0,
        )

    def to_body_config(self, device=None):
        from . import body

        b = self.body or BodySpec()
        return body.BodyConfig(
            duration_s=self.t,
            control_ms=b.control_ms,
            physics_dt=b.physics_dt,
            seed=self.seed,
            device=device or self.device,
            readout=b.readout,
            dn_tau_s=b.dn_tau_s,
            swap_antennae=b.swap_antennae,
            cross=b.cross,
            video=b.video,
            play_speed=b.play_speed,
            view=b.view,
            keep_s=b.keep_s or 0.0,
            odour_ref=b.odour_ref,
        )

    # ------------------------------------------------------------- хранение

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d):
        return from_dict(d)

    def name_for(self, slug_fn, fallback="run"):
        """Имя папки: заданное или отметка времени со слагом, как в CLI."""
        import time

        if self.out:
            return self.out
        base = fallback
        if self.kind == "stim" and self.stim and self.stim.activate:
            base = self.stim.activate[0].name or self.stim.activate[0].query
        elif self.kind in ("arena", "body", "lateral", "verify"):
            base = self.kind
        return time.strftime("%Y%m%d-%H%M%S") + "-" + slug_fn(base)


def from_dict(d):
    """Разобрать словарь обратно в спеку.

    Незнакомые ключи игнорируются нарочно: прогон, сохранённый старой
    сборкой, должен открываться новой, а не падать.
    """
    d = dict(d or {})
    known = {f.name for f in fields(RunSpec)}
    kind = d.get("kind", "stim")
    spec = RunSpec(kind=kind)
    for k, v in d.items():
        if k in known and k not in _BLOCKS:
            setattr(spec, k, v)
    for name, cls in _BLOCKS.items():
        raw = d.get(name)
        if raw is None:
            continue
        sub = cls()
        subknown = {f.name for f in fields(cls)}
        for k, v in dict(raw).items():
            if k in subknown:
                setattr(sub, k, v)
        if name == "stim" and sub.activate:
            sub.activate = [t if isinstance(t, StimTerm) else StimTerm(
                **{k: v for k, v in dict(t).items()
                   if k in {f.name for f in fields(StimTerm)}})
                for t in sub.activate]
        setattr(spec, name, sub)
    spec.ensure_block()
    return spec


# ---------------------------------------------------------------- демо

def _demo(name, title, about, query, rate, t, pulse=None, silence=None,
          explain=False):
    on, off = (pulse or (0.0, None))
    return RunSpec(
        kind="stim", out=name, title=title, about=about, t=t, bin_ms=10.0,
        explain=explain,
        stim=StimSpec(
            activate=[StimTerm(query=query, rate_hz=rate,
                               t_on_ms=on, t_off_ms=off)],
            silence=silence),
    )


# Пять готовых опытов. Раньше жили словарями в cli.py; теперь это спеки, и
# окно запускает ровно то же, что команда `fly demos`. Заголовки и пояснения
# перенесены дословно: по ним собираются отчёты.
DEMOS = [
    _demo("demo-sugar",
          "Сахар слева: вкусовые GRN, 200 Гц, 100–400 мс",
          "вкус: от рецепторов до мотонейронов хоботка",
          "class:gustatory+side:left", 200.0, 0.6, (100.0, 400.0),
          explain=True),
    _demo("demo-odour",
          "Запах слева: обонятельные рецепторы, 150 Гц",
          "запах: рецепторы → антеннальная долька → ГТ и латеральный рог",
          "class:olfactory+side:left", 150.0, 0.8, (100.0, 500.0)),
    _demo("demo-odour-noMB",
          "Тот же запах без грибовидного тела (клетки Кеньона заглушены)",
          "контроль к предыдущему: что уходит, если выключить память",
          "class:olfactory+side:left", 150.0, 0.8, (100.0, 500.0),
          silence="type:KC"),
    _demo("demo-looming",
          "Надвигающийся объект: детекторы LPLC2, 150 Гц",
          "зрение: путь избегания от лобулы вниз",
          "type:LPLC2", 150.0, 0.6, (100.0, 350.0)),
    # Апстрим зовёт их P9 и адресует по root_id; в номенклатуре FlyWire это
    # те же две клетки DNp09, по одной на полушарие.
    _demo("demo-p9",
          "Команда «вперёд»: нисходящие нейроны P9 (DNp09), 100 Гц",
          "движение: та самая пара нейронов из апстримного эксперимента",
          "type=DNp09", 100.0, 1.0),
]


# Готовые цели для агента. Это не украшение: свободный ввод цели пока
# невозможен, а формулировка задаёт весь ход опытов.
GOALS = [
    "найди путь от сахара до мотонейронов хоботка",
    "проверь, что уходит из ответа на запах без грибовидных тел",
    "выясни, какие нисходящие нейроны отвечают на надвигающийся объект",
    "найди, через что обонятельный вход доходит до центрального комплекса",
]
