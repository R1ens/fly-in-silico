"""LIF simulator for the whole-brain model, with time-resolved recording.

The neuron and synapse equations are those of Shiu et al. (Nature 2024) as
implemented in the upstream PyTorch backend
(``upstream/code/run_pytorch.py``): a leaky integrate-and-fire point neuron
with an exponentially decaying synaptic conductance, a fixed 1.8 ms axonal
delay, a 2.2 ms refractory period, and synaptic weights equal to the synapse
count of the connection times ``wScale`` millivolts.  Optogenetic activation is
modelled as Poisson spiking driven straight into the membrane voltage.

What this module adds over upstream:

* stimuli with on/off times, so a stimulus can be a pulse rather than a
  constant drive -- which is what makes the activity animation worth watching;
* silencing of arbitrary neurons by zeroing their rows and columns of W;
* recording of spike counts in time bins for every neuron, instead of a full
  spike list, so a 10 s run stays small enough to put in an HTML file.

Verified against the upstream backend by ``cli.py verify``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from . import data as D

# Shiu et al. defaults, copied from upstream run_pytorch.MODEL_PARAMS
MODEL_PARAMS = {
    "tauSyn": 5.0,  # ms, synaptic decay
    "tDelay": 1.8,  # ms, axonal delay
    "v0": -52.0,  # mV, initial voltage
    "vReset": -52.0,  # mV
    "vRest": -52.0,  # mV
    "vThreshold": -45.0,  # mV
    "tauMem": 20.0,  # ms, membrane time constant
    "tRefrac": 2.2,  # ms
    "scalePoisson": 250,  # unitless gain on the Poisson input
    "wScale": 0.275,  # mV per synapse
}
DT = 0.1  # ms


@dataclass
class Stimulus:
    """Poisson drive into a set of neurons, optionally time-windowed."""

    indices: np.ndarray
    rate_hz: float = 200.0
    t_on_ms: float = 0.0
    t_off_ms: float | None = None
    name: str = "stim"

    def to_dict(self, with_indices=False):
        d = {
            "name": self.name,
            "n": int(len(self.indices)),
            "rate_hz": self.rate_hz,
            "t_on_ms": self.t_on_ms,
            "t_off_ms": self.t_off_ms,
        }
        if with_indices:
            d["indices"] = [int(i) for i in self.indices]
        return d


# Имена режимов скорости. Держатся здесь, а не строками по месту.
FAST = "быстрый"
REFERENCE = "эталонный"

# --- чем считать умножение связей -------------------------------------
#
# Два пути к ОДНОМУ произведению `W @ spikes`, и они не два движка:
# результат обязан совпадать побитово, и это проверяется
# `probe/lif_event.py` (пять видов равенства) и `probe/lif_equal.py`.
#
# `SPMV_CSR` — прежний `torch.sparse.mm` по CSR. Читает все 15 091 983
# связи на каждом тике 0,1 мс. Остаётся эталоном и остаётся выбираемым
# всегда — он не требует на машине ничего, кроме torch.
#
# `SPMV_EVENT` — своё ядро (`flybrain/ev_spmv.cu`): складываются только
# строки спайкующих нейронов. В рабочей точке спайкуют 28…103 нейрона из
# 138 639, их связей 6 592…10 365 из 15 091 983, то есть `csrmv`
# умножает на ноль 99,96 % прочитанного.
#
# ТИХОГО ОТКАТА МЕЖДУ НИМИ НЕТ. Если событийное ядро просили, а собрать
# его нечем, движок поднимает `evspmv.EventKernelUnavailable` с
# указанием, чего не хватает. Причина — замеренная: `use_graphs` в этом
# же классе отказывает молча и оставляет шаг втрое медленнее, и второй
# такой механизм сделал бы любое будущее число непроверяемым.
SPMV_CSR = "эталонный"
SPMV_EVENT = "событийный"
SPMV_CHOICES = (SPMV_CSR, SPMV_EVENT)

# Выбор по умолчанию. Берётся из окружения, чтобы прогон можно было
# перевести на другую дорогу, не правя ни одного вызова:
#
#     set FLYBRAIN_SPMV=событийный
#
# Встроенное значение — ЭТАЛОННОЕ, и это ОТКАТ: прежде здесь стояло
# событийное. Причина не в найденном дефекте ядра, а в неразобранном
# отказе.
#
# Что было. Один раз при разборе продуктовой ветви зрения пришёл
# `CUDA illegal memory access`, и источником был назван `evspmv.Events`.
# Разбор — `KERNEL_FAULT.md`, проба — `probe/kernel_fault.py`,
# доказательства — `lab/bench/отказ-ядра/`. Итог разбора прямой:
# ПРИЧИНА НЕ НАЙДЕНА. Ядро чисто на всех своих границах (максимальный
# индекс столбца 138 638 при n = 138 639, буфер ровно `batch*n`, ряд
# 9 783 побитово равен csrmv), чисто под всеми четырьмя средствами
# `compute-sanitizer`, и отказ не воспроизвёлся ни в 150 повторных
# сборках движка, ни в трёх одновременных `vision_chain` под картой,
# занятой на 100 % (19 236 МБ из 24 564), ни в залпе на 1 918 736
# связей одним шагом против худшего реального 240.
#
# Почему тогда откат. Требование владельца: пока нет защитимого
# смягчения ПРИЧИНЫ и ограниченной нагрузочной проверки, затронутая
# ускоренная дорога не входит в приёмку выпуска. Причина не названа,
# значит и смягчения её нет. Две правки, сделанные при разборе
# (`Engine.use_graphs` возвращает состояние при срыве захвата,
# `Events._check_bounds` проверяет границы числами), закрывают два
# ДРУГИХ класса дефектов и `illegal memory access` не дают — выдавать
# их за смягчение разбираемого отказа было бы подлогом.
#
# ЧЕГО ЭТОТ ОТКАТ НЕ ДЕЛАЕТ — сказано прямо, потому что иначе он читался
# бы как исправление. Он не устраняет неизвестную причину. Замерено
# (`KERNEL_FAULT.md` §4): недопустимый доступ делается отдельным ранним
# запуском, а виноватым называется тот, кто первым спросит у карты
# ответ. Событийная дорога спрашивает шесть раз на каждую сборку
# (`exactness` — четыре, `Events.__init__` — два), эталонная — ни разу,
# потому что `_setup_spmv` выходит сразу. Поэтому «на событийной дороге
# всегда `evspmv`, на эталонной каждый раз другая строка `sim.py`» — это
# про место доклада, а не про место ошибки, и «эталонная не давала этой
# ошибки ни разу» означает, что она её НЕ ПОКАЗЫВАЕТ. Остаточный риск
# записан в `FULL_LOOP_VALIDATION.md` §8.
#
# Событийная дорога остаётся и выбирается явно
# (`FLYBRAIN_SPMV=событийный`). Её замеренные основания в силе и ни
# одним замером не отменены:
#
# * равнозначность побитовая: все четыре сочетания дорога×графы
#   дают один и тот же `v ddf796cd850722cb` при 19 захваченных
#   графах (`probe/lif_event.py`, 18 проверок, 0 провалов);
# * мозговая часть шага управления — в 2,679 раза ВНУТРИ настоящего
#   встроенного контура, а не в микробенчмарке;
# * довод, который не про скорость: мозговая часть перестаёт
#   зависеть от фона на видеокарте — у `csrmv` она гуляла в 4,4
#   раза при одной и той же сборке, у событийного в 1,13.
#
# `nvcc` и MSVC для ЗАПУСКА ей не нужны: ядро едет в дерево готовым PTX
# (`flybrain/ev_spmv.ptx`), а на машине его догружает JIT драйвера.
# Проверено подменой путей, а не рассуждением (`probe/kernel_ship.py`,
# `KERNEL_SHIP.md`). Прежняя дорога через `nvcc` осталась эталоном
# доставки и выбирается явно (`FLYBRAIN_EV_LOAD=nvcc`).
#
# Если выбрана событийная дорога, а артефакта нет, он испорчен или
# разошёлся с источником — сказано это ГРОМКО:
# `EventKernelUnavailable` называет, чего не хватает, где искали и как
# выбрать эталонную дорогу явно. В окне это видно как ошибка задачи:
# `worker._run_job` ловит и шлёт `ipc.Error` с причиной и следом, то
# есть окно не исчезает, а показывает, чего не хватает.
SPMV_DEFAULT = os.environ.get("FLYBRAIN_SPMV", SPMV_CSR)


@dataclass
class RunConfig:
    t_run_s: float = 1.0
    bin_ms: float = 10.0
    n_trials: int = 1
    seed: int = 0
    device: str | None = None
    params: dict = field(default_factory=lambda: dict(MODEL_PARAMS))
    # Режим скорости. "быстрый" — с CUDA-графами; "эталонный" — без них.
    # Это не два разных движка: графы дают побайтово тот же результат, и
    # это проверяется `probe/lif_equal.py` и `probe/bench_cmp.py`.
    # Эталонный режим оставлен, чтобы было с чем сравнивать и куда
    # отступить, если граф когда-нибудь поведёт себя иначе.
    speed: str = FAST
    # Чем считать умножение связей: `SPMV_CSR` или `SPMV_EVENT`. None —
    # по `SPMV_DEFAULT` (окружение `FLYBRAIN_SPMV`). Это тоже не два
    # движка: результат обязан совпадать побитово.
    spmv: str | None = None


class Result:
    """Outcome of one run: per-neuron rates plus a neuron x time-bin raster."""

    def __init__(self, counts, bins, config, stimuli, silenced, neurons, timing):
        self.counts = counts  # int64[N], spikes summed over trials
        self.bins = bins  # int16[n_bins, N], spikes per bin summed over trials
        self.config = config
        self.stimuli = stimuli
        self.silenced = silenced
        self.neurons = neurons
        self.timing = timing

    @property
    def n_bins(self):
        return self.bins.shape[0]

    @property
    def rates(self):
        """Firing rate in Hz per neuron, averaged over trials."""
        return self.counts / (self.config.t_run_s * self.config.n_trials)

    def active(self, min_rate=0.5):
        return np.nonzero(self.rates >= min_rate)[0]

    def rate_table(self, min_rate=0.5):
        import pandas as pd

        idx = self.active(min_rate)
        df = self.neurons.loc[idx, [
            "root_id", "label", "cell_class", "super_class", "region",
            "super_region", "side", "top_nt",
        ]].copy()
        df["rate_hz"] = self.rates[idx]
        df["n_spikes"] = self.counts[idx]
        return df.sort_values("rate_hz", ascending=False)

    def region_timeline(self):
        """DataFrame region x bin of mean firing rate in Hz."""
        import pandas as pd

        reg = self.neurons["region"].to_numpy()
        codes, uniq = pd.factorize(reg)
        n_reg = len(uniq)
        acc = np.zeros((n_reg, self.n_bins), dtype=np.float64)
        np.add.at(acc, codes, self.bins.T.astype(np.float64))
        sizes = np.bincount(codes, minlength=n_reg).astype(np.float64)
        bin_s = self.config.bin_ms / 1000.0
        rates = acc / (sizes[:, None] * bin_s * self.config.n_trials)
        return pd.DataFrame(rates, index=list(uniq))

    def meta(self, with_indices=False):
        return {
            "config": {
                k: v for k, v in asdict(self.config).items() if k != "params"
            },
            "params": self.config.params,
            "stimuli": [s.to_dict(with_indices) for s in self.stimuli],
            "silenced": [int(i) for i in self.silenced] if with_indices else None,
            "n_silenced": int(len(self.silenced)),
            "timing": self.timing,
            "n_neurons": int(len(self.neurons)),
            "n_active": int((self.rates >= 0.5).sum()),
            "total_spikes": int(self.counts.sum()),
        }


def _silence(weights, silenced, n):
    """Zero all incoming and outgoing synapses of the silenced neurons."""
    if len(silenced) == 0:
        return weights
    mask_vec = torch.zeros(n, dtype=torch.bool, device=weights.device)
    mask_vec[torch.as_tensor(np.asarray(silenced), device=weights.device)] = True
    w = weights.coalesce()
    idx, val = w.indices(), w.values()
    keep = ~(mask_vec[idx[0]] | mask_vec[idx[1]])
    return torch.sparse_coo_tensor(
        idx[:, keep], val[keep], w.shape, device=weights.device
    ).coalesce()


class Engine:
    """One brain, held in a state you can step forward.

    Separating the stepper from the experiment runner is what makes a closed
    loop possible: :func:`run` drives it from a fixed schedule, while
    ``loop.run_episode`` recomputes the input rates from the arena between
    blocks of steps.  The step body is the one checked against upstream by
    ``cli verify`` -- change it there and nothing else.
    """

    def __init__(self, weights, n, params=None, device="cpu", batch=1, seed=0,
                 no_refractory=None, spmv=None, allow_inexact=False):
        p = dict(params or MODEL_PARAMS)
        self.p = p
        self.n = n
        self.batch = batch
        self.device = device

        with D._quiet_sparse():
            self.w_csr = weights.to_sparse_csr()
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

        self.tau_syn_f = DT / p["tauSyn"]
        self.tau_mem_f = DT / p["tauMem"]
        self.prob_scale = DT / 1000.0
        self.poisson_gain = p["scalePoisson"]
        self.w_scale = p["wScale"]
        self.v_rest, self.v_reset = p["vRest"], p["vReset"]
        self.v_th = p["vThreshold"]

        # Upstream removes the refractory period of driven neurons so the
        # Poisson rate is reproduced faithfully; keep that behaviour.
        self.refrac_steps = torch.full(
            (n,), float(round(p["tRefrac"] / DT)), dtype=torch.float32,
            device=device,
        )
        if no_refractory is not None and len(no_refractory):
            if torch.is_tensor(no_refractory):
                free = no_refractory.to(device=device, dtype=torch.long)
            else:
                free = torch.as_tensor(
                    np.asarray(no_refractory, dtype=np.int64), device=device)
            self.refrac_steps[free] = 0.0

        self.buf_len = int(p["tDelay"] / DT) + 1
        self.conductance = torch.zeros(batch, n, device=device)
        self.delay_buffer = torch.zeros(batch, self.buf_len, n, device=device)
        self.ptr = 0
        self.v = torch.full((batch, n), p["v0"], device=device)
        self.spikes = torch.zeros(batch, n, device=device)
        self.refrac = self.refrac_steps.unsqueeze(0).repeat(batch, 1).clone()

        # --- вмешательства: три вектора-маски по нейрону ---
        #
        # Единица — обычная работа, ноль — вмешательство. Векторы, а не
        # правка матрицы: связей 15 091 983, и пересобирать их на каждое
        # включение нельзя. Маски живут в той же памяти всю жизнь движка
        # и меняются на месте, поэтому их видят и CUDA-графы: граф
        # читает те же тензоры, что и обычный шаг.
        #
        # При всех единицах результат ПОБАЙТОВО прежний: умножение на
        # 1.0 в IEEE754 точное для любого конечного числа.
        self.m_fire = torch.ones(batch, n, device=device)
        self.m_out = torch.ones(batch, n, device=device)
        self.m_in = torch.ones(batch, n, device=device)
        # Четвёртая маска — удержание мембраны. Пара: `m_hold` (1 —
        # обычная работа, 0 — удержание) и `v_hold`, куда положено
        # удерживаемое напряжение, умноженное на (1 - m_hold). Тогда
        # шаг пишется как `v = v*m_hold + v_hold`, и при всех единицах
        # это `v*1 + 0`, то есть побайтово прежнее.
        self.m_hold = torch.ones(batch, n, device=device)
        self.v_hold = torch.zeros(batch, n, device=device)

        # CUDA-графы: по одному на каждое положение кольца задержек.
        # Пусто, пока не позвали `use_graphs()`.
        self._graphs = None
        self._vstim = None
        self._pool = None

        # Чем считается `W @ spikes`. Эталонный путь не требует ничего;
        # событийный собирает своё ядро и отказывает ГРОМКО, если его
        # собрать нечем, — тихого отката здесь нет нарочно.
        self._ev = None
        self.spmv = self._setup_spmv(spmv, allow_inexact)

        # Счётчик вызовов шага. Это не отладка, а подпись стоимости: по
        # ней режимы воспроизведения различаются не на слово. Просмотр
        # записанного обязан не двигать движок вовсе (ноль вызовов),
        # пересчёт — звать его `round(control_ms / DT)` раз на шаг
        # управления. Подделать такую подпись, не посчитав, нельзя:
        # счётчик растёт внутри самого шага, а не рядом с ним.
        #
        # Разогрев захвата графов (`use_graphs`) считается отдельно и в
        # этот счётчик не идёт: он зовёт `_core` напрямую, минуя `step`,
        # и состояние вокруг него восстанавливается.
        self.calls = 0

    # --- умножение связей ----------------------------------------------

    def _setup_spmv(self, spmv, allow_inexact=False):
        """Выбрать путь умножения связей. Отказ — громкий, не тихий.

        Эталонный путь (`SPMV_CSR`) не требует ничего и возвращается
        сразу. Событийный (`SPMV_EVENT`) собирает своё ядро, и всякая
        нехватка поднимает `evspmv.EventKernelUnavailable` с указанием,
        чего именно нет. Это ровно то, чего НЕ делает `use_graphs` в этом
        же классе: там отказ захвата оставляет шаг втрое медленнее
        молча, и прогон об этом не узнаёт.

        Перед тем как согласиться считать событийно, проверяются три
        числа, из которых следует побитовое равенство с `csrmv`
        (`evspmv.exactness`): значения матрицы целые, спайк 0/1, а
        максимальная сумма модулей по строке ниже предела точности
        float32. Если условия не держатся — отказ, и тоже громкий:
        событийное умножение осталось бы быстрым, но перестало бы быть
        ТЕМ ЖЕ, а подменять равнозначность приближением молча нельзя.
        ``allow_inexact=True`` снимает этот отказ явно и записывается в
        отчёт прогона.
        """
        from . import evspmv

        name = SPMV_DEFAULT if spmv is None else str(spmv)
        if name not in SPMV_CHOICES:
            raise ValueError(
                "умножение связей %r не из списка %s" % (name, SPMV_CHOICES))
        self._spmv_exact = None
        self._spmv_inexact_ok = bool(allow_inexact)
        if name == SPMV_CSR:
            return name
        if self.device != "cuda":
            raise evspmv.EventKernelUnavailable(
                "событийное умножение связей считает только на CUDA, а "
                "движок собран на %r. Выберите эталонное умножение явно: "
                "spmv=\"%s\"." % (self.device, SPMV_CSR))
        exact = evspmv.exactness(self.w_csr)
        self._spmv_exact = exact
        if not exact["годится"] and not allow_inexact:
            raise evspmv.EventKernelUnavailable(
                "событийное умножение не обещает побитового равенства на "
                "ЭТОЙ матрице, и поэтому не включается: значения целые — "
                "%s (макс нецелая часть %g), макс сумма модулей по строке "
                "%.0f против предела точности float32 %d. Либо считайте "
                "эталонным умножением (spmv=\"%s\"), либо выберите "
                "приближение явно (allow_inexact=True) — и тогда это "
                "запишется в отчёт прогона как приближение."
                % (exact["значения целые"], exact["макс нецелая часть"],
                   exact["макс сумма модулей по строке"],
                   exact["предел точности float32"], SPMV_CSR))
        self._ev = evspmv.Events(self.w_csr, self.batch, self.device)
        return name

    def spmv_info(self):
        """Чем считалось умножение связей — для отчёта прогона.

        Пишется в каждый сохранённый прогон не для порядка: по готовому
        файлу должно быть видно, каким путём он посчитан, иначе числа
        скорости нечему приписать.
        """
        out = {"умножение связей": self.spmv}
        if self._spmv_exact is not None:
            out["побитовое равенство обеспечено"] = bool(
                self._spmv_exact["годится"])
            out["условия равенства"] = self._spmv_exact
        if self._spmv_inexact_ok:
            out["разрешено приближение"] = True
        if self._ev is not None:
            out["сборка ядра"] = self._ev.build_info
        return out

    # --- вмешательства -------------------------------------------------

    def set_masks(self, fire=None, out=None, inc=None, hold=None,
                  v_clamp=None):
        """Поставить маски вмешательств по спискам индексов нейронов.

        Ничего не пересобирает: три вектора заполняются единицами и
        обнуляются по индексам. Это десятки микросекунд на 138 639
        нейронов против пересборки 15 091 983 связей.

        ``fire`` — не выдают спайк вовсе; ``out`` — спайк есть, но никуда
        не уходит; ``inc`` — ничего не получают; ``hold`` — мембрана
        удерживается. Пустой список снимает соответствующее
        вмешательство.

        **Две разные семантики подавления, и путать их нельзя.**

        ``fire`` — историческая: гасится только ВЫДАЧА спайка, а
        мембранное уравнение не трогается. Нейрон продолжает копить
        потенциал, и при снятии вся популяция разряжается синхронно в
        один шаг. Замер: за 500 шагов подавления мембрана дошла до
        +1299,5 мВ при пороге −45 мВ, и в первом шаге после снятия
        выдали спайк 240 нейронов из 341 — в 64 раза больше обычного
        уровня. Это **вычислительный артефакт способа подавления**, а не
        нейронный rebound: биологического основания считать его ответом
        сети нет.

        ``hold`` — новая, с явно определённым уравнением. Пока
        вмешательство действует, мембранное уравнение нейрона заменяется
        на

            v(t) = v_clamp,   по умолчанию v_clamp = vReset,

        и подстановка делается ДО порога. Отсюда три следствия, и ни
        одно не навязано отдельно: спайка нет (``v_clamp`` ниже порога),
        накопления нет (мембрана не растёт), залпа при снятии нет
        (мембрана начинает с ``v_clamp``). Проводимость при этом
        считается как обычно — нейрон продолжает ПОЛУЧАТЬ вход, он
        только не может изменить своё напряжение; рефрактерный счётчик
        тоже идёт обычным порядком, и при снятии нейрон не находится в
        рефрактерности.

        Историческая семантика сохранена и воспроизводима: это ``fire``,
        и она ничем не изменилась.

        **Что при этом происходит с мембраной подавленного нейрона.**
        Сброс идёт по замаскированному спайку, то есть подавленный
        нейрон под ``fire`` не сбрасывается и продолжает копить
        потенциал. При снятии каждый такой нейрон выдаёт ОДИН спайк
        сразу (после него сброс срабатывает обычным порядком) — но один
        спайк на нейрон у всей популяции в один шаг и есть тот самый
        синхронный залп, и путать «один на нейрон» с «не залп» нельзя.
        Второй выбор — держать мембрану на ``vReset`` всё время
        подавления — не отброшен, а сделан отдельной операцией
        ``hold`` выше, потому что он меняет уравнение мембраны, а не
        только выдачу спайка. Обе семантики доступны, и обе измерены
        пробами `probe/lesion.py` и `probe/semantics.py`, а не оставлены
        на удивление.

        **Что с уже запланированными задержанными событиями** — см.
        :meth:`_core`: ``inc`` и ``hold`` действуют мгновенно, ``out``
        имеет хвост ровно ``buf_len`` = 19 шагов (1,9 мс), а ``fire`` —
        на шаг больше, 20 шагов (2,0 мс), потому что гасит спайк
        текущего шага, а тот попадает в кольцо следующим.
        """
        for vec, idx in ((self.m_fire, fire), (self.m_out, out),
                         (self.m_in, inc), (self.m_hold, hold)):
            if idx is None:
                continue
            vec.fill_(1.0)
            if len(idx):
                t = torch.as_tensor(np.asarray(idx, dtype=np.int64),
                                    device=self.device)
                vec.index_fill_(1, t, 0.0)
        if hold is not None:
            # Слагаемое удержания: ноль там, где нейрон работает обычно,
            # и `v_clamp` там, где удерживается. Значение по умолчанию —
            # `vReset`, то самое, на которое мембрана садится после
            # собственного спайка; произвольного числа здесь нет.
            self._v_clamp = (self.v_reset if v_clamp is None
                             else float(v_clamp))
            self.v_hold.zero_()
            if len(hold):
                t = torch.as_tensor(np.asarray(hold, dtype=np.int64),
                                    device=self.device)
                self.v_hold.index_fill_(1, t, self._v_clamp)

    def fan(self):
        """Сколько у каждого нейрона входящих и исходящих связей.

        Считается один раз и запоминается: окно показывает, сколько
        связей заденет вмешательство, а не только сколько нейронов, и
        пересчитывать для этого 15 млн связей на каждое движение
        ползунка нельзя.

        Матрица лежит как ``W[post, pre]`` в CSR, поэтому входящие — это
        длины строк, а исходящие — счётчик по столбцам.
        """
        got = getattr(self, "_fan", None)
        if got is None:
            crow = self.w_csr.crow_indices()
            cols = self.w_csr.col_indices()
            n_in = (crow[1:] - crow[:-1]).to(torch.int64)
            n_out = torch.bincount(cols.to(torch.int64), minlength=self.n)
            got = self._fan = (n_in, n_out)
        return got

    def mask_counts(self):
        """Сколько нейронов под каждой маской — для подтверждения и отчёта."""
        return {
            "не выдают спайк": int((self.m_fire[0] == 0).sum()),
            "исходящие отключены": int((self.m_out[0] == 0).sum()),
            "входящие отключены": int((self.m_in[0] == 0).sum()),
            "мембрана удержана": int((self.m_hold[0] == 0).sum()),
        }

    def masks_clear(self):
        """Вмешательств нет вовсе: все три маски — единицы."""
        return not (bool((self.m_fire == 0).any())
                    or bool((self.m_out == 0).any())
                    or bool((self.m_in == 0).any())
                    or bool((self.m_hold == 0).any()))

    def advance_ptr(self):
        self.ptr = (self.ptr + 1) % self.buf_len

    def use_graphs(self, enable=True):
        """Записать детерминированную часть шага в CUDA-графы.

        Зачем. Шаг упирается не в работу, а в число обращений к драйверу:
        73 ядра на один тик 0,1 мс, и при партии в шестнадцать раз большей
        время растёт всего в 1,74 раза (`probe/profile_lif.py`). Граф
        отправляет весь этот список одним вызовом.

        Почему графов девятнадцать, а не один. Слот кольца задержек
        выбирается питоновской переменной `ptr`, а она при записи
        застывает: один граф всю жизнь читал бы и писал один и тот же
        слот, и задержка перестала бы существовать. Поэтому пишется по
        графу на каждое из `buf_len` положений, а на прогоне берётся тот,
        что отвечает текущему `ptr`.

        Что НЕ попадает в граф: пуассоновский вход. Генератор при захвате
        получает собственный поток случайных чисел, и результат разошёлся
        бы с эталоном. Он считается как прежде, а в граф входит готовое
        `voltage_stim` через постоянный буфер.

        Арифметика не меняется ни на операцию — это те же ядра в том же
        порядке. Равенство проверяется `probe/lif_equal.py`.

        **Если захват сорвётся, состояние вернётся на место.** Захват
        может не дойти до девятнадцатого графа: под занятой картой не
        хватит памяти, или карту тронут из другой нити (torch
        захватывает в режиме ``capture_error_mode="global"``, и такой
        вызов рвёт захват). Прогрев перед захватом — настоящие три шага,
        поэтому при срыве без возврата движок молча оставался бы уехавшим
        вперёд. Замерено (`probe/kernel_fault.py --part срыв`): при срыве
        на седьмом графе число спайков за 30 шагов оставалось тем же,
        2048, а отпечаток состояния становился `b156cb46eec6eaf8` вместо
        `198729b7dbb0b927` — то есть расхождение было невидимым по
        счётчику спайков. Теперь при срыве состояние возвращается тем же
        `copy_`, `ptr` ставится назад, а пул и буфер входа отпускаются;
        исключение при этом поднимается дальше, тихого отката здесь нет.
        """
        if not enable:
            self._graphs = None
            self._vstim = None
            self._pool = None
            return False
        if self.device != "cuda" or not torch.cuda.is_available():
            return False
        if self._graphs is not None:
            return True

        self._vstim = torch.zeros(self.batch, self.n, device=self.device)

        # Прогрев перед захватом — это настоящие шаги: без него захват
        # ловит ленивую инициализацию библиотек. Но шаги меняют состояние,
        # и даже на нулевом входе счётчик рефрактерности уходит вперёд на
        # три. Поэтому состояние снимается до прогрева и возвращается
        # после — на место, в те же тензоры, на которые ссылаются графы.
        start_ptr = self.ptr
        saved = {name: getattr(self, name).clone()
                 for name in ("v", "conductance", "spikes", "refrac",
                              "delay_buffer")}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._core(self._vstim)
        torch.cuda.current_stream().wait_stream(side)
        self.ptr = start_ptr

        graphs = []
        try:
            for slot in range(self.buf_len):
                self.ptr = slot
                g = torch.cuda.CUDAGraph()
                # Один пул на все графы: иначе временные тензоры шага
                # размножатся на девятнадцать.
                if self._pool is None:
                    with torch.cuda.graph(g):
                        self._core(self._vstim)
                    self._pool = g.pool()
                else:
                    with torch.cuda.graph(g, pool=self._pool):
                        self._core(self._vstim)
                graphs.append(g)
        except BaseException:
            # Захват может сорваться НЕ на первом графе: не хватило
            # памяти под занятой картой, или кто-то тронул карту из
            # другой нити (torch захватывает в режиме
            # `capture_error_mode="global"`, и тогда это рвёт захват).
            # Тогда состояние обязано вернуться на место так же, как при
            # удачном захвате. Иначе движок МОЛЧА уезжает вперёд на три
            # шага прогрева: замерено (`probe/kernel_fault.py --part
            # срыв`) — при срыве на седьмом графе число спайков за 30
            # шагов остаётся тем же, 2048, а отпечаток состояния
            # становится `b156cb46eec6eaf8` вместо `198729b7dbb0b927`.
            # Такое расхождение не поймал бы ни один шлюз равенства,
            # потому что ловится оно только на пути отказа.
            self.ptr = start_ptr
            for name, value in saved.items():
                getattr(self, name).copy_(value)
            # Графов нет ни одного живого, поэтому и ручка пула, и буфер
            # входа уходят: следующая попытка обязана начинать заново, а
            # не с половины прошлой.
            self._vstim = None
            self._pool = None
            self._graphs = None
            raise
        self.ptr = start_ptr
        for name, value in saved.items():
            getattr(self, name).copy_(value)
        self._vstim.zero_()
        self._graphs = graphs
        return True

    def step(self, rates):
        """Advance one 0.1 ms tick.  ``rates`` is [batch, n] in Hz.

        Состояние обновляется НА МЕСТЕ: `v`, `conductance`, `spikes` и
        `refrac` всю жизнь движка лежат в одной и той же памяти. Это не
        украшение и не экономия аллокаций — без этого шаг нельзя записать
        в CUDA-граф: при пересвязывании `self.v = self.v + …` граф на
        повторе читал бы тензор, оставшийся от записи, и состояние не
        двигалось бы вовсе.

        Арифметика и её порядок не изменились ни на операцию. Там, где
        `torch.where` заменён умножением, равенство точное: `spikes` —
        это ровно 0.0 или 1.0, поэтому `(refrac + 1) * (1 - spikes)` даёт
        либо `refrac + 1` без изменения бита, либо ноль. Сверяется это не
        рассуждением, а `probe/lif_equal.py`: отпечатки состояния до и
        после правки совпали побайтово.
        """
        self.calls += 1
        if self._graphs is not None:
            # Вход считаем как обычно и кладём в постоянный буфер, из
            # которого его читает граф.
            self._vstim.copy_(self._poisson(rates))
            self._graphs[self.ptr].replay()
            self.advance_ptr()
            return self.spikes
        voltage_stim = self._poisson(rates)
        return self._core(voltage_stim)

    # --- состояние движка целиком ---------------------------------------

    # Что составляет состояние шага. Перечислено, а не выведено: по этому
    # списку состояние снимается, восстанавливается и — главное —
    # проверяется на полноту выбрасыванием по одному элементу
    # (`probe/resume_state.py`). Элемент, выбрасывание которого ничего не
    # меняет, в состояние не входит, и узнать это можно только замером.
    STATE_TENSORS = ("v", "conductance", "spikes", "refrac", "delay_buffer")
    MASK_TENSORS = ("m_fire", "m_out", "m_in", "m_hold", "v_hold")

    def snapshot(self):
        """Состояние шага как словарь массивов на процессоре.

        Сюда входит всё, что шаг читает и пишет, и ни одной настройки:
        веса, маски рефрактерности и число нейронов — это не состояние, а
        устройство движка, и восстанавливать их значит собрать другой
        движок, а не продолжить этот.

        Отдельно про два элемента, которые легко потерять, потому что они
        не тензоры состояния:

        * ``ptr`` — положение кольца аксональной задержки. Живёт
          питоновским целым, в CUDA-граф не попадает (графов поэтому 19,
          по одному на слот), и без него продолжение читает не тот слот:
          события задержки разъезжаются на шаг.
        * ``gen`` — состояние генератора пуассоновского входа. Без него
          продолжение идёт по другому потоку случайных чисел, и
          побитового совпадения нет со второго же шага.
        """
        import numpy as np

        out = {name: getattr(self, name).detach().cpu().numpy().copy()
               for name in self.STATE_TENSORS}
        for name in self.MASK_TENSORS:
            out[name] = getattr(self, name).detach().cpu().numpy().copy()
        out["ptr"] = np.int64(self.ptr)
        out["gen"] = self.gen.get_state().cpu().numpy().copy()
        out["calls"] = np.int64(self.calls)
        return out

    def restore(self, state, drop=()):
        """Поставить состояние на место. Возвращает список восстановленного.

        Копируется **на место** (``copy_``), а не присваивается: тензоры
        состояния адресуются захваченными CUDA-графами, и
        пересвязывание (``self.v = ...``) оставило бы граф читать старую
        память — состояние перестало бы двигаться вовсе.

        ``drop`` — имена элементов, которые нарочно НЕ восстанавливаются.
        Это не удобство, а инструмент проверки полноты: ровно так
        измеряется, что меняется от потери каждого элемента
        (`probe/resume_state.py`). Обычный вызов ``drop`` не передаёт.
        """
        import numpy as np
        import torch

        done = []
        for name in list(self.STATE_TENSORS) + list(self.MASK_TENSORS):
            if name in drop or name not in state:
                continue
            getattr(self, name).copy_(
                torch.as_tensor(np.asarray(state[name]),
                                device=self.device,
                                dtype=getattr(self, name).dtype))
            done.append(name)
        if "ptr" not in drop and "ptr" in state:
            self.ptr = int(state["ptr"]) % self.buf_len
            done.append("ptr")
        if "gen" not in drop and "gen" in state:
            self.gen.set_state(
                torch.as_tensor(np.asarray(state["gen"], dtype=np.uint8),
                                dtype=torch.uint8))
            done.append("gen")
        return done

    def _poisson(self, rates):
        """Пуассоновский вход в мембрану. Единственное место со случайностью.

        Вынесен из тела шага нарочно: всё остальное детерминировано и
        потому записывается в CUDA-граф, а генератор случайных чисел при
        захвате получил бы собственный поток — числа пошли бы другие, и
        сравнивать результат с эталоном стало бы не с чем.

        Два умножения оставлены двумя: `(x * gain) * scale` и
        `x * (gain * scale)` в общем случае дают разные последние биты.
        """
        poisson = torch.bernoulli(
            rates * self.prob_scale, generator=self.gen) * self.poisson_gain
        return self.w_scale * poisson

    def _core(self, voltage_stim):
        """Детерминированная часть шага. Её и пишем в граф.

        Три маски вмешательств стоят в трёх разных местах цепочки, и
        места выбраны не для удобства, а по смыслу.

        * ``m_out`` — перед матричным умножением, по ПРЕсинаптическому
          нейрону: спайк случается и считается, но никуда не уходит.
        * ``m_in`` — при ЧТЕНИИ кольца задержек, по ПОСТсинаптическому:
          нейрон не получает ничего, включая то, что уже летит к нему.
        * ``m_fire`` — после порога: нейрон не выдаёт спайк вовсе, и его
          нет ни в растре, ни в счётчиках, ни в передаче.

        Отсюда важное следствие, которое надо знать, а не выяснять на
        опыте. ``m_in`` действует МГНОВЕННО и в обе стороны: включили —
        и уже запланированные события до нейрона не доходят, выключили —
        и доставка возобновляется сразу, без накопленного залпа
        (замерено: обрыв и возобновление на шаге +0).

        ``m_out`` и ``m_fire`` так не умеют: в кольце лежит уже СУММА по
        всем пресинаптическим нейронам, и к моменту доставки отправителя
        в ней не различить. Значит у них есть хвост — события,
        записанные в кольцо до вмешательства, дойдут до получателей, —
        и длина хвоста у них РАЗНАЯ:

        * ``m_out`` — ровно ``buf_len`` шагов, 19 при ``tDelay`` 1,8 мс,
          то есть 1,9 мс модельного времени;
        * ``m_fire`` — на шаг больше, ``buf_len + 1`` = 20 шагов, 2,0 мс.

        Причина измерена, а не выведена: ``m_out`` гасит спайк ПРОШЛОГО
        шага в начале шага, перед матричным умножением, а ``m_fire`` —
        спайк ТЕКУЩЕГО шага в конце, и в кольцо этот спайк попадает уже
        следующим шагом. Столько же длится и пауза после снятия, прежде
        чем передача возобновится. Числа сняты `probe/lesion.py`
        (пункт 9) и там же проверяются.

        Очистить кольцо нельзя: вместе с событиями отключаемого нейрона
        оно выбросило бы события всех остальных.
        """
        # recurrent input: W[post, pre] @ spikes[pre]
        #
        # Две дороги к ОДНОМУ произведению, и обе здесь, рядом, а не в
        # двух версиях файла: эталонная читает все 15 091 983 связи,
        # событийная — только строки спайкующих нейронов. Умножение на
        # `w_scale` в обеих стоит ПОСЛЕ суммы и остаётся отдельной
        # операцией: внести его в ядро значило бы складывать уже
        # умноженные слагаемые, а это другие последние биты.
        if self._ev is None:
            recurrent = self.w_scale * torch.sparse.mm(
                self.w_csr, (self.spikes * self.m_out).t().contiguous()).t()
        else:
            # Маска `m_out` применяется внутри ядра, а не отдельным
            # умножением: и спайк, и маска — ровно 0.0 или 1.0, поэтому
            # произведение точное, а один полноразмерный временный
            # тензор и одна транспонированная копия на шаг не создаются.
            recurrent = self.w_scale * self._ev.mm(self.spikes, self.m_out)

        # refractory counter: reset on spike, otherwise count up
        self.refrac.add_(1.0).mul_(1.0 - self.spikes)
        gate = (self.refrac >= self.refrac_steps.unsqueeze(0)).float()

        # synaptic conductance with axonal delay.  The upstream code rolls a
        # (delay+1)-slot buffer every step; reading and then overwriting slot
        # ``step mod L`` of a ring buffer is the same thing without the copy.
        delayed = self.delay_buffer[:, self.ptr, :] * self.m_in
        conductance_new = self.conductance * (1.0 - self.tau_syn_f) + delayed * gate
        self.delay_buffer[:, self.ptr, :] = recurrent
        self.advance_ptr()

        # Membrane update uses the conductance from the PREVIOUS step, which is
        # what upstream's AlphaLIF.forward does -- it passes the old tensor to
        # the neuron and only afterwards swaps in the new one.  Keeping that
        # ordering is what makes `cli verify` match.
        self.v.add_(voltage_stim)
        self.v.add_(self.tau_mem_f * (
            self.conductance - (self.v - self.v_rest)))
        # Удержание мембраны — ДО порога, и это важно: «не выдаёт спайк»
        # здесь не навязано отдельно, а следует из уравнения. Удержание
        # заменяет мембранное уравнение на `v = v_clamp`, а раз
        # `v_clamp` ниже порога, спайка не будет сам собой.
        self.v.mul_(self.m_hold).add_(self.v_hold)
        self.spikes.copy_(self.v > self.v_th)
        self.spikes.mul_(self.m_fire)
        # Сброс мембраны идёт по УЖЕ замаскированному спайку: при
        # подавлении выдачи (`m_fire`) нейрон не сбрасывается и копит
        # потенциал. Это не оплошность, а историческая семантика, и у
        # неё измеренное последствие — см. `set_masks`.
        self.v.sub_((self.v - self.v_reset) * self.spikes)
        self.conductance.copy_(conductance_new
                               - conductance_new * self.spikes)
        return self.spikes


def build_engine(weights=None, neurons=None, silenced=None, params=None,
                 device=None, batch=1, seed=0, no_refractory=None,
                 graphs=True, spmv=None, allow_inexact=False):
    """Load what is needed and return a ready :class:`Engine`.

    `graphs` просит записать шаг в CUDA-графы. По умолчанию да: выигрыш
    замерен втрое, а результат побайтово тот же. На процессоре и при
    отказе захвата движок молча остаётся обычным — `use_graphs` вернёт
    False, и шаг пойдёт прежним путём.

    `spmv` выбирает дорогу умножения связей: `SPMV_CSR` (эталонная) или
    `SPMV_EVENT` (событийное ядро). None — значение `SPMV_DEFAULT`, то
    есть окружение `FLYBRAIN_SPMV`. В отличие от `graphs`, здесь тихого
    отката НЕТ: если событийное ядро просили и собрать его нечем,
    сборка движка падает громко.
    """
    if neurons is None:
        neurons = D.load_neurons()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(neurons)
    if weights is None:
        weights = D.load_weights(device=device)
    else:
        weights = weights.to(device)
    weights = _silence(weights, np.asarray(silenced if silenced is not None
                                           else [], dtype=np.int64), n)
    eng = Engine(weights, n, params=params, device=device, batch=batch,
                 seed=seed, no_refractory=no_refractory, spmv=spmv,
                 allow_inexact=allow_inexact)
    if graphs:
        eng.use_graphs()
    return eng


def run(
    stimuli,
    silenced=None,
    config: RunConfig | None = None,
    neurons=None,
    weights=None,
    progress=True,
    on_bin=None,
    should_stop=None,
):
    """Simulate and return a :class:`Result`.

    ``on_bin(bin_index, t_ms, counts)`` is called as each recording bin closes,
    with the per-neuron spike counts of that bin as a CPU int16 array; the live
    interface uses it to stream the brain while it is still computing.
    ``should_stop()`` is polled between bins so a run can be cancelled.
    """
    config = config or RunConfig()
    silenced = np.asarray([] if silenced is None else silenced, dtype=np.int64)
    if neurons is None:
        neurons = D.load_neurons()
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    p = config.params
    n = len(neurons)

    t0 = time.perf_counter()
    B = config.n_trials
    steps = int(round(config.t_run_s * 1000.0 / DT))
    bin_steps = max(1, int(round(config.bin_ms / DT)))
    n_bins = int(np.ceil(steps / bin_steps))

    # --- stimulus schedule: rate for every neuron, switched per step ---
    stim_events = []  # (step, indices tensor, rate)
    for s in stimuli:
        idx = torch.as_tensor(np.asarray(s.indices, dtype=np.int64), device=device)
        on = int(round(s.t_on_ms / DT))
        off = steps if s.t_off_ms is None else int(round(s.t_off_ms / DT))
        stim_events.append((on, idx, float(s.rate_hz)))
        if off < steps:
            stim_events.append((off, idx, 0.0))
    stim_events.sort(key=lambda e: e[0])
    stim_all = torch.cat([e[1] for e in stim_events]) if stim_events else None

    engine = build_engine(weights=weights, neurons=neurons, silenced=silenced,
                          params=p, device=device, batch=B, seed=config.seed,
                          no_refractory=stim_all,
                          graphs=(getattr(config, "speed", FAST) != REFERENCE),
                          spmv=getattr(config, "spmv", None))
    t_setup = time.perf_counter() - t0

    rates = torch.zeros(B, n, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    binned = torch.zeros(n_bins, n, dtype=torch.int16, device=device)
    # Накопитель бина держится во float32 — в том же типе, что и спайки:
    # так сложение идёт без приведения на каждом шаге, а в int16 он
    # приводится один раз на бин, когда уходит в растр.
    bin_acc = torch.zeros(n, dtype=torch.float32, device=device)

    ev = 0
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=steps, unit="step", ncols=78, desc="sim")
        except ImportError:
            bar = None

    t1 = time.perf_counter()
    for step in range(steps):
        while ev < len(stim_events) and stim_events[ev][0] == step:
            _, idx, rate = stim_events[ev]
            rates[:, idx] = rate
            ev += 1

        spikes = engine.step(rates)

        # Накопление идёт в том же типе, в каком приходят спайки, и общий
        # счётчик складывается раз в бин, а не на каждом шаге. Сумма та же:
        # счётчик — это сумма бинов, а бин — сумма шагов. Зато приведение в
        # int64 и сложение в int64 случаются сто раз реже; замер показал,
        # что на каждом шаге они стоили 0,12 мс из 0,93.
        bin_acc += spikes.sum(0)
        if (step + 1) % bin_steps == 0 or step == steps - 1:
            b = min(step // bin_steps, n_bins - 1)
            whole = bin_acc.to(torch.int16)
            binned[b] = whole
            counts += bin_acc.to(torch.int64)
            if on_bin is not None:
                on_bin(b, (step + 1) * DT, whole.cpu().numpy())
            bin_acc.zero_()
            if should_stop is not None and should_stop():
                steps = step + 1
                break

        if bar is not None and step % 200 == 0:
            bar.update(min(200, steps - step))
    if device == "cuda":
        torch.cuda.synchronize()
    t_sim = time.perf_counter() - t1
    if bar is not None:
        bar.close()

    timing = {
        "setup_s": round(t_setup, 3),
        "sim_s": round(t_sim, 3),
        "steps": steps,
        "device": device,
        "realtime_ratio": round(config.t_run_s / t_sim, 4) if t_sim else None,
        # Режим пишется в каждый прогон: по сохранённому файлу должно быть
        # видно, чем он посчитан.
        "speed": getattr(config, "speed", FAST),
        "graphs": engine._graphs is not None,
        # Каким путём считалось умножение связей. Пишется ДЕЙСТВИТЕЛЬНЫМ,
        # а не по просьбе: тихого отката нет, но просьба может приходить
        # из окружения, и по сохранённому файлу должно быть видно, чем
        # прогон посчитан.
        **engine.spmv_info(),
    }
    return Result(
        counts.cpu().numpy(),
        binned.cpu().numpy(),
        config,
        list(stimuli),
        silenced,
        neurons,
        timing,
    )


def save(result: Result, out_dir: Path, min_rate=0.5):
    """Write meta.json, rates.parquet and bins.npz into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(result.meta(with_indices=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.rate_table(min_rate).to_parquet(out_dir / "rates.parquet")
    keep = result.active(min_rate)
    # `t0_ms` — модельное время первой строки записи. У обычного прогона
    # это ноль, у длинной сессии с кольцом — начало последнего окна.
    # Поле новое, поэтому у прежних файлов его нет: читающий берёт ноль.
    np.savez_compressed(
        out_dir / "bins.npz",
        neuron_idx=keep.astype(np.int32),
        bins=result.bins[:, keep].astype(np.int16),
        bin_ms=np.float32(result.config.bin_ms),
        t0_ms=np.float32((result.timing or {}).get("запись с, мс", 0.0)),
    )
    return out_dir


def load(out_dir, neurons=None) -> Result:
    """Rebuild a :class:`Result` from a saved run directory."""
    import pandas as pd

    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    if neurons is None:
        neurons = D.load_neurons()
    n = len(neurons)

    rates_df = pd.read_parquet(out_dir / "rates.parquet")
    counts = np.zeros(n, dtype=np.int64)
    counts[rates_df.index.to_numpy()] = rates_df["n_spikes"].to_numpy()

    z = np.load(out_dir / "bins.npz")
    n_bins = z["bins"].shape[0]
    bins = np.zeros((n_bins, n), dtype=np.int16)
    bins[:, z["neuron_idx"]] = z["bins"]

    cfg = RunConfig(**meta["config"], params=meta["params"])
    stimuli = [
        Stimulus(
            indices=np.asarray(s.get("indices", []), dtype=np.int64),
            rate_hz=s["rate_hz"],
            t_on_ms=s["t_on_ms"],
            t_off_ms=s["t_off_ms"],
            name=s["name"],
        )
        for s in meta["stimuli"]
    ]
    silenced = np.asarray(meta.get("silenced") or [], dtype=np.int64)
    return Result(counts, bins, cfg, stimuli, silenced, neurons, meta["timing"])
