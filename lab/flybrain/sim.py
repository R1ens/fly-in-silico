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
                 no_refractory=None):
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

        # CUDA-графы: по одному на каждое положение кольца задержек.
        # Пусто, пока не позвали `use_graphs()`.
        self._graphs = None
        self._vstim = None
        self._pool = None

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
        if self._graphs is not None:
            # Вход считаем как обычно и кладём в постоянный буфер, из
            # которого его читает граф.
            self._vstim.copy_(self._poisson(rates))
            self._graphs[self.ptr].replay()
            self.advance_ptr()
            return self.spikes
        voltage_stim = self._poisson(rates)
        return self._core(voltage_stim)

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
        """Детерминированная часть шага. Её и пишем в граф."""
        # recurrent input: W[post, pre] @ spikes[pre]
        recurrent = self.w_scale * torch.sparse.mm(
            self.w_csr, self.spikes.t().contiguous()).t()

        # refractory counter: reset on spike, otherwise count up
        self.refrac.add_(1.0).mul_(1.0 - self.spikes)
        gate = (self.refrac >= self.refrac_steps.unsqueeze(0)).float()

        # synaptic conductance with axonal delay.  The upstream code rolls a
        # (delay+1)-slot buffer every step; reading and then overwriting slot
        # ``step mod L`` of a ring buffer is the same thing without the copy.
        delayed = self.delay_buffer[:, self.ptr, :]
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
        self.spikes.copy_(self.v > self.v_th)
        self.v.sub_((self.v - self.v_reset) * self.spikes)
        self.conductance.copy_(conductance_new
                               - conductance_new * self.spikes)
        return self.spikes


def build_engine(weights=None, neurons=None, silenced=None, params=None,
                 device=None, batch=1, seed=0, no_refractory=None,
                 graphs=True):
    """Load what is needed and return a ready :class:`Engine`.

    `graphs` просит записать шаг в CUDA-графы. По умолчанию да: выигрыш
    замерен втрое, а результат побайтово тот же. На процессоре и при
    отказе захвата движок молча остаётся обычным — `use_graphs` вернёт
    False, и шаг пойдёт прежним путём.
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
                 seed=seed, no_refractory=no_refractory)
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
                          graphs=(getattr(config, "speed", FAST) != REFERENCE))
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
