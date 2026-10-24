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


@dataclass
class RunConfig:
    t_run_s: float = 1.0
    bin_ms: float = 10.0
    n_trials: int = 1
    seed: int = 0
    device: str | None = None
    params: dict = field(default_factory=lambda: dict(MODEL_PARAMS))


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

    def step(self, rates):
        """Advance one 0.1 ms tick.  ``rates`` is [batch, n] in Hz."""
        # Poisson drive straight into the membrane voltage
        poisson = torch.bernoulli(
            rates * self.prob_scale, generator=self.gen) * self.poisson_gain
        voltage_stim = self.w_scale * poisson

        # recurrent input: W[post, pre] @ spikes[pre]
        recurrent = self.w_scale * torch.sparse.mm(
            self.w_csr, self.spikes.t().contiguous()).t()

        # refractory counter: reset on spike, otherwise count up
        self.refrac = torch.where(
            self.spikes > 0, torch.zeros_like(self.refrac), self.refrac + 1.0)
        gate = (self.refrac >= self.refrac_steps.unsqueeze(0)).float()

        # synaptic conductance with axonal delay.  The upstream code rolls a
        # (delay+1)-slot buffer every step; reading and then overwriting slot
        # ``step mod L`` of a ring buffer is the same thing without the copy.
        delayed = self.delay_buffer[:, self.ptr, :]
        conductance_new = self.conductance * (1.0 - self.tau_syn_f) + delayed * gate
        self.delay_buffer[:, self.ptr, :] = recurrent
        self.ptr = (self.ptr + 1) % self.buf_len

        # Membrane update uses the conductance from the PREVIOUS step, which is
        # what upstream's AlphaLIF.forward does -- it passes the old tensor to
        # the neuron and only afterwards swaps in the new one.  Keeping that
        # ordering is what makes `cli verify` match.
        self.v = self.v + voltage_stim
        self.v = self.v + self.tau_mem_f * (
            self.conductance - (self.v - self.v_rest))
        self.spikes = (self.v > self.v_th).float()
        self.v = self.v - (self.v - self.v_reset) * self.spikes
        self.conductance = conductance_new - conductance_new * self.spikes
        return self.spikes


def build_engine(weights=None, neurons=None, silenced=None, params=None,
                 device=None, batch=1, seed=0, no_refractory=None):
    """Load what is needed and return a ready :class:`Engine`."""
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
    return Engine(weights, n, params=params, device=device, batch=batch,
                  seed=seed, no_refractory=no_refractory)


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
                          no_refractory=stim_all)
    t_setup = time.perf_counter() - t0

    rates = torch.zeros(B, n, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    binned = torch.zeros(n_bins, n, dtype=torch.int16, device=device)
    bin_acc = torch.zeros(n, dtype=torch.int16, device=device)

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

        s_sum = spikes.sum(0).to(torch.int16)
        bin_acc += s_sum
        counts += s_sum.to(torch.int64)
        if (step + 1) % bin_steps == 0 or step == steps - 1:
            b = min(step // bin_steps, n_bins - 1)
            binned[b] = bin_acc
            if on_bin is not None:
                on_bin(b, (step + 1) * DT, bin_acc.cpu().numpy())
            bin_acc = torch.zeros_like(bin_acc)
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
    np.savez_compressed(
        out_dir / "bins.npz",
        neuron_idx=keep.astype(np.int32),
        bins=result.bins[:, keep].astype(np.int16),
        bin_ms=np.float32(result.config.bin_ms),
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
