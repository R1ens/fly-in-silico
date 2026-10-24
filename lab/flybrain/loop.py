"""Closed loop: arena -> sensory neurons -> whole brain -> descending -> arena.

Every control step (20 ms by default) the arena is read into firing rates for
the olfactory, gustatory and looming channels, the brain is stepped 200 times
at 0.1 ms, the descending neurons of each hemisphere are counted, and that
drives speed and turn.  Nothing about the loop is taken from the connectome
except the brain in the middle -- see :mod:`flybrain.world`.

A local model watches and comments as it goes, on a worker thread so the
simulation never waits for it.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from . import data as D, sim, world
from .world import Arena, Body, BodyParams, FlyState, MOTOR, PROBES, SENSORS


@dataclass
class EpisodeConfig:
    duration_s: float = 8.0
    control_ms: float = 20.0
    seed: int = 0
    device: str | None = None
    comment_every_ms: float = 1000.0
    comment: bool = True
    model: str | None = None


class Episode:
    """Everything an episode produced: the track, the brain, the commentary."""

    def __init__(self, arena, body_params, records, counts, bins, neurons,
                 config, comments, timing, metrics, sensor_idx):
        self.arena = arena
        self.body_params = body_params
        self.records = records
        self.counts = counts
        self.bins = bins
        self.neurons = neurons
        self.config = config
        self.comments = comments
        self.timing = timing
        self.metrics = metrics
        self.sensor_idx = sensor_idx

    def as_result(self):
        """Wrap as a :class:`sim.Result` so the usual report machinery works."""
        cfg = sim.RunConfig(
            # Остановленный эпизод короче заказанного, а rates считаются как
            # counts / t_run_s: с заказанной длительностью весь мозг выглядел
            # бы вялым. Фактическую длительность кладёт сюда ``run_episode``.
            t_run_s=self.timing.get("duration_s", self.config.duration_s),
            bin_ms=self.config.control_ms,
            n_trials=1,
            seed=self.config.seed,
            device=self.timing.get("device", "cpu"),
        )
        stimuli = [
            sim.Stimulus(idx, 0.0, name=name)
            for name, idx in self.sensor_idx.items()
        ]
        return sim.Result(self.counts, self.bins, cfg, stimuli,
                          np.array([], np.int64), self.neurons, self.timing)

    def meta(self):
        return {
            "arena": self.arena.to_dict(),
            "body": asdict(self.body_params),
            "config": asdict(self.config),
            "timing": self.timing,
            "metrics": self.metrics,
            "comments": self.comments,
            "n_steps": len(self.records),
        }


# ------------------------------------------------------------- commentator


COMMENT_SYSTEM = """Ты комментируешь симуляцию: коннектом мухи (138 639
нейронов FlyWire) замкнут на простую 2D-арену. Запах с двух антенн и вкус с
лапок подаются на настоящие сенсорные нейроны, а активность нисходящих
нейронов каждого полушария превращается в скорость и поворот.

Важно: превращение «нисходящие нейроны → ноги» придумано, грудного ганглия и
тела в модели нет. Не выдавай происходящее за поведение настоящей мухи.

Муха ходит по арене на ногах, она не летает.

Пиши по-русски, одно-два коротких предложения в настоящем времени: что муха
делает прямо сейчас и что в мозге этому соответствует. Каждый раз отмечай
что-то новое — смену курса, близкую стену, еду под ногами, конкретные типы
клеток — и не повторяй формулировки предыдущего комментария. Без вступлений,
без markdown."""


class Commentator:
    """Runs the local model on a worker thread; never blocks the simulation."""

    def __init__(self, model=None, enabled=True):
        self.enabled = enabled
        self.model = model
        self.out = []
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._last = ""
        self._thread = None
        self._errors = []
        if enabled:
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()

    def submit(self, t_ms, snapshot):
        if self.enabled:
            self._q.put((t_ms, snapshot))

    def _work(self):
        from . import llm

        while True:
            item = self._q.get()
            if item is None:
                return
            t_ms, snap = item
            msgs = [
                {"role": "system", "content": COMMENT_SYSTEM},
                {"role": "user", "content":
                    (f"Предыдущий комментарий: {self._last!r}\n" if self._last else "")
                    + f"t = {t_ms/1000:.1f} с. Состояние:\n```json\n"
                    + json.dumps(snap, ensure_ascii=False) + "\n```"},
            ]
            try:
                text = llm.chat(msgs, model=self.model, temperature=0.5).strip()
            except Exception as exc:  # noqa: BLE001
                self._errors.append(str(exc))
                continue
            text = " ".join(text.split())
            with self._lock:
                self._last = text
                self.out.append({"t_ms": t_ms, "text": text})
            print(f"  [{t_ms/1000:5.1f} с] {text}")

    def finish(self, timeout=180):
        if not self.enabled:
            return []
        deadline = time.time() + timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.2)
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, deadline - time.time()))
        with self._lock:
            return sorted(self.out, key=lambda c: c["t_ms"])

    @property
    def errors(self):
        return self._errors


# ------------------------------------------------------------------ runner


def run_episode(arena: Arena | None = None, config: EpisodeConfig | None = None,
                body_params: BodyParams | None = None, neurons=None,
                weights=None, start=None, progress=True, probes=False,
                on_step=None, should_stop=None):
    config = config or EpisodeConfig()
    arena = arena or world.default_arena(config.seed)
    body = Body(body_params)
    if neurons is None:
        neurons = D.load_neurons()
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(neurons)

    sensor_idx = {name: D.select(neurons, q) for name, q in SENSORS.items()}
    motor_idx = {
        "dn_left": D.select(neurons, body.p.motor_left),
        "dn_right": D.select(neurons, body.p.motor_right),
    }
    print(f"  рулёжка: {body.p.motor_left} / {body.p.motor_right}")
    for name, idx in list(sensor_idx.items()) + list(motor_idx.items()):
        if len(idx) == 0:
            raise SystemExit(f"канал {name!r} не нашёл ни одного нейрона")
        print(f"  {name:12s} {len(idx):6d} нейронов")

    sensor_all = np.unique(np.concatenate(list(sensor_idx.values())))
    t0 = time.perf_counter()
    engine = sim.build_engine(weights=weights, neurons=neurons, device=device,
                              seed=config.seed, no_refractory=sensor_all)
    t_setup = time.perf_counter() - t0

    sensor_t = {k: torch.as_tensor(v, device=device) for k, v in sensor_idx.items()}
    motor_t = {k: torch.as_tensor(v, device=device) for k, v in motor_idx.items()}

    probe_t = {}
    if probes:
        for name, (ql, qr) in PROBES.items():
            il, ir = D.select(neurons, ql), D.select(neurons, qr)
            if len(il) and len(ir):
                probe_t[name] = (torch.as_tensor(il, device=device),
                                 torch.as_tensor(ir, device=device),
                                 len(il), len(ir))

    dt_s = config.control_ms / 1000.0
    block_steps = max(1, int(round(config.control_ms / sim.DT)))
    n_steps = int(round(config.duration_s / dt_s))
    requested_steps = n_steps

    rates = torch.zeros(1, n, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    bins = torch.zeros(n_steps, n, dtype=torch.int16, device=device)

    fly = start or FlyState(arena.width * 0.5, arena.height * 0.12,
                            math.radians(80.0))
    commentator = Commentator(model=config.model, enabled=config.comment)
    records = []
    next_comment = 0.0

    if body.p.dn_ref_hz is None or body.p.turn_gain_rad_s is None:
        # A different readout population sits at a completely different
        # baseline, so measure it before deciding what counts as full drive.
        # Long enough for the readout's own low-pass to settle, so the sigma
        # measured here is the sigma the loop will actually see.
        warm = max(5, int(round(3000.0 * body.p.dn_tau_s / config.control_ms)))
        # Drive both antennae with the same rate during the warm-up: whatever
        # asymmetry remains is the network's own, not the stimulus.
        drives0 = body.sense(arena, fly)
        mean_orn = (drives0["odour_left"] + drives0["odour_right"]) / 2.0
        rates.zero_()
        rates[0, sensor_t["odour_left"]] = mean_orn
        rates[0, sensor_t["odour_right"]] = mean_orn
        bs = block_steps * sim.DT / 1000.0
        k = math.exp(-dt_s / max(body.p.dn_tau_s, 1e-6))
        el = er = None
        tot, asyms = 0.0, []
        for _ in range(warm):
            acc = torch.zeros(n, device=device)
            for _ in range(block_steps):
                acc += engine.step(rates)[0]
            bl = float(acc[motor_t["dn_left"]].sum()) / (len(motor_idx["dn_left"]) * bs)
            br = float(acc[motor_t["dn_right"]].sum()) / (len(motor_idx["dn_right"]) * bs)
            tot += (bl + br) / 2
            # Mirror the loop's own low-pass, otherwise the sigma measured here
            # is the raw per-block noise and the gain comes out far too high.
            el = bl if el is None else k * el + (1 - k) * bl
            er = br if er is None else k * er + (1 - k) * br
            asyms.append((el - er) / (el + er + 1e-6))
        settled = asyms[len(asyms) // 3:]
        if body.p.dn_ref_hz is None:
            body.p.dn_ref_hz = max(tot / warm * 1.3, 0.1)
            print(f"  опорная частота рулёжки: {body.p.dn_ref_hz:.2f} Гц")
        body.p.asym_bias = float(np.mean(settled)) if settled else 0.0
        if body.p.turn_gain_rad_s is None:
            sd = float(np.std(settled)) if len(settled) > 1 else 0.0
            body.p.turn_gain_rad_s = body.p.turn_target_rad_s / max(sd, 1e-6)
            print(f"  усиление поворота: {body.p.turn_gain_rad_s:.0f} рад/с "
                  f"(сигма сглаженной асимметрии {sd:.5f}, "
                  f"tau {body.p.dn_tau_s:g} с)")
        print(f"  собственный перекос сети: {body.p.asym_bias:+.5f}")
        body._dn_l = body._dn_r = None  # не тащить состояние разогрева в эпизод

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=n_steps, unit="шаг", ncols=78, desc="арена")
        except ImportError:
            bar = None

    t1 = time.perf_counter()
    for step in range(n_steps):
        t_ms = step * config.control_ms
        drives = body.sense(arena, fly)

        rates.zero_()
        for name, t in sensor_t.items():
            r = float(drives[name])
            if r > 0:
                rates[0, t] = r

        acc = torch.zeros(n, dtype=torch.float32, device=device)
        for _ in range(block_steps):
            acc += engine.step(rates)[0]

        bins[step] = acc.to(torch.int16)
        counts += acc.to(torch.int64)

        block_s = block_steps * sim.DT / 1000.0
        dn_l = float(acc[motor_t["dn_left"]].sum()) / (len(motor_idx["dn_left"]) * block_s)
        dn_r = float(acc[motor_t["dn_right"]].sum()) / (len(motor_idx["dn_right"]) * block_s)

        gx, gy = arena.gradient(fly.x, fly.y)
        gnorm = math.hypot(gx, gy)
        grad_cos = ((gx * math.cos(fly.theta) + gy * math.sin(fly.theta)) / gnorm
                    if gnorm > 1e-9 else 0.0)

        food = arena.on_food(fly.x, fly.y)
        v, omega, drive, asym = body.act(dn_l, dn_r, food is not None, dt_s)
        was_feeding = body.feeding
        if food is not None and not fly.feeding:
            fly.eaten += 1
        fly.feeding = body.feeding
        bounced = world.advance(arena, fly, v, omega, dt_s)

        rec = {
            "t_ms": t_ms,
            "x": round(fly.x, 3), "y": round(fly.y, 3),
            "theta": round(fly.theta, 4),
            "v": round(v, 3), "omega": round(omega, 4),
            "odour_l": round(drives["odour_left"], 1),
            "odour_r": round(drives["odour_right"], 1),
            "c_l": round(drives["_c_left"], 4),
            "c_r": round(drives["_c_right"], 4),
            "taste": round(drives["taste"], 1),
            "loom_l": round(drives["loom_left"], 1),
            "loom_r": round(drives["loom_right"], 1),
            "grad_cos": round(grad_cos, 4),
            "dn_l": round(dn_l, 2), "dn_r": round(dn_r, 2),
            "drive": round(drive, 3), "asym": round(asym, 3),
            "feeding": bool(was_feeding),
            "eaten": int(fly.eaten),
            "bounced": bool(bounced),
            "food_dist": round(arena.nearest_food_distance(fly.x, fly.y), 2),
            "spikes": int(acc.sum()),
        }
        for name, (il, ir, nl, nr) in probe_t.items():
            rec[f"p_{name}_l"] = round(float(acc[il].sum()) / (nl * block_s), 3)
            rec[f"p_{name}_r"] = round(float(acc[ir].sum()) / (nr * block_s), 3)
        records.append(rec)

        if config.comment and t_ms >= next_comment:
            commentator.submit(t_ms, _snapshot(rec, fly, arena, neurons, acc))
            next_comment = t_ms + config.comment_every_ms

        if on_step is not None:
            on_step(step, rec, bins[step].cpu().numpy(), commentator)
        if should_stop is not None and should_stop():
            n_steps = step + 1
            records = records[:n_steps]
            bins = bins[:n_steps]
            break

        if bar is not None:
            bar.update(1)
    if device == "cuda":
        torch.cuda.synchronize()
    t_sim = time.perf_counter() - t1
    if bar is not None:
        bar.close()

    print("\nжду комментатора...")
    comments = commentator.finish()
    if commentator.errors:
        print(f"  локальная модель: {commentator.errors[0][:200]}")

    # Длительность берём по фактически пройденным шагам только если эпизод
    # оборвали: n_steps * dt_s не обязано совпадать с заказанной длительностью
    # при неделимом control_ms, и подменять её без причины нельзя.
    dur_s = n_steps * dt_s if n_steps < requested_steps else config.duration_s
    timing = {
        "setup_s": round(t_setup, 3),
        "sim_s": round(t_sim, 3),
        "steps": n_steps * block_steps,
        "device": device,
        "realtime_ratio": round(dur_s / t_sim, 4) if t_sim else None,
    }
    if n_steps < requested_steps:
        timing["duration_s"] = round(dur_s, 6)
    metrics = _metrics(records)
    return Episode(arena, body.p, records, counts.cpu().numpy(),
                   bins.cpu().numpy(), neurons, config, comments, timing,
                   metrics, sensor_idx)


def lateral_probe(neurons=None, weights=None, device=None, duration_s=10.0,
                  control_ms=20.0, base_hz=75.0, depth=0.85, period_s=2.5,
                  seed=0, candidates=None, max_lag=12, progress=True,
                  on_step=None, should_stop=None):
    """Which populations encode which antenna smells more?

    The fly is held still and the olfactory drive is rocked slowly from side to
    side, so the only thing varying is the left/right imbalance.  That keeps
    the measurement out of the feedback loop: in a moving episode the readout
    steers the fly, the fly changes the stimulus, and a correlation no longer
    says who encodes what.

    Returns one row per population with the best correlation over lags up to
    ``max_lag`` control steps, since the signal needs time to arrive.

    ``on_step(step, t_ms, n_steps)`` зовётся на каждом шаге управления, а
    ``should_stop()`` проверяется там же: все популяции считаются одним
    прогоном, поэтому шаг — единственное место, где у замера вообще есть ход.
    После остановки корреляции считаются по накопленной части записи, и каждая
    строка помечается ``partial``: r посчитан не по заказанной длительности.

    Построчного колбэка здесь нет намеренно: строки берутся не из прогона, а из
    одной общей записи после него, и все 63 посчитались бы разом за десяток
    миллисекунд — ходом задачи это не является.
    """
    if neurons is None:
        neurons = D.load_neurons()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(neurons)
    candidates = candidates or world.probe_candidates(neurons)
    print(f"кандидатов: {len(candidates)}")

    orn_l = D.select(neurons, SENSORS["odour_left"])
    orn_r = D.select(neurons, SENSORS["odour_right"])
    driven = np.unique(np.concatenate([orn_l, orn_r]))
    engine = sim.build_engine(weights=weights, neurons=neurons, device=device,
                              seed=seed, no_refractory=driven)

    # One pooling matrix instead of two index-gathers per population per step:
    # 140 small reductions per step would mean 140 GPU syncs per step.
    names = list(candidates)
    pool = torch.zeros(len(names) * 2, n, device=device)
    sizes = {}
    for i, name in enumerate(names):
        ql, qr = candidates[name]
        il, ir = D.select(neurons, ql), D.select(neurons, qr)
        pool[2 * i, torch.as_tensor(il, device=device)] = 1.0 / len(il)
        pool[2 * i + 1, torch.as_tensor(ir, device=device)] = 1.0 / len(ir)
        sizes[name] = (len(il), len(ir))

    block_steps = max(1, int(round(control_ms / sim.DT)))
    block_s = block_steps * sim.DT / 1000.0
    n_steps = int(round(duration_s * 1000.0 / control_ms))

    rates = torch.zeros(1, n, device=device)
    tl = torch.as_tensor(orn_l, device=device)
    tr = torch.as_tensor(orn_r, device=device)
    pooled = torch.zeros(n_steps, len(names) * 2, device=device)
    drive = np.zeros(n_steps, dtype=np.float64)

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=n_steps, unit="шаг", ncols=78, desc="замер")
        except ImportError:
            bar = None

    done = n_steps
    for step in range(n_steps):
        t = step * control_ms / 1000.0
        a = depth * math.sin(2 * math.pi * t / period_s)
        left, right = base_hz * (1 + a), base_hz * (1 - a)
        drive[step] = left - right
        rates.zero_()
        rates[0, tl] = left
        rates[0, tr] = right

        acc = torch.zeros(n, device=device)
        for _ in range(block_steps):
            acc += engine.step(rates)[0]
        pooled[step] = pool @ (acc / block_s)
        if bar is not None:
            bar.update(1)
        if on_step is not None:
            on_step(step, t * 1000.0, n_steps)
        if should_stop is not None and should_stop():
            done = step + 1
            break
    if device == "cuda":
        torch.cuda.synchronize()
    if bar is not None:
        bar.close()

    partial = done < n_steps
    if partial:
        # Хвост записи остался нулевым, и если считать корреляцию по нему, то
        # сигма и r уедут к нулю на ровном месте. Считаем по тому, что успели.
        pooled = pooled[:done]
        drive = drive[:done]
        n_steps = done

    P = pooled.cpu().numpy()
    rows = []
    # Drop the first period: the network needs to settle before the
    # correlation means anything.
    warm = min(n_steps // 3, int(round(period_s * 1000.0 / control_ms)))
    for i, name in enumerate(names):
        left, right = P[warm:, 2 * i], P[warm:, 2 * i + 1]
        asym = left - right
        d = drive[warm:]
        best_r, best_lag = 0.0, 0
        for lag in range(max_lag + 1):
            a_, b_ = (d[:-lag], asym[lag:]) if lag else (d, asym)
            # Оборванная запись может быть короче лага: на пустом срезе
            # corrcoef тихо отдаёт nan, а сравнение nan с лучшим r его
            # пропускает — но numpy при этом сыплет предупреждениями.
            if len(a_) < 2 or a_.std() < 1e-9 or b_.std() < 1e-9:
                continue
            r = float(np.corrcoef(a_, b_)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        nl, nr = sizes[name]
        row = {
            "population": name,
            "r": round(best_r, 3),
            "lag_ms": best_lag * control_ms,
            "asym_sd_hz": round(float(asym.std()), 3),
            "mean_hz": round(float((left.mean() + right.mean()) / 2), 2),
            "n_left": nl,
            "n_right": nr,
            "selectors": list(candidates[name]),
        }
        if partial:
            # Признак неполноты едет в самой строке: её сортируют, режут и
            # сохраняют по отдельности, и оговорка не должна при этом теряться.
            row["partial"] = True
            row["measured_s"] = round(done * control_ms / 1000.0, 3)
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["r"]))
    return rows


def probe_report(records):
    """Where does the left/right difference in the stimulus stop being visible?

    For each recorded population, the correlation between the asymmetry of the
    olfactory drive and the asymmetry of that population's firing, over the
    whole episode.  ORN is the driven population itself, so its correlation is
    the ceiling; anything far below it has lost the side information.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    d_in = (df["odour_l"] - df["odour_r"]).to_numpy()
    out = []
    for col in df.columns:
        if not col.startswith("p_") or not col.endswith("_l"):
            continue
        name = col[2:-2]
        d_out = (df[f"p_{name}_l"] - df[f"p_{name}_r"]).to_numpy()
        if d_out.std() < 1e-9 or d_in.std() < 1e-9:
            r = float("nan")
        else:
            r = float(np.corrcoef(d_in, d_out)[0, 1])
        out.append({
            "population": name,
            "r": round(r, 3),
            "asym_sd_hz": round(float(d_out.std()), 3),
            "mean_hz": round(float((df[f"p_{name}_l"] + df[f"p_{name}_r"]).mean() / 2), 2),
        })
    return out


def _snapshot(rec, fly, arena, neurons, acc):
    """The few numbers the commentator actually needs."""
    top = torch.topk(acc, 6)
    idx = top.indices.cpu().numpy()
    names = neurons.loc[idx, "label"].tolist()
    regions = neurons.loc[idx, "region"].tolist()
    return {
        "позиция": [rec["x"], rec["y"]],
        "курс_град": round(math.degrees(fly.theta), 1),
        "скорость_мм_с": rec["v"],
        "поворот_град_с": round(math.degrees(rec["omega"]), 1),
        "запах_слева_Гц": rec["odour_l"],
        "запах_справа_Гц": rec["odour_r"],
        "вкус_Гц": rec["taste"],
        "стена_впереди": rec["loom_l"] + rec["loom_r"] > 1,
        "нисходящие_слева_Гц": rec["dn_l"],
        "нисходящие_справа_Гц": rec["dn_r"],
        "до_еды_мм": rec["food_dist"],
        "ест": rec["feeding"],
        "съедено": fly.eaten,
        "самые_активные": [f"{a} ({b})" for a, b in zip(names, regions)],
    }


def _metrics(records):
    """Did the loop do anything, or is it a random walk?

    The headline number is ``grad_alignment``: the mean cosine between the
    heading and the up-gradient direction of the odour field.  A random walk
    averages 0 whatever the arena looks like, so it is comparable between runs
    in a way a distance-to-food number is not.

    ``turn_to_odour`` is kept as a secondary number, but read it carefully:
    turns persist for about 100 ms and the gradient direction persists for
    seconds, so consecutive steps are nowhere near independent and the
    effective sample size is far below ``turn_steps_counted``.  Compare
    conditions with :func:`bench`, which treats a whole episode as one sample.
    """
    toward = 0
    counted = 0
    for r in records:
        dc = r["c_l"] - r["c_r"]
        if abs(dc) < 1e-4 or abs(r["omega"]) < 1e-3:
            continue
        counted += 1
        if (dc > 0) == (r["omega"] > 0):
            toward += 1
    half = max(1, len(records) // 2)
    d_first = float(np.mean([r["food_dist"] for r in records[:half]]))
    d_second = float(np.mean([r["food_dist"] for r in records[half:]]))
    return {
        "steps": len(records),
        "grad_alignment": round(float(np.mean([r["grad_cos"] for r in records])), 4),
        "turn_to_odour": round(toward / counted, 3) if counted else None,
        "turn_steps_counted": counted,
        "food_dist_first_half_mm": round(d_first, 2),
        "food_dist_second_half_mm": round(d_second, 2),
        "food_visits": int(records[-1]["eaten"]) if records else 0,
        "food_steps": int(sum(1 for r in records if r["taste"] > 0)),
        "mean_speed_mm_s": round(float(np.mean([r["v"] for r in records])), 2),
        "mean_dn_hz": round(float(np.mean([(r["dn_l"] + r["dn_r"]) / 2
                                           for r in records])), 2),
        "wall_bounces": int(sum(1 for r in records if r["bounced"])),
    }


def bench(n_runs=6, duration_s=4.0, neurons=None, weights=None, device=None,
          body_params=None, seed0=100, on_episode=None, should_stop=None):
    """Normal versus swapped antennae, one number per episode.

    Each episode is a single independent sample, which is the only way these
    conditions can honestly be compared: within an episode the heading and the
    gradient both drift slowly, so per-step counts look far more significant
    than they are.

    ``on_episode(cond, i, n, metrics)`` зовётся после каждого эпизода, где
    ``cond`` — название условия, ``i`` — номер эпизода с единицы.
    ``should_stop()`` проверяется между эпизодами и уходит внутрь эпизода,
    иначе «стоп» ждал бы конца текущего прогона. У оборванной серии в
    ``n_runs`` стоит фактическое число эпизодов, а не заказанное: по нему
    считаются sd и sem, и соврать здесь значит соврать про значимость.
    """
    if neurons is None:
        neurons = D.load_neurons()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if weights is None:
        weights = D.load_weights(device=device)

    base = body_params or BodyParams()
    out = {}
    stopped = False
    for label, swap in (("норма", False), ("перестановка антенн", True)):
        vals, dists, visits, eps = [], [], [], []
        for i in range(n_runs):
            bp = BodyParams(**{**asdict(base), "swap_antennae": swap})
            cfg = EpisodeConfig(duration_s=duration_s, comment=False,
                                seed=seed0 + i, device=device)
            ep = run_episode(config=cfg, body_params=bp, neurons=neurons,
                             weights=weights, progress=False,
                             should_stop=should_stop)
            m = ep.metrics
            vals.append(m["grad_alignment"])
            dists.append(m["food_dist_second_half_mm"] - m["food_dist_first_half_mm"])
            visits.append(m["food_visits"])
            # Сид эпизода записывается рядом со значением. Без него два
            # набора эпизодов нельзя сложить: непересечение проверить
            # нечем, а повторить отдельный эпизод — тем более.
            eps.append({"i": i + 1, "seed": int(cfg.seed),
                        "grad_alignment": round(m["grad_alignment"], 4),
                        "food_dist_change_mm": round(
                            m["food_dist_second_half_mm"]
                            - m["food_dist_first_half_mm"], 2),
                        "food_visits": int(m["food_visits"])})
            print(f"  {label:20s} [{i+1}/{n_runs}] "
                  f"совпадение с градиентом {m['grad_alignment']:+.3f}")
            if on_episode is not None:
                on_episode(label, i + 1, n_runs, m)
            if should_stop is not None and should_stop():
                stopped = True
                break
        n_done = len(vals)
        out[label] = {
            "n_runs": n_done,
            "duration_s": duration_s,
            # Записываем, с какой популяции рулили: без этого по сохранённой
            # серии уже не понять, что именно сравнивалось.
            "readout": base.motor_left.replace("+side:left", ""),
            "dn_tau_s": base.dn_tau_s,
            "grad_alignment_mean": round(float(np.mean(vals)), 4),
            "grad_alignment_sd": round(float(np.std(vals, ddof=1)), 4),
            "grad_alignment_sem": round(
                float(np.std(vals, ddof=1) / math.sqrt(n_done)), 4),
            "runs": [round(v, 4) for v in vals],
            "seeds": [int(e["seed"]) for e in eps],
            "seed0": int(seed0),
            "episodes": eps,
            "food_dist_change_mm": round(float(np.mean(dists)), 2),
            "food_visits_total": int(sum(visits)),
        }
        if stopped:
            # Второе условие не считаем: сравнивать его с оборванным первым
            # нечестно, а делать вид, что серия целая, — тем более.
            break
    return out


def load(out_dir, neurons=None) -> Episode:
    """Rebuild an episode from a saved directory, for re-rendering."""
    import pandas as pd

    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "episode.json").read_text(encoding="utf-8"))
    records = pd.read_parquet(out_dir / "track.parquet").to_dict("records")

    a = meta["arena"]
    arena = Arena(
        a["width"], a["height"],
        [world.Food(f["x"], f["y"], f["r"]) for f in a["foods"]],
        [world.Odour(o["x"], o["y"], o["s"], o["sigma"]) for o in a["odours"]],
    )
    result = sim.load(out_dir, neurons=neurons)
    return Episode(
        arena, BodyParams(**meta["body"]), records, result.counts, result.bins,
        result.neurons, EpisodeConfig(**meta["config"]), meta["comments"],
        meta["timing"], meta["metrics"],
        {s.name: s.indices for s in result.stimuli},
    )


def save(episode: Episode, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode.json").write_text(
        json.dumps(episode.meta(), ensure_ascii=False, indent=2), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(episode.records).to_parquet(out_dir / "track.parquet")
    result = episode.as_result()
    sim.save(result, out_dir)
    return out_dir
