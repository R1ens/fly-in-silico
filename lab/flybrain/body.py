"""Мозг в теле: коннектом FlyWire управляет ногами NeuroMechFly v2.

Отличие от :mod:`flybrain.loop` — там арена и физика были написаны нами, а
«нисходящие нейроны → скорость и поворот» я придумывал сам. Здесь обе
половины заменены на опубликованные:

* запах берётся с антенн мухи в MuJoCo, а не считается из гауссова шлейфа;
* нисходящий выход уходит в ``HybridTurningController`` — тот самый
  контроллер походки из NeuroMechFly v2, который ожидает ровно два числа,
  «нисходящий сигнал поворота». Ноги, их фазы, прилипание и спотыкание
  считает он.

Придуманными остаются два места, и оба отмечены в коде: пересчёт
интенсивности запаха в частоту рецепторов и пересчёт асимметрии нисходящих
в амплитуду шага. Заместителем грудного ганглия работает контроллер FlyGym.

Семантика команды (прочитана из turning_controller.py): ``action[0]`` задаёт
амплитуду шага левых трёх ног, ``action[1]`` — правых, знак задаёт
направление. Замер показал, что более сильные левые ноги уводят муху вправо,
поэтому связь мозга с ногами перекрёстная: левое полушарие — правые ноги.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

import numpy as np

from . import data as D, sim, world


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

    # --- запись видео ---
    video: bool = False
    fps: int = 30
    play_speed: float = 0.2  # замедление: муха быстрая, иначе ничего не видно

    # --- живое окно MuJoCo ---
    view: bool = False
    view_every: int = 1  # обновлять окно раз в столько шагов управления


DEFAULT_SOURCES = np.array([
    [25.0, 12.0, 1.5],
    [38.0, -16.0, 1.5],
])


class EmbodiedRun:
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
        return {
            "config": asdict(self.config),
            "sources": self.sources.tolist(),
            "timing": self.timing,
            "metrics": self.metrics,
            "calibration": self.calib,
            "n_steps": len(self.records),
        }


def _build_body(cfg: BodyConfig, sources, video_path=None):
    from flygym_gymnasium import Fly, YawOnlyCamera
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import default_leg_sensor_placements
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    arena = OdorArena(
        odor_source=sources,
        peak_odor_intensity=np.ones((len(sources), 1)),
    )
    fly = Fly(
        enable_adhesion=True,
        enable_olfaction=True,
        contact_sensor_placements=default_leg_sensor_placements,
        init_pose="stretch",
        control="position",
        spawn_pos=(0.0, 0.0, 0.2),
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
                video_path=None, on_body=None):
    """Эпизод в теле. ``on_body(body)`` зовётся один раз, когда тело собрано
    и сброшено: живому окну нужен доступ к физической модели, чтобы снимать
    кадры своим ``mujoco.Renderer`` — камера FlyGym отмеряет кадры по
    модельному времени и для живого вида не годится.
    """
    import torch

    cfg = cfg or BodyConfig()
    sources = DEFAULT_SOURCES if sources is None else np.asarray(sources, float)
    if neurons is None:
        neurons = D.load_neurons()
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(neurons)

    # --- кого слушаем и кого зажигаем ---
    cands = world.probe_candidates(neurons)
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

    dt_s = cfg.control_ms / 1000.0
    brain_steps = max(1, int(round(cfg.control_ms / sim.DT)))
    body_steps = max(1, int(round(dt_s / cfg.physics_dt)))
    block_s = brain_steps * sim.DT / 1000.0
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
    body, cams = _build_body(cfg, sources, video_path if cfg.video else None)
    obs, _ = body.reset(seed=cfg.seed)
    t_setup = time.perf_counter() - t0
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
        return rl, rr

    def step_brain():
        acc = torch.zeros(n, device=device)
        for _ in range(brain_steps):
            acc += engine.step(rates)[0]
        dl = float(acc[t_ml].sum()) / (len(motor_l) * block_s)
        dr = float(acc[t_mr].sum()) / (len(motor_r) * block_s)
        return acc, dl, dr

    # --- разогрев: симметричный вход, меряем опору, перекос и шум ---
    li, ri = read_odour(obs)
    odour_ref = max((li + ri) / 2.0 * cfg.odour_ref_factor, 1e-9)
    warm = max(5, int(round(3000.0 * cfg.dn_tau_s / cfg.control_ms)))
    k = math.exp(-dt_s / max(cfg.dn_tau_s, 1e-6))
    mean_i = (li + ri) / 2.0
    drive_brain(mean_i, mean_i, 0.0, odour_ref)
    el = er = None
    tot, asyms = 0.0, []
    stopped = False
    for _ in range(warm):
        _, dl, dr = step_brain()
        tot += (dl + dr) / 2
        el = dl if el is None else k * el + (1 - k) * dl
        er = dr if er is None else k * er + (1 - k) * dr
        asyms.append((el - er) / (el + er + 1e-9))
        if should_stop is not None and should_stop():
            # Разогрев при большой tau длиннее самого эпизода, и ждать его
            # конца значит не отвечать на «стоп» десяток секунд.
            stopped = True
            break
    settled = asyms[len(asyms) // 3:]
    bias = float(np.mean(settled)) if settled else 0.0
    noise = float(np.std(settled)) if len(settled) > 1 else 1e-4
    calib = {
        "odour_ref": round(odour_ref, 8),
        "dn_ref_hz": round(tot / warm, 4),
        "asym_bias": round(bias, 6),
        "asym_noise": round(noise, 6),
        "warm_steps": warm,
    }
    print(f"  опора {calib['dn_ref_hz']} Гц, перекос {bias:+.5f}, "
          f"шум {noise:.5f}, яркость насыщения {odour_ref:.5f}")

    # --- эпизод ---
    if stopped:
        # Один шаг всё же делаем: на пустой записи не считаются ни метрики,
        # ни итог, и вызывающему пришлось бы разбирать особый случай.
        n_steps = min(n_steps, 1)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    bins = torch.zeros(n_steps, n, dtype=torch.int16, device=device)
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
        nearest = int(np.argmin(d))
        li, ri = read_odour(obs)
        taste = cfg.grn_hz if d[nearest] < cfg.food_radius_mm else 0.0
        rl, rr = drive_brain(li, ri, taste, odour_ref)

        acc, dl, dr = step_brain()
        bins[step] = acc.to(torch.int16)
        counts += acc.to(torch.int64)

        el = dl if el is None else k * el + (1 - k) * dl
        er = dr if er is None else k * er + (1 - k) * dr
        asym = (el - er) / (el + er + 1e-9) - bias
        # Придуманная часть: нормируем на собственный шум и разводим ноги.
        z = float(np.clip(asym / max(noise, 1e-9), -3.0, 3.0))
        gain = cfg.amp_span / 3.0
        amp_l = cfg.amp_base - gain * z   # перекрёстно: левое полушарие -> правые ноги
        amp_r = cfg.amp_base + gain * z
        if not cfg.cross:
            amp_l, amp_r = amp_r, amp_l
        action = np.array([np.clip(amp_l, cfg.amp_min, cfg.amp_max),
                           np.clip(amp_r, cfg.amp_min, cfg.amp_max)])

        for _ in range(body_steps):
            obs, _, term, trunc, _ = body.step(action)
            if cfg.video:
                body.render()
            if term or trunc:
                break
        if viewer is not None and step % cfg.view_every == 0:
            if not viewer.is_running():
                print("  окно закрыто, останавливаюсь")
                n_steps = step + 1
                bins = bins[:n_steps]
                break
            viewer.sync()

        nx, ny, _ = _pose(obs)
        to_src = sources[nearest, :2] - np.array([nx, ny])
        norm = float(np.linalg.norm(to_src))
        grad_cos = float((to_src @ np.array([math.cos(theta), math.sin(theta)])) / norm) \
            if norm > 1e-6 else 0.0

        rec = {
            "t_ms": step * cfg.control_ms,
            "x": round(nx, 3), "y": round(ny, 3), "theta": round(theta, 4),
            "odour_l": round(li, 7), "odour_r": round(ri, 7),
            "orn_l": round(rl, 1), "orn_r": round(rr, 1),
            "taste": round(taste, 1),
            "dn_l": round(dl, 3), "dn_r": round(dr, 3),
            "asym": round(asym, 5), "z": round(z, 3),
            "amp_l": round(float(action[0]), 3), "amp_r": round(float(action[1]), 3),
            "dist": round(float(np.linalg.norm(sources[nearest, :2] - [nx, ny])), 2),
            "grad_cos": round(grad_cos, 4),
            "spikes": int(acc.sum()),
        }
        records.append(rec)
        if on_step is not None:
            on_step(step, rec, bins[step].cpu().numpy())
        if should_stop is not None and should_stop():
            n_steps = step + 1
            bins = bins[:n_steps]
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
    dur_s = n_steps * dt_s if cut else cfg.duration_s
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
    metrics = _metrics(records, dt_s)
    return EmbodiedRun(records, counts.cpu().numpy(), bins.cpu().numpy(),
                       neurons, cfg, sources, timing, metrics, calib, sensor_idx)


def _metrics(records, dt_s):
    if not records:
        return {}
    # При одном шаге (отмена на разогреве) вторая половина выходила
    # пустой, и `np.mean` отдавал nan с предупреждением. Тогда обе
    # половины — это сам единственный шаг.
    half = len(records) // 2
    d = [r["dist"] for r in records]
    d_first = d[:half] if half else d
    d_second = d[half:] if half < len(d) else d
    path = sum(math.dist((records[i]["x"], records[i]["y"]),
                         (records[i - 1]["x"], records[i - 1]["y"]))
               for i in range(1, len(records)))
    return {
        "steps": len(records),
        "grad_alignment": round(float(np.mean([r["grad_cos"] for r in records])), 4),
        "dist_first_half_mm": round(float(np.mean(d_first)), 2),
        "dist_second_half_mm": round(float(np.mean(d_second)), 2),
        "dist_start_mm": round(d[0], 2),
        "dist_end_mm": round(d[-1], 2),
        "path_mm": round(path, 2),
        "speed_mm_s": round(path / max(len(records) * dt_s, 1e-9), 2),
        "food_steps": int(sum(1 for r in records if r["taste"] > 0)),
        "mean_dn_hz": round(float(np.mean([(r["dn_l"] + r["dn_r"]) / 2
                                           for r in records])), 3),
    }


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
    return EmbodiedRun(records, res.counts, res.bins, res.neurons, cfg,
                       sources, timing, meta.get("metrics") or {},
                       meta.get("calibration") or {}, sensor_idx)


def save(run: EmbodiedRun, out_dir):
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embodied.json").write_text(
        json.dumps(run.meta(), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(run.records).to_parquet(out_dir / "track.parquet")
    sim.save(run.as_result(), out_dir)
    return out_dir
