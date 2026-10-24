"""Реестр задач: спека на входе, прогон на выходе.

Единственное место, где прогон исполняется. И командная строка, и окно
собирают `RunSpec` и отдают его сюда — поэтому новая возможность заводится
один раз, а не дважды.

Контекст (`Ctx`) отвечает за то, куда уходят ход и сообщения. В командной
строке он печатает, в работнике — складывает в очередь к окну. Сами задачи
про это не знают.

Отдельная забота — отмена. Движок опрашивает `should_stop` на границе бина
или шага, и при остановке числа надо чинить: `sim.run` оставляет в
`config.t_run_s` полную длительность, из-за чего `rates = counts / t_run_s`
выходят заниженными, а в хвосте `bins` остаются нулевые бины. Окно будет
останавливать прогоны постоянно, так что это чинится здесь, а не оставляется
на совесть вызывающего.
"""

import inspect
import io
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import data as D
from . import spec as S

__all__ = ["Ctx", "PrintCtx", "JobResult", "run", "KINDS", "estimate"]

KINDS = S.KINDS


def _call(fn, *args, **kw):
    """Позвать, отбросив именованные параметры, которых у функции нет.

    Колбэки прогресса добавляются в разные модули постепенно, и реестр не
    должен падать из-за того, что в одной функции `on_episode` уже есть, а в
    другой ещё нет.
    """
    try:
        ok = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        ok = None
    if ok is not None:
        kw = {k: v for k, v in kw.items() if k in ok}
    return fn(*args, **kw)


class Ctx:
    """Куда уходят ход, сообщения и находки задачи."""

    def __init__(self, emit=None, should_stop=None, neurons=None,
                 weights=None, device=None, job_id="job", out_dir=None):
        self._emit = emit
        self._stop = should_stop
        self.job_id = job_id
        self.device = device
        self._neurons = neurons
        self._weights = weights
        # Папка прогона известна ДО счёта, а не после. Иначе видео
        # эпизода писать некуда, и оно уходило в текущий каталог под
        # одним и тем же именем: каждый следующий эпизод затирал
        # предыдущий.
        self.out_dir = Path(out_dir) if out_dir else None
        # Разбор команд на ходу. Ставится работником; без него задача
        # ведёт себя как прежде и ни на какой паузе не стоит.
        self._drain = None
        self.paused = False
        self.stepping = 0
        self.applied = []
        self._later = []          # просятся на будущее модельное время
        self._asked_stop = False
        self._t0 = time.perf_counter()

    # -- данные ---------------------------------------------------------

    @property
    def neurons(self):
        if self._neurons is None:
            self._neurons = D.load_neurons()
        return self._neurons

    def weights(self):
        if self._weights is None:
            self._weights = D.load_weights(device=self.device)
        return self._weights

    # -- наружу ---------------------------------------------------------

    def emit(self, msg):
        if self._emit is not None:
            self._emit(msg)

    def status(self, state, phase="", progress=0.0, text=""):
        from . import ipc

        self.emit(ipc.Status(self.job_id, state, phase, progress, text))

    def log(self, text, level="info"):
        from . import ipc

        self.emit(ipc.Log(self.job_id, level, str(text)))

    def frame(self, t_ms, counts, extra=None):
        from . import ipc

        idx, val = ipc.pack_frame(counts)
        self.emit(ipc.Frame(self.job_id, float(t_ms), idx, val, extra))

    def series(self, t_ms, values):
        from . import ipc

        self.emit(ipc.Series(self.job_id, float(t_ms), dict(values)))

    def artifact(self, kind, path):
        from . import ipc

        self.emit(ipc.Artifact(self.job_id, kind, str(path)))

    def should_stop(self):
        if self._asked_stop:
            return True
        return bool(self._stop()) if self._stop is not None else False

    # -- команды на ходу ------------------------------------------------

    def control_point(self, t_ms, state=None):
        """Безопасная граница шага: разобрать очередь и, если надо, встать.

        Зовётся задачей один раз на шаг управления — там, где состояние
        модели целостно. Шаг физики посреди вычисления прерывать нельзя,
        поэтому всё, что пришло за время шага, применяется здесь.

        Правило порядка детерминировано и простое: команды применяются в
        том порядке, в каком пришли в очередь, и если несколько меняют
        одно свойство, побеждает последняя. Переставлять их нельзя и
        терять нельзя — каждая получает подтверждение со своим исходом.

        На паузе очередь продолжает разбираться, но модель стоит: новые
        события копятся и применяются там же, на границе, когда придёт
        `resume` или `step`.
        """
        if self._drain is None:
            return False
        self._take(t_ms, state)
        while self.paused and not self.should_stop():
            if self.stepping > 0:
                self.stepping -= 1
                return True
            time.sleep(0.005)
            self._take(t_ms, state)
        return not self.paused

    def _take(self, t_ms, state):
        """Разобрать то, что успело прийти. Ничего не ждёт.

        Сначала созревшие отложенные — они пришли раньше, — потом новые.
        Порядок внутри каждой группы тот же, в каком команды легли в
        очередь.
        """
        from . import ipc

        ready = [c for c in self._later
                 if c.at_t_ms is None or t_ms + 1e-9 >= float(c.at_t_ms)]
        if ready:
            self._later = [c for c in self._later if c not in ready]
        for cmd in ready + list(self._drain()):
            outcome, values, why = "applied", None, ""
            if cmd.at_t_ms is not None and t_ms + 1e-9 < float(cmd.at_t_ms):
                # Рано: команда просится на будущее модельное время.
                # Возвращаем её в собственную очередь ожидания.
                self._later.append(cmd)
                continue
            if cmd.action == "pause":
                self.paused = True
            elif cmd.action == "resume":
                self.paused = False
                self.stepping = 0
            elif cmd.action == "step":
                self.paused = True
                self.stepping += int((cmd.payload or {}).get("n", 1))
            elif cmd.action == "stop":
                self.paused = False
                self._asked_stop = True
            elif cmd.action == "set":
                values, why = self._apply_set(cmd.payload, state, t_ms)
                outcome = "applied" if not why else "rejected"
            else:
                outcome, why = "rejected", "неизвестное действие %r" % cmd.action
            rec = ipc.Applied(cmd.cmd_id, cmd.action, float(t_ms),
                              round(time.perf_counter() - self._t0, 4),
                              outcome, values, why)
            self.applied.append(rec)
            self.emit(rec)

    def _apply_set(self, payload, state, t_ms=None):
        """Изменить параметр идущего расчёта. Пока — только то, что есть.

        Проверка значений делается здесь, в работнике: из окна приходит
        словарь, и верить ему нельзя. Чего `state` не умеет менять, то
        отклоняется с причиной, а не применяется наполовину.

        Применённая правка получает модельное время. Без него по
        сохранённому прогону нельзя сказать, на какой миллисекунде она
        сработала: время жило только в подтверждении, которое уходит в
        окно и на диск не попадает, — а бриф требует фиксировать именно
        время применения.
        """
        if state is None:
            return None, "этот вид прогона менять на ходу не умеет"
        try:
            values = state.apply(dict(payload or {}))
        except (ValueError, TypeError, KeyError) as exc:
            return None, str(exc)[:160]
        log = getattr(state, "log", None)
        if log and t_ms is not None:
            log[-1]["модельное, мс"] = round(float(t_ms), 3)
        return values, ""

    # -- крючки для работника -------------------------------------------

    def on_body(self, body):
        """Тело собрано. Работник цепляет сюда свой рендерер MuJoCo.

        Камера FlyGym для живого вида не годится: она отмеряет кадры по
        модельному времени и отдала три кадра на двадцать один вызов.
        """

    def step(self, rec):
        """Шаг управления сделан. Работник кладёт сюда кадр картинки."""


class PrintCtx(Ctx):
    """Контекст командной строки: ход печатается, кадры никому не нужны."""

    def __init__(self, **kw):
        super().__init__(**kw)

    def status(self, state, phase="", progress=0.0, text=""):
        if text:
            print(text)

    def log(self, text, level="info"):
        print(text)

    def frame(self, t_ms, counts, extra=None):
        pass

    def series(self, t_ms, values):
        pass

    def artifact(self, kind, path):
        pass


class JobResult:
    """Что задача отдала: объект прогона, сводка и где он лёг."""

    def __init__(self, kind, obj=None, summary=None, out_dir=None,
                 cancelled=False):
        self.kind = kind
        self.obj = obj
        self.summary = dict(summary or {})
        self.out_dir = out_dir
        self.cancelled = cancelled


# ---------------------------------------------------------------- правка чисел

def _truncate(result, steps_done, total_steps):
    """Починить `Result` после ранней остановки.

    Длительность в конфиге остаётся заказанной, а частоты считаются делением
    на неё — значит после отмены все частоты занижены, причём тем сильнее,
    чем раньше остановились. Плюс хвост `bins` остаётся нулевым и портит
    тепловую карту.
    """
    if total_steps <= 0 or steps_done >= total_steps:
        return result
    frac = max(steps_done / total_steps, 1e-6)
    result.config.t_run_s = result.config.t_run_s * frac
    n_bins = max(1, int(round(result.bins.shape[0] * frac)))
    result.bins = result.bins[:n_bins]
    return result


# ---------------------------------------------------------------- задачи

def _job_stim(spec, ctx):
    from . import sim

    neurons = ctx.neurons
    ctx.status("setup", "стимулы", 0.05)
    stimuli = spec.to_stimuli(neurons)
    silenced = spec.silenced(neurons)
    for s in stimuli:
        ctx.log(f"{s.name} ({len(s.indices)} @ {s.rate_hz:g} Гц)")
    if len(silenced):
        ctx.log(f"заглушено: {len(silenced)} нейронов")

    cfg = spec.to_run_config()
    n_bins = max(1, int(round(cfg.t_run_s * 1000.0 / cfg.bin_ms)))
    if spec.max_bins and n_bins * len(neurons) > spec.max_bins:
        raise ValueError(
            f"слишком длинный прогон: {n_bins} бинов на {len(neurons)} нейронов. "
            "Увеличьте бин или сократите длительность.")

    seen = {"bins": 0}

    def on_bin(i, t_ms, counts):
        seen["bins"] = i + 1
        ctx.frame(t_ms, counts)
        ctx.status("running", "счёт", (i + 1) / n_bins)
        # У стимуляции безопасная граница — конец бина: внутри бина
        # движок крутится блоком, и состояние трогать нельзя.
        ctx.control_point(t_ms)

    ctx.status("running", "счёт", 0.0)
    res = sim.run(stimuli, silenced=silenced, config=cfg, neurons=neurons,
                  weights=ctx._weights, progress=False,
                  on_bin=on_bin, should_stop=ctx.should_stop)
    cancelled = seen["bins"] < n_bins
    if cancelled:
        res = _truncate(res, seen["bins"], n_bins)
        ctx.log(f"остановлено на {seen['bins']} бине из {n_bins}")

    summary = {
        "kind": "stim",
        "n_active": int(len(res.active())),
        "total_spikes": int(res.counts.sum()),
        "timing": res.timing,
        "cancelled": cancelled,
    }
    return JobResult("stim", res, summary, cancelled=cancelled)


def _job_arena(spec, ctx):
    from . import live, loop, world

    neurons = ctx.neurons
    weights = ctx.weights()
    body = spec.to_body_params(neurons)

    if spec.repeat:
        return _series(spec, ctx, loop.bench, neurons, weights,
                       body_params=body)

    cfg = spec.to_episode_config()
    # Арена собирается здесь, а не внутри `run_episode`, ровно по той же
    # команде и с тем же семенем. Нужна она снаружи для одного: живые
    # условия должны менять ТОТ объект, который модель читает на каждом
    # шаге, а не его копию. То же и с `body` — `world.Body` держит
    # `BodyParams` ссылкой и перечитывает его в `sense` каждый шаг.
    arena = world.default_arena(cfg.seed)
    state = live.ArenaLive(arena, body)
    n_steps = max(1, int(round(cfg.duration_s * 1000.0 / cfg.control_ms)))
    seen = {"n": 0}

    def on_step(step, rec, bins_row, commentator=None):
        seen["n"] = step + 1
        ctx.frame(rec["t_ms"], bins_row, extra=rec)
        ctx.series(rec["t_ms"], {k: rec[k] for k in
                                 ("dn_l", "dn_r", "odour_l", "odour_r",
                                  "v", "omega", "grad_cos", "food_dist")
                                 if k in rec})
        ctx.step(rec)
        ctx.status("running", "эпизод", (step + 1) / n_steps)
        ctx.control_point(rec["t_ms"], state=state)

    ctx.status("warmup", "разогрев", 0.0)
    ep = _call(loop.run_episode, arena=arena, config=cfg, body_params=body,
               neurons=neurons, weights=weights, progress=False,
               probes=spec.arena.probes if spec.arena else False,
               on_step=on_step, should_stop=ctx.should_stop)
    _record_edits(ep.timing, state)
    summary = {"kind": "arena", "metrics": ep.metrics, "timing": ep.timing,
               "cancelled": seen["n"] < n_steps}
    return JobResult("arena", ep, summary, cancelled=summary["cancelled"])


def _record_edits(timing, state):
    """Живые правки — часть результата, а не история нажатий.

    Без этого эпизод, у которого посреди счёта поменяли усиление
    рецепторов или поменяли антенны местами, на диске выглядел бы как
    чистый. Журнал уезжает в `timing`, оттуда в `episode.json`, в отчёт и
    в каталог; помеченные правки отдельно собираются в оговорку.
    """
    if not getattr(state, "log", None):
        return
    timing["живые правки"] = list(state.log)
    marked = sorted({m for r in state.log for m in r.get("помечено", [])})
    if marked:
        timing.setdefault("оговорки", []).append(
            "условия менялись на ходу, и среди правок есть меняющие смысл "
            "прогона: %s. С эталонными прогонами этот результат не "
            "сравнивается." % ", ".join(marked))


def _job_body(spec, ctx):
    from . import body as B, live

    neurons = ctx.neurons
    weights = ctx.weights()
    cfg = spec.to_body_config(ctx.device)

    if spec.repeat:
        # Серия в теле получает папку: без неё каждый эпизод писал
        # `walk.mp4` в текущий каталог поверх предыдущего.
        return _series(spec, ctx, B.bench, neurons, weights, base=cfg,
                       out_dir=ctx.out_dir)

    endless = cfg.duration_s is not None and cfg.duration_s <= 0
    n_steps = (0 if endless
               else max(1, int(round(cfg.duration_s * 1000.0
                                     / cfg.control_ms))))
    seen = {"n": 0}
    # Живые условия сеанса. Состояние создаётся заранее, а ссылки на
    # настоящую модель MuJoCo получает внутри `run_episode`, когда тело
    # собрано и сброшено. Исходные условия (`live_initial`) применяются
    # там же — до разогрева, поэтому они и есть условия, а не правка.
    state = live.BodyLive(initial=(spec.body.live_initial
                                   if spec.body else None))

    def on_step(step, rec, bins_row):
        seen["n"] = step + 1
        # Кадр снимается ДО отправки состояния: тогда поза камеры,
        # которую работник кладёт в `last_cam`, относится к тому самому
        # кадру, что уехал в общую память, и окно может перевести щелчок
        # по сцене в координаты модели. Иначе поза отставала бы на шаг.
        ctx.step(rec)
        cam = getattr(ctx, "last_cam", None)
        # Поза едет отдельной копией, а не в самой записи шага: записи
        # уходят в `track.parquet`, и словарь в столбце там не нужен.
        ctx.frame(rec["t_ms"], bins_row,
                  extra=(dict(rec, cam=cam) if cam else rec))
        ctx.series(rec["t_ms"], {k: rec[k] for k in
                                 ("dn_l", "dn_r", "odour_l", "odour_r",
                                  "amp_l", "amp_r", "dist", "grad_cos")
                                 if k in rec})
        if endless:
            # Доли у сеанса без конца нет, и показывать её нечестно:
            # вместо неё окно получает прожитое модельное время.
            ctx.status("running", "сеанс", 0.0,
                       text="" if step % 50 else
                       "сеанс идёт: %.1f с модельного времени"
                       % (rec["t_ms"] / 1000.0))
        else:
            ctx.status("running", "эпизод", (step + 1) / n_steps)
        # Граница шага управления: здесь состояние целостно, и здесь
        # разбираются команды, пришедшие за время шага.
        ctx.control_point(rec["t_ms"], state=state)

    ctx.status("setup", "сборка тела", 0.0)
    video = None
    if cfg.video:
        if ctx.out_dir is None:
            raise ValueError("видео просят, а папку прогона не задали: "
                             "укажите имя папки в out/, иначе файл некуда "
                             "положить")
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        video = ctx.out_dir / "walk.mp4"
    run = _call(B.run_episode, cfg=cfg, neurons=neurons, weights=weights,
                sources=(spec.body.sources if spec.body else None),
                progress=False, on_step=on_step, should_stop=ctx.should_stop,
                on_body=ctx.on_body, video_path=video, live=state)
    _record_edits(run.timing, state)
    if state.log:
        # Исходные условия — часть результата: без них по журналу правок
        # не восстановить, от чего сеанс отталкивался. Но писать их в
        # КАЖДЫЙ эпизод тела нельзя: у обычного прогона никаких исходных
        # условий не задавалось, и новый блок в научном результате
        # появлялся бы без причины. Признак — непустой журнал: он
        # непуст ровно тогда, когда что-то применяли.
        run.timing["исходные условия"] = state.start
    cancelled = (ctx.should_stop() if endless else seen["n"] < n_steps)
    summary = {"kind": "body", "metrics": run.metrics, "timing": run.timing,
               "calibration": run.calib, "cancelled": cancelled}
    return JobResult("body", run, summary, cancelled=cancelled)


def _series(spec, ctx, bench_fn, neurons, weights, **kw):
    """Серия эпизодов: оба `bench` зовутся отсюда одинаково."""
    done = {"n": 0}
    total = spec.repeat * 2        # две условные ветки: норма и контроль

    def on_episode(cond, i, n, metrics):
        done["n"] += 1
        ctx.status("running", f"{cond} {i + 1}/{n}", done["n"] / max(total, 1))
        align = (metrics or {}).get("grad_alignment")
        ctx.log(f"{cond} {i + 1}/{n}: совпадение "
                + (f"{align:+.3f}" if align is not None else "—"))

    res = _call(bench_fn, n_runs=spec.repeat, duration_s=spec.episode_t,
                neurons=neurons, weights=weights, device=ctx.device,
                seed0=spec.repeat_seed0, progress=False,
                on_episode=on_episode, should_stop=ctx.should_stop, **kw)
    summary = {"kind": spec.kind + "-bench", "bench": res,
               "cancelled": ctx.should_stop()}
    return JobResult(spec.kind, res, summary, cancelled=summary["cancelled"])


def _job_lateral(spec, ctx):
    from . import loop

    neurons = ctx.neurons
    weights = ctx.weights()
    lat = spec.lateral or S.LateralSpec()
    n_steps = max(1, int(round(spec.t * 1000.0 / lat.control_ms)))

    def on_step(step, t_ms, total):
        ctx.status("running", "перебор популяций",
                   (step + 1) / max(total or n_steps, 1))

    ctx.status("running", "перебор популяций", 0.0)
    rows = _call(loop.lateral_probe, neurons=neurons, weights=weights,
                 device=ctx.device, duration_s=spec.t,
                 control_ms=lat.control_ms, base_hz=lat.base_hz,
                 depth=lat.depth, period_s=lat.period_s, seed=spec.seed,
                 max_lag=lat.max_lag, progress=False,
                 on_step=on_step, should_stop=ctx.should_stop)
    for r in rows[:lat.top]:
        ctx.log(f"{r['population']:16s} r {r['r']:+.3f}  "
                f"лаг {r['lag_ms']:g} мс  {r['mean_hz']:g} Гц")
    summary = {"kind": "lateral", "n": len(rows),
               "top": rows[:lat.top], "cancelled": ctx.should_stop()}
    return JobResult("lateral", rows, summary, cancelled=summary["cancelled"])


def _job_verify(spec, ctx):
    """Сверка с апстримом. Печать подпроцесса уходит в журнал, а не в никуда."""
    import os
    import subprocess

    from . import sim

    up = D.UPSTREAM
    sys.path.insert(0, str(up / "code"))
    try:
        import benchmark  # noqa: F401  из upstream/code
        exp = benchmark.get_experiment((spec.verify or S.VerifySpec()).experiment)
        allowed = list(getattr(benchmark, "T_RUN_VALUES_SEC", []) or [])
    finally:
        sys.path.pop(0)

    # Длительность проверяется ДО счёта и по настоящему списку апстрима.
    # Он на чужом значении печатает «not in allowed values» и выходит с
    # нулём: наш прогон посчитался бы впустую, а сравнение подхватило бы
    # прошлый файл и выдало бы чужие числа за сверку.
    if allowed and float(spec.t) not in [float(v) for v in allowed]:
        raise ValueError(
            "апстрим принимает только %s с, а заказано %g с: при другом "
            "значении он ничего не считает и выходит с нулём"
            % (", ".join("%g" % float(v) for v in allowed), float(spec.t)))

    neurons = ctx.neurons
    # Апстрим хранит в эксперименте не индексы, а `root_id` нейронов FlyWire:
    # ключ называется `neu_exc`, частота — `stim_rate`. Перевод root_id в
    # индексы — тот же, что в `cli.cmd_verify`, чтобы у окна и у командной
    # строки не расходились числа на одном прогоне.
    name = (spec.verify or S.VerifySpec()).experiment
    idx = neurons[neurons.root_id.isin(exp["neu_exc"])].idx.to_numpy()
    if not len(idx):
        raise ValueError(f"эксперимент {name!r}: ни один из "
                         f"{len(exp['neu_exc'])} root_id не нашёлся в таблице")
    ctx.log(f"эксперимент {exp.get('name', name)}: {len(idx)} нейронов")
    ctx.status("running", "мой прогон", 0.0)
    stim = [sim.Stimulus(indices=np.asarray(idx, np.int64),
                         rate_hz=float(exp.get("stim_rate", 200.0)),
                         name=name)]
    cfg = sim.RunConfig(t_run_s=spec.t, bin_ms=50.0, n_trials=1, seed=0,
                        device=ctx.device)
    n_bins = max(1, int(round(spec.t * 1000.0 / 50.0)))
    seen = {"bins": 0}

    def on_bin(i, t_ms, counts):
        seen["bins"] = i + 1
        ctx.frame(t_ms, counts)
        ctx.status("running", "мой прогон", 0.5 * (i + 1) / n_bins)

    mine = sim.run(stim, config=cfg, neurons=neurons, weights=ctx._weights,
                   progress=False, on_bin=on_bin,
                   should_stop=ctx.should_stop)
    base = _verify_head(spec, name, exp, idx, cfg, mine)
    cancelled = seen["bins"] < n_bins
    if cancelled:
        # Сверять обрезанный прогон с полным прогоном апстрима нельзя:
        # частоты несравнимы. Останавливаемся честно, без сравнения, и
        # записываем это как отмену, а не как пройденную сверку.
        mine = _truncate(mine, seen["bins"], n_bins)
        ctx.log(f"остановлено на {seen['bins']} бине из {n_bins}: "
                "апстрим не запускался, сравнивать не с чем")
        base.update(outcome="cancelled", compared=False,
                    why="остановлено на %d бине из %d: апстрим не "
                        "запускался" % (seen["bins"], n_bins),
                    bins_done=seen["bins"], bins_requested=n_bins,
                    t_run_s=float(mine.config.t_run_s))
        return JobResult("verify", mine,
                         {"kind": "verify", "cancelled": True,
                          "n_active": int(len(mine.active())),
                          "total_spikes": int(mine.counts.sum()),
                          "verify": base},
                         cancelled=True)

    ctx.status("running", "прогон апстрима", 0.5)
    # Запоминаем, что лежало в результатах апстрима ДО запуска: иначе
    # нечем отличить свежий файл от прошлогоднего.
    res_dir = up / "data" / "results"
    before = {q: q.stat().st_mtime for q in res_dir.glob("**/pytorch_*.parquet")}
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "main.py", "--pytorch", "--t_run", str(spec.t),
         "--n_run", "1", "--no_log_file"],
        cwd=str(up), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env)
    for line in (proc.stdout or "").splitlines()[-12:]:
        ctx.log(line)
    summary = {"kind": "verify", "rc": proc.returncode,
               "experiment": name, "n_stimulated": int(len(idx)),
               "n_active": int(len(mine.active())),
               "total_spikes": int(mine.counts.sum()),
               "cancelled": False}
    base["rc"] = int(proc.returncode)
    if proc.returncode != 0:
        for line in (proc.stderr or "").splitlines()[-8:]:
            ctx.log(line)
        summary["compared"] = False
        summary["why"] = "апстрим вернул код %d" % proc.returncode
        base.update(outcome="upstream_failed", compared=False,
                    why=summary["why"],
                    stderr_tail=(proc.stderr or "").splitlines()[-4:])
        summary["verify"] = base
        ctx.log("апстрим упал: сравнивать не с чем")
        return JobResult("verify", mine, summary)

    ctx.status("running", "сравнение", 0.9)
    cmp_, why = _compare_upstream(up, mine, neurons, spec.t, before)
    if cmp_ is None:
        summary["compared"] = False
        summary["why"] = why
        base.update(outcome="no_matching_upstream", compared=False, why=why,
                    upstream_seen=sorted(q.name for q in
                                         res_dir.glob("**/pytorch_*.parquet")))
        summary["verify"] = base
        ctx.log(why)
        return JobResult("verify", mine, summary)
    summary.update(cmp_)
    summary["compared"] = True
    base.update(cmp_)
    base.update(outcome="ok", compared=True)
    summary["verify"] = base
    ctx.log("активных: мой %d, апстрим %d"
            % (cmp_["n_active_mine"], cmp_["n_active_upstream"]))
    ctx.log("Jaccard %.3f, корреляция частот %.4f"
            % (cmp_["jaccard"], cmp_["corr"]))
    return JobResult("verify", mine, summary)


def _verify_head(spec, name, exp, idx, cfg, mine):
    """Паспорт сверки: что с чем сравнивалось и на каких настройках.

    Без него результат сверки жил только в сводке задачи и на диск не
    попадал: по сохранённой папке нельзя было узнать ни версию данных, ни
    исход проверки, ни даже то, что это вообще была сверка.
    """
    import hashlib

    comp = D.PATH_COMP
    conn = D.PATH_CONN

    def stamp(path):
        try:
            st = path.stat()
            return {"файл": path.name, "байт": int(st.st_size),
                    "изменён": time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(st.st_mtime))}
        except OSError:
            return {"файл": path.name, "байт": None, "изменён": None}

    ids = ",".join(str(x) for x in exp.get("neu_exc", []))
    return {
        "version": 1,
        "experiment": name,
        "experiment_name": exp.get("name", name),
        "dataset": {
            "release": "FlyWire v783",
            "completeness": stamp(comp),
            "connectivity": stamp(conn),
            "n_neurons": int(len(mine.neurons)),
        },
        "stimulus": {
            "n_root_ids": len(exp.get("neu_exc", [])),
            "n_matched": int(len(idx)),
            "root_ids_sha1": hashlib.sha1(ids.encode()).hexdigest()[:16],
            "rate_hz": float(exp.get("stim_rate", 200.0)),
        },
        "params": {"t_run_s": float(cfg.t_run_s), "bin_ms": float(cfg.bin_ms),
                   "n_trials": int(cfg.n_trials), "seed": int(cfg.seed),
                   "device": str(cfg.device)},
        "threshold_hz": 0.5,
        "limits": [
            "сравниваются средние частоты за весь прогон, а не отдельные "
            "спайки: времена спайков апстрим и этот движок пишут по-разному",
            "порог активности 0,5 Гц задан здесь же и к биологии "
            "отношения не имеет",
            "апстрим считается подпроцессом с его собственными "
            "настройками по умолчанию: совпадают только t_run и n_run",
        ],
        "errors": [],
    }


def _compare_upstream(up, mine, neurons, t_run_s, before=None):
    """Сравнить свой прогон с parquet апстрима ТОЙ ЖЕ длительности.

    Порог активности, Jaccard и корреляция считаются ровно теми же
    выражениями, что в `cli.cmd_verify`: иначе у окна и у командной
    строки получались бы разные числа на одном прогоне.

    Подбирать «последний попавшийся» parquet нельзя. Частоты апстрима
    считаются делением его спайков на НАШУ длительность, и файл от
    прогона другой длины даёт частоты, завышенные ровно в отношение
    длительностей: при `--t 0.2` сравнение шло с файлом на 1 с, и
    суммарные спайки расходились впятеро. Корреляция это переживает —
    она не зависит от масштаба, — а вот всё остальное нет.
    """
    import pandas as pd

    res_dir = up / "data" / "results"
    pat = f"**/pytorch_t{t_run_s}s_n1.parquet"
    cands = sorted(res_dir.glob(pat))
    if not cands:
        seen = sorted(q.name for q in res_dir.glob("**/pytorch_*.parquet"))
        return None, (
            "апстрим не оставил файла на %g с: искал %s, нашёл %s. "
            "Сравнивать с файлом другой длительности нельзя — частоты "
            "апстрима делятся на нашу длительность и разойдутся во "
            "столько же раз." % (t_run_s, pat.replace("**/", ""),
                                 ", ".join(seen) if seen else "ничего"))
    path = cands[-1]
    stale = bool(before) and before.get(path, -1.0) == path.stat().st_mtime
    spikes = pd.read_parquet(path)
    up_counts = np.zeros(len(neurons), dtype=np.int64)
    np.add.at(up_counts, spikes["neuron_index"].to_numpy(), 1)
    up_rates = up_counts / float(t_run_s)
    my_rates = mine.rates
    a = my_rates >= 0.5
    b = up_rates >= 0.5
    both = a | b
    corr = (float(np.corrcoef(my_rates[both], up_rates[both])[0, 1])
            if both.sum() > 2 else float("nan"))
    out = {"upstream_file": path.name,
           "upstream_mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(
                                               path.stat().st_mtime)),
           "upstream_fresh": not stale,
           "n_active_mine": int(a.sum()),
           "n_active_upstream": int(b.sum()),
           "jaccard": float((a & b).sum() / max(int((a | b).sum()), 1)),
           "corr": corr,
           "spikes_mine": float(my_rates.sum() * t_run_s),
           "spikes_upstream": float(up_rates.sum() * t_run_s)}
    if stale:
        # Файл подходит по длительности, но его не трогали: апстрим
        # ничего не пересчитал, и мы сравниваемся с прошлым прогоном.
        out["stale_warning"] = (
            "файл апстрима не обновился на этом запуске: сравнение идёт "
            "с прежним его прогоном той же длительности")
    return out, ""


def _job_agent(spec, ctx):
    """Агент сам порождает прогоны: вложенных задач не заводим.

    Его опыты сохраняются папками `exp-NN`, как и раньше, а окно видит их
    как сообщения и артефакты — иначе пришлось бы делать задачу, которая
    порождает задачи, и отменять такое было бы нечем.
    """
    from . import agent as A

    a = spec.agent or S.AgentSpec()
    if not a.goal:
        raise ValueError("цель не задана")

    def on_event(kind, payload):
        payload = payload or {}
        if kind == "experiment":
            path = payload.get("report") or payload.get("dir")
            if path:
                ctx.artifact("report", path)
            ctx.log("опыт: " + str(payload.get("note") or payload.get("dir")))
        elif kind in ("thought", "observation", "answer", "error"):
            text = str(payload.get("text") or payload)[:400]
            if text:
                ctx.log(text, level="error" if kind == "error" else "info")
        ctx.status("running", "шаг агента", -1.0)

    res = _call(A.run_loop, a.goal, model=spec.model, max_steps=a.max_steps,
                t_run_s=a.t_run_s, open_reports=False, out_name=spec.out,
                on_event=on_event, should_stop=ctx.should_stop)
    summary = {"kind": "agent", "goal": a.goal,
               "answer": (res or {}).get("answer", ""),
               "experiments": len((res or {}).get("experiments") or []),
               "dir": str((res or {}).get("dir") or "")}
    if summary["dir"]:
        ctx.artifact("run", summary["dir"])
    return JobResult("agent", res, summary)


_REGISTRY = {
    "stim": _job_stim,
    "arena": _job_arena,
    "body": _job_body,
    "lateral": _job_lateral,
    "verify": _job_verify,
    "agent": _job_agent,
}


# ---------------------------------------------------------------- запуск

def run(spec, ctx=None, capture=True):
    """Исполнить спеку. Возвращает `JobResult`.

    `capture` перехватывает печать задач: `print` и tqdm внутри `loop` и
    `body` иначе уходят в никуда, когда задача живёт в работнике.
    """
    if isinstance(spec, dict):
        spec = S.from_dict(spec)
    ctx = ctx or PrintCtx()
    fn = _REGISTRY.get(spec.kind)
    if fn is None:
        raise ValueError(f"неизвестный вид задачи {spec.kind!r}")

    t0 = time.perf_counter()
    ctx.status("setup", spec.kind, 0.0)
    old = (sys.stdout, sys.stderr)
    if capture and not isinstance(ctx, PrintCtx):
        sys.stdout = sys.stderr = _LogStream(ctx)
    try:
        res = fn(spec, ctx)
    except (Exception, SystemExit) as exc:                     # noqa: BLE001
        # `SystemExit` ловится наравне с обычной ошибкой нарочно. Счётные
        # модули сообщают о неверных входных данных через `SystemExit`
        # (так удобно командной строке), а он наследует `BaseException` и
        # проходил мимо: процесс работника умирал молча, окно оставалось
        # в «Подготовке» навсегда и даже не узнавало, что считать больше
        # некому.
        sys.stdout, sys.stderr = old
        tb = "".join(traceback.format_exception(exc))
        text = str(exc) or exc.__class__.__name__
        ctx.status("error", spec.kind, 1.0, text[:200])
        return JobResult(spec.kind, None, {"error": text, "tb": tb[-1200:]})
    finally:
        sys.stdout, sys.stderr = old

    res.summary.setdefault("elapsed_s", round(time.perf_counter() - t0, 2))
    ctx.status("done" if not res.cancelled else "done", spec.kind, 1.0)
    return res


class _LogStream(io.TextIOBase):
    """Поток, который режет печать задачи на строки и шлёт их как сообщения.

    Режет и по `\\n`, и по `\\r`: tqdm обновляет строку возвратом каретки, и
    без этого прогресс-бар пришёл бы одной бесконечной строкой.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.buf = ""

    def write(self, s):
        self.buf += s
        while True:
            i = max(self.buf.find("\n"), self.buf.find("\r"))
            if i < 0:
                break
            line, self.buf = self.buf[:i].strip(), self.buf[i + 1:]
            if line:
                self.ctx.log(line)
        return len(s)

    def flush(self):
        if self.buf.strip():
            self.ctx.log(self.buf.strip())
            self.buf = ""


def estimate(spec):
    """Грубая оценка времени счёта, секунды.

    Нужна окну, чтобы не предлагать молча запустить получасовую серию.
    Основание — замеры на этой машине: 1167 шагов движка в секунду, тело
    замедляет ещё примерно вдвое.
    """
    steps_per_s = 1167.0
    if spec.kind in ("stim", "verify"):
        return spec.t * 10000.0 / steps_per_s
    if spec.kind == "arena":
        n = spec.repeat or 1
        return n * (2 if spec.repeat else 1) * spec.episode_t * 10000.0 / steps_per_s
    if spec.kind == "body":
        n = (spec.repeat or 1) * (2 if spec.repeat else 1)
        return n * (spec.episode_t * 10000.0 / steps_per_s * 2.0 + 10.0)
    if spec.kind == "lateral":
        return spec.t * 10000.0 / steps_per_s + 15.0
    if spec.kind == "agent":
        a = spec.agent
        steps = a.max_steps if a else 6
        # шаг это ответ модели плюс короткий прогон
        return steps * (12.0 + (a.t_run_s if a else 0.5) * 10000.0 / steps_per_s)
    return 0.0
