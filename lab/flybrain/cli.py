"""Command line for the lab.

Запускается через fly.cmd в корне проекта (или console.cmd, если мышкой).

    fly doctor
    fly demos
    fly select "class:gustatory+side:left"
    fly run --activate "class:gustatory+side:left" --rate 200 --t 1.0 --pulse 100 600
    fly run --activate "class:olfactory+side:left" --silence "type:KC" --explain
    fly arena --t 5 --motor ALPN --dn-tau 0.5
    fly lateral --t 10 --top 30
    fly agent "найди путь от сахара до мотонейронов хоботка"
    fly index
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

from . import data as D


def _slug(text, limit=48):
    s = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "-", str(text)).strip("-").lower()
    return s[:limit] or "run"


def _vision_presets():
    """Имена зрительных пресетов — для подсказки argparse.

    Берутся у `spec`, а не перечисляются здесь: два списка на одно
    множество неизбежно расходятся, и расходиться они будут в
    подсказке, то есть незаметно.
    """
    from . import spec as S

    return tuple(sorted(S.VISION_PRESETS))


def _object_presets():
    from . import objects as O

    return tuple(sorted(O.PRESETS))


# Подсказка собирается на построении парсера, то есть при каждом вызове
# `fly`. Оба импорта дешёвые: `spec` и `objects` не тянут ни MuJoCo, ни
# torch.
S_VISION_PRESETS = _vision_presets()


def _track_from_args(args):
    """Траектория объектов из `--track`: имя пресета или файл json.

    Разворачивается ЗДЕСЬ, в список отрезков: в `spec.json` прогона
    должно лежать то, что считалось, а не имя, у которого завтра другой
    смысл. Отказ громкий — молча посчитать эпизод без объекта хуже, чем
    не посчитать его вовсе.
    """
    from . import objects as O

    raw = getattr(args, "vis_track", None)
    if not raw:
        return None
    if str(raw) in O.PRESETS:
        return O.preset(str(raw))
    p = Path(str(raw))
    if not p.exists():
        raise SystemExit(
            "траектория %r — ни имя пресета, ни существующий файл. "
            "Пресеты: %s" % (raw, ", ".join(sorted(O.PRESETS))))
    try:
        legs = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit("траектория %s не читается: %s" % (p, exc))
    # Проверяется здесь же, до всякой сборки тела: отказ на двадцатой
    # секунде сборки MuJoCo стоит дороже, чем на нулевой.
    O.Track(legs)
    return legs


def cmd_fetch_annotations(args):
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
        "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
    )
    D.CACHE.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, D.PATH_ANNOT)
    print(f"saved {D.PATH_ANNOT} ({D.PATH_ANNOT.stat().st_size/1e6:.1f} MB)")
    D.load_neurons(rebuild=True)
    print("neuron table rebuilt")


def cmd_select(args):
    df = D.load_neurons()
    idx = D.select(df, args.query)
    print(f"{len(idx)} нейронов по запросу {args.query!r}")
    if len(idx) == 0:
        return
    sub = df.loc[idx]
    print("\nпо областям:")
    print(sub["region"].value_counts().head(15).to_string())
    print("\nпо типам:")
    print(sub["label"].value_counts().head(20).to_string())
    if args.list:
        import pandas as pd

        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(D.describe(df, idx[:500]).to_string())


def cmd_regions(args):
    df = D.load_neurons()
    from . import regions as R

    rows = []
    for name, cnt in df["region"].value_counts().items():
        rows.append((name, cnt, R.super_region(name), R.NEUROPILS.get(name, ("", ""))[0]))
    width = max(len(r[0]) for r in rows)
    print(f"{'обл.':<{width}}  {'нейронов':>8}  {'группа':<16}  название")
    for name, cnt, grp, ru in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<{width}}  {cnt:>8}  {grp:<16}  {ru}")
    print(f"\nвсего {len(rows)} областей, {len(df)} нейронов")
    print("\nисточник метки области:")
    print(df["region_src"].value_counts().to_string())


def _spec_from_args(args, kind):
    """Собрать `RunSpec` из разобранных флагов.

    Одна схема на командную строку и на окно: иначе всякая новая возможность
    заводится дважды и они расходятся. Выражения ниже перенесены из прежних
    `cmd_*` слово в слово, поведение не меняется.
    """
    from . import spec as S

    sp = S.RunSpec(
        kind=kind,
        out=getattr(args, "out", None),
        title=getattr(args, "title", None),
        open_report=not getattr(args, "no_open", False),
        t=getattr(args, "t", 1.0),
        seed=getattr(args, "seed", 0),
        device=getattr(args, "device", None),
        explain=bool(getattr(args, "explain", False)),
        model=getattr(args, "model", None),
    )

    if kind == "stim":
        pulse = args.pulse or [None, None]
        terms = []
        for i, q in enumerate(args.activate or []):
            rate = args.rate[i] if i < len(args.rate) else args.rate[-1]
            name = args.name[i] if args.name and i < len(args.name) else q
            terms.append(S.StimTerm(query=q, rate_hz=rate,
                                    t_on_ms=pulse[0] or 0.0,
                                    t_off_ms=pulse[1], name=name))
        sp.bin_ms = args.bin
        sp.stim = S.StimSpec(activate=terms, silence=args.silence,
                             trials=args.trials)
    elif kind == "arena":
        sp.bin_ms = args.control_ms
        sp.comment = not args.no_comment
        sp.comment_every_ms = args.comment_every
        sp.repeat = args.bench
        sp.repeat_t = args.bench_t
        sp.repeat_seed0 = args.bench_seed
        sp.arena = S.ArenaSpec(
            readout=args.motor, swap_antennae=args.control,
            turn_sign=args.turn_sign, dn_ref_hz=args.dn_ref,
            turn_gain_rad_s=args.turn_gain, dn_tau_s=args.dn_tau,
            turn_target_rad_s=args.turn_target, probes=args.probe)
    elif kind == "body":
        sp.bin_ms = args.control_ms
        sp.repeat = args.bench
        sp.repeat_seed0 = args.bench_seed
        sp.body = S.BodySpec(
            readout=args.readout, dn_tau_s=args.dn_tau,
            control_ms=args.control_ms, physics_dt=args.physics_dt,
            swap_antennae=args.control, cross=not args.no_cross,
            video=args.video, play_speed=args.play_speed, view=args.view,
            motor=getattr(args, "motor_decoder", "legacy"),
            motor_weights=getattr(args, "motor_weights", None),
            gf_bridge=getattr(args, "gf_bridge", "none"),
            head_stab=bool(getattr(args, "head_stab", False)),
            neck_kp=getattr(args, "neck_kp", None),
            probes=not getattr(args, "no_probes", False),
            spmv=getattr(args, "spmv", None),
            vision=bool(getattr(args, "vision", False)),
            vision_hz=getattr(args, "vision_hz", 100.0),
            vision_adapter=getattr(args, "vision_adapter", "retina"),
            vision_max_hz=getattr(args, "vision_max_hz", 150.0),
            vision_ref=getattr(args, "vision_ref", 0.5),
            vision_channel=getattr(args, "vision_channel", "max"),
            vision_eyes=getattr(args, "vision_eyes", "both"),
            vision_mode=getattr(args, "vision_mode", "whiten"),
            vision_assign=getattr(args, "vision_assign", "capacity"),
            vision_shuffle=bool(getattr(args, "vision_shuffle", False)),
            vision_light=getattr(args, "vision_light", 1.0),
            vision_contrast=getattr(args, "vision_contrast", 1.0),
            vision_record=getattr(args, "vision_record", None),
            vision_replay=getattr(args, "vision_replay", None),
            vision_bridge=getattr(args, "vision_bridge", None),
            vision_bridge_boundary=getattr(
                args, "vision_bridge_boundary", "T4/T5"),
            vision_bridge_gain=getattr(args, "vision_bridge_gain", 200.0),
            vision_bridge_max_hz=getattr(
                args, "vision_bridge_max_hz", 1000.0),
            vision_bridge_delay_ms=getattr(
                args, "vision_bridge_delay_ms", 30.0),
            vision_bridge_shuffle=bool(getattr(
                args, "vision_bridge_shuffle", False)),
            vision_bridge_subtypes_shuffle=bool(getattr(
                args, "vision_bridge_subtypes_shuffle", False)),
            vis_objects=int(getattr(args, "vis_objects", 0) or 0),
            vis_track=_track_from_args(args))
        # Пресет накладывается ПОВЕРХ флагов и в самом конце, а не
        # вместо них: иначе `--vision-preset loom-dark --vision-hz 250`
        # молча считался бы на сотне. Поля, которых пресет не называет,
        # остаются как заданы флагами.
        preset = getattr(args, "vision_preset", None)
        if preset:
            for k, v in S.vision_preset(preset).items():
                setattr(sp.body, k, v)
    elif kind == "lateral":
        sp.lateral = S.LateralSpec(base_hz=args.base, depth=args.depth,
                                   period_s=args.period, top=args.top)
    return sp


def _build_stimuli(args, df):
    """Стимулы из флагов. Пустой запрос — выход с сообщением, как и было."""
    sp = _spec_from_args(args, "stim")
    try:
        return sp.to_stimuli(df)
    except ValueError as exc:
        raise SystemExit(str(exc))


def cmd_run(args):
    from . import report, sim

    df = D.load_neurons()
    stimuli = _build_stimuli(args, df)
    silenced = D.select(df, args.silence) if args.silence else np.array([], np.int64)

    for s in stimuli:
        print(f"стимул {s.name!r}: {len(s.indices)} нейронов @ {s.rate_hz:g} Гц")
    if len(silenced):
        print(f"заглушено: {len(silenced)} нейронов")

    cfg = _spec_from_args(args, "stim").to_run_config()
    result = sim.run(stimuli, silenced=silenced, config=cfg, neurons=df)
    print("время:", json.dumps(result.timing))

    name = args.out or (
        time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(stimuli[0].name if stimuli else "idle")
    )
    out_dir = D.OUT / name
    sim.save(result, out_dir)

    rt = result.rate_table()
    print(f"\nактивных нейронов: {len(rt)}, спайков: {result.counts.sum()}")
    print("\nтоп областей (ср. частота среди активных):")
    g = (
        rt.groupby("region")
        .agg(n=("rate_hz", "size"), mean_hz=("rate_hz", "mean"), max_hz=("rate_hz", "max"))
        .sort_values("n", ascending=False)
    )
    print(g.head(12).round(1).to_string())
    print("\nтоп нейронов:")
    print(rt.head(15)[["label", "cell_class", "region", "rate_hz"]].round(1).to_string())

    narrative = ""
    if args.explain:
        from . import llm

        print("\nпрошу локальную модель разобрать результат...")
        try:
            narrative = llm.explain(result, model=args.model)
            (out_dir / "narrative.md").write_text(narrative, encoding="utf-8")
            print("\n" + narrative)
        except Exception as exc:  # noqa: BLE001
            print(f"локальная модель недоступна: {exc}")

    html = report.render(
        result, out_dir / "report.html", narrative=narrative, title=args.title
    )
    print(f"\nотчёт: {html}")
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())
    return result


# Набор готовых опытов живёт в schema: окно и команда должны запускать
# ровно одно и то же.
from .spec import DEMOS  # noqa: E402


def cmd_demos(args):
    """Run the whole demo set, loading the connectome once."""
    import torch

    from . import report, sim

    df = D.load_neurons()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("загружаю матрицу связей...")
    weights = D.load_weights(device=device)

    for i, demo in enumerate(DEMOS, 1):
        print(f"\n{'='*70}\n[{i}/{len(DEMOS)}] {demo.out} — {demo.about}")
        try:
            stimuli = demo.to_stimuli(df)
        except ValueError as exc:
            raise SystemExit(f"демо {demo.out}: {exc}")
        silenced = demo.silenced(df)
        for s in stimuli:
            print(f"  стимул {s.name}: {len(s.indices)} нейронов @ {s.rate_hz:g} Гц")
        if len(silenced):
            print(f"  заглушено: {len(silenced)}")

        cfg = sim.RunConfig(t_run_s=demo.t, bin_ms=10.0, device=device)
        res = sim.run(stimuli, silenced=silenced, config=cfg, neurons=df,
                      weights=weights)
        out_dir = D.OUT / demo.out
        sim.save(res, out_dir)
        rt = res.rate_table()
        print(f"  активных: {len(rt)}, спайков: {res.counts.sum()}, "
              f"счёт {res.timing['sim_s']} c")

        narrative = ""
        if demo.explain and not args.no_explain:
            from . import llm

            try:
                narrative = llm.explain(res, model=args.model)
                (out_dir / "narrative.md").write_text(narrative, encoding="utf-8")
                print("  разбор локальной модели записан")
            except Exception as exc:  # noqa: BLE001
                print(f"  локальная модель недоступна: {exc}")

        html = report.render(res, out_dir / "report.html",
                             narrative=narrative, title=demo.title)
        print(f"  отчёт: {html}")

    print(f"\nготово, отчёты в {D.OUT}")


def cmd_app(args):
    """Нативное окно: свой растеризатор на Taichi, без браузера и без Qt.

    Два окна живут рядом, пока новое не согласовано: `--old` открывает
    прежнее. Возврат к прежнему интерфейсу должен оставаться возможным до
    конца проверки, и стоит он одной ветки.
    """
    if getattr(args, "old", False):
        from .ui import app

        app.run(device=args.device)
        return
    from .desk import app

    app.run(device=args.device,
            seconds=float(getattr(args, "seconds", 0) or 0),
            selftest=getattr(args, "selftest", None))


def cmd_font(args):
    """Испечь атлас глифов для окна.

    Атлас — артефакт, а не кэш рантайма: набор символов задан списком, и
    дырки в покрытии гарнитуры должны всплывать здесь, а не в окне.
    """
    from .ui import font as F

    roles = args.roles or list(F.ROLES)
    for role in roles:
        for size in args.sizes:
            a = F.bake(size=size, font_path=args.font, force=args.force,
                       role=role)
            print(f"{role:7s} {a.font} кегль {a.size}: {len(a.index)} глифов, "
                  f"ремень {a.pixels.shape[0]}×{a.pixels.shape[1]} "
                  f"= {a.pixels.nbytes / 1e6:.3f} МБ, строка {a.gh}")
            if a.missing:
                print("  НЕТ ГЛИФОВ: " + " ".join(a.missing))
    print(f"\nготово, атласы в {D.CACHE}")


def cmd_ui(args):
    """Окно управления: коннектом остаётся в памяти между прогонами."""
    from . import server

    server.serve(port=args.port, open_browser=not args.no_open,
                 device=args.device)


def cmd_index(args):
    """Build the hub page over everything in out/."""
    from . import index, loop, report, sim

    if args.rebuild:
        df = D.load_neurons()
        targets = [p.parent for p in sorted(D.OUT.glob("**/report.html"))]
        for i, run_dir in enumerate(targets, 1):
            depth = len(run_dir.relative_to(D.OUT).parts)
            home = "../" * depth + "index.html"
            try:
                if (run_dir / "episode.json").exists():
                    ep = loop.load(run_dir, neurons=df)
                    res, arena = ep.as_result(), report.build_arena_payload(ep)
                elif (run_dir / "meta.json").exists():
                    res, arena = sim.load(run_dir, neurons=df), None
                else:
                    continue
                nf = run_dir / "narrative.md"
                report.render(
                    res, run_dir / "report.html", arena=arena, home=home,
                    title=report.title_of(run_dir),
                    narrative=nf.read_text(encoding="utf-8") if nf.exists() else "",
                )
                print(f"  [{i}/{len(targets)}] {run_dir.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(targets)}] {run_dir.name}: {str(exc)[:90]}")

    path = index.render()
    entries = index.scan()
    kinds = {}
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"собрано {len(entries)} записей: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"страница: {path}")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())


def cmd_doctor(args):
    """Check that everything this project needs is actually in place."""
    import shutil

    ok = True
    rows = []

    def check(name, good, detail, hint=""):
        nonlocal ok
        if not good:
            ok = False
        rows.append((name, good, detail, hint if not good else ""))

    check("python", True, sys.executable)

    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}"
        if cuda:
            free, total = torch.cuda.mem_get_info(0)
            detail += (f", CUDA есть, {torch.cuda.get_device_name(0)}, "
                       f"{free/1024**3:.1f}/{total/1024**3:.1f} ГБ свободно")
        else:
            detail += ", CUDA НЕТ — считать будет на процессоре, очень медленно"
        check("torch", True, detail,
              "переставьте: uv pip install --index-url "
              "https://download.pytorch.org/whl/cu128 --reinstall torch")
        if not cuda:
            rows[-1] = ("torch", False, detail, rows[-1][3])
            ok = False
    except Exception as exc:  # noqa: BLE001
        check("torch", False, str(exc)[:120], "окружение не собрано")

    for mod in ("pandas", "numpy", "pyarrow", "scipy", "tqdm"):
        try:
            m = __import__(mod)
            check(mod, True, getattr(m, "__version__", "есть"))
        except ImportError as exc:
            check(mod, False, str(exc)[:80],
                  f"uv pip install {mod}")

    for label, path, hint in (
        ("данные модели", D.PATH_COMP, "клонируйте upstream заново"),
        ("связи", D.PATH_CONN, "клонируйте upstream заново"),
        ("аннотации FlyWire", D.PATH_ANNOT, "fly fetch-annotations"),
    ):
        exists = path.exists()
        size = f"{path.stat().st_size/1e6:.1f} МБ" if exists else "нет файла"
        check(label, exists, f"{size}  {path}", hint)

    for label, path, hint in (
        ("таблица нейронов", D.NEURONS_CACHE, "соберётся при первом запуске"),
        ("кэш весов", D.WEIGHT_CACHE, "соберётся при первом запуске (~30 с)"),
    ):
        exists = path.exists()
        size = f"{path.stat().st_size/1e6:.1f} МБ" if exists else "будет собран"
        rows.append((label, True, size, "" if exists else hint))

    try:
        df = D.load_neurons()
        check("нейроны читаются", len(df) > 100000,
              f"{len(df)} нейронов, {df['region'].nunique()} областей")
    except Exception as exc:  # noqa: BLE001
        check("нейроны читаются", False, str(exc)[:120], "fly fetch-annotations")

    try:
        from . import llm

        models = llm.available_models()
        has = llm.DEFAULT_MODEL in models
        rows.append(("Ollama", True,
                     f"{len(models)} моделей: {', '.join(models[:4])}"
                     + ("…" if len(models) > 4 else ""), ""))
        rows.append((f"модель {llm.DEFAULT_MODEL}", has,
                     "есть" if has else "нет",
                     "" if has else f"ollama pull {llm.DEFAULT_MODEL}, "
                                    "или задайте --model из списка выше"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("Ollama", False, str(exc)[:100],
                     "запустите Ollama; без неё считается всё, "
                     "кроме комментариев и агента"))

    D.OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(D.ROOT).free / 1024 ** 3
    check("место на диске", free > 2.0, f"{free:.1f} ГБ свободно",
          "освободите пару гигабайт под отчёты")

    width = max(len(r[0]) for r in rows)
    print()
    for name, good, detail, hint in rows:
        mark = "OK  " if good else "СБОЙ"
        print(f"  [{mark}] {name:<{width}}  {detail}")
        if hint:
            print(f"         {' ' * width}  -> {hint}")
    print()
    if ok:
        print("всё на месте, можно работать:  fly arena --t 5 --motor ALPN --dn-tau 0.5")
    else:
        print("есть проблемы — см. строки СБОЙ выше")
    return 0 if ok else 1


def cmd_body(args):
    """Мозг в теле NeuroMechFly v2: ноги считает MuJoCo, запах даёт симулятор."""
    import torch

    from . import body, jobs, report

    df = D.load_neurons()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("загружаю матрицу связей...")
    weights = D.load_weights(device=device)

    cfg = _spec_from_args(args, "body").to_body_config(device)

    if args.bench:
        # Папка серии известна заранее: иначе видео эпизодов писать некуда,
        # и каждый следующий эпизод затирал бы предыдущий.
        out = D.OUT / (args.out or "body-bench")
        out.mkdir(parents=True, exist_ok=True)
        res = body.bench(n_runs=args.bench, duration_s=args.t, neurons=df,
                         weights=weights, device=device, base=cfg,
                         seed0=args.bench_seed, out_dir=out)
        print()
        for label, r in res.items():
            print(f"{label:22s} градиент {r['grad_alignment_mean']:+.3f}"
                  f" ± {r['grad_alignment_sem']:.3f}"
                  f" (sd {r['grad_alignment_sd']:.3f}, {r['n_runs']} прогонов)")
            print(f"{'':22s} до еды {r['dist_change_mm']:+.1f} мм, "
                  f"шагов на еде {r['food_steps_total']}")
        (out / "bench.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсохранено: {out / 'bench.json'}")
        return

    name = args.out or (time.strftime("%Y%m%d-%H%M%S") + "-body")
    out_dir = D.OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Лента таймлайна собирается и здесь, а не только на пути окна.
    # Иначе прогон из командной строки ложится на диск без
    # `timeline.npz`, и в виде «Таймлайн» такой сеанс не открывается
    # вовсе — а он ничем не хуже посчитанного окном.
    tape = jobs.tape_for("body", df, cfg.control_ms,
                         keep_s=getattr(cfg, "keep_s", None),
                         readout=cfg.readout, seed=cfg.seed,
                         origin="командная строка")
    run = body.run_episode(cfg=cfg, neurons=df, weights=weights,
                           tape=tape,
                           video_path=out_dir / "walk.mp4" if args.video else None)
    body.save(run, out_dir)

    print("\nитог:")
    for k, v in run.metrics.items():
        print(f"  {k:24s} {v}")
    print("время:", json.dumps(run.timing))

    html = report.render(
        run.as_result(), out_dir / "report.html",
        title=f"В теле: выход {cfg.readout}, {cfg.duration_s:g} с"
              + (" — КОНТРОЛЬ: антенны переставлены" if cfg.swap_antennae else ""),
    )
    print(f"отчёт: {html}")
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())


def cmd_lateral(args):
    """Screen bilateral populations for an odour-side signal, open loop."""
    import torch

    from . import loop

    df = D.load_neurons()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("загружаю матрицу связей...")
    weights = D.load_weights(device=device)

    rows = loop.lateral_probe(neurons=df, weights=weights, device=device,
                              duration_s=args.t, base_hz=args.base,
                              depth=args.depth, period_s=args.period,
                              seed=args.seed)

    print("\nкто кодирует сторону запаха "
          "(муха неподвижна, асимметрия входа качается синусом):")
    print(f"  {'популяция':24s} {'r':>7s} {'лаг':>6s} "
          f"{'сигма L-R':>10s} {'средн.':>8s}  размер")
    for r in rows[:args.top]:
        print(f"  {r['population']:24s} {r['r']:+7.3f} {r['lag_ms']:5.0f}м "
              f"{r['asym_sd_hz']:10.3f} {r['mean_hz']:8.2f}  "
              f"{r['n_left']}/{r['n_right']}")

    out = D.OUT / (args.out or "lateral-probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lateral.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out / 'lateral.json'}")


def cmd_arena(args):
    """Closed loop: the brain drives a fly around a 2D arena."""
    import torch

    from . import jobs, loop, report, world

    df = D.load_neurons()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("загружаю матрицу связей...")
    weights = D.load_weights(device=device)

    try:
        body = _spec_from_args(args, "arena").to_body_params(df)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.bench:
        res = loop.bench(n_runs=args.bench, duration_s=args.bench_t,
                         neurons=df, weights=weights, device=device,
                         body_params=body, seed0=args.bench_seed)
        print()
        for label, r in res.items():
            print(f"{label:22s} совпадение с градиентом "
                  f"{r['grad_alignment_mean']:+.3f} ± {r['grad_alignment_sem']:.3f} "
                  f"(sd {r['grad_alignment_sd']:.3f}, {r['n_runs']} прогонов)")
            print(f"{'':22s} до еды {r['food_dist_change_mm']:+.1f} мм, "
                  f"визитов {r['food_visits_total']}")
        out = D.OUT / (args.out or "arena-bench")
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсохранено: {out / 'bench.json'}")
        return

    cfg = loop.EpisodeConfig(
        duration_s=args.t,
        control_ms=args.control_ms,
        seed=args.seed,
        device=device,
        comment=not args.no_comment,
        comment_every_ms=args.comment_every,
        model=args.model,
    )
    print(f"эпизод {cfg.duration_s} с, шаг управления {cfg.control_ms} мс")
    tape = jobs.tape_for("arena", df, cfg.control_ms,
                         keep_s=getattr(cfg, "keep_s", None),
                         motor=body, seed=cfg.seed,
                         origin="командная строка")
    ep = loop.run_episode(config=cfg, body_params=body, neurons=df,
                          weights=weights, probes=args.probe, tape=tape)

    name = args.out or (time.strftime("%Y%m%d-%H%M%S") + "-arena")
    out_dir = D.OUT / name
    loop.save(ep, out_dir)

    print("\nитог:")
    for k, v in ep.metrics.items():
        print(f"  {k:28s} {v}")
    print("время:", json.dumps(ep.timing))

    if args.probe:
        rows = loop.probe_report(ep.records)
        print("\nгде теряется сторона запаха "
              "(корреляция асимметрии входа и асимметрии популяции):")
        print(f"  {'популяция':10s} {'r':>7s} {'сигма L-R, Гц':>14s} {'средн., Гц':>11s}")
        for r in rows:
            print(f"  {r['population']:10s} {r['r']:+7.3f} "
                  f"{r['asym_sd_hz']:14.3f} {r['mean_hz']:11.2f}")
        (out_dir / "probes.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    title = args.title or (
        f"Замкнутый цикл: арена {ep.arena.width:.0f}×{ep.arena.height:.0f} мм, "
        f"{cfg.duration_s:g} с"
        + (" — КОНТРОЛЬ: антенны переставлены" if args.control else "")
    )
    html = report.render(
        ep.as_result(), out_dir / "report.html",
        title=title,
        arena=report.build_arena_payload(ep),
    )
    print(f"отчёт: {html}")
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())


def cmd_report(args):
    """Rebuild the HTML from a saved run, without simulating again."""
    from . import report, sim

    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = D.OUT / args.run
    narrative = ""
    nf = run_dir / "narrative.md"
    if nf.exists():
        narrative = nf.read_text(encoding="utf-8")

    arena = None
    if (run_dir / "episode.json").exists():
        from . import loop

        ep = loop.load(run_dir)
        result = ep.as_result()
        arena = report.build_arena_payload(ep)
    elif (run_dir / "embodied.json").exists():
        # Прогон в теле собирается своим загрузчиком: через `sim.load`
        # терялись и длительность по факту, и имена сенсорных каналов.
        # Панель арены ему не строится — в записи по шагам нет ни
        # скорости, ни поворота, ни кормления, и рисовать их нечем.
        from . import body as B

        run = B.load(run_dir)
        result = run.as_result()
        print(f"  эпизод в теле: {len(run.records)} шагов, "
              f"{result.config.t_run_s:g} с по факту "
              f"(заказано {run.config.duration_s:g} с)")
        print("  панель арены не строится: в track.parquet тела нет "
              "скорости, поворота и кормления — их в этом виде не пишут")
    else:
        result = sim.load(run_dir)

    title = args.title
    if title is None:
        title = report.title_of(run_dir)
    html = report.render(result, run_dir / "report.html",
                         narrative=narrative, title=title, arena=arena)
    print(f"отчёт: {html}")
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())


def cmd_explain(args):
    """Add an LLM narrative to an already saved run."""
    from . import llm

    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = D.OUT / args.run
    text = llm.explain_dir(run_dir, model=args.model)
    (run_dir / "narrative.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_agent(args):
    from . import agent

    agent.run_loop(
        args.goal,
        model=args.model,
        max_steps=args.steps,
        t_run_s=args.t,
        open_reports=not args.no_open,
        out_name=args.out,
    )


def cmd_verify(args):
    """Compare this simulator against the upstream PyTorch backend."""
    import subprocess

    from . import sim

    up = D.UPSTREAM
    sys.path.insert(0, str(up / "code"))
    import benchmark as up_bench  # noqa: E402

    # Апстрим принимает только свой список длительностей, а на чужом
    # печатает ошибку и ВОЗВРАЩАЕТ НОЛЬ. Проверяем до того, как потратим
    # минуты на собственный прогон, которому не с чем будет сравниваться.
    allowed = [float(v) for v in getattr(up_bench, "T_RUN_VALUES_SEC", [])]
    if allowed and float(args.t) not in allowed:
        raise SystemExit(
            "апстрим принимает только %s с, а задано %g: при другом "
            "значении он ничего не считает и выходит с нулём"
            % (", ".join("%g" % v for v in allowed), float(args.t)))

    exp = up_bench.get_experiment("sugar")
    df = D.load_neurons()
    idx = df[df.root_id.isin(exp["neu_exc"])].idx.to_numpy()
    print(f"эксперимент {exp['name']}: {len(idx)} нейронов")

    cfg = sim.RunConfig(t_run_s=args.t, bin_ms=50, n_trials=1, seed=0)
    mine = sim.run([sim.Stimulus(idx, exp["stim_rate"], name="sugar")], config=cfg,
                   neurons=df)
    my_rates = mine.rates

    print("\nзапускаю апстрим (python main.py --pytorch)...")
    cmd = [
        sys.executable, "main.py", "--pytorch",
        "--t_run", str(args.t), "--n_run", "1", "--no_log_file",
    ]
    # Апстрим печатает «✓», которого нет в cp1251: без этого дочерний питон
    # падает на кодировке. Переменная ставится только ему, глобально трогать
    # кодовую страницу нельзя — см. примечание в fly.cmd.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, cwd=up, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    print(proc.stdout[-2500:])
    if proc.returncode != 0:
        print(proc.stderr[-2500:])
        raise SystemExit("апстрим упал")

    import pandas as pd

    # Строго той же длительности. Частоты апстрима считаются делением его
    # спайков на НАШУ длительность, поэтому файл от прогона другой длины
    # даёт частоты, завышенные ровно в отношение длительностей — раньше
    # сюда молча подставлялся «последний попавшийся» файл.
    cands = sorted((up / "data" / "results").glob(f"**/pytorch_t{args.t}s_n1.parquet"))
    if not cands:
        seen = sorted(q.name for q in
                      (up / "data" / "results").glob("**/pytorch_*.parquet"))
        raise SystemExit(
            f"апстрим не оставил файла на {args.t} с: искал "
            f"pytorch_t{args.t}s_n1.parquet, нашёл "
            + (", ".join(seen) if seen else "ничего")
            + ". Сравнивать с файлом другой длительности нельзя.")
    spikes = pd.read_parquet(cands[-1])
    print(f"апстрим: {cands[-1].name}, {len(spikes)} спайков")
    up_counts = np.zeros(len(df), dtype=np.int64)
    np.add.at(up_counts, spikes["neuron_index"].to_numpy(), 1)
    up_rates = up_counts / args.t

    a = my_rates >= 0.5
    b = up_rates >= 0.5
    jac = (a & b).sum() / max((a | b).sum(), 1)
    both = a | b
    corr = np.corrcoef(my_rates[both], up_rates[both])[0, 1] if both.sum() > 2 else float("nan")
    print(f"\nактивных: мой {a.sum()}, апстрим {b.sum()}")
    print(f"Jaccard по активным нейронам: {jac:.3f}")
    print(f"корреляция частот: {corr:.4f}")
    print(f"суммарные спайки: мой {my_rates.sum()*args.t:.0f}, "
          f"апстрим {up_rates.sum()*args.t:.0f}")


def build_parser():
    p = argparse.ArgumentParser(prog="fly", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="КОМАНДА")

    s = sub.add_parser("app", help="нативное окно: карта мозга, прогоны, тело")
    s.add_argument("--device", default=None)
    s.add_argument("--old", action="store_true",
                   help="прежний интерфейс, пока новый не согласован")
    s.add_argument("--seconds", type=float, default=0.0,
                   help="закрыть окно самому через N секунд (для проверок)")
    s.add_argument("--selftest", default=None, metavar="ОТЧЁТ.json",
                   help="короткий сценарий самопроверки окна: четыре "
                        "пространства, опыт, сохранение; отчёт в файл")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("ui", help="окно управления в браузере")
    s.add_argument("--port", type=int, default=8750)
    s.add_argument("--device", default=None)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("font", help="испечь атлас глифов для окна")
    s.add_argument("--sizes", type=int, nargs="+",
                   default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 26])
    s.add_argument("--font", default=None, help="путь к ttf; по умолчанию Consolas")
    s.add_argument("--force", action="store_true", help="перепечь, даже если есть")
    s.add_argument("--roles", nargs="+", default=None,
                   help="какие начертания печь: ui, strong, num")
    s.set_defaults(func=cmd_font)

    s = sub.add_parser("doctor", help="проверить, что всё на месте")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("select", help="что находит запрос")
    s.add_argument("query")
    s.add_argument("--list", action="store_true", help="вывести сами нейроны")
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("regions", help="список областей мозга")
    s.set_defaults(func=cmd_regions)

    s = sub.add_parser("run", help="прогнать эксперимент")
    s.add_argument("--activate", "-a", action="append", default=[],
                   help="запрос нейронов для активации (можно несколько раз)")
    s.add_argument("--rate", "-r", type=float, nargs="+", default=[200.0],
                   help="частота стимуляции, Гц (по стимулу)")
    s.add_argument("--silence", "-s", default=None, help="запрос нейронов для глушения")
    s.add_argument("--t", type=float, default=1.0, help="длительность, с")
    s.add_argument("--bin", type=float, default=10.0, help="шаг бина записи, мс")
    s.add_argument("--pulse", type=float, nargs=2, default=None,
                   metavar=("ON_MS", "OFF_MS"), help="окно стимула в мс")
    s.add_argument("--trials", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--name", action="append", default=None, help="имя стимула")
    s.add_argument("--title", default=None, help="заголовок отчёта")
    s.add_argument("--out", default=None, help="имя папки в out/")
    s.add_argument("--explain", action="store_true", help="разбор локальной моделью")
    s.add_argument("--model", default=None, help="модель Ollama")
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("index", help="страница со всеми прогонами")
    s.add_argument("--rebuild", action="store_true",
                   help="пересобрать все отчёты перед сборкой страницы")
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("body", help="мозг в теле NeuroMechFly v2 (MuJoCo)")
    s.add_argument("--t", type=float, default=4.0, help="длительность, с")
    s.add_argument("--readout", default="ALPN", help="популяция для рулёжки")
    s.add_argument("--dn-tau", type=float, default=0.5, help="интегрирование, с")
    s.add_argument("--control-ms", type=float, default=20.0)
    s.add_argument("--physics-dt", type=float, default=2e-4, help="шаг физики, с")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--control", action="store_true",
                   help="контроль: переставить антенны")
    s.add_argument("--no-cross", action="store_true",
                   help="прямая связь полушарий с ногами вместо перекрёстной")
    s.add_argument("--view", action="store_true",
                   help="живое окно MuJoCo: видно, как муха идёт прямо во "
                        "время счёта (закрытие окна останавливает эпизод)")
    s.add_argument("--video", action="store_true",
                   help="записать видео ходьбы в walk.mp4")
    s.add_argument("--play-speed", type=float, default=0.2,
                   help="замедление видео, 0.2 = в пять раз медленнее")
    s.add_argument("--bench", type=int, default=0,
                   help="серия эпизодов: норма против перестановки антенн")
    s.add_argument("--bench-seed", type=int, default=100)
    # --- инженерный моторный путь; по умолчанию всё как было ---
    s.add_argument("--motor-decoder", default="legacy",
                   choices=("legacy", "dn_linear", "dn_trained"),
                   help="чем активность становится командой: legacy — "
                        "прежнее среднее по полушарию; dn_linear — прямые "
                        "зонды нисходящих с объявленными коэффициентами; "
                        "dn_trained — те же зонды, веса из файла")
    s.add_argument("--motor-weights", default=None,
                   help="файл весов для --motor-decoder dn_trained")
    s.add_argument("--spmv", default=None,
                   choices=("эталонный", "событийный"),
                   help="чем считать умножение связей в движке LIF: "
                        "эталонный — csrmv по всем 15 091 983 связям; "
                        "событийный — своё ядро только по строкам "
                        "спайкующих (требует nvcc и MSVC, при их "
                        "отсутствии падает громко). По умолчанию — "
                        "окружение FLYBRAIN_SPMV")
    s.add_argument("--gf-bridge", default="none",
                   choices=("none", "gain_v1"),
                   help="ИНЖЕНЕРНЫЙ мост через обрыв «Giant Fiber → "
                        "мотонейроны». По умолчанию его нет: пути в v783 "
                        "нет вовсе")
    s.add_argument("--head-stab", action="store_true",
                   help="опубликованная ОБУЧЕННАЯ стабилизация головы "
                        "(инженерная деталь тела, обходит коннектом)")
    s.add_argument("--neck-kp", type=float, default=None,
                   help="жёсткость приводов шеи при --head-stab (по "
                        "умолчанию 500, опубликованное значение)")
    s.add_argument("--no-probes", action="store_true",
                   help="не писать readout_* прямых зондов в запись")
    # --- зрение ---
    #
    # Зрение ВЫКЛЮЧЕНО по умолчанию: прежний опыт, пересчитанный новой
    # сборкой, обязан совпасть с сохранённым. Включается явно — флагом
    # или готовым пресетом.
    s.add_argument("--vision", action="store_true",
                   help="подать кадр обоих сложных глаз в фоторецепторы "
                        "коннектома (по умолчанию выключено)")
    s.add_argument("--vision-preset", default=None,
                   help="готовый зрительный опыт: " + ", ".join(
                       sorted(S_VISION_PRESETS)) + ". Ставит сразу все "
                        "условия, включая зрение и объект в сцене")
    s.add_argument("--vision-hz", type=float, default=100.0,
                   help="частота обновления зрения, Гц. РАБОЧАЯ ТОЧКА, а "
                        "не биологическая постоянная: 100 Гц выбраны по "
                        "замеру цены шага")
    s.add_argument("--vision-adapter", default="retina",
                   help="зрительный переходник по имени "
                        "(flybrain.vision.ADAPTERS)")
    s.add_argument("--vision-max-hz", type=float, default=150.0,
                   help="частота фоторецепторов при насыщении, Гц")
    s.add_argument("--vision-ref", type=float, default=0.5,
                   help="яркость кадра, считающаяся насыщением")
    s.add_argument("--vision-channel", default="max",
                   choices=("max", "pale", "yellow"),
                   help="что делать с двумя каналами фоторецептора")
    s.add_argument("--vision-eyes", default="both",
                   choices=("both", "left", "right"),
                   help="оба глаза или один (один — контроль)")
    s.add_argument("--vision-mode", default="whiten",
                   choices=("whiten", "rigid"),
                   help="совмещение облаков в привязке омматидиев")
    s.add_argument("--vision-assign", default="capacity",
                   choices=("capacity", "nearest"),
                   help="назначение нейрона омматидию")
    s.add_argument("--vision-shuffle", action="store_true",
                   help="ВСТРЕЧНЫЙ ЗАМЕР: перемешать привязку с "
                        "сохранением числа нейронов на омматидий — "
                        "суммарный вход тот же до последнего разряда")
    s.add_argument("--vision-light", type=float, default=1.0,
                   help="множитель к свету арены (настоящие источники "
                        "MuJoCo: меняется сам кадр глаза)")
    s.add_argument("--vision-contrast", type=float, default=1.0,
                   help="контраст кадра: clip(0,5 + (I − 0,5)·k). "
                        "Объявленное преобразование кадра, не свойство "
                        "сцены")
    s.add_argument("--vision-record", default=None,
                   help="записать зрительный вход в ленту (.npz рядом с "
                        "манифестом .json). ЖИВОЙ рендер не "
                        "воспроизводим между процессами — первая "
                        "расходящаяся ступень сырые пиксели; "
                        "воспроизведение такой ленты от видеокарты не "
                        "зависит и даёт побитовое равенство")
    s.add_argument("--vision-replay", default=None,
                   help="взять зрительный вход из ленты вместо живого "
                        "рендера. Прогон из записи НЕ является прогоном "
                        "с живым зрением, и в происхождении это "
                        "записано отдельным полем")
    s.add_argument("--vision-bridge", default=None, choices=("гибрид",),
                   help="ГИБРИДНЫЙ зрительный переходник: направленные "
                        "сигналы T4/T5 считаются ВНЕ сети, а дальше "
                        "сигнал идёт по настоящим связям FlyWire "
                        "(T4/T5 -> LPLC2, LC4 -> DNp01). Это ИНЖЕНЕРНЫЙ "
                        "интерфейс, а не передача по глубокому "
                        "зрительному пути оригинальной сети: без него за "
                        "третьим синапсом ровно 0,000 Гц")
    s.add_argument("--vision-bridge-boundary", default="T4/T5",
                   choices=("T4/T5", "LC4/LPLC2"),
                   help="где переходник стыкуется с сетью. При "
                        "LC4/LPLC2 отклики самих LC4 и LPLC2 НЕ "
                        "принадлежат никаким синапсам — они поданы")
    s.add_argument("--vision-bridge-gain", type=float, default=200.0,
                   help="усиление подачи переходника, Гц")
    s.add_argument("--vision-bridge-max-hz", type=float, default=1000.0,
                   help="потолок подачи переходника, Гц. Объявленное "
                        "правило min(hz, потолок); выше 10 000 Гц "
                        "пуассоновский вход перестаёт быть "
                        "пуассоновским (sim.MAX_RATE_HZ)")
    s.add_argument("--vision-bridge-delay-ms", type=float, default=30.0,
                   help="плечо задержки коррелятора, мс")
    s.add_argument("--vision-bridge-shuffle", action="store_true",
                   help="ВСТРЕЧНЫЙ ЗАМЕР: те же числа тем же "
                        "популяциям, но нейронам в случайном порядке — "
                        "суммарный вход тот же, разрушена только "
                        "пространственная структура")
    s.add_argument("--vision-bridge-subtypes-shuffle", action="store_true",
                   help="ВСТРЕЧНЫЙ ЗАМЕР: перемешать соответствие "
                        "«подтип T4/T5 — направление» — разрушена "
                        "только направленная структура")
    s.add_argument("--objects", type=int, default=0, dest="vis_objects",
                   help="сколько слотов зрительных объектов вписать в "
                        "сцену (тела MuJoCo создаются до компиляции)")
    s.add_argument("--track", default=None, dest="vis_track",
                   help="готовая траектория объекта: " + ", ".join(
                       sorted(_object_presets())) + "; или путь к json "
                        "со списком отрезков")
    s.add_argument("--device", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_body)

    s = sub.add_parser("lateral", help="кто кодирует сторону запаха (открытый контур)")
    s.add_argument("--t", type=float, default=10.0, help="длительность замера, с")
    s.add_argument("--base", type=float, default=75.0, help="средняя частота ORN, Гц")
    s.add_argument("--depth", type=float, default=0.85, help="глубина качания, 0..1")
    s.add_argument("--period", type=float, default=2.5, help="период качания, с")
    s.add_argument("--top", type=int, default=25, help="сколько строк показать")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_lateral)

    s = sub.add_parser("arena", help="замкнутый цикл: мозг водит муху по арене")
    s.add_argument("--t", type=float, default=12.0, help="длительность эпизода, с")
    s.add_argument("--control-ms", type=float, default=20.0,
                   help="шаг управления, мс")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--control", action="store_true",
                   help="контроль: переставить антенны местами")
    s.add_argument("--turn-sign", type=float, default=1.0,
                   help="+1: больше активности слева — поворот налево")
    s.add_argument("--motor", default=None,
                   help="популяция для рулёжки (имя из вывода `fly lateral`), "
                        "по умолчанию все нисходящие")
    s.add_argument("--dn-ref", type=float, default=None,
                   help="опорная частота рулёжки, Гц; без неё измеряется сама")
    s.add_argument("--turn-gain", type=float, default=None,
                   help="усиление поворота, рад/с; без него калибруется само")
    s.add_argument("--dn-tau", type=float, default=None,
                   help="постоянная интегрирования выхода, с (по умолчанию 0.12)")
    s.add_argument("--turn-target", type=float, default=None,
                   help="поворот на одну сигму шума, рад/с (по умолчанию 0.6)")
    s.add_argument("--probe", action="store_true",
                   help="записать активность ORN/ALPN/LH/MBON/DN по сторонам "
                        "и показать, где теряется сторона запаха")
    s.add_argument("--no-comment", action="store_true",
                   help="без комментария локальной модели")
    s.add_argument("--comment-every", type=float, default=1000.0,
                   help="как часто просить комментарий, мс")
    s.add_argument("--model", default=None, help="модель Ollama")
    s.add_argument("--bench", type=int, default=0,
                   help="серия эпизодов: норма против перестановки антенн")
    s.add_argument("--bench-t", type=float, default=4.0,
                   help="длительность эпизода в серии, с")
    s.add_argument("--bench-seed", type=int, default=100,
                   help="первый сид серии; смените, чтобы набрать независимые "
                        "прогоны и объединить с прежними")
    s.add_argument("--title", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_arena)

    s = sub.add_parser("demos", help="прогнать готовый набор из пяти опытов")
    s.add_argument("--device", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--no-explain", action="store_true")
    s.set_defaults(func=cmd_demos)

    s = sub.add_parser("report", help="пересобрать HTML из сохранённого прогона")
    s.add_argument("run")
    s.add_argument("--title", default=None)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("explain", help="разбор сохранённого прогона локальной моделью")
    s.add_argument("run")
    s.add_argument("--model", default=None)
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("agent", help="локальная модель сама ставит опыты")
    s.add_argument("goal")
    s.add_argument("--model", default=None)
    s.add_argument("--steps", type=int, default=6)
    s.add_argument("--t", type=float, default=0.5)
    s.add_argument("--out", default=None, help="имя папки в out/")
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_agent)

    s = sub.add_parser("verify", help="сверка с апстримным бэкендом")
    s.add_argument("--t", type=float, default=1.0)
    s.set_defaults(func=cmd_verify)

    # Разовая установочная команда — последней, чтобы не мешала в списке.
    s = sub.add_parser("fetch-annotations", help="скачать аннотации FlyWire")
    s.set_defaults(func=cmd_fetch_annotations)
    return p


def main(argv=None):
    # Консоль может быть в cp866 или cp1251, и тогда любой символ вне её
    # таблицы (например «✓» из вывода апстрима) валит процесс. Лучше
    # напечатать вопросительный знак, чем упасть.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser.print_help()
        print("\nбыстрый старт:")
        print("  fly doctor                                   проверить окружение")
        print("  fly demos                                    пять готовых опытов")
        print("  fly arena --t 5 --motor ALPN --dn-tau 0.5     муха на арене")
        return 0
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
