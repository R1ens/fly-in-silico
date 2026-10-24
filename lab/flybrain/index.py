"""One page that gathers everything the lab has produced.

Each run writes its own report, but the results of a session live in several
shapes: single stimulations, closed-loop episodes, benchmark series, the
open-loop screen for side-coding populations, and the agent's journal.  This
builds a hub over ``out/`` so all of it is reachable and comparable from one
place, instead of only through file names.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import data as D

# ------------------------------------------------------------------ scanning


def read_json(path):
    """Прочитать JSON прогона. Вернуть ``(данные, ошибка)``.

    Отсутствие файла — не ошибка: у перебора популяций нет ни `meta.json`,
    ни `episode.json`, и это нормально. А вот битый файл ошибкой является,
    и молчать о нём нельзя: раньше любое исключение глушилось и папка либо
    исчезала из указателя целиком, либо меняла вид прогона на другой —
    битый `episode.json` превращал эпизод арены в обычную стимуляцию.
    """
    if not path.exists():
        return None, ""
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except json.JSONDecodeError as exc:
        return None, ("не разбирается как JSON: строка %d, столбец %d, %s"
                      % (exc.lineno, exc.colno, exc.msg))
    except UnicodeDecodeError as exc:
        return None, "не читается как UTF-8: %s" % exc
    except OSError as exc:
        return None, "не открывается: %s" % exc


def _top_regions(run_dir, limit=6):
    """Top regions of a run, straight from rates.parquet."""
    path = run_dir / "rates.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=["region", "rate_hz"])
    except Exception:  # noqa: BLE001
        return []
    g = (df.groupby("region")
           .agg(n=("rate_hz", "size"), mean_hz=("rate_hz", "mean"),
                max_hz=("rate_hz", "max"))
           .sort_values("mean_hz", ascending=False)
           .head(limit))
    return [{"region": r, "n": int(v.n), "mean_hz": round(float(v.mean_hz), 1),
             "max_hz": round(float(v.max_hz), 1)} for r, v in g.iterrows()]


def _title_from_html(run_dir):
    """The title the report gave itself, without re-rendering it."""
    path = run_dir / "report.html"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(4096)
    except Exception:  # noqa: BLE001
        return None
    i = head.find("<title>")
    if i < 0:
        return None
    j = head.find("</title>", i)
    return head[i + 7:j].strip() if j > i else None


def scan(out_dir=None):
    out_dir = Path(out_dir or D.OUT)
    entries = []
    for run in sorted(out_dir.iterdir(), key=lambda p: -p.stat().st_mtime):
        if not run.is_dir():
            continue
        e = {"name": run.name, "dir": run,
             "report": (run / "report.html").exists(),
             "mtime": run.stat().st_mtime}

        # Вид прогона решается по тому, какой файл ЛЕЖИТ в папке, а не по
        # тому, какой удалось разобрать. Иначе битый файл молча понижает
        # прогон до другого вида, и в указателе он выглядит исправным.
        errors = {}
        parsed = {}
        for fname in ("bench.json", "lateral.json", "episode.json",
                      "embodied.json", "meta.json"):
            data, err = read_json(run / fname)
            parsed[fname] = data
            if err:
                errors[fname] = err
        bench = parsed["bench.json"]
        lateral = parsed["lateral.json"]
        episode = parsed["episode.json"]
        embodied = parsed["embodied.json"]
        meta = parsed["meta.json"]
        journal = run / "journal.md"
        e["errors"] = errors

        marker = next((f for f in ("bench.json", "lateral.json",
                                   "episode.json", "embodied.json",
                                   "meta.json") if (run / f).exists()), None)
        if marker is not None and marker in errors:
            # Папка остаётся видимой, но честно названа повреждённой:
            # ни успехом, ни другим видом прогона она не притворяется.
            e["kind"] = "broken"
            e["title"] = "Повреждён: " + run.name
            e["broken_file"] = marker
            entries.append(e)
            continue

        if bench:
            e["kind"] = "bench"
            e["bench"] = bench
        elif lateral:
            e["kind"] = "lateral"
            e["lateral"] = lateral
        elif journal.exists():
            e["kind"] = "agent"
            e["journal"] = journal.read_text(encoding="utf-8")
            e["experiments"] = sorted(
                p.name for p in run.glob("exp-*") if (p / "report.html").exists())
            e["meta"] = meta
            e["regions"] = _top_regions(run)
        elif embodied:
            # Прогон в теле писал `embodied.json`, но ветки на него не было,
            # и он до сих пор попадал в указатель как обычная стимуляция —
            # без траектории, без метрик и в чужом разделе.
            e["kind"] = "embodied"
            e["embodied"] = embodied
            e["regions"] = _top_regions(run)
        elif episode:
            e["kind"] = "episode"
            e["episode"] = episode
            e["regions"] = _top_regions(run)
        elif meta:
            e["kind"] = "run"
            e["meta"] = meta
            e["regions"] = _top_regions(run)
        else:
            continue

        e["title"] = _title_from_html(run) or run.name
        if e["kind"] == "agent":
            # The agent's own report is rendered from its last experiment, so
            # its <title> says nothing useful; the goal is the real title.
            e["title"] = e["journal"].split("\n", 1)[0].replace("#", "").strip()
        elif e["kind"] == "bench":
            e["title"] = "Серия: " + run.name
        elif e["kind"] == "lateral":
            e["title"] = "Перебор популяций: " + run.name
        elif e["kind"] == "episode":
            # The report's own title is the stimulus list, which for a closed
            # loop is just the sensory channels at 0 Hz and says nothing.
            a = episode.get("arena", {})
            cfg = episode.get("config", {})
            e["title"] = (f"Арена {a.get('width', 0):.0f}×{a.get('height', 0):.0f} мм, "
                          f"{cfg.get('duration_s', '?')} с")
            if (episode.get("body") or {}).get("swap_antennae"):
                e["title"] += " — контроль"
        narrative = run / "narrative.md"
        if narrative.exists():
            e["narrative"] = narrative.read_text(encoding="utf-8")
        entries.append(e)
    return entries


def _card_broken(e):
    """Карточка повреждённого прогона: что именно не читается."""
    rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(why)}</td></tr>"
        for name, why in (e.get("errors") or {}).items())
    table = ("<table class='tbl'><thead><tr><th>файл</th><th>что с ним</th>"
             f"</tr></thead><tbody>{rows}</tbody></table>")
    return _card(e, "повреждён",
                 "папка найдена, но главный файл вида "
                 f"(<b>{_esc(e.get('broken_file'))}</b>) не читается. "
                 "Остальные файлы прогона на месте и не тронуты.",
                 [], table, no_report=True)


# ----------------------------------------------------------------- rendering


def _esc(x):
    return html.escape(str(x), quote=True)


def _fmt(n):
    try:
        return f"{int(n):,}".replace(",", " ")
    except (TypeError, ValueError):
        return str(n)


def _region_chips(regions):
    if not regions:
        return ""
    cells = "".join(
        f"<span class='chip'><b>{_esc(r['region'])}</b> {r['mean_hz']:g} Гц"
        f"<i>{_fmt(r['n'])}</i></span>" for r in regions)
    return f"<div class='chips'>{cells}</div>"


def _card_run(e):
    m = e.get("meta") or {}
    cfg = m.get("config", {})
    stim = ", ".join(
        f"{_esc(s.get('name', '?'))} ({_fmt(s.get('n', 0))} @ {s.get('rate_hz', 0):g} Гц)"
        for s in m.get("stimuli", [])) or "—"
    facts = [
        ("активных", _fmt(m.get("n_active", "—"))),
        ("спайков", _fmt(m.get("total_spikes", "—"))),
        ("длительность", f"{cfg.get('t_run_s', '—')} с"),
        ("заглушено", _fmt(m.get("n_silenced", 0))),
        ("счёт", f"{(m.get('timing') or {}).get('sim_s', '—')} с"),
    ]
    return _card(e, "опыт", stim, facts, _region_chips(e.get("regions")),
                 e.get("narrative"))


def _card_episode(e):
    ep = e["episode"]
    met = ep.get("metrics", {})
    body = ep.get("body", {})
    cfg = ep.get("config", {})
    motor = (body.get("motor_left") or "").replace("+side:left", "")
    motor = motor or "все нисходящие"
    sub = (f"выход: <b>{_esc(motor)}</b>, "
           f"tau {body.get('dn_tau_s', '?')} с"
           + (", <b class='warn'>антенны переставлены</b>"
              if body.get("swap_antennae") else ""))
    align = met.get("grad_alignment")
    facts = [
        ("совпадение с градиентом",
         f"<b class='{'good' if (align or 0) > 0.05 else ''}'>{align:+.3f}</b>"
         if align is not None else "—"),
        ("до еды",
         f"{met.get('food_dist_first_half_mm', '?')} → "
         f"{met.get('food_dist_second_half_mm', '?')} мм"),
        ("визитов", met.get("food_visits", "—")),
        ("скорость", f"{met.get('mean_speed_mm_s', '—')} мм/с"),
        ("длительность", f"{cfg.get('duration_s', '—')} с"),
        ("комментариев", len(ep.get("comments") or [])),
    ]
    return _card(e, "арена", sub, facts, _region_chips(e.get("regions")))


def _card_embodied(e):
    """Прогон в теле NeuroMechFly: мозг ведёт шесть ног, а не точку на плоскости."""
    run = e["embodied"]
    met = run.get("metrics", {})
    cfg = run.get("config", {})
    cal = run.get("calibration", {})
    sub = (f"съём: <b>{_esc(str(cfg.get('readout', '?')))}</b>, "
           f"tau {cfg.get('dn_tau_s', '?')} с"
           + (", <b class='warn'>антенны переставлены</b>"
              if cfg.get("swap_antennae") else "")
           + ("" if cfg.get("cross", True) else ", <b class='warn'>без перекрёста</b>"))
    align = met.get("grad_alignment")
    facts = [
        ("совпадение с градиентом",
         f"<b class='{'good' if (align or 0) > 0.05 else ''}'>{align:+.3f}</b>"
         if align is not None else "—"),
        ("до источника",
         f"{met.get('dist_start_mm', '?')} → {met.get('dist_end_mm', '?')} мм"),
        ("прошла", f"{met.get('path_mm', '—')} мм"),
        ("скорость", f"{met.get('speed_mm_s', '—')} мм/с"),
        ("длительность", f"{cfg.get('duration_s', '—')} с"),
        ("перекос на разогреве", f"{cal.get('asym_bias', '—')}"),
    ]
    return _card(e, "тело", sub, facts, _region_chips(e.get("regions")))


def _welch(a, b):
    """Welch's t-test without pulling in scipy for a two-sample comparison."""
    import math

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return None
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:  # noqa: BLE001
        p = None
    return {"t": t, "df": df, "p": p, "diff": ma - mb,
            "se": math.sqrt(se2)}


def _card_bench(e):
    # Серия в теле пишет свои имена: `dist_change_mm` вместо
    # `food_dist_change_mm` и `food_steps_total` вместо `food_visits_total`.
    # Раньше карточка знала только про арену и роняла весь указатель, стоило
    # появиться папке body-bench.
    rows = []
    embodied = any("dist_change_mm" in r for r in e["bench"].values())
    for label, r in e["bench"].items():
        mean = r.get("grad_alignment_mean")
        sem = r.get("grad_alignment_sem")
        spread = f" ± {sem:.3f}" if sem is not None else ""
        moved = r.get("food_dist_change_mm", r.get("dist_change_mm"))
        near = r.get("food_visits_total", r.get("food_steps_total"))
        rows.append(
            f"<tr><td>{_esc(label)}</td>"
            f"<td class='n'><b>{mean:+.3f}</b>{spread}</td>"
            f"<td class='n'>{r.get('n_runs', '—')}</td>"
            f"<td class='n'>{moved:+.1f} мм</td>"
            f"<td class='n'>{near}</td></tr>")
    last = "шагов у еды" if embodied else "визитов"
    table = ("<table class='tbl'><thead><tr><th>условие</th>"
             "<th class='n'>совпадение с градиентом</th><th class='n'>прогонов</th>"
             f"<th class='n'>до цели</th><th class='n'>{last}</th></tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table>")

    conds = list(e["bench"].values())
    facts = []
    if len(conds) == 2 and conds[0].get("runs") and conds[1].get("runs"):
        w = _welch(conds[0]["runs"], conds[1]["runs"])
        if w:
            if w["p"] is None:
                verdict, cls = f"t = {w['t']:.2f}", ""
            elif w["p"] < 0.05:
                verdict, cls = f"p = {w['p']:.3f}", "good"
            else:
                verdict, cls = f"p = {w['p']:.2f}", "dim"
            facts = [
                ("разница", f"<b>{w['diff']:+.3f}</b> ± {w['se']:.3f}"),
                ("Welch t", f"{w['t']:.2f}"),
                ("значимость", f"<b class='{cls}'>{verdict}</b>"),
            ]

    first = conds[0]
    readout = first.get("readout")
    sub = (f"выход: <b>{_esc(readout)}</b>, " if readout else "")
    sub += f"эпизодов по {first['duration_s']:g} с, один эпизод — одно наблюдение"
    return _card(e, "серия", sub, facts, table, no_report=True)


def _card_lateral(e):
    rows = "".join(
        f"<tr><td>{_esc(r['population'])}</td>"
        f"<td class='n'><b>{r['r']:+.3f}</b></td>"
        f"<td class='n'>{r['lag_ms']:g} мс</td>"
        f"<td class='n'>{r['mean_hz']:g} Гц</td>"
        f"<td class='n'>{r['n_left']}/{r['n_right']}</td></tr>"
        for r in e["lateral"][:20])
    table = ("<table class='tbl'><thead><tr><th>популяция</th><th class='n'>r</th>"
             "<th class='n'>лаг</th><th class='n'>средн.</th><th class='n'>размер</th>"
             f"</tr></thead><tbody>{rows}</tbody></table>")
    return _card(e, "кто кодирует сторону",
                 "муха неподвижна, асимметрия входа качается синусом", [],
                 table, no_report=True)


def _card_agent(e):
    text = e["journal"]
    goal = ""
    for line in text.split("\n"):
        if line.startswith("модель"):
            goal = line.strip()
            break
    concl = ""
    if "### Вывод" in text:
        concl = text.split("### Вывод", 1)[1].strip()
    steps = text.count("\n## Шаг")
    links = " ".join(
        f"<a class='pill' href='{_esc(e['name'])}/{_esc(x)}/report.html'>{_esc(x)}</a>"
        for x in e.get("experiments", []))
    extra = (f"<div class='note'>{links}</div>" if links else "")
    if concl:
        extra += f"<div class='quote'>{_esc(concl)}</div>"
    facts = [("шагов", steps), ("опытов", len(e.get("experiments", [])))]
    return _card(e, "агент", _esc(goal), facts, extra)


def _card(e, kind, subtitle, facts, extra="", narrative=None, no_report=False):
    fact_html = "".join(
        f"<div class='f'><span>{_esc(k)}</span><b>{v}</b></div>" for k, v in facts)
    link = ""
    if e.get("report") and not no_report:
        link = (f"<a class='open' href='{_esc(e['name'])}/report.html'>"
                "открыть отчёт →</a>")
    narr = (f"<div class='quote'>{_esc(narrative.strip())}</div>"
            if narrative else "")
    return f"""<article class="card" data-kind="{_esc(kind)}">
  <header><span class="kind">{_esc(kind)}</span>
    <h3>{_esc(e['title'])}</h3>
    <code>{_esc(e['name'])}</code></header>
  <p class="sub">{subtitle}</p>
  <div class="facts">{fact_html}</div>
  {extra}{narr}
  {link}
</article>"""


SECTIONS = [
    ("run", "Опыты: активация и заглушение",
     "Одиночные прогоны. Выбранные нейроны получают пуассоновский вход, "
     "дальше смотрим, что загорелось и в каком порядке."),
    ("episode", "Замкнутый цикл: муха на арене",
     "Арена кормит сенсорные нейроны, нисходящие (или другой выбранный выход) "
     "рулят. Мозг настоящий, тело придумано."),
    ("embodied", "Мозг в теле: NeuroMechFly",
     "То же замыкание, но тело настоящее: шесть ног, физика MuJoCo, запах "
     "берётся из симулятора. Выход мозга идёт в амплитуды походки."),
    ("bench", "Серии с контролем",
     "Один эпизод — одно наблюдение. Контроль: антенны переставлены местами."),
    ("lateral", "Поиск выхода для рулёжки",
     "Открытый контур: кто из популяций вообще кодирует, какая антенна нюхает."),
    ("agent", "Агент-экспериментатор",
     "Локальная модель сама ставит опыты на коннектоме и подводит итог."),
    ("broken", "Повреждённые результаты",
     "Папка есть, но файл, по которому прогон опознаётся, не читается. "
     "Такой прогон не выбрасывается и не выдаётся за другой вид."),
]

CARD_RENDERERS = {
    "run": _card_run,
    "episode": _card_episode,
    "embodied": _card_embodied,
    "bench": _card_bench,
    "lateral": _card_lateral,
    "agent": _card_agent,
    "broken": _card_broken,
}


def render(out_dir=None, path=None):
    out_dir = Path(out_dir or D.OUT)
    entries = scan(out_dir)
    by_kind = {}
    for e in entries:
        by_kind.setdefault(e["kind"], []).append(e)

    blocks = []
    nav = []
    for kind, title, blurb in SECTIONS:
        items = by_kind.get(kind, [])
        if not items:
            continue
        nav.append(f"<a href='#{kind}'>{_esc(title)} <i>{len(items)}</i></a>")
        cards = "\n".join(CARD_RENDERERS[kind](e) for e in items)
        blocks.append(
            f"<section id='{kind}'><h2>{_esc(title)}</h2>"
            f"<p class='blurb'>{_esc(blurb)}</p>"
            f"<div class='grid'>{cards}</div></section>")

    page = TEMPLATE.replace("<!--NAV-->", "".join(nav))
    page = page.replace("<!--BODY-->", "\n".join(blocks))
    page = page.replace("<!--COUNT-->", str(len(entries)))

    path = Path(path or (out_dir / "index.html"))
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мозг мухи — все прогоны</title>
<style>
  :root{--bg:#07090d;--panel:#0e1219;--line:#1e2633;--fg:#e6edf6;
    --dim:#8b97a8;--accent:#ffb259;--good:#6ee7a8;--warn:#fc8181}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header.top{position:sticky;top:0;z-index:5;background:rgba(10,13,19,.92);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line);
    padding:12px 24px}
  header.top h1{margin:0 0 6px;font-size:16px;letter-spacing:.2px}
  header.top .sub{color:var(--dim);font-size:12.5px}
  nav{display:flex;gap:14px;flex-wrap:wrap;margin-top:9px}
  nav a{color:var(--dim);text-decoration:none;font-size:12.5px;
    border-bottom:1px solid transparent;padding-bottom:2px}
  nav a:hover{color:var(--fg);border-color:var(--accent)}
  nav a i{color:var(--accent);font-style:normal;font-variant-numeric:tabular-nums}
  main{padding:22px 24px 60px;max-width:1500px;margin:0 auto}
  section{margin-bottom:38px}
  h2{font-size:14px;text-transform:uppercase;letter-spacing:.9px;
    color:var(--dim);margin:0 0 4px;font-weight:600}
  .blurb{color:var(--dim);font-size:12.5px;margin:0 0 14px;max-width:70ch}
  .grid{display:grid;gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:14px 16px;display:flex;flex-direction:column;gap:9px}
  .card header{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
  .card h3{margin:0;font-size:13.5px;font-weight:600;line-height:1.35}
  .card code{color:#5f6b7c;font-size:11px}
  .kind{background:#172030;color:var(--accent);border-radius:4px;
    padding:1px 7px;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}
  .sub{margin:0;color:var(--dim);font-size:12.5px}
  .facts{display:flex;flex-wrap:wrap;gap:4px 18px}
  .f{display:flex;flex-direction:column;font-size:12px}
  .f span{color:var(--dim);font-size:11px}
  .f b{font-weight:600;font-variant-numeric:tabular-nums}
  .good{color:var(--good)} .warn{color:var(--warn)} .dim{color:var(--dim)}
  .chips{display:flex;flex-wrap:wrap;gap:5px}
  .chip{background:#121926;border:1px solid var(--line);border-radius:5px;
    padding:2px 7px;font-size:11.5px;color:var(--dim)}
  .chip b{color:var(--fg);margin-right:5px}
  .chip i{color:#5f6b7c;font-style:normal;margin-left:5px}
  .tbl{width:100%;border-collapse:collapse;font-size:12px}
  .tbl th{text-align:left;color:var(--dim);font-weight:600;padding:3px 5px;
    border-bottom:1px solid var(--line)}
  .tbl td{padding:3px 5px;border-bottom:1px solid #141a24}
  .tbl .n{text-align:right;font-variant-numeric:tabular-nums}
  .quote{background:#0b1017;border:1px solid var(--line);border-radius:6px;
    padding:9px 11px;font-size:12px;color:#aab6c6;white-space:pre-wrap;
    max-height:180px;overflow:auto}
  .note{display:flex;flex-wrap:wrap;gap:5px}
  .pill{background:#172030;color:var(--fg);border:1px solid var(--line);
    border-radius:5px;padding:2px 8px;font-size:11.5px;text-decoration:none}
  .pill:hover{border-color:var(--accent)}
  .open{color:var(--accent);text-decoration:none;font-size:12.5px;
    margin-top:auto;padding-top:4px}
  .open:hover{text-decoration:underline}
</style>
</head>
<body>
<header class="top">
  <h1>Мозг мухи — все прогоны</h1>
  <div class="sub">Коннектом FlyWire, 138 639 нейронов · <!--COUNT--> записей в out/</div>
  <nav><!--NAV--></nav>
</header>
<main><!--BODY--></main>
</body>
</html>
"""
