"""Какие графики по сохранённому прогону можно построить честно.

Список строится по тому, что **действительно лежит в папке**, а не по виду
прогона: у двух эпизодов арены он разный, потому что популяции по сторонам
пишутся только при включённой записи зондов.

Временной ряд рисуется лишь там, где в файлах есть ось времени. Таких мест
ровно два:

* `track.parquet`, колонка `t_ms` — настоящая ось, шаг равен шагу управления;
* `bins.npz` — ось неявная, но точная: строка `i` покрывает
  `[i·bin_ms, (i+1)·bin_ms)`, а сам `bin_ms` лежит в том же файле.

Всё остальное на диске — агрегаты за весь прогон: `rates.parquet`,
`metrics` в `episode.json`, `probes.json`, `lateral.json`, `bench.json`. Из
них временных рядов не делается: времени там нет, и рисовать его значило бы
выдумывать данные.

У каждого графика есть единицы, описание величины и имя источника с точностью
до колонки — это требование, а не украшение: без них «0,07» на экране не
значит ничего.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runs import bins_npz, detail, rates_table, series_table

__all__ = ["Plot", "available", "load"]


def _finite(v):
    """Число ли это вообще. `nan` в файлах встречается и ломает отрисовку."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


@dataclass(frozen=True)
class Plot:
    """Описание графика: что за величина, в чём она и откуда взята."""

    key: str
    title: str
    what: str                # что это за величина, словами
    unit: str                # единицы; пусто — безразмерная
    source: str              # файл и колонки
    kind: str = "line"       # line | track | bar | strip | hist
    caveat: str = ""         # оговорка, которую нельзя не показать


# Ряды арены и тела: ключ -> подпись, колонки `track.parquet`, единицы, смысл.
_LINES = {
    "arena": (
        ("speed", "Скорость", ("v",), "мм/с",
         "мгновенная скорость центра тела"),
        ("omega", "Поворот", ("omega",), "рад/с",
         "угловая скорость курса"),
        ("odour", "Запах на антеннах", ("odour_l", "odour_r"), "Гц",
         "частота, подаваемая в рецепторы левой и правой антенны"),
        ("dn", "Съём", ("dn_l", "dn_r"), "Гц",
         "средняя частота нейронов съёма слева и справа"),
        ("asym", "Асимметрия съёма", ("asym",), "",
         "разность сторон, делённая на сумму, за вычетом собственного "
         "перекоса сети, измеренного на разогреве"),
        ("grad", "Совпадение с градиентом", ("grad_cos",), "",
         "косинус между курсом и направлением роста запаха, от -1 до +1"),
        ("food", "Расстояние до еды", ("food_dist",), "мм",
         "до ближайшего источника"),
        ("spikes", "Спайков за шаг", ("spikes",), "шт",
         "сумма спайков по всему мозгу за один шаг управления"),
    ),
    "body": (
        ("dist", "Расстояние до еды", ("dist",), "мм",
         "до источника запаха"),
        ("amp", "Амплитуда ног", ("amp_l", "amp_r"), "",
         "множитель качания левой и правой стороны"),
        ("odour", "Запах на антеннах", ("odour_l", "odour_r"), "Гц",
         "частота, подаваемая в рецепторы"),
        ("dn", "Съём", ("dn_l", "dn_r"), "Гц",
         "средняя частота нейронов съёма слева и справа"),
        ("asym", "Асимметрия съёма", ("asym",), "",
         "разность сторон, делённая на сумму"),
        ("grad", "Совпадение с градиентом", ("grad_cos",), "",
         "косинус между курсом и направлением на еду, от -1 до +1"),
        ("spikes", "Спайков за шаг", ("spikes",), "шт",
         "сумма спайков по всему мозгу за один шаг управления"),
    ),
}


def available(run):
    """Какие графики можно построить честно. Ничего не читает с диска."""
    out = []
    if run.has("bins.npz"):
        out.append(Plot(
            "activity", "Активность мозга во времени",
            "сумма спайков всех записанных нейронов в бине", "шт/бин",
            "bins.npz: сумма bins по нейронам, шаг — bin_ms оттуда же",
            caveat="записаны только нейроны от 0,5 Гц: молчащих в файле "
                   "нет вовсе (sim.save с min_rate 0.5)"))
    if run.has("rates.parquet"):
        out.append(Plot(
            "regions", "Области по средней частоте",
            "средняя частота записанных нейронов области за весь прогон",
            "Гц", "rates.parquet: region, rate_hz", kind="bar",
            caveat="среднее по НАЙДЕННЫМ нейронам области, а не по всей "
                   "области: молчащие в файл не попали, и доля попавших у "
                   "областей разная"))
        out.append(Plot(
            "spread", "Распределение частот",
            "сколько нейронов на какой частоте", "Гц",
            "rates.parquet: rate_hz", kind="hist",
            caveat="слева обрезано порогом записи 0,5 Гц"))
    if run.kind in ("arena", "body") and run.has("track.parquet"):
        out.append(Plot(
            "track", "Траектория", "путь центра тела", "мм",
            "track.parquet: x, y" + ("; арена и еда из episode.json"
                                     if run.kind == "arena" else
                                     "; источники запаха из embodied.json"),
            kind="track"))
        for key, title, cols, unit, what in _LINES[run.kind]:
            out.append(Plot(key, title, what, unit,
                            "track.parquet: " + ", ".join(cols)
                            + "; ось времени — t_ms"))
    if run.kind == "arena" and run.has("probes.json"):
        out.append(Plot(
            "probes", "Популяции: связь с перекосом входа",
            "корреляция Пирсона асимметрии популяции с асимметрией входа",
            "", "probes.json: population, r", kind="bar",
            caveat="ORN — это сама подаваемая популяция, её корреляция "
                   "потолочная, и сравнивать с ней остальные нельзя"))
    if run.kind == "lateral" and run.has("lateral.json"):
        out.append(Plot(
            "lat_r", "Популяции по кодированию стороны",
            "корреляция асимметрии популяции с качанием входа", "",
            "lateral.json: population, r", kind="bar"))
        out.append(Plot(
            "lat_lag", "Задержка максимума корреляции",
            "на сколько ответ популяции отстаёт от входа", "мс",
            "lateral.json: population, lag_ms", kind="bar",
            caveat="одно число на популяцию: сама кросс-корреляция не "
                   "сохраняется, и восстановить её по лагу нельзя"))
    if run.kind == "bench" and run.has("bench.json"):
        out.append(Plot(
            "episodes", "Эпизоды серии по условиям",
            "совпадение курса с градиентом, по одному числу на эпизод", "",
            "bench.json: условие -> runs", kind="strip",
            caveat="поэпизодные значения сохранены только для совпадения с "
                   "градиентом; смещение до еды и визиты свёрнуты в среднее "
                   "и сумму ещё до записи"))
    return tuple(out)


def load(run, key):
    """Данные одного графика. Тяжёлые файлы открываются только здесь."""
    import numpy as np

    spec = next((p for p in available(run) if p.key == key), None)
    if spec is None:
        return None
    base = {"plot": spec}

    if key == "activity":
        z = bins_npz(run)
        if z is None:
            return None
        bins = z["bins"]
        bin_ms = float(z["bin_ms"])
        # Запись не обязана начинаться с нуля: у длинной сессии это
        # последнее окно, и ось, начатая с нуля, показала бы чужое время.
        t0_ms = float(z["t0_ms"]) if "t0_ms" in z.files else 0.0
        y = bins.sum(axis=1).astype("float64")
        x = t0_ms + (np.arange(len(y), dtype="float64") + 0.5) * bin_ms
        note = "бин %g мс, записанных нейронов %d" % (bin_ms, bins.shape[1])
        if t0_ms > 0:
            note += "; запись — окно с %.0f мс, не вся сессия" % t0_ms
        base.update(kind="line", x=x, xlabel="время, мс",
                    series=[("спайков в бине", y)], note=note)
        return base

    if key in ("regions", "spread"):
        cols = ["region", "rate_hz"] if key == "regions" else ["rate_hz"]
        df = rates_table(run, columns=cols)
        if df is None or not len(df):
            return None
        if key == "regions":
            g = (df.groupby("region")["rate_hz"].agg(["size", "mean"])
                   .sort_values("mean", ascending=False).head(18))
            base.update(kind="bar", labels=[str(i) for i in g.index],
                        values=[float(v) for v in g["mean"]],
                        counts=[int(v) for v in g["size"]],
                        note="18 областей с наибольшим средним; в скобках — "
                             "сколько нейронов области попало в файл")
            return base
        v = df["rate_hz"].to_numpy("float64")
        lo, hi = float(v.min()), float(v.max())
        n, edges = np.histogram(v, bins=24, range=(lo, max(hi, lo + 1e-6)))
        base.update(kind="hist", edges=edges.astype("float64"),
                    counts=n.astype("float64"),
                    note="нейронов в файле %d, медиана %.2f Гц, максимум "
                         "%.1f Гц" % (len(v), float(np.median(v)), hi))
        return base

    if key == "track":
        df = series_table(run, columns=["x", "y"])
        if df is None or not len(df):
            return None
        d = detail(run)
        arena = (d.get("episode.json") or {}).get("arena") or {}
        foods = [(float(f["x"]), float(f["y"]), float(f.get("r") or 1.0))
                 for f in (arena.get("foods") or [])]
        if not foods:
            for src in ((d.get("embodied.json") or {}).get("sources") or []):
                try:
                    foods.append((float(src[0]), float(src[1]), 1.0))
                except (TypeError, IndexError, ValueError):
                    pass
        base.update(kind="track", x=df["x"].to_numpy("float64"),
                    y=df["y"].to_numpy("float64"), foods=foods,
                    bounds=(float(arena.get("width") or 0.0),
                            float(arena.get("height") or 0.0)),
                    note="точек %d" % len(df))
        return base

    if key == "episodes":
        b = (detail(run) or {}).get("bench.json") or {}
        groups = []
        for cond, rec in b.items():
            if not isinstance(rec, dict):
                continue
            vals = [float(v) for v in (rec.get("runs") or [])
                    if _finite(v)]
            if vals:
                # sem у серии из одного эпизода записан как `nan`: это не
                # число, и дальше по нему ничего не рисуется.
                sem = rec.get("grad_alignment_sem")
                groups.append((cond, vals,
                               float(rec.get("grad_alignment_mean") or 0.0),
                               float(sem) if _finite(sem) else None))
        if not groups:
            return None
        base.update(kind="strip", groups=groups,
                    note="по одному числу на эпизод; сами эпизоды серии на "
                         "диск не сохраняются")
        return base

    if key in ("lat_r", "lat_lag", "probes"):
        d = detail(run)
        rows = d.get("lateral.json" if key.startswith("lat")
                     else "probes.json") or []
        field = "lag_ms" if key == "lat_lag" else "r"
        rows = [x for x in rows if isinstance(x, dict) and field in x]
        rows.sort(key=lambda x: -abs(float(x.get("r") or 0.0)))
        rows = rows[:18]
        if not rows:
            return None
        base.update(kind="bar",
                    labels=[str(x.get("population")) for x in rows],
                    values=[float(x[field]) for x in rows],
                    counts=[int(x.get("n_left") or 0) for x in rows],
                    note="сортировка по модулю корреляции; показаны первые "
                         "%d" % len(rows))
        return base

    cols = next((c for k, _t, c, _u, _w in _LINES.get(run.kind, ())
                 if k == key), None)
    if cols is None:
        return None
    df = series_table(run, columns=["t_ms"] + list(cols))
    if df is None or not len(df):
        return None
    base.update(kind="line", x=df["t_ms"].to_numpy("float64"),
                xlabel="время, мс",
                series=[(c, df[c].to_numpy("float64")) for c in cols
                        if c in df.columns],
                note="шагов %d" % len(df))
    return base
