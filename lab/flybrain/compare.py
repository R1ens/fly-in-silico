"""Сравнение сохранённых прогонов: что сопоставимо, а что только выглядит так.

Складывать числа из разных прогонов можно не всегда, и причин несколько.

* **Разные виды.** Стимуляция и эпизод в арене не меряют одно и то же:
  у первой нет ни траектории, ни градиента, у второй нет растра по бинам в
  том же смысле. Общими остаются настройки и время счёта.
* **Одинаковые имена при разном смысле.** `food_visits_total` в арене — это
  сколько раз муха оказалась у еды, а `food_steps_total` в теле — сколько
  шагов она там простояла. Обе величины «штук», обе про еду, и складывать их
  нельзя.
* **Разные имена при одном смысле.** `food_dist_change_mm` и
  `dist_change_mm` — обе «на сколько миллиметров приблизилась», и их
  сопоставить можно, но с оговоркой: в арене это до ближайшей еды, в теле —
  до источника запаха.
* **Ноль в знаменателе.** Процент изменения при нулевом исходном значении не
  выводится: это не «бесконечный рост», это отсутствие опоры для доли.

Поэтому каждая строка сравнения несёт не только значения, но и признак
сопоставимости и причину, по которой он такой.
"""

from __future__ import annotations

from .runs import detail

__all__ = ["MEASURES", "table", "percent"]

# Величина: подпись, единицы, оговорка. Ключ — как она названа в файле.
MEASURES = {
    "grad_alignment": ("Совпадение с градиентом", "",
                       "среднее по шагам эпизода, от -1 до +1"),
    "turn_to_odour": ("Поворотов к запаху", "доля",
                      "шаги внутри эпизода зависимы: доля считается по "
                      "шагам, и за размер выборки её брать нельзя"),
    "mean_speed_mm_s": ("Средняя скорость", "мм/с", ""),
    "speed_mm_s": ("Средняя скорость", "мм/с", ""),
    "mean_dn_hz": ("Средняя частота съёма", "Гц", ""),
    "food_dist_first_half_mm": ("До еды, первая половина", "мм", ""),
    "food_dist_second_half_mm": ("До еды, вторая половина", "мм", ""),
    "dist_first_half_mm": ("До цели, первая половина", "мм", ""),
    "dist_second_half_mm": ("До цели, вторая половина", "мм", ""),
    "dist_start_mm": ("До цели в начале", "мм", ""),
    "dist_end_mm": ("До цели в конце", "мм", ""),
    "path_mm": ("Длина пути", "мм", ""),
    "food_visits": ("Визитов к еде", "шт",
                    "визит — вход в зону еды; это не то же, что шаги в ней"),
    "food_steps": ("Шагов у еды", "шт", ""),
    "wall_bounces": ("Ударов о стену", "шт", ""),
    "steps": ("Шагов управления", "шт", ""),
    "n_active": ("Нейронов от 0,5 Гц", "шт",
                 "порог записи, а не биологический порог"),
    "total_spikes": ("Спайков всего", "шт", ""),
    "n_neurons": ("Нейронов в модели", "шт", ""),
}

# Одно и то же под разными именами: можно сопоставлять, но с оговоркой.
_TARGET = ("в арене это расстояние до ближайшей еды, в теле — до источника "
           "запаха: величина одна, цель разная")
_SAME = {
    "food_dist_change_mm": ("moved", "Смещение к цели", "мм", _TARGET),
    "dist_change_mm": ("moved", "Смещение к цели", "мм", _TARGET),
    "mean_speed_mm_s": ("speed", "Средняя скорость", "мм/с",
                        "в арене скорость задаётся моделью движения, в теле "
                        "она получается из ходьбы: числа сопоставимы, "
                        "происхождение разное"),
    "speed_mm_s": ("speed", "Средняя скорость", "мм/с",
                   "в арене скорость задаётся моделью движения, в теле она "
                   "получается из ходьбы: числа сопоставимы, происхождение "
                   "разное"),
    "food_dist_first_half_mm": ("half1", "До цели, первая половина", "мм",
                                _TARGET),
    "dist_first_half_mm": ("half1", "До цели, первая половина", "мм",
                           _TARGET),
    "food_dist_second_half_mm": ("half2", "До цели, вторая половина", "мм",
                                 _TARGET),
    "dist_second_half_mm": ("half2", "До цели, вторая половина", "мм",
                            _TARGET),
}

# Разное под похожими именами: сопоставлять нельзя.
_DIFFERENT = {
    "food_visits_total": ("Визитов к еде за серию", "шт",
                          "это ВХОДЫ в зону еды; у серии в теле на том же "
                          "месте стоят ШАГИ у еды — другая величина"),
    "food_steps_total": ("Шагов у еды за серию", "шт",
                         "это ШАГИ в зоне еды; у серии в арене на том же "
                         "месте стоят ВИЗИТЫ — другая величина"),
}


def _metrics_of(run, d=None):
    """Измеренные величины прогона: из его главного файла, как есть."""
    d = d if d is not None else detail(run)
    for name in ("episode.json", "embodied.json"):
        m = (d.get(name) or {}).get("metrics")
        if isinstance(m, dict):
            return dict(m)
    m = d.get("meta.json") or {}
    if m:
        return {k: m[k] for k in ("n_active", "total_spikes", "n_neurons")
                if k in m}
    b = d.get("bench.json") or {}
    if isinstance(b, dict) and b:
        out = {}
        for cond, rec in b.items():
            if isinstance(rec, dict):
                for k in ("grad_alignment_mean", "food_dist_change_mm",
                          "dist_change_mm", "food_visits_total",
                          "food_steps_total", "n_runs"):
                    if k in rec:
                        out["%s · %s" % (cond, k)] = rec[k]
        return out
    return {}


def _params_of(run, d=None):
    """Настройки прогона: то, что заказывали, а не то, что намерили."""
    d = d if d is not None else detail(run)
    out = {"вид": run.kind_title, "состояние": run.state_word}
    spec = run.spec or {}
    if spec:
        for k in ("t", "seed", "bin_ms", "device", "repeat", "repeat_t"):
            if spec.get(k) is not None:
                out[k] = spec[k]
        for block in ("stim", "arena", "body", "lateral", "verify"):
            for k, v in (spec.get(block) or {}).items():
                if k == "activate":
                    v = ", ".join(str(t.get("query")) for t in v
                                  if isinstance(t, dict))
                if v is not None:
                    out["%s.%s" % (block, k)] = v
    ep = d.get("episode.json") or d.get("embodied.json") or {}
    for k, v in (ep.get("config") or {}).items():
        if v is not None and k not in ("model",):
            out.setdefault(k, v)
    body = ep.get("body") or (ep.get("config") or {})
    for k in ("dn_ref_hz", "turn_gain_rad_s", "dn_tau_s", "asym_bias",
              "swap_antennae", "motor_left", "turn_sign"):
        if body.get(k) is not None:
            out.setdefault("измерено/задано: " + k, body[k])
    # Нефизиологичные детали моторного пути. В сверку они идут отдельной
    # строкой на деталь, с именем и версией: два прогона с разными
    # декодерами не должны выглядеть как два прогона одного опыта.
    eng = ep.get("engineered") or {}
    if eng:
        out["инженерное: версия пути"] = eng.get("версия моторного пути")
        for det in (eng.get("детали") or []):
            if not isinstance(det, dict):
                continue
            label = "инженерное: " + str(det.get("имя"))
            val = "v%s" % det.get("версия")
            if det.get("вид"):
                val += ", " + str(det["вид"])
            if det.get("sha256"):
                val += ", sha256 " + str(det["sha256"])[:12]
            out[label] = val
    m = d.get("meta.json") or {}
    for s in (m.get("stimuli") or []):
        out.setdefault("стимул", "%s, %d нейронов, %g Гц"
                       % (s.get("name"), s.get("n") or 0,
                          s.get("rate_hz") or 0.0))
    cfg = m.get("config") or {}
    for k in ("t_run_s", "bin_ms", "n_trials", "seed"):
        if cfg.get(k) is not None:
            out.setdefault("meta." + k, cfg[k])
    b = d.get("bench.json") or {}
    if isinstance(b, dict):
        for cond, rec in b.items():
            if isinstance(rec, dict):
                out.setdefault("серия: условие", cond)
                for k in ("readout", "dn_tau_s", "duration_s", "n_runs"):
                    if rec.get(k) is not None:
                        out.setdefault("серия." + k, rec[k])
                break
    return out


def percent(a, b):
    """Изменение в процентах, если знаменатель это позволяет."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None, "не числа"
    if a == 0.0:
        return None, "исходное значение ноль: доли от него не бывает"
    if abs(a) < 1e-9:
        return None, "исходное значение почти ноль: процент был бы шумом"
    return (b - a) / abs(a) * 100.0, ""


def table(runs):
    """Сравнение нескольких прогонов.

    Возвращает `{"runs", "params", "measures", "notes", "kinds"}`, где
    каждая строка — `(подпись, единицы, [значения], сопоставимо, почему)`.
    """
    if len(runs) < 2:
        return {"runs": list(runs), "params": [], "measures": [],
                "notes": ["Для сравнения нужно выбрать хотя бы два прогона."],
                "kinds": []}
    dets = [detail(r) for r in runs]
    kinds = [r.kind for r in runs]
    notes = []
    if len(set(kinds)) > 1:
        notes.append("Виды прогонов разные (%s): общими остаются настройки "
                     "и время счёта, измеренные величины у них разные по "
                     "определению." % ", ".join(
                         dict.fromkeys(r.kind_title for r in runs)))
    if any(r.state == "cancelled" for r in runs):
        notes.append("Среди выбранных есть остановленный прогон: его числа "
                     "относятся к укороченному времени, и сравнивать их с "
                     "полным прогоном напрямую нельзя.")
    if any(r.state == "unknown" for r in runs):
        notes.append("У части прогонов исход не записан: они сделаны "
                     "командной строкой. Был ли такой прогон доведён до "
                     "конца, из файлов не видно.")

    params = []
    keys = []
    maps = [_params_of(r, d) for r, d in zip(runs, dets)]
    for m in maps:
        for k in m:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [m.get(k) for m in maps]
        same = len({str(v) for v in vals}) == 1
        params.append((k, "", vals, True,
                       "" if same else "значения расходятся"))

    measures = []
    mm = [_metrics_of(r, d) for r, d in zip(runs, dets)]
    keys = []
    for m in mm:
        for k in m:
            if k not in keys:
                keys.append(k)
    seen_groups = {}
    for k in keys:
        if k in _DIFFERENT:
            label, unit, why = _DIFFERENT[k]
            measures.append((label, unit, [m.get(k) for m in mm], False, why))
            continue
        if k in _SAME:
            group, label, unit, why = _SAME[k]
            if group in seen_groups:
                continue
            seen_groups[group] = True
            vals = []
            for m in mm:
                v = None
                for alias, (g, _l, _u, _w) in _SAME.items():
                    if g == group and alias in m:
                        v = m[alias]
                        break
                vals.append(v)
            measures.append((label, unit, vals, True, why))
            continue
        label, unit, why = MEASURES.get(k, (k, "", ""))
        vals = [m.get(k) for m in mm]
        missing = sum(1 for v in vals if v is None)
        ok = missing == 0
        if not ok:
            why = (why + ("; " if why else "")
                   + "величина есть не у всех выбранных прогонов")
        measures.append((label, unit, vals, ok, why))
    return {"runs": list(runs), "params": params, "measures": measures,
            "notes": notes, "kinds": kinds}
