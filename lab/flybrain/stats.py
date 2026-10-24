"""Статистика по сериям: что из сохранённого можно посчитать и чего нельзя.

Данные здесь одни — `bench.json`: на каждое условие число эпизодов, среднее,
sd, sem и **список сырых поэпизодных значений** `runs`. Сырые значения есть
только у одной величины, совпадения курса с градиентом; смещение до еды и
визиты свёрнуты в среднее и сумму ещё до записи, и переанализировать их по
эпизодам нельзя.

Правила, которых этот модуль придерживается и которые видны в его выводе:

* **Одно наблюдение — один эпизод.** Шаги внутри эпизода зависимы: поворот
  держится около сотни миллисекунд, градиент — секунды. Считать шаги за
  наблюдения значит завысить выборку в сотни раз; так прямо написано в
  `loop._metrics`, и здесь это соблюдается.
* **Парность решается по семенам, а не по порядку.** `loop.bench` и
  `body.bench` берут для обоих условий один и тот же набор семян
  (`seed0 + i`), то есть эпизод нормы и эпизод контроля под одним номером —
  это одна и та же расстановка арены и те же случайные числа, то есть
  настоящая пара. Но в файлах семена появились только сейчас: у прогонов,
  сохранённых раньше, их нет, и парность там **непроверяема**. Поэтому
  критерий выбирается по данным: где пары доказаны семенами — парный, где
  нет — непарный, и в обоих случаях это сказано словами. Угадывать парность
  по одинаковой длине списков нельзя.
* **p считается только когда есть что считать.** Меньше трёх наблюдений на
  условие — p не выводится вовсе, а не выводится «p = 1,0».
* **Ничего не объявляется доказанным.** Функция возвращает числа и словесную
  оговорку; слова «значимо», «подтверждает», «доказывает» не употребляются.
  Малое p на шести эпизодах, не повторившееся на десяти, — это повод
  присмотреться, а не вывод.
"""

from __future__ import annotations

import math

from .index import _welch

__all__ = ["condition_stats", "series_stats", "pairing",
           "paired_stats", "pool", "verdict", "warm", "CAVEATS"]

CAVEATS = (
    "Одно наблюдение — один эпизод. Шаги внутри эпизода зависимы "
    "(поворот держится ~100 мс, градиент — секунды), и за наблюдения их "
    "здесь не считают.",
    "Парность берётся из семян, записанных в файл, а не из порядка чисел "
    "в списке. Где семян нет — там парность непроверяема, и критерий "
    "непарный.",
    "Разные файлы не делают эпизоды независимыми. Независимость серий "
    "подтверждается только непересечением семян; без них она неизвестна.",
    "Сырые значения по эпизодам сохранены только для совпадения с "
    "градиентом. Смещение до еды и визиты свёрнуты в среднее и сумму.",
    "`n_runs` в файле — сколько эпизодов ВЫПОЛНЕНО, а не сколько заказано: "
    "оборванная серия неотличима от короткой, если в ней оба условия.",
    "Значимость этим окном не объявляется. Числа — это числа.",
)


# Ссылка на scipy держится здесь, а не берётся импортом на каждый вызов.
# Первый `from scipy import stats` стоит больше секунды, и он случался
# прямо в кадре окна — на первой же серии интерфейс замирал на 1,2 с.
# `warm()` переносит эту секунду в понятный момент: открытие каталога.
_SCIPY = []


def warm():
    """Подтянуть scipy заранее. Возвращает, получилось ли."""
    if not _SCIPY:
        try:
            from scipy import stats as sp_stats

            _SCIPY.append(sp_stats)
        except Exception:                                     # noqa: BLE001
            _SCIPY.append(None)
    return _SCIPY[0] is not None


def _t_crit(df, p=0.95):
    """Квантиль Стьюдента. Без scipy — нормальное приближение, и об этом
    говорится вслух: на пяти наблюдениях разница между 2,78 и 1,96
    заметная."""
    if warm():
        try:
            return float(_SCIPY[0].t.ppf(0.5 + p / 2.0, df)), True
        except Exception:                                     # noqa: BLE001
            pass
    return (1.959963985 if p > 0.9 else 1.0), False


def condition_stats(values):
    """Среднее, sd, sem и доверительный интервал по списку эпизодов."""
    vals = [float(v) for v in values]
    n = len(vals)
    if not n:
        return None
    mean = sum(vals) / n
    out = {"n": n, "mean": mean, "min": min(vals), "max": max(vals),
           "sd": None, "sem": None, "ci": None, "exact_t": True}
    if n < 2:
        out["why"] = ("одно наблюдение: разброса нет, sd и sem не "
                      "определены")
        return out
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    out["sd"], out["sem"] = sd, sem
    if n >= 3:
        t, exact = _t_crit(n - 1)
        out["ci"] = (mean - t * sem, mean + t * sem)
        out["exact_t"] = exact
    else:
        out["why"] = ("два наблюдения: интервал по ним настолько широк, "
                      "что не несёт смысла, и здесь он не считается")
    return out


def pairing(conds):
    """Доказана ли парность эпизодов. Вернуть (состояние, почему, пары).

    ``proven``   — у обоих условий записаны семена, и они совпадают
                   поэлементно: эпизод под номером i — одна и та же
                   расстановка и те же случайные числа;
    ``broken``   — семена записаны, но не совпадают: пар нет;
    ``unknown``  — семян нет. Так выглядят все серии, сохранённые до того,
                   как их начали писать.
    """
    if len(conds) != 2:
        return "unknown", "условий не два: парность не про это", 0
    a, b = conds
    sa, sb = a.get("seeds") or [], b.get("seeds") or []
    if not sa or not sb:
        return ("unknown",
                "семена эпизодов в файле не сохранены: проверить, что "
                "эпизод нормы и эпизод контроля — одна и та же "
                "расстановка, нечем", 0)
    n = min(len(sa), len(sb))
    if n and list(sa[:n]) == list(sb[:n]):
        return ("proven",
                "семена обоих условий совпадают поэлементно: эпизоды "
                "образуют пары (`bench` берёт seed0 + i для обоих "
                "условий)", n)
    return ("broken",
            "семена условий не совпадают: пар нет, и сравнивать эпизоды "
            "по номеру нельзя", 0)


def paired_stats(a_vals, b_vals):
    """Разности внутри пар. Это не новый критерий, а тот же t по d_i."""
    n = min(len(a_vals), len(b_vals))
    if n < 3:
        return None
    d = [float(a_vals[i]) - float(b_vals[i]) for i in range(n)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n) if n else 0.0
    if sem <= 0:
        return None
    t = mean / sem
    t_c, exact = _t_crit(n - 1)
    pv = None
    if warm():
        try:
            pv = float(2 * _SCIPY[0].t.sf(abs(t), n - 1))
        except Exception:                                     # noqa: BLE001
            pv = None
    return {"name": "Стьюдент по разностям пар, двусторонний, парный",
            "n_pairs": n, "diff": mean, "sd": sd, "se": sem,
            "ci": (mean - t_c * sem, mean + t_c * sem),
            "t": t, "df": float(n - 1), "p": pv, "exact_t": exact,
            "effect": (mean / sd) if sd > 0 else None,
            "pairs": d}


def series_stats(bench):
    """Разобрать `bench.json` в условия с числами и пригодностью выборки.

    Возвращает словарь: условия, можно ли сравнивать, и почему нет.
    """
    conds = []
    for name, rec in (bench or {}).items():
        if not isinstance(rec, dict):
            continue
        raw = [float(v) for v in (rec.get("runs") or [])]
        st = condition_stats(raw) or {"n": 0}
        conds.append({
            "name": name,
            "done": int(rec.get("n_runs") or len(raw)),
            "usable": len(raw),
            "seeds": list(rec.get("seeds") or []),
            "seed0": rec.get("seed0"),
            "episode_s": rec.get("duration_s"),
            "readout": rec.get("readout"),
            "dn_tau_s": rec.get("dn_tau_s"),
            "stats": st,
            "written": {"mean": rec.get("grad_alignment_mean"),
                        "sd": rec.get("grad_alignment_sd"),
                        "sem": rec.get("grad_alignment_sem")},
            "moved_mm": rec.get("food_dist_change_mm",
                                rec.get("dist_change_mm")),
            "near": rec.get("food_visits_total",
                            rec.get("food_steps_total")),
            "raw": raw,
        })
    out = {"conditions": conds, "comparable": False, "why": "",
           "test": None, "unpaired": None, "pairing": "unknown",
           "pairing_why": "", "n_pairs": 0,
           "embodied": any("dist_change_mm" in (r or {})
                           for r in (bench or {}).values()
                           if isinstance(r, dict))}
    if len(conds) < 2:
        out["why"] = ("в файле одно условие: сравнивать не с чем. Серия "
                      "считает норму и контроль, и одно условие значит, "
                      "что она не дошла до второго")
        return out
    if len(conds) > 2:
        out["why"] = "условий больше двух: какое с каким сравнивать — неясно"
        return out
    a, b = conds[0], conds[1]
    # Парность — свойство данных, а не критерия: её надо назвать даже
    # тогда, когда считать по этим эпизодам нечего.
    state, why, n_pairs = pairing(conds)
    out["pairing"], out["pairing_why"], out["n_pairs"] = state, why, n_pairs
    if a["usable"] < 3 or b["usable"] < 3:
        out["why"] = ("наблюдений слишком мало: %d и %d. Критерий на таких "
                      "данных ничего не покажет, и считать его нечестно"
                      % (a["usable"], b["usable"]))
        return out
    w = _welch(a["raw"], b["raw"])
    if w is None:
        out["why"] = "разброс нулевой: критерий не определён"
        return out
    diff_sd = math.sqrt((a["stats"]["sd"] ** 2 + b["stats"]["sd"] ** 2) / 2.0)
    t_c, exact = _t_crit(w["df"])
    unpaired = {
        "name": "Уэлч, двусторонний, непарный",
        "t": w["t"], "df": w["df"], "p": w["p"],
        "diff": w["diff"], "se": w["se"],
        "ci": (w["diff"] - t_c * w["se"], w["diff"] + t_c * w["se"]),
        "exact_t": exact,
        "effect": (w["diff"] / diff_sd) if diff_sd > 0 else None,
        "pair": why,
    }
    out["comparable"] = True
    out["unpaired"] = unpaired
    if state == "proven":
        # Пары доказаны семенами — значит непарный критерий здесь
        # неуместен: он выбрасывает ровно ту связь, ради которой серия
        # и гоняется с одинаковыми семенами.
        pt = paired_stats(a["raw"], b["raw"])
        if pt is not None:
            pt["pair"] = why
            out["test"] = pt
            return out
        # Пары есть, а критерия нет: либо пар меньше трёх, либо все
        # разности одинаковы. Подменять его непарным нельзя — непарный
        # на тех же данных даст число, которое к парным данным не
        # относится. Поэтому критерия нет, и сказано почему.
        same = (list(a["raw"][:n_pairs]) == list(b["raw"][:n_pairs]))
        out["comparable"] = False
        out["test"] = None
        out["why"] = (
            "пары доказаны семенами, но критерий по ним не определён: "
            + ("условия дали поэпизодно одни и те же числа — разности "
               "нулевые, различать нечего (так бывает, когда управление "
               "не влияет на поведение: например, усиление поворота ноль)"
               if same else
               "пар меньше трёх, на таком числе критерий ничего не "
               "покажет"))
        return out
    out["test"] = unpaired
    return out


def verdict(test, pooled=False):
    """Словами — ровно то, что получилось, без заявлений о доказанности."""
    if test is None:
        return "Сравнение не считалось."
    tail = ("" if not pooled else
            " Выборка собрана из нескольких серий вручную: такой результат "
            "не подтверждение гипотезы, а повод проверить её отдельным "
            "набором эпизодов.")
    p = test.get("p")
    if p is None:
        return ("p не посчитан: нет scipy, а без неё квантиль Стьюдента "
                "брать неоткуда. t = %.2f при df = %.1f." % (test["t"],
                                                             test["df"])
                + tail)
    if p < 0.05:
        return ("p = %.3f — разница заметнее разброса. Это не "
                "доказательство: одна серия повторяемости не даёт, и "
                "проверять её надо отдельным набором эпизодов." % p + tail)
    return ("p = %.2f — разница не выделяется из разброса. Это не "
            "доказательство отсутствия эффекта: при таком числе эпизодов "
            "слабый эффект был бы незаметен в любом случае." % p + tail)


def pool(benches):
    """Сложить несколько серий в одну выборку по условиям.

    Делается только по просьбе и только для совпадающих настроек: это
    ручное объединение, и проверить, что семена не пересекаются, нельзя —
    `repeat_seed0` в файлы не пишется. Возвращает (данные, оговорки).
    """
    merged = {}
    notes = []
    keys = None
    for name, b in benches:
        if not isinstance(b, dict):
            continue
        k = tuple(sorted(b.keys()))
        if keys is None:
            keys = k
        elif k != keys:
            notes.append("у «%s» другие условия (%s): не складываю"
                         % (name, ", ".join(k)))
            continue
        for cond, rec in b.items():
            if not isinstance(rec, dict):
                continue
            merged.setdefault(cond, {"runs": [], "n_runs": 0,
                                     "duration_s": rec.get("duration_s"),
                                     "readout": rec.get("readout"),
                                     "dn_tau_s": rec.get("dn_tau_s")})
            m = merged[cond]
            m["runs"].extend(float(v) for v in (rec.get("runs") or []))
            m["n_runs"] += int(rec.get("n_runs") or 0)
            for field in ("duration_s", "readout", "dn_tau_s"):
                if m[field] != rec.get(field):
                    notes.append("условие «%s»: %s расходится (%s против "
                                 "%s) — складывать такое нельзя"
                                 % (cond, field, m[field], rec.get(field)))
    # Независимость серий проверяется семенами, а не тем, что они лежат
    # в разных файлах. Где семена есть — проверяем честно; где их нет —
    # так и говорим.
    seen = {}
    have_seeds = True
    for name, b in benches:
        if not isinstance(b, dict):
            continue
        for cond, rec in b.items():
            if not isinstance(rec, dict):
                continue
            sd = list(rec.get("seeds") or [])
            if not sd:
                have_seeds = False
                continue
            prev = seen.setdefault(cond, {})
            clash = sorted(set(sd) & set(prev))
            if clash:
                notes.append("условие «%s»: семена %s уже встречались в "
                             "«%s» — это те же эпизоды, складывать их "
                             "нельзя" % (cond, clash[:6], prev[clash[0]]))
            for v in sd:
                prev.setdefault(v, name)
    if have_seeds and seen:
        notes.append("Семена серий не пересекаются: эпизоды разные. "
                     "Это проверено по записанным семенам, а не "
                     "предположено по тому, что файлы разные.")
    else:
        notes.append("Семена эпизодов записаны не везде: проверить, что "
                     "эпизоды не повторяются, нечем. Независимость здесь "
                     "НЕ установлена — разные файлы её не доказывают.")
    notes.append("Объединение ручное: это не подтверждение гипотезы, а "
                 "повод проверить её отдельным набором эпизодов.")
    return merged, notes
