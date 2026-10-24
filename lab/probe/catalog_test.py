"""Проверка справочника `flybrain.catalog` против настоящего кода.

Справочник — это данные о коде, и его легко разойтись с кодом молча: кто-то
переименует поле в `spec.py`, а интерфейс будет спокойно показывать прежнее.
Поэтому здесь сверяется не «работает ли», а «совпадает ли»:

1. каждое поле справочника есть в своём dataclass'е, и значение по умолчанию то
   же самое — сверка через `dataclasses.fields`;
2. обратно: каждое поле dataclass'а либо есть в справочнике, либо явно записано
   в `catalog.NOT_IN_UI` с причиной, почему в интерфейс не вынесено;
3. `spec.from_dict` на словаре, собранном по справочнику, даёт спеку с теми же
   значениями, и `to_dict` переживает круговой обход;
4. `validate` ловит неизвестный вид, число за пределом, пустой обязательный
   селектор и заведомо битый селектор, а на честной спеке молчит;
5. `limits` даёт осмысленные числа: таблица «длительность × шаг бина → память
   bins» сверяется прямым расчётом;
6. для каждого вида печатается, что он даст на выходе.

Работник не поднимается, коннектом не грузится, прогонов нет: проверка должна
отрабатывать за секунды.

Запуск:
    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8 && set PYTHONUTF8=1
    ..\\.venv\\Scripts\\python.exe probe\\catalog_test.py
"""

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import catalog as C, spec as S                    # noqa: E402

FAILS = []


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def fail(text):
    FAILS.append(text)
    print("  ПРОВАЛ: " + text)


def ok(text):
    print("  ок: " + text)


# Какой dataclass стоит за каким блоком справочника.
BLOCK_CLASS = {
    None: S.RunSpec,
    "stim": S.StimSpec,
    "arena": S.ArenaSpec,
    "body": S.BodySpec,
    "lateral": S.LateralSpec,
    "verify": S.VerifySpec,
    "agent": S.AgentSpec,
    "stim.activate": S.StimTerm,
}

_NODEFAULT = object()


def dc_default(cls, name):
    """Значение по умолчанию поля dataclass'а, с учётом фабрик."""
    for f in dataclasses.fields(cls):
        if f.name != name:
            continue
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:   # type: ignore[misc]
            return f.default_factory()                     # type: ignore[misc]
        return _NODEFAULT
    return KeyError


def same(a, b):
    """Равенство значений по умолчанию.

    Отдельно про числа: `False == 0` в питоне истинно, а для поля это
    расхождение — `bool` и `int` в интерфейсе разные вещи.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


# ---------------------------------------------------------------- 1 и 2


def check_names_and_defaults():
    section("1. поля справочника против dataclass'ов")
    all_fields = list(C.STIM_TERM_FIELDS)
    for kind in C.KINDS:
        all_fields.extend(C.fields_of(kind))

    checked = 0
    seen = {}
    for f in all_fields:
        cls = BLOCK_CLASS.get(f.block)
        if cls is None:
            fail(f"поле {f.block}.{f.key}: неизвестный блок {f.block!r}")
            continue
        names = {x.name for x in dataclasses.fields(cls)}
        if f.key not in names:
            fail(f"{cls.__name__}.{f.key} — такого поля в dataclass'е нет")
            continue
        real = dc_default(cls, f.key)
        if real is _NODEFAULT:
            if not f.required:
                fail(f"{cls.__name__}.{f.key} без значения по умолчанию, "
                     "но в справочнике не помечено обязательным")
            else:
                ok(f"{cls.__name__}.{f.key}: без умолчания, помечено "
                   "обязательным")
            checked += 1
            continue
        if not same(f.default, real):
            fail(f"{cls.__name__}.{f.key}: в справочнике {f.default!r}, "
                 f"в dataclass'е {real!r}")
        else:
            prev = seen.get((f.block, f.key))
            if prev is None:
                seen[(f.block, f.key)] = f.default
            elif not same(prev, f.default):
                fail(f"{cls.__name__}.{f.key}: в разных видах разные "
                     f"умолчания {prev!r} и {f.default!r}")
        checked += 1
    print(f"  сверено полей: {checked} "
          f"(уникальных: {len(seen) + len(C.STIM_TERM_FIELDS)})")

    section("2. поля dataclass'ов, которых в справочнике нет")
    in_catalog = {}
    for f in all_fields:
        in_catalog.setdefault(f.block, set()).add(f.key)

    for block, cls in BLOCK_CLASS.items():
        have = in_catalog.get(block, set())
        missing = [x.name for x in dataclasses.fields(cls)
                   if x.name not in have]
        if not missing:
            print(f"  {cls.__name__}: все {len(dataclasses.fields(cls))} полей "
                  "в справочнике")
            continue
        print(f"  {cls.__name__}: не вынесено {len(missing)} из "
              f"{len(dataclasses.fields(cls))}")
        for name in missing:
            why = C.NOT_IN_UI.get((cls.__name__, name))
            if why:
                print(f"    - {name}: {why}")
            else:
                fail(f"{cls.__name__}.{name} не вынесено в справочник и не "
                     "записано в catalog.NOT_IN_UI — причина не задокументирована")

    section("2б. какие общие поля RunSpec показывает каждый вид")
    root_all = sorted(in_catalog.get(None, set()))
    for kind in C.KINDS:
        mine = {f.key for f in C.fields_of(kind) if f.block is None}
        absent = [k for k in root_all if k not in mine]
        print(f"  {kind:8s} показывает {len(mine):2d}: {', '.join(sorted(mine))}")
        if absent:
            print(f"           не показывает: {', '.join(absent)}")


# ------------------------------------------------------------------- 3


def check_round_trip():
    section("3. словарь по справочнику -> RunSpec -> словарь")
    for kind in C.KINDS:
        d = C.demo_spec_dict(kind, source="cli")
        sp = S.from_dict(d)
        if sp.kind != kind:
            fail(f"{kind}: from_dict дал вид {sp.kind!r}")
            continue

        bad = []
        for f in C.fields_of(kind):
            want = d[f.key] if f.block is None else d[f.block][f.key]
            holder = sp if f.block is None else getattr(sp, f.block)
            got = getattr(holder, f.key)
            if f.type == "list":
                if len(got) != len(want):
                    bad.append(f"{f.key}: {len(got)} вместо {len(want)}")
                for i, (g, w) in enumerate(zip(got, want)):
                    for tf in C.STIM_TERM_FIELDS:
                        gv, wv = getattr(g, tf.key), w[tf.key]
                        if not same(gv, wv):
                            bad.append(f"{f.key}[{i}].{tf.key}: {gv!r} "
                                       f"вместо {wv!r}")
            elif not same(got, want):
                bad.append(f"{f.key}: {got!r} вместо {want!r}")
        if bad:
            for b in bad:
                fail(f"{kind}: {b}")
        else:
            ok(f"{kind}: все {len(C.fields_of(kind))} полей доехали до спеки")

        # круговой обход: to_dict роняет None, from_dict их восстанавливает
        again = S.from_dict(sp.to_dict())
        diffs = []
        for f in C.fields_of(kind):
            h1 = sp if f.block is None else getattr(sp, f.block)
            h2 = again if f.block is None else getattr(again, f.block)
            a, b = getattr(h1, f.key), getattr(h2, f.key)
            if f.type == "list":
                if len(a) != len(b):
                    diffs.append(f"{f.key}: длина {len(a)} -> {len(b)}")
                    continue
                for i, (x, y) in enumerate(zip(a, b)):
                    for tf in C.STIM_TERM_FIELDS:
                        if not same(getattr(x, tf.key), getattr(y, tf.key)):
                            diffs.append(f"{f.key}[{i}].{tf.key}: "
                                         f"{getattr(x, tf.key)!r} -> "
                                         f"{getattr(y, tf.key)!r}")
            elif not same(a, b):
                diffs.append(f"{f.key}: {a!r} -> {b!r}")
        if diffs:
            for x in diffs:
                fail(f"{kind}: круговой обход меняет {x}")
        else:
            ok(f"{kind}: to_dict -> from_dict ничего не потерял")


# ------------------------------------------------------------------- 4


def check_validate():
    section("4. validate: что считается годным, а что нет")

    for kind in C.KINDS:
        d = C.demo_spec_dict(kind, source="cli")
        errs = C.validate(d)
        if errs:
            for e in errs:
                fail(f"{kind}: честная спека забракована — {e}")
        else:
            ok(f"{kind}: спека с умолчаниями командной строки проходит")

    cases = [
        ("неизвестный вид", {"kind": "кракен"}, "неизвестный вид"),
        ("длительность за пределом",
         dict(C.demo_spec_dict("stim"), t=100000.0), "больше максимума"),
        ("отрицательный сид",
         dict(C.demo_spec_dict("stim"), seed=-1), "меньше минимума"),
        ("частота за пределом Пуассона",
         _with_term(C.demo_spec_dict("stim"), rate_hz=50000.0),
         "больше максимума"),
        ("ни одного стимула",
         _without_terms(C.demo_spec_dict("stim")), "хотя бы один"),
        ("пустой обязательный селектор",
         _with_term(C.demo_spec_dict("stim"), query=""), "пустой селектор"),
        ("неизвестный ключ селектора",
         _with_term(C.demo_spec_dict("stim"), query="кракен:left"),
         "unknown selector key"),
        ("битое регулярное выражение в селекторе",
         _with_term(C.demo_spec_dict("stim"), query="type:["), "error"),
        ("битый селектор глушения",
         _with_silence(C.demo_spec_dict("stim"), "класс:KC"),
         "unknown selector key"),
        ("импульс гаснет раньше, чем зажигается",
         _with_term(C.demo_spec_dict("stim"), t_on_ms=400.0, t_off_ms=100.0),
         "не позже включения"),
        ("стимул включается после конца прогона",
         _with_term(dict(C.demo_spec_dict("stim"), t=0.5), t_on_ms=900.0),
         "выходит за прогон"),
        ("шаг физики крупнее шага управления",
         _with_block(C.demo_spec_dict("body"), "body", physics_dt=0.05),
         "крупнее шага управления"),
        ("знак поворота не ±1",
         _with_block(C.demo_spec_dict("arena"), "arena", turn_sign=3.0),
         "не из списка"),
        ("неизвестный опыт апстрима",
         _with_block(C.demo_spec_dict("verify"), "verify",
                     experiment="сахарок"), "не из списка"),
        ("цель агента не задана",
         _with_block(C.demo_spec_dict("agent"), "agent", goal=""),
         "цель не задана"),
        ("вход перебора за пределом Пуассона",
         _with_block(C.demo_spec_dict("lateral"), "lateral", base_hz=9000.0),
         "не больше"),
        ("предохранитель max_bins",
         dict(C.demo_spec_dict("stim"), t=300.0, bin_ms=1.0,
              max_bins=10_000_000), "слишком длинный прогон"),
    ]

    width = max(len(name) for name, _, _ in cases)
    for name, spec, needle in cases:
        errs = C.validate(spec)
        hit = [e for e in errs if needle.lower() in e.lower()]
        if hit:
            print(f"  ок: {name:<{width}}  -> {hit[0][:92]}")
        else:
            fail(f"{name}: ждали фразу про {needle!r}, получили {errs!r}")


def _with_term(d, **kw):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    terms = [dict(t) for t in d["stim"]["activate"]]
    if not terms:
        terms = [{f.key: f.default for f in C.STIM_TERM_FIELDS}]
    terms[0].update(kw)
    d["stim"]["activate"] = terms
    return d


def _without_terms(d):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    d["stim"]["activate"] = []
    return d


def _with_silence(d, q):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    d["stim"]["silence"] = q
    return d


def _with_block(d, block, **kw):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    d[block] = dict(d.get(block) or {})
    d[block].update(kw)
    return d


# ------------------------------------------------------------------- 5


def check_limits():
    section("5. limits: память под bins и время по стене")

    N = C.N_NEURONS
    print(f"  плотный bins = torch.zeros(n_bins, {N}, int16) — sim.py\n")
    print("  стимуляция: длительность × шаг бина -> бинов -> память, МБ")
    print(f"  {'t, с':>7} | " + " | ".join(f"{b:>5g} мс" for b in
                                           (1.0, 5.0, 10.0, 20.0, 50.0)))
    print("  " + "-" * 62)
    for t in (0.5, 1.0, 5.0, 10.0, 60.0):
        cells = []
        for bin_ms in (1.0, 5.0, 10.0, 20.0, 50.0):
            d = dict(C.demo_spec_dict("stim"), t=t, bin_ms=bin_ms)
            cells.append(f"{C.bins_bytes(d) / 1e6:7.1f}")
        print(f"  {t:>7g} | " + " | ".join(cells))

    # Прямой расчёт одной клетки: t = 10 с, бин 10 мс.
    # steps = 10 * 1000 / 0.1 = 100 000; bin_steps = round(10 / 0.1) = 100;
    # n_bins = 100 000 / 100 = 1000; байт = 1000 * N * 2.
    d = dict(C.demo_spec_dict("stim"), t=10.0, bin_ms=10.0)
    want_bins = int(10.0 * 1000.0 / 0.1) // int(round(10.0 / 0.1))
    want_bytes = want_bins * N * 2
    got_bins, got_bytes = C.n_bins_of(d), C.bins_bytes(d)
    print(f"\n  прямой расчёт для t = 10 с, бин 10 мс: {want_bins} бинов, "
          f"{want_bytes} байт = {want_bytes / 1e6:.3f} МБ")
    if (got_bins, got_bytes) == (want_bins, want_bytes):
        ok(f"справочник даёт то же: {got_bins} бинов, {got_bytes} байт")
    else:
        fail(f"справочник даёт {got_bins} бинов и {got_bytes} байт вместо "
             f"{want_bins} и {want_bytes}")

    # Арена и тело считают бины по шагу управления, а не по бину записи.
    da = dict(C.demo_spec_dict("arena"), t=12.0, bin_ms=20.0)
    if C.n_bins_of(da) == 600:
        ok("арена: 12 с при шаге 20 мс -> 600 шагов, "
           f"{C.bins_bytes(da) / 1e6:.1f} МБ")
    else:
        fail(f"арена: ждали 600 шагов, получили {C.n_bins_of(da)}")
    db = dict(C.demo_spec_dict("body"), t=4.0)
    if C.n_bins_of(db) == 200:
        ok("тело: 4 с при шаге 20 мс -> 200 шагов, "
           f"{C.bins_bytes(db) / 1e6:.1f} МБ")
    else:
        fail(f"тело: ждали 200 шагов, получили {C.n_bins_of(db)}")

    print("\n  оценка времени по стене (без загрузки коннектома):")
    rows = [
        ("stim, 1 с, бин 10 мс", dict(C.demo_spec_dict("stim"), t=1.0)),
        ("stim, 10 с, бин 10 мс", dict(C.demo_spec_dict("stim"), t=10.0)),
        ("arena, 12 с", C.demo_spec_dict("arena")),
        ("arena, серия 6 × 4 с",
         dict(C.demo_spec_dict("arena"), repeat=6, repeat_t=4.0)),
        ("body, 4 с", C.demo_spec_dict("body")),
        ("body, серия 6 × 3 с",
         dict(C.demo_spec_dict("body"), repeat=6, repeat_t=3.0)),
        ("lateral, 10 с", C.demo_spec_dict("lateral")),
        ("verify, 1 с", C.demo_spec_dict("verify")),
        ("agent, 6 шагов", C.demo_spec_dict("agent")),
    ]
    width = max(len(n) for n, _ in rows)
    for name, spec in rows:
        sec = C.estimate_wall_s(spec)
        print(f"    {name:<{width}}  {sec:8.1f} с = {sec / 60:6.2f} мин")
        if sec <= 0:
            fail(f"{name}: оценка времени {sec}")

    # Серия должна множить: 6 × 2 эпизода не могут стоить меньше одного.
    one = C.estimate_wall_s(dict(C.demo_spec_dict("body"), t=3.0))
    many = C.estimate_wall_s(dict(C.demo_spec_dict("body"), repeat=6,
                                  repeat_t=3.0))
    if many > one * 10:
        ok(f"серия множит: один эпизод {one:.0f} с, 6 × 2 эпизода "
           f"{many:.0f} с (отношение {many / one:.1f})")
    else:
        fail(f"серия не множится: {one:.0f} с против {many:.0f} с")

    print("\n  сами предупреждения:")
    for kind in C.KINDS:
        warns = C.limits(C.demo_spec_dict(kind))
        print(f"\n  [{kind}]")
        for w in warns:
            print(f"    - {w}")
        if not warns:
            fail(f"{kind}: limits ничего не сказал")


# ------------------------------------------------------------------- 6


def check_measures():
    section("6. что каждый вид даёт на выходе")
    for kind, k in C.KINDS.items():
        m = C.measures_of(kind)
        if m is None:
            fail(f"{kind}: нет записи в MEASURES")
            continue
        n_fields = len(C.fields_of(kind))
        mark = "  (порождает подпрогоны)" if k.spawns_runs else ""
        print(f"\n  {kind} — {k.title}{mark}")
        print(f"    {k.about}")
        print(f"    полей для настройки: {n_fields}, "
              f"из них обязательных: "
              f"{sum(1 for f in C.fields_of(kind) if f.required)}, "
              f"с осмысленным None: "
              f"{sum(1 for f in C.fields_of(kind) if f.none_means)}")
        print(f"    файлы в out/<прогон>/:")
        for name, when, what in m.files:
            print(f"      {name:16s} [{when}] {what}")
        print(f"    в summary: "
              + ", ".join(key for key, _ in m.summary))
        print(f"    ряды в окно: "
              + (", ".join(m.series) if m.series else "нет"))
        print(f"    кадры активности: {'да' if m.activity_frames else 'нет'}; "
              f"кадры картинки: {'да' if m.picture_frames else 'нет'}")
        print(f"    в указателе (index.scan): {m.index_kind or '—'}")
        if k.notes:
            print("    оговорки по коду:")
            for n in k.notes:
                print(f"      • {n}")
        if not m.files:
            fail(f"{kind}: ни одного файла на выходе")


def check_series_against_jobs():
    section("6б. ряды справочника против jobs.py")
    import inspect

    from flybrain import jobs as J

    src = inspect.getsource(J)
    for kind in ("arena", "body"):
        missing = [s for s in C.series_of(kind) if f'"{s}"' not in src]
        if missing:
            fail(f"{kind}: рядов {missing} в jobs.py нет")
        else:
            ok(f"{kind}: все {len(C.series_of(kind))} рядов встречаются "
               "в jobs.py")
    for kind in ("stim", "lateral", "verify", "agent"):
        if C.series_of(kind):
            fail(f"{kind}: в jobs.py ctx.series не зовётся, а справочник "
                 f"обещает ряды {C.series_of(kind)}")
        else:
            ok(f"{kind}: рядов нет — так же, как в jobs.py")

    # Виды справочника должны совпадать с реестром задач и со схемой.
    if tuple(C.KINDS) != tuple(S.KINDS):
        fail(f"виды справочника {tuple(C.KINDS)} не совпадают со spec.KINDS "
             f"{tuple(S.KINDS)}")
    else:
        ok(f"виды совпадают со spec.KINDS: {', '.join(C.KINDS)}")
    if set(C.KINDS) != set(J._REGISTRY):
        fail(f"виды справочника не совпадают с реестром задач "
             f"{sorted(J._REGISTRY)}")
    else:
        ok("виды совпадают с реестром задач jobs._REGISTRY")


# ------------------------------------------------------------------ итог


def summary_table():
    section("сводка: вид -> полей -> что даёт на выходе")
    head = f"  {'вид':8s} {'полей':>6s} {'ряды':>5s} {'файлов':>7s}  на выходе"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for kind, k in C.KINDS.items():
        m = C.measures_of(kind)
        main = ", ".join(n for n, _, _ in m.files[:3])
        print(f"  {kind:8s} {len(C.fields_of(kind)):>6d} "
              f"{len(m.series):>5d} {len(m.files):>7d}  {main}")

    section("где пределы мои, а не из кода")
    # Один и тот же ключ в разных видах может иметь разные пределы (`t` у тела
    # уже, чем у стимуляции), поэтому различаем именно сочетания.
    rows, keys = [], set()
    for kind in C.KINDS:
        for f in list(C.fields_of(kind)) + (
                list(C.STIM_TERM_FIELDS) if kind == "stim" else []):
            if f.limits_from != "мой":
                continue
            if f.min is None and f.max is None:
                continue
            row = (f.block or "RunSpec", f.key, f.min, f.max, f.unit)
            if row not in rows:
                rows.append(row)
            keys.add((f.block, f.key))
    for block, key, lo, hi, unit in rows:
        print(f"  {block}.{key}: {lo} .. {hi} {unit}".rstrip())
    print(f"  всего таких полей: {len(keys)}, "
          f"разных сочетаний пределов: {len(rows)}")

    section("где пределы следуют из кода")
    rows = []
    for kind in C.KINDS:
        for f in list(C.fields_of(kind)) + (
                list(C.STIM_TERM_FIELDS) if kind == "stim" else []):
            if f.limits_from != "код":
                continue
            row = (f.block or "RunSpec", f.key, f.min, f.max, f.unit,
                   f.choices)
            if row not in rows:
                rows.append(row)
    for block, key, lo, hi, unit, choices in rows:
        where = (f"из списка {choices}" if choices
                 else f"{lo} .. {hi} {unit}".rstrip())
        print(f"  {block}.{key}: {where}")

    section("поля, где None — это поведение")
    seen = set()
    for kind in C.KINDS:
        for f in list(C.fields_of(kind)) + list(C.STIM_TERM_FIELDS):
            if f.none_means and (f.block, f.key) not in seen:
                seen.add((f.block, f.key))
                print(f"  {f.block or 'RunSpec'}.{f.key}: {f.none_means}")


def main():
    t0 = time.perf_counter()
    check_names_and_defaults()
    check_round_trip()
    check_validate()
    check_limits()
    check_measures()
    check_series_against_jobs()
    summary_table()

    section("итог")
    print(f"  время проверки: {time.perf_counter() - t0:.2f} с")
    if FAILS:
        print(f"  ПРОВАЛОВ: {len(FAILS)}")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  всё сошлось")
    return 0


if __name__ == "__main__":
    sys.exit(main())
