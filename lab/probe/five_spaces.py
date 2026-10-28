"""Рабочих пространств ровно пять: проверка числом, а не обещанием.

Сеанс песочницы смотрится **внутри** «Анализа», видом прогона, а не
шестым пространством. Это условие проекта, и проверять его надо числом:
сколько записей в `desk.state.WORKSPACES`, какие у них ключи и подписи,
столько ли пространств собрано в `desk.app.SPACES` и столько ли модулей
`ws_*.py` лежит в каталоге.

Главное здесь — **встречный замер на сам критерий**. Проверка «пять
записей» сама по себе ничего не стоит: такая же строчка прошла бы и на
списке из шести, если в ней ошибка. Поэтому один и тот же судья
(:func:`judge`) гоняется трижды: на настоящем списке (обязан пройти), на
том же списке с подложенным шестым пространством (обязан провалиться) и
на урезанном до четырёх (обязан провалиться). Если судья не ловит
подлог, об этом говорится отдельно — и тогда провалом считается он, а не
список.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\five_spaces.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
LAB = Path(__file__).resolve().parents[1]
DESK = LAB / "flybrain" / "desk"

fails = []
rep = {"проверки": {}, "замеры": {}, "встречный замер": {}, "находки": []}

# Пять пространств, как они объявлены владельцем проекта. Перечислены
# здесь буквально, а не выведены из кода: инвариант, сверяемый с самим
# же кодом, не инвариант.
WANT = (("experiment", "Эксперимент"), ("brain", "Мозг"), ("body", "Тело"),
        ("sandbox", "Песочница"), ("analysis", "Анализ"))


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def judge(pairs):
    """Судья инварианта: пять записей, те самые ключи, те самые подписи.

    Возвращает ``(прошло, причина)``. Отдельной функцией — чтобы её
    можно было прогнать на подложенном списке и увидеть, ловит ли она
    подлог вовсе.
    """
    pairs = tuple(tuple(p) for p in pairs)
    if len(pairs) != len(WANT):
        return False, "записей %d, а должно быть %d" % (len(pairs), len(WANT))
    for i, (got, want) in enumerate(zip(pairs, WANT)):
        if got != want:
            return False, "запись %d: %r, а должна быть %r" % (i, got, want)
    return True, "записей %d: %s" % (len(pairs),
                                     ", ".join(n for _, n in pairs))


def part_list():
    """Сам список: число записей, ключи, подписи."""
    from flybrain.desk import state as ST

    got = tuple(tuple(p) for p in ST.WORKSPACES)
    rep["замеры"]["записей в WORKSPACES"] = len(got)
    rep["замеры"]["WORKSPACES"] = [list(p) for p in got]
    ok, why = judge(got)
    check("WORKSPACES: ровно пять записей, те самые ключи и подписи",
          ok, why)

    # Переключатель пространств берёт имена из этого же списка, то есть
    # шестая запись появилась бы в окне сама. Проверяется, что другого
    # источника имён нет.
    names = [n for _, n in got]
    check("подписи в окне берутся из этого списка",
          len(set(names)) == 5 and "" not in names,
          "имён %d, все разные и непустые" % len(set(names)))


def part_spaces():
    """Собранные пространства: `desk.app.SPACES` и модули `ws_*.py`."""
    from flybrain.desk import app as A, state as ST

    keys = [k for k, _ in ST.WORKSPACES]
    rep["замеры"]["ключей в SPACES"] = len(A.SPACES)
    check("SPACES: собрано ровно пять пространств",
          len(A.SPACES) == 5 and sorted(A.SPACES) == sorted(keys),
          "ключей %d: %s" % (len(A.SPACES), ", ".join(sorted(A.SPACES))))

    mods = sorted(p.name for p in DESK.glob("ws_*.py"))
    rep["замеры"]["модулей ws_*.py"] = mods
    check("модулей пространств на диске ровно пять",
          len(mods) == 5, "%d: %s" % (len(mods), ", ".join(mods)))

    # Самопроверка окна ходит по пространствам по тому же списку.
    from flybrain.desk import selftest as SELF

    src = (DESK / "selftest.py").read_text(encoding="utf-8")
    check("самопроверка окна берёт пространства из WORKSPACES, а не своим "
          "списком",
          "WORKSPACES" in src and bool(SELF),
          "в selftest.py есть ссылка на WORKSPACES")


def part_session_is_a_run():
    """Сеанс песочницы — прогон внутри «Анализа», а не пространство.

    Проверяется тремя числами: вид прогона остаётся прежним (не
    появилось нового), таймлайн — вид внутри «Анализа» (запись в
    `ws_analysis.VIEWS`), и ни одного модуля пространства под сеанс не
    завелось.
    """
    from flybrain import runs as RN
    from flybrain.desk import ws_analysis as WA

    kinds = tuple(RN.KIND_TITLES)
    rep["замеры"]["видов прогона"] = len(kinds)
    rep["замеры"]["виды прогона"] = list(kinds)
    check("новых видов прогона под сеанс не завелось",
          "session" not in kinds and "sandbox" not in kinds
          and "replay" not in kinds,
          "видов %d: %s" % (len(kinds), ", ".join(kinds)))

    rep["замеры"]["видов в «Анализе»"] = len(WA.VIEWS)
    rep["замеры"]["виды в «Анализе»"] = list(WA.VIEWS)
    check("таймлайн сеанса — вид внутри «Анализа»",
          "Таймлайн" in WA.VIEWS,
          "видов %d: %s" % (len(WA.VIEWS), ", ".join(WA.VIEWS)))

    # Признак сеанса живёт на прогоне, а не на пространстве.
    check("признак сеанса — поле прогона (`Run.session`), а не "
          "пространство",
          hasattr(RN.Run, "session") and hasattr(RN.Run, "has_timeline"),
          "Run.session и Run.has_timeline есть")

    # Ни один модуль не объявляет собственного списка пространств.
    bad = []
    pat = re.compile(r"(Эксперимент|Песочница)[\"'].{0,40}[\"'](Анализ|Мозг)")
    for p in sorted((LAB / "flybrain").rglob("*.py")):
        if "__pycache__" in p.parts or p.name == "state.py":
            continue
        if pat.search(p.read_text(encoding="utf-8")):
            bad.append(str(p.relative_to(LAB)))
    rep["замеры"]["вторых списков пространств"] = bad
    check("второго списка пространств в коде нет",
          not bad, "найдено %d" % len(bad) + (": " + ", ".join(bad)
                                              if bad else ""))


def part_counter():
    """Встречный замер: ловит ли судья подложенное шестое пространство.

    Без этого «пять записей» — утверждение о коде, а не о нём замер:
    пустой судья напечатал бы то же самое.
    """
    cases = {
        "настоящий список": (WANT, True),
        "подложено шестое пространство": (
            WANT + (("replay", "Воспроизведение"),), False),
        "подложен шестой вид под другим именем": (
            WANT + (("session", "Сеанс"),), False),
        "убрано одно пространство": (WANT[:-1], False),
        "переставлены местами": (WANT[1:] + WANT[:1], False),
        "подменена подпись": (
            WANT[:4] + (("analysis", "Воспроизведение"),), False),
    }
    for name, (pairs, want_ok) in cases.items():
        ok, why = judge(pairs)
        rep["встречный замер"][name] = {"судья пропустил": bool(ok),
                                        "должен был": bool(want_ok),
                                        "причина": why}
        check("встречный замер — %s: судья %s"
              % (name, "пропускает" if want_ok else "ловит"),
              ok == want_ok, why)


def main():
    print("— список пространств —")
    part_list()
    print("\n— собранные пространства —")
    part_spaces()
    print("\n— сеанс песочницы: прогон, а не пространство —")
    part_session_is_a_run()
    print("\n— встречный замер на самого судью —")
    part_counter()
    (OUT / "five_spaces.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    for n in rep["находки"]:
        print("  замечание: " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
