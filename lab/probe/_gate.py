"""Общая обвязка шлюзов: судья, который обязан уметь провалиться.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. В этом проекте четыре пробы уже отчитались «ок»
на пустом механизме: подделка «выход из записи» дала 838,1× ускорения
при побитово совпавших отпечатках и НУЛЕ вызовов движка; проба печатала
«150 Гц» вообще без раздражителя, потому что смотрела на число кадров;
подмена дороги умножения обёрткой на питоне не поймалась ни разу за 40
шагов. Общее у всех четырёх одно: **проверка не умела провалиться**.

Отсюда устройство этого модуля. Шлюз здесь — не «печать ок», а
:func:`gate`, и у него три части, из которых обязательны все три:

1. **замер** — словарь чисел, снятый с настоящей системы;
2. **правила** — функции от этого словаря, каждая со своим числом и
   порогом; их объединение и есть шлюз;
3. **подлоги** — переделки ТОГО ЖЕ замера, каждая с указанием, какое
   именно правило обязана уронить. Подлог, который судья пропустил,
   считается провалом САМОГО СУДЬИ, а не замера.

Третья часть — главная. Без неё «шлюз пройден» ничего не значит, и
проверить это можно только числом: подлог либо роняет правило, либо нет.

РАЗНИЦА МЕЖДУ ДВУМЯ ВЕРДИКТАМИ, И ПУТАТЬ ИХ НЕЛЬЗЯ.

* ``судья исправен`` — все подлоги уронили то правило, которое обязаны
  были. Это утверждение о проверке, и его провал — провал пробы (код
  возврата 1).
* ``шлюз на нынешнем v2`` — прошёл ли шлюз на настоящем замере. Это
  ИСХОДНОЕ СОСТОЯНИЕ, и его «не пройден» — честный результат, а не
  дефект пробы. Шлюз C на нынешнем v2 не проходит по построению: сигнал
  до LC4 и DNp01 не доходит, и проба обязана это записать, а не
  подогнать порог.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["Report", "fp", "gate", "rule"]

OUT = Path(__file__).parent


def fp(a) -> str:
    """Отпечаток массива: первые 16 знаков sha256 от его байтов."""
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def rule(name, fn, нужно=""):
    """Одно правило шлюза.

    ``fn(m)`` возвращает ``(ok, строка с числом)``. Строка обязана
    содержать ЧИСЛО: правило без числа нельзя перенести в отчёт и нельзя
    сравнить с «до».
    """
    return {"имя": name, "функция": fn, "нужно": нужно}


def forge(name, fn, роняет, зачем=""):
    """Один подлог: переделка замера и правило, которое обязано упасть.

    ``fn(m)`` возвращает НОВЫЙ словарь замера (исходный не меняется).
    ``роняет`` — имя правила из набора шлюза.
    """
    return {"имя": name, "функция": fn, "роняет": роняет, "зачем": зачем}


class Report:
    """Отчёт пробы: шлюзы, замеры, находки, код возврата."""

    def __init__(self, title, path):
        self.title = title
        self.path = Path(path)
        self.t0 = time.perf_counter()
        self.d = {"проба": title, "шлюзы": {}, "замеры": {},
                  "находки": [], "сорвалось": []}
        self.fails = []
        print(title, flush=True)

    def note(self, text):
        print("  НАХОДКА " + text, flush=True)
        self.d["находки"].append(text)

    def broke(self, text):
        print("  СОРВАЛОСЬ " + text, flush=True)
        self.d["сорвалось"].append(text)

    def measure(self, name, value):
        self.d["замеры"][name] = value
        return value

    def gate(self, letter, title, m, rules, forgeries):
        """Прогнать один шлюз: правила на настоящем замере и подлоги.

        Возвращает словарь шлюза. В ``self.fails`` попадает только то,
        что сломано В ПРОБЕ: подлог, который судья пропустил, или
        правило, упавшее с ошибкой. Непройденный шлюз на настоящем
        замере в ``fails`` НЕ попадает — это исходное состояние.
        """
        print("\n=== ШЛЮЗ %s. %s" % (letter, title), flush=True)
        got = {"буква": letter, "что проверяет": title,
               "правила": {}, "подлоги": {}}

        # --- 1. правила на настоящем замере ---
        real = {}
        for r in rules:
            try:
                ok, info = r["функция"](m)
            except Exception as e:                       # noqa: BLE001
                ok, info = None, "ОШИБКА ПРАВИЛА: %r" % (e,)
                self.fails.append("правило %r упало с ошибкой: %r"
                                  % (r["имя"], e))
            real[r["имя"]] = bool(ok) if ok is not None else None
            got["правила"][r["имя"]] = {
                "пройдено": real[r["имя"]], "нужно": r["нужно"],
                "замер": str(info)}
            mark = ("  ok     " if ok else
                    ("  ОШИБКА " if ok is None else "  НЕ ПРОШЛО "))
            print(mark + r["имя"] + "\n           " + str(info), flush=True)

        passed = all(v is True for v in real.values())
        got["шлюз на нынешнем v2"] = bool(passed)
        got["правил всего"] = len(rules)
        got["правил не прошло"] = sum(1 for v in real.values()
                                      if v is not True)

        # --- 2. подлоги: судья обязан уронить каждое названное правило ---
        by_name = {r["имя"]: r for r in rules}
        judge_ok = True
        for f in forgeries:
            target = by_name.get(f["роняет"])
            if target is None:
                self.fails.append("подлог %r ссылается на правило %r, "
                                  "которого в шлюзе нет"
                                  % (f["имя"], f["роняет"]))
                judge_ok = False
                continue
            try:
                m2 = f["функция"](dict(m))
                ok2, info2 = target["функция"](m2)
            except Exception as e:                       # noqa: BLE001
                ok2, info2 = None, "ОШИБКА ПОДЛОГА: %r" % (e,)
            dropped = (ok2 is False)
            got["подлоги"][f["имя"]] = {
                "роняет правило": f["роняет"],
                "судья уронил": bool(dropped),
                "зачем": f["зачем"],
                "замер на подлоге": str(info2)}
            print(("  ПОДЛОГ уронён    " if dropped
                   else "  ПОДЛОГ ПРОПУЩЕН ")
                  + f["имя"] + " -> " + f["роняет"]
                  + "\n           " + str(info2), flush=True)
            if not dropped:
                judge_ok = False
                self.fails.append(
                    "судья шлюза %s пропустил подлог %r (обязан был "
                    "уронить правило %r)" % (letter, f["имя"], f["роняет"]))

        got["судья исправен"] = bool(judge_ok)
        got["подлогов"] = len(forgeries)
        print("  ИТОГ шлюза %s: на нынешнем v2 %s; судья %s (%d подлогов)"
              % (letter, "ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН",
                 "исправен" if judge_ok else "СЛОМАН", len(forgeries)),
              flush=True)
        self.d["шлюзы"][letter] = got
        return got

    def finish(self):
        self.d["проба заняла, с"] = round(time.perf_counter() - self.t0, 1)
        self.path.write_text(
            json.dumps(self.d, ensure_ascii=False, indent=1,
                       default=str),
            encoding="utf-8")
        g = self.d["шлюзы"]
        print("\nшлюзов %d; пройдено на v2 %d; судей исправно %d; "
              "подлогов %d; за %.0f с"
              % (len(g),
                 sum(1 for v in g.values() if v["шлюз на нынешнем v2"]),
                 sum(1 for v in g.values() if v["судья исправен"]),
                 sum(v["подлогов"] for v in g.values()),
                 self.d["проба заняла, с"]), flush=True)
        print("отчёт: " + str(self.path), flush=True)
        if self.fails:
            print("ПРОВАЛЫ ПРОБЫ (сломана проверка, а не замер):",
                  flush=True)
            for f in self.fails:
                print("  " + f, flush=True)
        return 1 if self.fails else 0


def main_guard(fn):
    """Обёртка ``main``: срыв записывается в отчёт, а не теряется."""
    def run():
        try:
            return fn()
        except KeyboardInterrupt:
            print("прервано", flush=True)
            return 130
    return run


def cli_exit(code):
    sys.exit(code)
