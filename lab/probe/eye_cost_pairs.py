# -*- coding: utf-8 -*-
r"""Пары «v3 против v2» чередующимися процессами.

Мерить две сборки кода в одном процессе нельзя — вторую не импортировать
поверх первой, — а мерить их в разное время бессмысленно: машина дышит,
и в этом проекте уже замерено, что разброс шага на полном контуре даёт
отношение ×1,385 сам по себе. Отсюда правило, которому следуют все
замеры скорости здесь: **чередующиеся пары, отношение внутри пары,
медиана и размах по парам, и порог различимости 1,45**.

Порядок внутри пары МЕНЯЕТСЯ (v3, v2, затем v2, v3 и так далее): иначе
первый прогон пары всегда платил бы за прогрев, и отношение поехало бы
в одну сторону — ровно та ошибка, которую поймал встречный замер в
`eye_view.py`.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\eye_cost_pairs.py --pairs 4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB3 = HERE.parent
LAB2 = (LAB3.parent / ".claude" / "worktrees" / "v2-base" / "lab")
PY = LAB3.parent / ".venv" / "Scripts" / "python.exe"


def once(lab, steps, seed):
    out = subprocess.run(
        [str(PY), str(HERE / "eye_cost.py"), "--lab", str(lab),
         "--steps", str(steps), "--seed", str(seed)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    for line in (out.stdout or "").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print(out.stdout[-2000:])
    print(out.stderr[-2000:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    if not (LAB2 / "flybrain" / "body.py").exists():
        print("дерева v2 нет: " + str(LAB2))
        return 1
    rows, ratios = [], []
    for i in range(args.pairs):
        order = [("v3", LAB3), ("v2", LAB2)]
        if i % 2:
            order.reverse()
        got = {}
        for tag, lab in order:
            r = once(lab, args.steps, seed=i)
            if r is None:
                print("пара %d: прогон %s сорвался" % (i + 1, tag))
                break
            got[tag] = r
            print("  %s %s" % (tag, json.dumps(
                {k: r[k] for k in ("мс на шаг управления",
                                   "доля реального на шагах",
                                   "сборка тела, с",
                                   "крючок есть", "кадров в кольцо")},
                ensure_ascii=False)), flush=True)
        if len(got) == 2:
            ratio = (got["v3"]["мс на шаг управления"]
                     / got["v2"]["мс на шаг управления"])
            ratios.append(ratio)
            rows.append({"пара": i + 1, "порядок": [t for t, _ in order],
                         "v3, мс": got["v3"]["мс на шаг управления"],
                         "v2, мс": got["v2"]["мс на шаг управления"],
                         "отношение v3/v2": round(ratio, 4)})
            print("  пара %d: отношение %.4f" % (i + 1, ratio), flush=True)
    rep = {"пары": rows}
    if ratios:
        ratios.sort()
        mid = ratios[len(ratios) // 2] if len(ratios) % 2 else (
            0.5 * (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]))
        rep["медиана отношения"] = round(mid, 4)
        rep["размах"] = [round(ratios[0], 4), round(ratios[-1], 4)]
        rep["порог различимости"] = 1.45
        rep["вывод"] = ("ниже порога различимости: транспорт кадра глаз "
                        "на скорости счёта не сказался"
                        if mid < 1.45 else
                        "выше порога: транспорт заметно дороже")
        print("\nмедиана отношения v3/v2: %.4f, размах %.4f…%.4f"
              % (mid, ratios[0], ratios[-1]))
        print(rep["вывод"])
    (HERE / "eye_cost_pairs.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
