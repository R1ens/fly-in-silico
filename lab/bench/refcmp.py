"""Сверка выхода уровня 4 со ВСЕЙ цепочкой снимков эталона.

Отдельный файл, а не строчка в отчёте, потому что это ловушка, на
которой легко ошибиться дважды. Снимок, который называется «эталон»,
снят ДО сознательной правки чтения антенн (`SPEED.md` §5: цепочка
`эталон → быстрый → графы → антенны → выборка → кольцо`). Сверка с ним
одним даёт четыре расхождения на исправном коде — и дальше либо автора
правки обвиняют зря, либо настоящее расхождение списывают на «это же
антенны».

Поэтому сверяется вся цепочка, и печатается, с какого места отпечатки
сходятся.

    ..\\.venv\\Scripts\\python.exe lab\\bench\\refcmp.py bench7_базовая.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROBE = HERE.parent / "probe"
sys.stdout.reconfigure(encoding="utf-8")

FIELDS = (
    ("отпечаток counts", ("активность", "отпечаток counts")),
    ("отпечаток bins", ("активность", "отпечаток bins")),
    ("отпечаток траектории", ("тело", "отпечаток траектории")),
    ("отпечаток моторных команд", ("моторные команды", "отпечаток dn")),
)


def main(path, level="4 closed"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    v = d["уровни"][level]
    got = v["выход прогона"]
    steps = got.get("шагов")
    print("уровень %s стенда: шагов %s, спайков %s"
          % (level, steps, got.get("суммарные спайки")))
    for name, _ in FIELDS:
        print("  %-26s %s" % (name, got.get(name)))
    if steps != 50:
        print("\nсверять нечего: у сценария body эталона 50 шагов, "
              "а здесь %s" % steps)
        return 1
    rows = []
    for p in sorted(PROBE.glob("bench_v2_*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        sc = (s.get("сценарии") or {}).get("body")
        if sc is None:
            continue
        same, pairs = 0, []
        for name, (blk, key) in FIELDS:
            ref = (sc.get(blk) or {}).get(key)
            ok = bool(ref and ref == got.get(name))
            same += ok
            pairs.append((name, ref, ok))
        rows.append((s.get("когда") or "", p.name, s.get("метка"), same,
                     pairs))
    rows.sort()
    print("\n| снимок | метка | когда | совпало полей из 4 |")
    print("|---|---|---|---|")
    for when, name, tag, same, _ in rows:
        print("| %s | %s | %s | **%d** |" % (name, tag, when, same))
    first_full = next((r for r in rows if r[3] == len(FIELDS)), None)
    print()
    if first_full:
        print("отпечатки сходятся побитово начиная со снимка %r (%s) и "
              "совпадают во всех %d более поздних"
              % (first_full[2], first_full[0],
                 sum(1 for r in rows if r[0] >= first_full[0]
                     and r[3] == len(FIELDS))))
    else:
        print("ни с одним снимком не совпало: это расхождение, а не "
              "объявленная правка")
    return 0 if first_full else 1


if __name__ == "__main__":
    args = sys.argv[1:] or ["bench7_базовая.json"]
    p = Path(args[0])
    if not p.is_absolute():
        p = HERE / p
    raise SystemExit(main(p, *(args[1:] or ["4 closed"])))
