"""Свести журналы шлюзов в сводку: код возврата и строки с числами."""

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")

KEEP = re.compile(r"провалов|проверок|совпало|побитово|расхождени|уровень |"
                  r"ПРОВАЛ|сверено|из \d+|отпечатк|в \d+[.,]\d+ раза|"
                  r"пропущено|папок|файлов|шагов из")


def main():
    codes = {}
    cf = HERE / "gates_codes.txt"
    if cf.exists():
        for line in cf.read_text(encoding="utf-8").splitlines():
            p = line.split("|")
            if len(p) == 3:
                codes[p[1]] = p[2]
    print("## коды возврата")
    for k, v in codes.items():
        print("  %-24s %s" % (k, v))
    for log in ("gates.log", "gates2.log", "gates3.log"):
        p = HERE / log
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        print("\n## %s" % log)
        cur = None
        for line in txt.splitlines():
            if line.startswith("--- "):
                cur = line.split()[1]
                print("\n=== %s (код %s)" % (cur, codes.get(cur, "?")))
                continue
            if KEEP.search(line):
                print("   " + line.strip()[:240])


if __name__ == "__main__":
    main()
