"""Командная строка цела: те же команды, что в README, настоящими прогонами.

Окно и командная строка делят один путь исполнения, и правки ради окна
ломаются именно здесь. Каждая команда запускается отдельным процессом —
так же, как её запускает человек, — и проверяется код возврата и то, что
она действительно написала на диск.

Команды, которым нужна Ollama (`explain`, `agent`), не запускаются: без
запущенной модели они упали бы не по своей вине. Это сказано в отчёте.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\cli_check.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe import _isolate  # noqa: E402  прогоны — во временный каталог

LAB = Path(__file__).resolve().parents[1]
PY = LAB.parent / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    # Из рабочего дерева git окружения рядом нет: оно одно на весь
    # репозиторий. Берётся тот интерпретатор, которым проба и запущена —
    # это он же.
    PY = Path(sys.executable)
OUT = Path(__file__).parent
fails = []
rep = {"команды": {}, "проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def run(args, limit=900):
    # Подкоманды тоже пишут во временный каталог: иначе проверка CLI
    # насыпала бы папок поверх научных результатов.
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               FLYBRAIN_OUT=str(_isolate.OUT))
    t0 = time.perf_counter()
    p = subprocess.run([str(PY), "-m", "flybrain.cli", *args], cwd=str(LAB),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=limit)
    took = time.perf_counter() - t0
    rep["команды"][" ".join(args)] = {
        "код": p.returncode, "с": round(took, 1),
        "хвост": (p.stdout or "").strip().splitlines()[-3:],
        "ошибки": (p.stderr or "").strip().splitlines()[-3:],
    }
    return p, took


def main():
    from flybrain import data as D

    print("python:", PY)
    cases = [
        (["select", "region:AL"], "выбор нейронов по запросу"),
        (["regions"], "список областей"),
        (["doctor"], "диагностика"),
        (["run", "-a", "class:gustatory+side:left", "--t", "0.3",
          "--no-open", "--out", "cli-check-stim"], "стимуляция"),
        (["arena", "--t", "2", "--no-open", "--out", "cli-check-arena"],
         "эпизод арены"),
        (["body", "--t", "0.5", "--no-open", "--out", "cli-check-body"],
         "эпизод в теле"),
        (["report", "cli-check-body", "--no-open"], "отчёт по телу"),
        (["report", "cli-check-arena", "--no-open"], "отчёт по арене"),
        (["verify", "--t", "0.1"], "сверка с апстримом"),
        (["index"], "указатель"),
    ]
    for args, what in cases:
        print("  запускаю: fly " + " ".join(args))
        try:
            p, took = run(args)
        except subprocess.TimeoutExpired:
            check("fly %s — %s" % (" ".join(args), what), False,
                  "не уложилась в отведённое время")
            continue
        tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
        check("fly %s — %s" % (" ".join(args), what), p.returncode == 0,
              "код %d за %.0f с | %s" % (p.returncode, took, tail[0][:70]))

    # что команды реально положили на диск
    for name, want in (("cli-check-stim",
                        ["bins.npz", "meta.json", "rates.parquet",
                         "report.html"]),
                       ("cli-check-arena",
                        ["episode.json", "track.parquet", "report.html"]),
                       ("cli-check-body",
                        ["embodied.json", "track.parquet", "report.html"])):
        d = D.OUT / name
        got = sorted(x.name for x in d.iterdir()) if d.exists() else []
        miss = [w for w in want if w not in got]
        check("%s: файлы на месте" % name, not miss,
              ("нет: " + ", ".join(miss)) if miss else ", ".join(got))

    # отчёт по телу не должен уходить в ветку стимуляции
    log = rep["команды"].get("report cli-check-body --no-open", {})
    check("отчёт по телу собран своим загрузчиком",
          any("эпизод в теле" in s for s in log.get("хвост", [])
              + log.get("ошибки", [])),
          "; ".join(log.get("хвост", []))[:90])

    # сверка: та же корреляция, что из окна
    v = rep["команды"].get("verify --t 0.1", {})
    line = "; ".join(v.get("хвост", []))
    check("сверка напечатала корреляцию", "корреляц" in line, line[:90])

    check("указатель перестроился",
          (D.OUT / "index.html").exists(),
          "%d байт" % (D.OUT / "index.html").stat().st_size
          if (D.OUT / "index.html").exists() else "нет файла")

    rep["не запускалось"] = [
        "fly explain и fly agent — нужна запущенная Ollama; без неё они "
        "упали бы не по своей вине",
        "fly ui и fly app --old — прежний интерфейс, он отживает",
        "fly fetch-annotations — разовая установочная команда, лезет в сеть",
    ]
    (OUT / "cli_check.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
