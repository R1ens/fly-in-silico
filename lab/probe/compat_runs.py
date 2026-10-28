"""Обратная совместимость: прежние каталоги в `out/` открываются как раньше.

Требование владельца буквальное: научные прогоны, посчитанные до этой
работы, обязаны открываться **так же**. Проверяется это не осмотром, а
сравнением двух обходов каталога:

* каталог читается **прежним** кодом — пакет `lab/flybrain` разворачи­
  вается из коммита-основания во временную папку (`git archive`, рабочее
  дерево не трогается);
* тот же каталог читается кодом этой ветки;
* два обхода сравниваются **побайтово**, поле в поле.

Трогать `out/` нельзя ни байтом, поэтому отдельно снимается sha256
каждого файла каждого прогона до обхода и после: обход — это чтение, и
оно обязано ничего не менять. Это и есть встречный замер: без него
«открывается как раньше» прошло бы и на коде, который при чтении
дописывает в папку кэш.

    cd D:\\fly-brain\\lab
    ..\\.venv\\Scripts\\python.exe probe\\compat_runs.py --base ccd321c
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
LAB = HERE.parents[1]
REPO = LAB.parent
PY = REPO.parent / ".venv" / "Scripts" / "python.exe"
if not PY.exists():                      # из рабочего дерева агента
    PY = Path(r"D:\fly-brain\.venv\Scripts\python.exe")

# Каталог научных прогонов основного дерева. Только чтение.
MAIN_OUT = Path(r"D:\fly-brain\out")

OUT = HERE.parent
fails = []
rep = {"проверки": {}, "замеры": {}, "расхождения": [], "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# ------------------------------------------------------------- обход


def walk(out_dir):
    """Что каталог показывает о прогонах. Только чтение, ничего не пишет."""
    from flybrain import plots as PL, runs as RN

    got = {}
    for r in RN.scan(out_dir):
        d = {
            "вид": r.kind,
            "подпись вида": r.kind_title,
            "имя": r.name,
            "файлов": sorted(r.files),
            "состояние": getattr(r, "state", None),
            "когда": r.when,
            "ошибки": {k: str(v) for k, v in (r.errors or {}).items()},
            "оговорки": list(r.caveats or ()),
        }
        for extra in ("steps", "duration_s", "device", "seed", "title",
                      "note", "size", "n_steps"):
            if hasattr(r, extra):
                d[extra] = getattr(r, extra)
        try:
            d["подробно"] = sorted((RN.detail(r) or {}).keys())
        except Exception as exc:                              # noqa: BLE001
            d["подробно"] = "ошибка: %s" % exc
        try:
            d["графики"] = sorted(p.key for p in PL.available(r))
        except Exception as exc:                              # noqa: BLE001
            d["графики"] = "ошибка: %s" % exc
        got[r.name] = d
    return got


def digest(root):
    """sha256 каждого файла каталога: до обхода и после."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def dump_mode(out_dir):
    print(json.dumps(walk(out_dir), ensure_ascii=False, sort_keys=True,
                     default=str))
    return 0


def run_dump(lab_dir, out_dir):
    """Запустить обход в отдельном процессе с указанным пакетом."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(lab_dir)
    # Прогоны проб — в свой каталог, чтобы никакой импорт не создал
    # ничего рядом с научными.
    env.setdefault("FLYBRAIN_OUT", str(Path(tempfile.gettempdir())
                                       / "flybrain-probe-out"))
    # В пути только тот пакет, который проверяется: иначе обход «прежним
    # кодом» мог бы подхватить новый `flybrain` и сравнить код сам с
    # собой. Сам файл пробы запускается по пути, импорта ему не нужно.
    code = ("import sys; sys.path.insert(0, %r);\n"
            "import runpy; sys.argv=['compat_runs.py','--dump',%r];\n"
            "runpy.run_path(%r, run_name='__main__')"
            % (str(lab_dir), str(out_dir), str(HERE)))
    p = subprocess.run([str(PY), "-c", code], capture_output=True, env=env,
                       cwd=str(lab_dir))
    if p.returncode != 0:
        raise RuntimeError("обход не удался (%s):\n%s"
                           % (lab_dir, p.stderr.decode("utf-8", "replace")
                              [-2000:]))
    text = p.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
    return json.loads(text)


def base_lab(rev):
    """Развернуть `lab/flybrain` из коммита-основания во временную папку."""
    tmp = Path(tempfile.mkdtemp(prefix="flybrain-base-"))
    tar = tmp / "base.tar"
    with open(tar, "wb") as f:
        p = subprocess.run(["git", "-C", str(REPO), "archive", rev,
                            "lab/flybrain"], stdout=f, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("git archive: "
                           + p.stderr.decode("utf-8", "replace"))
    subprocess.run(["tar", "-xf", str(tar), "-C", str(tmp)], check=True)
    return tmp / "lab"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=None, metavar="КАТАЛОГ")
    ap.add_argument("--base", default="ccd321c",
                    help="коммит-основание: код ДО этой работы")
    ap.add_argument("--out", default=str(MAIN_OUT))
    args = ap.parse_args()
    if args.dump:
        return dump_mode(Path(args.dump))

    out_dir = Path(args.out)
    dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    rep["замеры"]["каталог"] = str(out_dir)
    rep["замеры"]["папок прогонов"] = len(dirs)
    rep["замеры"]["записей в каталоге всего"] = len(list(out_dir.iterdir()))
    print("— каталог %s: папок %d —" % (out_dir, len(dirs)))

    before = digest(out_dir)
    rep["замеры"]["файлов в каталоге"] = len(before)

    print("  обход прежним кодом (%s)" % args.base)
    old_lab = base_lab(args.base)
    t0 = time.perf_counter()
    old = run_dump(old_lab, out_dir)
    rep["замеры"]["обход прежним кодом, с"] = round(time.perf_counter() - t0,
                                                    2)
    print("  обход кодом этой ветки")
    t0 = time.perf_counter()
    new = run_dump(LAB, out_dir)
    rep["замеры"]["обход новым кодом, с"] = round(time.perf_counter() - t0, 2)

    after = digest(out_dir)
    check("обход не изменил в каталоге ни байта",
          before == after,
          "файлов %d, sha256 всех совпали" % len(before)
          if before == after else
          "разошлось файлов %d"
          % len({k for k in set(before) | set(after)
                 if before.get(k) != after.get(k)}))
    check("прогоны найдены все до одного",
          len(old) == len(dirs) and len(new) == len(dirs),
          "папок %d, прежний код нашёл %d, новый %d"
          % (len(dirs), len(old), len(new)))

    diff = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name), new.get(name)
        if a is None or b is None:
            diff.append("%s: %s" % (name, "нет у нового" if b is None
                                    else "нет у прежнего"))
            continue
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diff.append("%s / %s: было %r, стало %r"
                            % (name, k, a.get(k), b.get(k)))
    rep["расхождения"] = diff
    check("каталог показывает прежние прогоны в точности так же",
          not diff, "прогонов %d, полей сверено %d, расхождений %d"
          % (len(new), sum(len(v) for v in new.values()), len(diff)))
    for d in diff[:20]:
        print("    - " + d)

    kinds = {}
    for v in new.values():
        kinds[v["вид"]] = kinds.get(v["вид"], 0) + 1
    rep["замеры"]["виды прогонов"] = kinds
    check("ни один прежний прогон не стал «неопознанным»",
          kinds.get("unknown", 0) == 0,
          "виды: %s" % ", ".join("%s %d" % kv for kv in sorted(kinds.items())))

    # Прежние прогоны таймлайна не несут, и это не поломка: вид
    # «Таймлайн» обязан сказать это словами, а не показать пустоту.
    with_tl = [n for n, v in new.items() if "timeline.npz" in v["файлов"]]
    rep["замеры"]["прежних прогонов с таймлайном"] = len(with_tl)
    check("прежние прогоны таймлайна не несут и не обязаны",
          not with_tl, "с таймлайном %d из %d" % (len(with_tl), len(new)))

    (OUT / "compat_runs.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
