"""Свести отчёты стенда в таблицы для BENCH.md. Ничего не считает заново."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")

NAMES = ["мозг: LIF, синхронизация, учёт", "чистая физика MuJoCo",
         "походка и прослойка dm_control", "хвост шага: поза, метрики, запись",
         "кадр живого вида", "обмен через IPC", "прочее в обработчике шага"]


def g(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def levels(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print("# %s  метка %s" % (Path(path).name, d.get("метка")))
    bg = d.get("фон до замера (стенд простаивал 6 с)", {})
    print("ФОН ДО ЗАМЕРА: видеокарта %s %% (мин %s, макс %s), машина %s %% "
          "ядер, память видеокарты занято машиной %s МиБ"
          % (g(bg, "видеокарта", "загрузка видеокарты, % медиана"),
             g(bg, "видеокарта", "загрузка видеокарты, % мин"),
             g(bg, "видеокарта", "загрузка видеокарты, % макс"),
             g(bg, "процессор", "вся машина, %"),
             g(bg, "видеокарта", "память видеокарты занято машиной, МиБ медиана")))
    print("загрузка данных: %s" % json.dumps(d.get("загрузка данных, с"),
                                             ensure_ascii=False))

    print("\n## шаг управления по уровням")
    print("| уровень | медиана, мс | среднее | p95 | макс | хвост p95/мед | "
          "размах макс/мин | модельных с на настенную | медленнее РВ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in d["уровни"].items():
        s = v.get("шаг целиком", {})
        sp = v.get("скорость", {})
        rt = (sp.get("модельных секунд на настенную (по сумме окон шагов)")
              or sp.get("модельных секунд на настенную (по всему окну)"))
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (k, s.get("медиана"), s.get("среднее"), s.get("p95"),
                 s.get("макс"), s.get("хвост p95/медиана"),
                 s.get("размах макс/мин"), rt,
                 sp.get("во сколько раз медленнее реального времени")))

    print("\n## разложение (медианы, мс / доля шага)")
    cols = []
    for k, v in d["уровни"].items():
        cols.append((k, v.get("разложение (разогретые шаги)", {})))
    keys = []
    for _, r in cols:
        for n in NAMES:
            if n in r and n not in keys:
                keys.append(n)
    print("| составляющая | " + " | ".join(k for k, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for n in keys:
        row = []
        for _, r in cols:
            s = r.get(n)
            row.append("—" if not s else "%s / %s"
                       % (s.get("медиана"), s.get("доля от медианы шага")))
        print("| %s | %s |" % (n, " | ".join(row)))
    row = []
    for _, r in cols:
        nv = r.get("невязка", {})
        row.append("%s (%s %%)" % (nv.get("медиана, мс"),
                                   nv.get("медиана со знаком, % шага")))
    print("| **невязка со знаком** | %s |" % " | ".join(row))
    row = []
    for _, r in cols:
        row.append(str(r.get("шагов в разложении")))
    print("| шагов в разложении | %s |" % " | ".join(row))

    print("\n## фон, загрузка, память по уровням")
    print("| уровень | видеокарта в окне, % мед (мин…макс) | машина, % ядер | "
          "свой процесс, ядер | своя память GPU пик, МБ | GPU занято машиной, "
          "МиБ | процесс пик, МБ |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d["уровни"].items():
        gg, ld, mm = (v.get("видеокарта", {}), v.get("загрузка", {}),
                      v.get("память процесса", {}))
        print("| %s | %s (%s…%s) | %s | %s | %s | %s | %s |"
              % (k, gg.get("загрузка видеокарты, % медиана"),
                 gg.get("загрузка видеокарты, % мин"),
                 gg.get("загрузка видеокарты, % макс"),
                 ld.get("вся машина, %"), ld.get("свой процесс, ядер"),
                 gg.get("своя, выделено пик, МБ"),
                 gg.get("память видеокарты занято машиной, МиБ медиана"),
                 mm.get("пик, МБ")))

    print("\n## контакты (уровни 1-3)")
    for k, v in d["уровни"].items():
        if "контакты" in v:
            c = v["контакты"]
            print("  %-12s медиана %s, шаг 0 %s, последний %s"
                  % (k, c.get("в конце шага", {}).get("медиана"),
                     c.get("на шаге 0"), c.get("на последнем шаге")))

    print("\n## подготовка (в шаг не входит), с")
    for k, v in d["уровни"].items():
        print("  %-12s %s" % (k, json.dumps(v.get("подготовка, с"),
                                            ensure_ascii=False)))

    print("\n## первый шаг против разогретого")
    for k, v in d["уровни"].items():
        cw = v.get("первый шаг против разогретого", {})
        print("  === %s" % k)
        print("    шаг 0:   %s" % json.dumps(cw.get("шаг 0"),
                                             ensure_ascii=False))
        for kk, vv in cw.items():
            if kk != "шаг 0":
                print("    %s: %s" % (kk, json.dumps(vv, ensure_ascii=False)))

    print("\n## счётчики")
    for k, v in d["уровни"].items():
        c = v.get("счётчики", {})
        pr = {kk: vv.get("встретилось") for kk, vv in
              (c.get("проверки") or {}).items()}
        print("  %-12s всё сошлось %s | %s | генератор %s | арифметика %s"
              % (k, c.get("всё сошлось"), json.dumps(pr, ensure_ascii=False),
                 json.dumps(c.get("генератор"), ensure_ascii=False),
                 json.dumps(c.get("арифметика вызовов"), ensure_ascii=False)))

    print("\n## выход прогона и сверка с эталоном")
    for k, v in d["уровни"].items():
        if "выход прогона" in v:
            o = v["выход прогона"]
            print("  === %s шагов %s, спайков %s, поза (%s, %s, %s)"
                  % (k, o.get("шагов"), o.get("суммарные спайки"),
                     o.get("конец x, мм"), o.get("конец y, мм"),
                     o.get("конец курса, рад")))
            print("      отпечатки: counts %s bins %s траектория %s "
                  "моторные %s"
                  % (o.get("отпечаток counts"), o.get("отпечаток bins"),
                     o.get("отпечаток траектории"),
                     o.get("отпечаток моторных команд")))
            print("      сверка: %s"
                  % json.dumps(v.get("сверка с сохранённым эталоном"),
                               ensure_ascii=False))
        for extra in ("зрение", "правки сводка", "запись ленты"):
            if extra in v:
                print("      %s: %s" % (extra, json.dumps(
                    v[extra], ensure_ascii=False)))
    cnt = d.get("встречный замер на счётчики")
    if cnt:
        print("\n## встречный замер на счётчики")
        print(json.dumps(cnt, ensure_ascii=False, indent=1))


def pairsfile(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print("# %s  предмет %s" % (Path(path).name, d.get("предмет")))
    print("фон до: %s" % json.dumps(
        d.get("фон до замера (стенд простаивал 6 с)"), ensure_ascii=False))
    print("фон в окне: %s" % json.dumps(d.get("фон в окне пар"),
                                        ensure_ascii=False))
    print(json.dumps(d.get("пары"), ensure_ascii=False, indent=1))


def longfile(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    v = list(d["уровни"].values())[0]
    print("# %s" % Path(path).name)
    print("фон до: %s" % json.dumps(
        d.get("фон до замера (стенд простаивал 6 с)"), ensure_ascii=False))
    for k in ("устройство замера", "шаг управления, мс", "дрейф по сеансу",
              "скорость", "память: рост", "кольцо записи", "счётчики",
              "загрузка", "видеокарта", "память процесса", "оговорки прогона"):
        if k in v:
            print("%s = %s" % (k, json.dumps(v[k], ensure_ascii=False)))
    print("правок %d: %s" % (len(v.get("правки") or []),
                             json.dumps(v.get("правки"), ensure_ascii=False)))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_absolute():
            p = HERE / arg
        name = p.name
        if "pairs" in name:
            pairsfile(p)
        elif "long" in name:
            longfile(p)
        else:
            levels(p)
        print()
