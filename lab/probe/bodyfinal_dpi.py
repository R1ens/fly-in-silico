"""Пункт 1: масштабирование по DPI в оболочке без окна.

Системного масштабирования Windows в этой среде нет: процесс объявлен
DPI-неосведомлённым (`flybrain/desk/shell.py`, `dpi_report`), окно
растягивает сама Windows, и своего коэффициента у оболочки не возникает.
Поэтому здесь меряется то, что измеримо: `Offscreen` при dpi 1,0 / 1,25 /
1,5 / 2,0 — кегли атласов, высоты зон раскладки и толщины линий, и
отношения к dpi 1,0.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\bodyfinal_dpi.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
DPIS = (1.0, 1.25, 1.5, 2.0)
SIZE = (1500, 880)


def main():
    from flybrain.desk import layout as L
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen, dpi_report
    from flybrain.desk.tokens import S
    from probe.app_shot import FakeSession

    rep = {"dpi_report": dpi_report(None), "строки": []}
    print("осведомлённость процесса о DPI: "
          + json.dumps(rep["dpi_report"], ensure_ascii=False))

    sh = Offscreen(SIZE)
    ses = FakeSession()
    ses.alive = True
    ses.ready = True
    app = App(sh, ses)
    app.st.worker.readouts = ("DNa02", "ALPN")
    app.st.readout = "DNa02"

    rows = {}
    for dpi in DPIS:
        sh.resize(SIZE[0], SIZE[1], dpi)
        app.frame()
        app.frame()
        sc = sh.sc
        dock = app.st.ws.dock()
        lay = L.compute(sh.w, sh.h, sc, dock)
        m = {}
        # кегли: заказанный размер и настоящая высота строки атласа
        for role in ("workspace", "heading", "body", "small", "micro",
                     "metric"):
            m["кегль " + role] = sc.size(role)
            m["строка " + role] = int(getattr(sh.fonts, role).gh)
        # высоты и ширины зон
        m["зона topbar h"] = lay.topbar.h
        m["зона footer h"] = lay.footer.h
        m["зона timeline h"] = lay.timeline.h if lay.has("timeline") else 0
        m["зона inspector w"] = lay.inspector.w if lay.has("inspector") else 0
        m["зона secondary w"] = lay.secondary.w if lay.has("secondary") else 0
        m["центр w"] = lay.centre.w
        m["центр h"] = lay.centre.h
        # толщины и отступы
        m["линия hair"] = sc.hair()
        m["отступ xs"] = sc.px(S.xs)
        m["отступ md"] = sc.px(S.md)
        m["кнопка control"] = sc.px(S.control)
        m["кнопка control_lg"] = sc.px(S.control_lg)
        m["разделитель"] = sc.px(S.splitter)
        m["коэффициент k"] = round(sc.k, 4)
        rows[dpi] = m
        # прямоугольники деталей: ещё одна независимая проверка роста
        for wid in ("tl/run", "body/tb/reset", "b/out", "b/dist",
                    "zone/inspector"):
            r = sh.ctx.rect_of(wid)
            if r is not None:
                m["деталь %s w" % wid] = r.w
                m["деталь %s h" % wid] = r.h

    base = rows[1.0]
    keys = list(base)
    print("\n%-28s %8s %8s %8s %8s | %6s %6s %6s"
          % ("величина", "1.00", "1.25", "1.50", "2.00",
             "×1.25", "×1.50", "×2.00"))
    bad = []
    for k in keys:
        vals = [rows[d].get(k) for d in DPIS]
        b = base.get(k) or 0
        rat = []
        for d in DPIS[1:]:
            v = rows[d].get(k)
            rat.append(None if not b or v is None else round(v / b, 3))
        line = "%-28s %8s %8s %8s %8s | %6s %6s %6s" % (
            k, vals[0], vals[1], vals[2], vals[3],
            *(("—" if r is None else "%.3f" % r) for r in rat))
        print(line)
        rep["строки"].append({"величина": k,
                              "значения": dict(zip(map(str, DPIS), vals)),
                              "отношения": dict(zip(("1.25", "1.5", "2.0"),
                                                    rat))})
        # Отклонение считаем от ожидаемого отношения, допуск — округление
        # до целого пикселя: при базе b пиксель это 1/b.
        for d, r in zip(DPIS[1:], rat):
            if r is None or not b:
                continue
            tol = max(0.02, 1.2 / b)
            if abs(r - d) > tol:
                bad.append((k, d, r, round(tol, 3)))
    rep["отклонения"] = [{"величина": k, "dpi": d, "отношение": r,
                          "допуск": t} for k, d, r, t in bad]
    print("\nвеличин проверено: %d, отклонений вне допуска округления: %d"
          % (len(keys), len(bad)))
    for k, d, r, t in bad:
        print("  !! %s при dpi %.2f: отношение %.3f, допуск ±%.3f"
              % (k, d, r, t))
    (OUT / "bodyfinal_dpi.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/bodyfinal_dpi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
