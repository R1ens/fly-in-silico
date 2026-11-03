r"""Куда уходит шаг физики: разбор СОБСТВЕННЫМИ таймерами MuJoCo.

Прежние разборы делили шаг снаружи — `mj_step1` против `mj_step2`, — и
дальше гадать было нечем. MuJoCo ведёт свои счётчики по ступеням
(`mjData.timer`), и это не оценка профилировщика, а время, замеренное
самим движком внутри своего же конвейера.

Отдельно берётся расход решателя: `solver_niter` и число островов. В
пробе `mj_solver.py` это поле читалось как одно число и дало 108
итераций при любом допуске и любом пределе — то есть читалось неверно.
Здесь оно читается как массив по островам, и вместе с
`solver_nisland` его уже можно понять.

Разбор нужен затем, чтобы знать, какую из двух оставшихся ручек
прошлой кампании трогать: бюджет решателя или модель контактов. Допуск
проверен и отвергнут (`mj_solver.py`: ослабление с 1e-12 до 1e-4 не
даёт ничего, а срез итераций до десяти уводит позу на 1,6 мм). Значит
вопрос к контактам: в модели **2316 заранее заданных пар** и
**69 сеток**.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\mj_inside.py
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = Path(__file__).resolve().parents[2] / ".venv-mj-model"
XML = MODEL / "fly.xml"
STATE = MODEL / "state.npz"
SUB = 100
ROUNDS = 7

ИМЕНА = [n for n in dir(mujoco.mjtTimer)
         if n.startswith("mjTIMER_")]

rep = {"замеры": {}, "проверки": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    if not (XML.exists() and STATE.exists()):
        print("нет модели или рабочей точки; сперва probe\\model_export.py")
        return 1
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    z = np.load(STATE)
    buf, ctrl, warm = z["состояние"], z["ctrl"], z["разогрев"]
    print("mujoco %s" % mujoco.__version__, flush=True)
    # Без обратного вызова времени счётчики движка остаются нулями, и
    # первая редакция этой пробы честно напечатала таблицу из нулей.
    # `mjcb_time` обязан отдавать МИЛЛИСЕКУНДЫ.
    mujoco.set_mjcb_time(lambda: time.perf_counter() * 1000.0)
    rep["замеры"]["модель"] = {
        "тел": int(m.nbody), "степеней свободы": int(m.nv),
        "геометрий": int(m.ngeom), "сеток": int(m.nmesh),
        "заданных пар": int(m.npair), "исключений": int(m.nexclude),
    }

    def встать():
        mujoco.mj_setState(m, d, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
        d.ctrl[:] = ctrl
        d.qacc_warmstart[:] = warm

    встать()
    mujoco.mj_forward(m, d)
    rep["замеры"]["контактов"] = int(d.ncon)
    print("  контактов %d, сеток %d, заданных пар %d"
          % (d.ncon, m.nmesh, m.npair), flush=True)

    # --- расход решателя: поле оказалось ненадёжным -----------------
    #
    # `solver_niter` отдаёт 101 при пределе 1000, 20 и даже 5 — то есть
    # не отражает ни предела, ни расхода. Поле читается и пишется в
    # отчёт как есть, но ВЫВОДОВ по нему не делается: расход итераций
    # определён косвенно, перебором предела в `mj_solver.py` (пределы
    # 1000, 50 и 20 не меняют ни времени, ни траектории, а 10 ломает её
    # на 1,6 мм — значит решатель тратит от десяти до двадцати).
    # --- расход решателя, прочитанный как есть ----------------------
    встать()
    ниты, острова = [], []
    for _ in range(SUB):
        mujoco.mj_step(m, d)
        n = np.asarray(d.solver_niter)
        ni = int(getattr(d, "solver_nisland", 1) or 1)
        острова.append(ni)
        # Итерации лежат ПО ОСТРОВАМ, и значимы только первые `nisland`
        # записей: остальные — мусор прошлых шагов. Суммировать весь
        # массив значило бы складывать живое с протухшим, и ровно это
        # дало прежние «108 итераций при любом пределе».
        ниты.append(int(np.asarray(n)[:max(ni, 1)].sum()))
    a = np.asarray(ниты)
    rep["замеры"]["итераций решателя"] = {
        "медиана": float(np.median(a)), "наименьшая": int(a.min()),
        "наибольшая": int(a.max()), "предел модели": int(m.opt.iterations),
        "островов, медиана": float(np.median(острова)),
    }
    print("  итераций решателя: медиана %.0f, размах %d…%d; островов %.0f"
          % (np.median(a), a.min(), a.max(), np.median(острова)), flush=True)
    rep["находки"].append(
        "solver_niter отдаёт %d при пределе %d, и то же самое при "
        "пределе 5 — поле ненадёжно, выводы по нему не делаются"
        % (int(np.median(a)), int(m.opt.iterations)))

    # --- таймеры самого движка --------------------------------------
    кругов = []
    for _ in range(ROUNDS + 2):
        встать()
        for i in range(len(d.timer)):
            d.timer[i].duration = 0
            d.timer[i].number = 0
        t = time.perf_counter()
        mujoco.mj_step(m, d, nstep=SUB)
        стена = (time.perf_counter() - t) * 1000.0
        круг = {"стена, мс": стена}
        for имя in ИМЕНА:
            if имя in ("mjNTIMER",):
                continue
            k = int(getattr(mujoco.mjtTimer, имя))
            if k >= len(d.timer):
                continue
            # Длительность уже в МИЛЛИСЕКУНДАХ: столько отдаёт наш
            # обратный вызов. Деления на тысячу здесь не нужно, и
            # первая редакция на нём же напечатала доли процента.
            круг[имя[8:].lower()] = d.timer[k].duration
        кругов.append(круг)
    кругов = кругов[2:]

    ключи = [k for k in кругов[0] if k != "стена, мс"]
    стена = float(np.median([c["стена, мс"] for c in кругов]))
    строки = []
    for k in ключи:
        v = float(np.median([c[k] for c in кругов]))
        строки.append({"ступень": k, "мс на 100 шагов": round(v, 3),
                       "доля шага, %": round(100.0 * v / стена, 1)})
    строки.sort(key=lambda s: -s["мс на 100 шагов"])
    rep["замеры"]["стена, мс на 100 шагов"] = round(стена, 3)
    rep["замеры"]["ступени"] = строки
    print("\n  %-18s %10s %8s" % ("ступень", "мс/100", "доля"), flush=True)
    for s in строки:
        print("  %-18s %9.3f %7.1f %%"
              % (s["ступень"], s["мс на 100 шагов"], s["доля шага, %"]),
              flush=True)

    мед = {s["ступень"]: s["мс на 100 шагов"] for s in строки}
    check("таймеры движка сходятся со стеной",
          abs(мед.get("step", 0.0) - стена) < 0.25 * стена,
          "step %.1f против стены %.1f" % (мед.get("step", 0.0), стена))
    столк = мед.get("col_narrow", 0.0) + мед.get("col_broad", 0.0)
    rep["замеры"]["столкновения, мс"] = round(столк, 3)
    rep["замеры"]["решатель (constraint), мс"] = мед.get("constraint", 0.0)
    rep["находки"].append(
        "столкновения %.1f мс (%.0f %%), сборка ограничений %.1f мс, "
        "решатель %.1f мс на 100 шагов"
        % (столк, 100.0 * столк / max(стена, 1e-9), мед.get("pos_make", 0.0),
           мед.get("constraint", 0.0)))
    print("  " + rep["находки"][-1], flush=True)

    (HERE / "mj_inside.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
