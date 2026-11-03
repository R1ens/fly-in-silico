r"""Чего стоит допуск решателя: 108 итераций на шаг при 18 контактах.

Прежняя кампания оставила две ручки непроверенными — бюджет решателя и
модель контактов (`PERF_RT.md` §1). Эта проба берёт первую и делает с
ней то, что полагается делать с ручкой разряда B: **меряет, сколько она
даёт и чего стоит**, не принимая никаких решений.

Отчего вопрос вообще возник. Модель NeuroMechFly настроена на
`iterations = 1000` при `tolerance = 1e-12`. Замер показал, что предел
не достигается (расход 101…113), то есть ограничивает не он, а допуск:
решатель Ньютона с восемнадцатью контактами обычно сходится за единицы
итераций, а здесь он доводит невязку до 1e-12 и тратит на это сотню.

Что меряется на каждой паре «допуск, предел итераций»:

* **время** ста шагов физики из одной и той же рабочей точки;
* **расход итераций** — он и объясняет время;
* **расхождение траектории** от эталона после ста шагов, по `qpos` и
  `qvel`. Это не приёмка поведения, а первая прикидка: если ослабление
  допуска уводит позу на микрон, разговор один, если на миллиметр —
  другой.

Решения здесь не принимаются. Ручка разряда B принимается только через
поведенческую приёмку с порогами, объявленными до опыта, — опора на
ногах, контакты, походка, наведение.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\mj_solver.py
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
    эт_доп, эт_ит = float(m.opt.tolerance), int(m.opt.iterations)
    print("mujoco %s; эталон: допуск %.0e, предел итераций %d"
          % (mujoco.__version__, эт_доп, эт_ит), flush=True)

    def встать():
        mujoco.mj_setState(m, d, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
        d.ctrl[:] = ctrl
        d.qacc_warmstart[:] = warm

    встать()
    mujoco.mj_forward(m, d)
    rep["замеры"]["контактов"] = int(d.ncon)

    def прогон():
        """Сто шагов из рабочей точки: время, итерации, куда пришли."""
        встать()
        ниты = []
        t = time.perf_counter()
        for _ in range(SUB):
            mujoco.mj_step(m, d)
            n = np.asarray(d.solver_niter)
            ниты.append(int(n.sum()) if n.ndim else int(n))
        ms = (time.perf_counter() - t) * 1000.0
        return ms, float(np.median(ниты)), \
            np.array(d.qpos, copy=True), np.array(d.qvel, copy=True)

    # --- эталон -----------------------------------------------------
    for _ in range(2):
        прогон()
    эт = sorted(прогон()[0] for _ in range(ROUNDS))
    _, эт_нит, q_эт, v_эт = прогон()
    эт_мс = эт[len(эт) // 2]
    rep["замеры"]["эталон"] = {
        "допуск": эт_доп, "предел итераций": эт_ит,
        "мс на 100 шагов": round(эт_мс, 3),
        "итераций, медиана": эт_нит,
    }
    print("  эталон: %.2f мс, итераций %.0f, контактов %d"
          % (эт_мс, эт_нит, d.ncon), flush=True)

    # --- перебор ----------------------------------------------------
    строки = []
    случаи = [(t, i) for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
              for i in (1000, 50, 20, 10)]
    for доп, ит in случаи:
        m.opt.tolerance = доп
        m.opt.iterations = ит
        for _ in range(2):
            прогон()
        ts = sorted(прогон()[0] for _ in range(ROUNDS))
        мс = ts[len(ts) // 2]
        _, нит, q, v = прогон()
        dq = float(np.abs(q - q_эт).max())
        dv = float(np.abs(v - v_эт).max())
        строки.append({
            "допуск": доп, "предел итераций": ит,
            "мс на 100 шагов": round(мс, 3),
            "ускорение": round(эт_мс / мс, 3),
            "итераций, медиана": нит,
            "расхождение qpos": dq, "расхождение qvel": dv,
        })
        print("  допуск %.0e, предел %4d: %7.2f мс  ×%.2f  итераций %4.0f"
              "  |dqpos| %.3e" % (доп, ит, мс, эт_мс / мс, нит, dq),
              flush=True)
    m.opt.tolerance, m.opt.iterations = эт_доп, эт_ит
    rep["замеры"]["перебор"] = строки

    # --- что из этого следует ---------------------------------------
    точное = [s for s in строки
              if s["допуск"] == эт_доп and s["предел итераций"] == эт_ит]
    check("эталонная пара воспроизводит сама себя побитово",
          точное and точное[0]["расхождение qpos"] == 0.0,
          "|dqpos| %.1e" % точное[0]["расхождение qpos"] if точное else "—")
    лучший = max(строки, key=lambda s: s["ускорение"])
    rep["замеры"]["наибольшее ускорение"] = лучший
    check("ослабление допуска вообще что-то даёт",
          лучший["ускорение"] > 1.1,
          "×%.2f при допуске %.0e" % (лучший["ускорение"], лучший["допуск"]))
    # Главное разграничение: есть ли ускорение БЕЗ расхождения.
    даром = [s for s in строки
             if s["расхождение qpos"] == 0.0 and s["ускорение"] > 1.05]
    rep["замеры"]["ускорение без расхождения"] = даром
    if даром:
        rep["находки"].append(
            "есть пары, где быстрее и траектория побитово та же: "
            + json.dumps(даром, ensure_ascii=False))
    else:
        rep["находки"].append(
            "побитово даром ускорения нет: любое ослабление допуска "
            "уводит траекторию, то есть это разряд B и только через "
            "поведенческую приёмку")
    print("  " + rep["находки"][-1][:200], flush=True)

    (HERE / "mj_solver.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
