r"""Выгрузить ТУ САМУЮ модель мухи в XML — чтобы мерить её другим MuJoCo.

Сравнивать версии MuJoCo можно только на одной и той же модели, иначе
сравниваются две разные мухи. Производственное окружение закреплено на
`mujoco == 3.2.7` требованием самого `flygym-gymnasium`, поэтому новая
версия может быть только в отдельном окружении, а модель туда надо
как-то передать.

MJB не годится: он привязан к версии. Остаётся XML, и его надо не
просто выгрузить, а **доказать, что выгруженное — то же самое**. Проба
это и делает: собирает тело ровно как песочница, выгружает корень mjcf
вместе с ресурсами, тут же читает выгруженное обратно **тем же** MuJoCo
и сверяет с исходной моделью по тому, что определяет счёт: число тел,
степеней свободы, приводов, геометрий, пар исключений, шаг
интегрирования, тип решателя, число итераций, допуск, гравитация,
массы.

Сверх того сверяется поведение: сто шагов физики из одного и того же
состояния на обеих моделях обязаны дать **побитово одинаковые** `qpos`
и `qvel`. Если XML потерял хоть что-нибудь существенное, это всплывёт
здесь, а не в выводах о скорости.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\model_export.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_model_export_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

ПОЛЯ = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart",
         "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat")

OUT = Path(__file__).parent
DST = Path(__file__).resolve().parents[2] / ".venv-mj-model"
SUB = 100

rep = {"проверки": {}, "замеры": {}}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def опись(m):
    """Всё, что определяет счёт. Сравнивается целиком, а не на глаз."""
    return {
        "тел": int(m.nbody), "степеней свободы": int(m.nv),
        "обобщённых координат": int(m.nq), "приводов": int(m.nu),
        "геометрий": int(m.ngeom), "суставов": int(m.njnt),
        "датчиков": int(m.nsensor), "сеток": int(m.nmesh),
        "исключённых пар": int(m.nexclude), "пар контактов": int(m.npair),
        "шаг": float(m.opt.timestep),
        "решатель": int(m.opt.solver), "итераций": int(m.opt.iterations),
        "допуск": float(m.opt.tolerance),
        "итераций ls": int(m.opt.ls_iterations),
        "целостность": int(m.opt.integrator),
        "гравитация": [float(v) for v in m.opt.gravity],
        "масса": round(float(m.body_mass.sum()), 12),
        "флаги": int(m.opt.disableflags), "включено": int(m.opt.enableflags),
    }


def main():                                                   # noqa: C901
    import mujoco
    from dm_control import mjcf

    from flybrain import body as B

    print("собираю тело", flush=True)
    cfg = B.BodyConfig(duration_s=0.4, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN",
                       vision=True, vision_hz=100.0)
    sources = [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]]
    sim, _ = B._build_body(cfg, sources)
    m0 = sim.physics.model.ptr
    d0 = sim.physics.data.ptr
    rep["замеры"]["исходная модель"] = опись(m0)
    print("  " + json.dumps(опись(m0), ensure_ascii=False), flush=True)

    корень = getattr(sim.arena, "root_element", None)
    check("корень mjcf найден", корень is not None, type(корень).__name__)
    if корень is None:
        return

    DST.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mjcf.export_with_assets(корень, str(DST), "fly.xml")
    xml = DST / "fly.xml"
    check("XML выгружен", xml.exists(),
          "%s, %.1f КБ, за %.1f с"
          % (xml, xml.stat().st_size / 1024, time.perf_counter() - t0))
    файлов = len(list(DST.iterdir()))
    rep["замеры"]["файлов в выгрузке"] = файлов
    rep["замеры"]["каталог"] = str(DST)

    m1 = mujoco.MjModel.from_xml_path(str(xml))
    d1 = mujoco.MjData(m1)
    rep["замеры"]["перечитанная модель"] = опись(m1)
    a, b = опись(m0), опись(m1)
    разошлось = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    rep["замеры"]["расхождения описи"] = разошлось
    check("опись модели совпала целиком", not разошлось,
          json.dumps(разошлось, ensure_ascii=False) if разошлось
          else "%d полей" % len(a))

    # --- поведение: сто шагов из одного состояния -------------------
    #
    # Состояние переносится тем же набором полей, каким его переносит
    # `replay.py`: иначе сравнивались бы две модели из разных точек.
    size = mujoco.mj_stateSize(m0, mujoco.mjtState.mjSTATE_INTEGRATION)
    buf = np.zeros(size, dtype=np.float64)
    for _ in range(SUB):
        sim.step(np.array([0.9, 0.9]))
    mujoco.mj_getState(m0, d0, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
    ctrl = np.array(d0.ctrl, copy=True)
    warm = np.array(d0.qacc_warmstart, copy=True)

    def прогнать(m, d):
        mujoco.mj_setState(m, d, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
        d.ctrl[:] = ctrl
        d.qacc_warmstart[:] = warm
        mujoco.mj_step(m, d, nstep=SUB)
        return (np.array(d.qpos, copy=True), np.array(d.qvel, copy=True))

    # Рабочая точка — та, в которой муха СТОИТ НА НОГАХ после ста
    # настоящих шагов контроллера. Сохраняется для сравнения версий, и
    # сохраняется именно отсюда, а не осаждением свободным падением:
    # замер показал, что осевшая муха даёт 112 контактов и 11 мс на шаг
    # против 0,63 мс в настоящем прогоне, то есть осаждение мерило бы
    # совсем другую механику.
    # Состояние сохраняется ПОЛЯМИ ПО ИМЕНАМ, а не одним буфером
    # `mjSTATE_INTEGRATION`. Причина замерена: у mujoco 3.17 этот буфер
    # длиной 916 против 826 у 3.2.7 — в него добавились поля, и перенос
    # между версиями стал невозможен. Имена же означают одно и то же.
    поля = {k: np.array(getattr(d0, k), copy=True) for k in ПОЛЯ}
    np.savez(DST / "state.npz", время=np.array([d0.time]), **поля)
    mujoco.mj_forward(m0, d0)
    rep["замеры"]["контактов в рабочей точке"] = int(d0.ncon)
    print("  рабочая точка сохранена: контактов %d" % d0.ncon, flush=True)

    q0, v0 = прогнать(m0, d0)
    q1, v1 = прогнать(m1, d1)
    dq = float(np.abs(q0 - q1).max()) if q0.shape == q1.shape else float("inf")
    dv = float(np.abs(v0 - v1).max()) if v0.shape == v1.shape else float("inf")
    rep["замеры"]["расхождение после 100 шагов"] = {
        "qpos": dq, "qvel": dv}
    check("сто шагов физики совпали побитово", dq == 0.0 and dv == 0.0,
          "макс |разн| qpos %.3e, qvel %.3e" % (dq, dv))

    (OUT / "model_export.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
