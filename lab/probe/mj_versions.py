r"""Один и тот же шаг физики разными версиями MuJoCo.

Запускается ДВАЖДЫ: в рабочем окружении (`mujoco 3.2.7`, закреплено
требованием `flygym-gymnasium`) и в отдельном, где стоит современная
версия. Рабочее окружение при этом не трогается ничем.

Модель одна и та же — выгруженная `probe/model_export.py` и там же
проверенная на совпадение описи и на побитовое совпадение ста шагов.
Состояние тоже одно: первый запуск его сохраняет, второй читает, иначе
версии считали бы из разных точек, то есть с разным числом контактов.

Зачем это вообще. Прежняя кампания закрыла направление многопоточного
`mj_step` так: «`mjThreadPool` в привязках `mujoco` 3.2.7 не выставлен»
(`PERF_RT.md` §7). С тех пор вышло двадцать три выпуска, и в **3.10.0**
появился `mju_threadpool` — создание пула потоков на `mjData`, после
чего распараллеливаются обнаружение столкновений и решение ограничений
по островам. Это и есть новое основание открыть направление заново, а
не повторение отвергнутого.

    rem рабочее окружение
    ..\.venv\Scripts\python.exe probe\mj_versions.py
    rem отдельное
    ..\.venv-mj\Scripts\python.exe probe\mj_versions.py
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ПОЛЯ = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart",
         "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat")

HERE = Path(__file__).resolve().parent
MODEL = Path(__file__).resolve().parents[2] / ".venv-mj-model"
XML = MODEL / "fly.xml"
STATE = MODEL / "state.npz"
SUB = 100            # шагов физики на шаг управления
ROUNDS = 9


def main():                                                   # noqa: C901
    if not XML.exists():
        print("модели нет: " + str(XML))
        print("сперва: ..\\.venv\\Scripts\\python.exe probe\\model_export.py")
        return 1
    ver = mujoco.__version__
    print("mujoco %s, python %s" % (ver, sys.version.split()[0]), flush=True)

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    рез = {
        "версия mujoco": ver,
        "python": sys.version.split()[0],
        "модель": {"тел": int(m.nbody), "обобщённых координат": int(m.nq), "степеней свободы": int(m.nv),
                   "приводов": int(m.nu), "геометрий": int(m.ngeom),
                   "шаг": float(m.opt.timestep),
                   "решатель": int(m.opt.solver),
                   "итераций": int(m.opt.iterations),
                   "допуск": float(m.opt.tolerance)},
    }

    if STATE.exists():
        z = np.load(STATE)
        поля = {k: z[k] for k in ПОЛЯ if k in z.files}
        время = float(z["время"][0]) if "время" in z.files else 0.0
        print("  состояние прочитано из %s: %d полей"
              % (STATE.name, len(поля)), flush=True)
        # Поле другой формы — повод остановиться, но не всегда. У
        # mujoco 3.17 тот же XML компилируется в 65 тел вместо 80
        # (компилятор сращивает неподвижные), и `xfrc_applied` имеет
        # форму по телам. Степеней свободы, обобщённых координат,
        # приводов и геометрий при этом ровно столько же, то есть
        # динамика та же. Пропустить такое поле можно ТОЛЬКО если оно
        # целиком нулевое: тогда переносить нечего. Ненулевое поле
        # другой формы — это разные модели, и дальше идти нельзя.
        чужие = {k: поля[k] for k in list(поля)
                 if np.asarray(getattr(d, k)).shape != поля[k].shape}
        плохие = {k: int((v != 0).sum()) for k, v in чужие.items()
                  if (v != 0).any()}
        if плохие:
            print("  поля другой формы и ненулевые: %s — модели разные"
                  % плохие)
            return 1
        for k in чужие:
            поля.pop(k)
        if чужие:
            рез["поля пропущены (другая форма, нули)"] = sorted(чужие)
            print("  пропущено как нулевое: %s" % sorted(чужие), flush=True)
    else:
        # Своей рабочей точки у этой пробы нет и быть не должно.
        # Осаждение свободным падением выглядит невинно, а даёт совсем
        # другую механику: замерено 112 контактов и 11,0 мс на шаг
        # против 0,63 мс в настоящем прогоне, где муха стоит на ногах,
        # — в семнадцать раз. Поэтому точка берётся только из настоящей
        # симуляции, и кладёт её `model_export.py`.
        print("состояния нет: " + str(STATE))
        print(r"сперва: ..\.venv\Scripts\python.exe probe\model_export.py")
        return 1

    def встать():
        mujoco.mj_resetData(m, d)
        for k, v in поля.items():
            getattr(d, k)[:] = v
        d.time = время

    def отрезок():
        встать()
        t = time.perf_counter()
        mujoco.mj_step(m, d, nstep=SUB)
        return (time.perf_counter() - t) * 1000.0

    # --- сколько контактов в этой точке ----------------------------
    встать()
    mujoco.mj_forward(m, d)
    рез["контактов в точке замера"] = int(d.ncon)
    print("  контактов: %d" % d.ncon, flush=True)

    # --- сколько итераций решатель тратит на самом деле -------------
    #
    # Модель настроена на 1000 итераций при допуске 1e-12. Это предел, а
    # не расход: если решатель сходится за двадцать, предел не стоит
    # ничего, и трогать его бессмысленно. Поэтому расход меряется, а не
    # выводится из настройки.
    встать()
    ниты, остатки = [], []
    for _ in range(SUB):
        mujoco.mj_step(m, d)
        try:
            n = np.asarray(d.solver_niter)
            ниты.append(int(n.sum()) if n.ndim else int(n))
        except Exception:                                  # noqa: BLE001
            pass
        try:
            остатки.append(float(np.asarray(d.solver_fwdinv).max()))
        except Exception:                                  # noqa: BLE001
            pass
    if ниты:
        a_ = np.asarray(ниты)
        рез["итераций решателя на шаг"] = {
            "медиана": float(np.median(a_)),
            "наименьшая": int(a_.min()), "наибольшая": int(a_.max()),
            "предел модели": int(m.opt.iterations),
            "шагов на пределе": int((a_ >= m.opt.iterations).sum()),
        }
        print("  итераций решателя: медиана %.0f, размах %d…%d при "
              "пределе %d" % (np.median(a_), a_.min(), a_.max(),
                              m.opt.iterations), flush=True)
    if остатки:
        рез["невязка решателя"] = {
            "медиана": float(np.median(остатки)),
            "наибольшая": float(max(остатки))}

    # --- один поток -------------------------------------------------
    for _ in range(3):
        отрезок()
    один = sorted(отрезок() for _ in range(ROUNDS))
    рез["один поток, мс на 100 шагов"] = {
        "медиана": round(один[len(один) // 2], 3),
        "наименьшая": round(один[0], 3),
        "наибольшая": round(один[-1], 3),
    }
    print("  один поток: медиана %.3f мс, размах %.3f…%.3f"
          % (один[len(один) // 2], один[0], один[-1]), flush=True)

    # --- пул потоков, если он есть ---------------------------------
    пул = getattr(mujoco, "mju_threadpool", None)
    рез["пул потоков выставлен"] = пул is not None
    if пул is None:
        print("  mju_threadpool в этой версии не выставлен", flush=True)
    else:
        рез["многопоточно"] = {}
        for n in (2, 4, 8, 16):
            try:
                пул(d, n)
            except Exception as exc:                       # noqa: BLE001
                print("  пул на %d: %s" % (n, repr(exc)[:120]), flush=True)
                рез["многопоточно"][str(n)] = {"отказ": repr(exc)[:200]}
                continue
            for _ in range(3):
                отрезок()
            ts = sorted(отрезок() for _ in range(ROUNDS))
            med = ts[len(ts) // 2]
            рез["многопоточно"][str(n)] = {
                "медиана": round(med, 3),
                "наименьшая": round(ts[0], 3),
                "наибольшая": round(ts[-1], 3),
                "ускорение": round(один[len(один) // 2] / med, 3),
            }
            print("  %2d потоков: медиана %.3f мс, ускорение ×%.3f"
                  % (n, med, один[len(один) // 2] / med), flush=True)

    # --- куда пришли после ста шагов: для сверки версий -------------
    встать()
    mujoco.mj_step(m, d, nstep=SUB)
    рез["после 100 шагов"] = {
        "qpos[:6]": [round(float(v), 12) for v in d.qpos[:6]],
        "сумма |qvel|": round(float(np.abs(d.qvel).sum()), 9),
    }

    путь = HERE / ("mj_versions_%s.json" % ver.replace(".", "_"))
    путь.write_text(json.dumps(рез, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("  записано в %s" % путь.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
