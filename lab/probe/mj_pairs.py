r"""Последняя названная ручка: модель контактов.

Узкая фаза столкновений стоит **17,69 мс на сто шагов** (28 % шага
физики), а широкая — 0,11 мс, то есть почти ноль. Это и есть ключ:
заранее заданные пары (`<pair>` в MJCF) **минуют широкую фазу** и
проверяются узкой каждый шаг, все до одной. В модели их **2316**.

Вопрос поэтому простой и проверяемый: сколько из этих пар
когда-нибудь дают контакт? Если большинство не даёт никогда, цена
узкой фазы — это цена проверок, которые заведомо ни к чему не ведут.

Проба считает, а не рассуждает: гоняет настоящий эпизод и ведёт
счётчик по **подписи пары геометрий**. Отдельно меряет, сколько
контактов бывает на шаге.

Решения здесь не принимаются. Снятие пар — изменение модели
контактов, то есть разряд B, и принимать его можно только через
поведенческую приёмку: опора на ногах, число и силы контактов,
проскальзывание, фазы походки, высота и крен корпуса.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\mj_pairs.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_mj_pairs_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
ШАГОВ = 1500        # шагов физики: пятнадцать шагов управления
rep = {"проверки": {}, "замеры": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    import mujoco

    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN",
                       vision=False, vision_hz=100.0)
    sources = [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]]
    sim, _ = B._build_body(cfg, sources)
    m = sim.physics.model.ptr
    d = sim.physics.data.ptr
    rep["замеры"]["заданных пар"] = int(m.npair)
    rep["замеры"]["геометрий"] = int(m.ngeom)
    print("  заданных пар %d, геометрий %d" % (m.npair, m.ngeom), flush=True)

    # Подписи заданных пар: по ним и считаем. Подпись — пара номеров
    # геометрий в том же порядке, в каком её хранит модель.
    заданные = set()
    for i in range(m.npair):
        заданные.add((int(m.pair_geom1[i]), int(m.pair_geom2[i])))

    трогали = {}
    контактов = []
    action = np.array([0.9, 0.9], dtype=float)
    t0 = time.perf_counter()
    for k in range(ШАГОВ):
        sim.step(action)
        n = int(d.ncon)
        контактов.append(n)
        for j in range(n):
            c = d.contact[j]
            пара = (int(c.geom1), int(c.geom2))
            if пара not in трогали:
                трогали[пара] = 0
            трогали[пара] += 1
    стена = time.perf_counter() - t0

    из_заданных = {p: v for p, v in трогали.items() if p in заданные}
    не_заданные = {p: v for p, v in трогали.items() if p not in заданные}
    rep["замеры"]["шагов физики"] = ШАГОВ
    rep["замеры"]["по стене, с"] = round(стена, 1)
    rep["замеры"]["контактов на шаге"] = {
        "медиана": float(np.median(контактов)),
        "наименьшее": int(min(контактов)),
        "наибольшее": int(max(контактов)),
    }
    rep["замеры"]["пар трогали хоть раз"] = len(трогали)
    rep["замеры"]["из них заданных"] = len(из_заданных)
    rep["замеры"]["найденных широкой фазой"] = len(не_заданные)
    доля = 100.0 * len(из_заданных) / max(int(m.npair), 1)
    rep["замеры"]["доля заданных пар, которые касались, %"] = round(доля, 2)
    print("  контактов на шаге: медиана %.0f, размах %d…%d"
          % (np.median(контактов), min(контактов), max(контактов)),
          flush=True)
    print("  пар касалось %d, из них заданных %d из %d (%.2f %%)"
          % (len(трогали), len(из_заданных), m.npair, доля), flush=True)

    check("контакты вообще были", len(трогали) > 0, "%d пар" % len(трогали))
    check("большинство заданных пар не касается никогда", доля < 50.0,
          "%.2f %% из %d" % (доля, m.npair))
    rep["находки"].append(
        "узкая фаза проверяет %d заданных пар каждый шаг, а касаются "
        "за %d шагов только %d — то есть %.1f %% проверок заведомо ни к "
        "чему не ведут" % (m.npair, ШАГОВ, len(из_заданных),
                           100.0 - доля))
    print("  " + rep["находки"][-1], flush=True)
    rep["находки"].append(
        "снятие этих пар — изменение модели контактов, разряд B: "
        "«никогда не касалась за %d шагов» не то же самое, что «не "
        "может коснуться». Принимать только через поведенческую "
        "приёмку." % ШАГОВ)
    print("  " + rep["находки"][-1], flush=True)

    # --- избыточны ли заданные пары вовсе ---------------------------
    #
    # Заданная пара (`<pair>`) делает две вещи: минует широкую фазу и
    # несёт СВОИ параметры контакта. Если её параметры совпадают с
    # теми, что MuJoCo вывел бы из двух геометрий сам, и маски
    # `contype`/`conaffinity` позволяют им столкнуться обычным путём,
    # то пара не добавляет ничего, кроме проверки каждый шаг. Тогда её
    # снятие — разряд A, а не изменение модели.
    #
    # Правила смешивания — из руководства MuJoCo: condim, margin и gap
    # берутся наибольшими, трение — поэлементно наибольшее, solref и
    # solimp смешиваются по solmix, а при разных знаках solref[0]
    # берётся поэлементный минимум.
    def смешать(g1, g2):
        s1, s2 = float(m.geom_solmix[g1]), float(m.geom_solmix[g2])
        w = s1 / (s1 + s2) if (s1 + s2) > 0 else 0.5
        r1, r2 = m.geom_solref[g1], m.geom_solref[g2]
        i1, i2 = m.geom_solimp[g1], m.geom_solimp[g2]
        if r1[0] > 0 and r2[0] > 0:
            solref = w * r1 + (1.0 - w) * r2
            solimp = w * i1 + (1.0 - w) * i2
        else:
            solref = np.minimum(r1, r2)
            solimp = np.minimum(i1, i2)
        return {
            "condim": max(int(m.geom_condim[g1]), int(m.geom_condim[g2])),
            "margin": max(float(m.geom_margin[g1]),
                          float(m.geom_margin[g2])),
            "gap": max(float(m.geom_gap[g1]), float(m.geom_gap[g2])),
            "friction": np.maximum(
                np.concatenate([m.geom_friction[g1],
                                m.geom_friction[g1][1:]])[:5],
                np.concatenate([m.geom_friction[g2],
                                m.geom_friction[g2][1:]])[:5]),
            "solref": np.asarray(solref), "solimp": np.asarray(solimp),
        }

    не_совпали, масок_нет = 0, 0
    примеры = []
    for i in range(int(m.npair)):
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        маска = (int(m.geom_contype[g1]) & int(m.geom_conaffinity[g2])) or                 (int(m.geom_contype[g2]) & int(m.geom_conaffinity[g1]))
        if not маска:
            масок_нет += 1
        св = смешать(g1, g2)
        плохо = []
        if int(m.pair_dim[i]) != св["condim"]:
            плохо.append("condim %d против %d"
                         % (m.pair_dim[i], св["condim"]))
        if abs(float(m.pair_margin[i]) - св["margin"]) > 1e-12:
            плохо.append("margin")
        if abs(float(m.pair_gap[i]) - св["gap"]) > 1e-12:
            плохо.append("gap")
        if not np.allclose(m.pair_friction[i], св["friction"],
                           rtol=0, atol=1e-12):
            плохо.append("friction")
        if not np.allclose(m.pair_solref[i], св["solref"],
                           rtol=0, atol=1e-12):
            плохо.append("solref")
        if not np.allclose(m.pair_solimp[i], св["solimp"],
                           rtol=0, atol=1e-12):
            плохо.append("solimp")
        if плохо:
            не_совпали += 1
            if len(примеры) < 5:
                примеры.append({"пара": i, "расхождения": плохо})
    rep["замеры"]["пар с иными параметрами"] = не_совпали
    rep["замеры"]["пар, которым маски запрещают обычное столкновение"] =         масок_нет
    rep["замеры"]["примеры расхождений"] = примеры
    print("  пар с иными параметрами: %d из %d; масок не хватает у %d"
          % (не_совпали, m.npair, масок_нет), flush=True)
    избыточны = (не_совпали == 0 and масок_нет == 0)
    # Проверка записана ОТ ОТВЕТА, а не от надежды: вопрос был
    # «избыточны ли пары», и ответ «нет» — такой же ответ. Поэтому
    # проверка требует не избыточности, а ОПРЕДЕЛЁННОСТИ: либо все
    # пары избыточны, либо ни одна. Промежуточный случай означал бы,
    # что модель собрана непоследовательно, и вот это стоило бы
    # разобрать.
    check("ответ про избыточность определённый",
          избыточны or (не_совпали == int(m.npair)
                        and масок_нет == int(m.npair)),
          "иные параметры у %d из %d, масок нет у %d"
          % (не_совпали, m.npair, масок_нет))
    rep["находки"].append(
        "заданные пары %s: %d из %d несут параметры, отличные от "
        "выводимых из геометрий, и у %d маски запрещают обычное "
        "столкновение — значит снятие пары %s"
        % ("избыточны" if избыточны else "НЕ избыточны",
           не_совпали, m.npair, масок_нет,
           "ничего бы не изменило" if избыточны
           else "изменило бы модель контактов"))
    print("  " + rep["находки"][-1], flush=True)

    # Какие именно пары касаются — по именам геометрий, для разбора.
    def имя(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)

    верх = sorted(из_заданных.items(), key=lambda kv: -kv[1])[:20]
    rep["замеры"]["чаще всего касались"] = [
        {"пара": "%s + %s" % (имя(a), имя(b)), "раз": n} for (a, b), n in верх]

    (OUT / "mj_pairs.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
