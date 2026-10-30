"""Что ФИЗИЧЕСКИ умеет механическая модель и её приводы. Со встречными замерами.

Зачем эта проба существует
--------------------------
Владелец запретил заявлять поддержку полёта или прыжка, не проверив
настоящую модель и приводы. `MOTOR.md` до этой пробы не содержал ни
одного числа про механику: там измерены нейроны, команда и её
происхождение, а про тело сказано только то, что приводы на шею
добавляются при стабилизации головы. То есть утверждение «полёт не
поддержан» было бы таким же необоснованным, как «поддержан».

Здесь измеряется ровно это, и ничего про нейроны: сколько в модели
степеней свободы, какие приводы, какие пределы, может ли тело оторваться
от пола и есть ли крылья как приводы или только как геометрия.

Устройство проверок
-------------------
У каждой проверки есть ВСТРЕЧНЫЙ ЗАМЕР, и он здесь важнее обычного:
почти все утверждения этой пробы ОТРИЦАТЕЛЬНЫЕ («приводов на крыльях
нет», «жидкостных сил нет», «в воздух не поднимается»). Отрицательный
результат ничего не стоит, пока не показано, что измеритель умеет дать
положительный:

* «на крыльях нет шарниров» — тот же запрос на бедре средней ноги
  обязан найти шарнир и привод;
* «аэродинамики нет» — на КОПИИ модели с ненулевой плотностью та же
  проверка обязана увидеть ненулевую пассивную силу;
* «в воздух не поднимается» — поднятая на 3 мм муха обязана дать нуль
  лап в контакте, то есть счётчик контактов умеет показать воздух;
* «реакция пола равна весу» — у подвешенной мухи она обязана стать
  нулём.

Единицы модели
--------------
Длина — миллиметры, время — секунды, ускорение свободного падения
9810 мм/с². Масса в собственных единицах NeuroMechFly; абсолютное
значение здесь не нужно, потому что всё сравнивается с ВЕСОМ самой мухи,
посчитанным в тех же единицах, а калибровка измерителя силы проверяется
отдельно (§4): стоячая реакция пола обязана совпасть с весом.

    cd <рабочее дерево>
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_mech.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent
G = 9810.0          # мм/с², как в самой модели
SETTLE_STEPS = 2000  # 0,4 с физики: за это время походка выходит на ритм

fails = []
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(key, value):
    rep["замеры"][key] = value


# --------------------------------------------------------------- стенд


class Mech:
    """Тело проекта и прямой доступ к его модели MuJoCo.

    Собирается ровно тем же `body._build_body`, которым считаются
    прогоны, и с теми же настройками по умолчанию. Крючки ускорения
    выключены (`acc_forward=False`, `fast_obs=False`) не ради скорости, а
    ради чистоты: они подменяют `Physics.step`, а здесь физика — предмет
    измерения, и мерить её через свою же надстройку незачем. На числа
    модели это не влияет: крючки меняют, что пересчитывается после шага,
    а не сам шаг.
    """

    LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

    def __init__(self, **kw):
        from flybrain import body as B

        cfg = B.BodyConfig(duration_s=1.0, video=False, acc_forward=False,
                           fast_obs=False, **kw)
        self.cfg = cfg
        self.env, _ = B._build_body(cfg, B.DEFAULT_SOURCES.copy())
        self.m = self.env.physics.model.ptr
        m = self.m
        self.a_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                        for i in range(m.nu)]
        self.j_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(m.njnt)]
        self.b_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                        for i in range(m.nbody)]
        self.g_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
                        for i in range(m.ngeom)]
        # Привод по виду передачи: 0 — на шарнир (позиционный), 5 — на
        # тело (прилипание лап). Вид берётся из модели, а не из имени.
        self.pos_act = [i for i in range(m.nu) if m.actuator_trntype[i] == 0]
        self.adh_act = [i for i in range(m.nu) if m.actuator_trntype[i] == 5]
        self.fly_bodies = [i for i, s in enumerate(self.b_names)
                           if s and s.startswith("0/")]
        self.fly_geoms = {g for g in range(m.ngeom)
                          if m.geom_bodyid[g] in self.fly_bodies}
        self.ground = [g for g, s in enumerate(self.g_names) if s == "ground"]
        self.mass = float(m.body_mass[self.fly_bodies].sum())
        self.weight = self.mass * G
        self.thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                        "0/Thorax")

    # ---- измерители ----

    @property
    def d(self):
        return self.env.physics.data.ptr

    def ground_contacts(self):
        """Сколько контактов «муха — пол» и у скольких РАЗНЫХ ног."""
        d, m = self.d, self.m
        n, legs = 0, set()
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            pair = {g1, g2}
            if not (pair & set(self.ground)):
                continue
            other = (g2 if g1 in self.ground else g1)
            if other not in self.fly_geoms:
                continue
            n += 1
            nmb = self.b_names[m.geom_bodyid[other]] or ""
            for leg in self.LEGS:
                if nmb.startswith("0/" + leg):
                    legs.add(leg)
        return n, len(legs)

    def ground_reaction(self):
        """Вертикальная составляющая суммарной реакции пола, в силе модели."""
        d, m = self.d, self.m
        f6 = np.zeros(6)
        tot = 0.0
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            pair = {g1, g2}
            if not (pair & set(self.ground)):
                continue
            other = (g2 if g1 in self.ground else g1)
            if other not in self.fly_geoms:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            frame = np.array(d.contact[c].frame).reshape(3, 3)
            fw = frame.T @ f6[:3]
            # Кадр контакта смотрит от geom1 к geom2. Сила, которую пол
            # приложил к мухе, положительна вверх, когда муха — geom2.
            tot += (1.0 if other == g2 else -1.0) * float(fw[2])
        return tot

    def thorax_z(self):
        return float(self.d.xpos[self.thorax][2])

    # ---- прогоны ----

    def walk(self, action, steps):
        """Шагать штатным интерфейсом: ровно два числа, как из декодера."""
        a = np.asarray(action, dtype=np.float64)
        out = {"z": [], "legs": [], "react": [], "ctrl": [], "force": []}
        out["air"] = []
        for _ in range(steps):
            self.env.step(a)
            n, legs = self.ground_contacts()
            out["z"].append(self.thorax_z())
            out["legs"].append(legs)
            # «В воздухе» — это НОЛЬ контактов с полом вообще, а не ноль
            # лап. Разница не придирка: при сильной ступени на приводы
            # муха теряет все шесть лап, но ложится на грудь и брюшко, и
            # счётчик по лапам отчитался бы о полёте там, где тело
            # просто лежит. Счётчик по всем телам этого не позволяет.
            out["air"].append(n == 0)
            out["react"].append(self.ground_reaction())
            out["ctrl"].append(np.array(self.d.ctrl[self.pos_act]))
            out["force"].append(np.abs(
                np.array(self.d.actuator_force[self.pos_act])).max())
        return out

    def reset(self, seed=0):
        return self.env.reset(seed=seed)


def stand(mech, action=(1.0, 1.0), steps=SETTLE_STEPS, seed=0):
    """Довести муху до походной позы её же контроллером и замереть состояние."""
    mech.reset(seed=seed)
    tr = mech.walk(action, steps)
    d = mech.d
    return {
        "z": mech.thorax_z(),
        "ctrl": np.array(d.ctrl),
        "qpos": np.array(d.qpos),
        "qvel": np.array(d.qvel),
        "legs": mech.ground_contacts()[1],
        "react": mech.ground_reaction(),
        "track": tr,
    }


# ====================================================== 1. инвентарь


def part1(mech):
    print("\n1. Инвентарь модели: степени свободы и приводы")
    m = mech.m
    free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    note("модель", {"nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
                    "njnt": int(m.njnt), "nbody": int(m.nbody),
                    "ngeom": int(m.ngeom),
                    "шаг физики, с": float(m.opt.timestep),
                    "g, мм/с²": float(-m.opt.gravity[2])})
    check("ровно один свободный шарнир (6 DOF корня)", len(free) == 1,
          "свободных %d, имя %r, nq %d, nv %d"
          % (len(free), mech.j_names[free[0]] if free else None,
             m.nq, m.nv))
    # Встречный замер к «одному свободному»: остальные 87 обязаны быть
    # вращательными. Если бы запрос по типу читал не то поле, это бы
    # здесь и вылезло.
    hinge = [i for i in range(m.njnt)
             if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    check("встречный: остальные шарниры — вращательные",
          len(hinge) == m.njnt - 1, "вращательных %d из %d"
          % (len(hinge), m.njnt))

    n_pos, n_adh = len(mech.pos_act), len(mech.adh_act)
    check("приводов 48: 42 позиционных на суставы ног + 6 прилипания лап",
          n_pos == 42 and n_adh == 6 and m.nu == 48,
          "позиционных %d, прилипания %d, всего %d" % (n_pos, n_adh, m.nu))
    # Встречный замер НЕЗАВИСИМЫМ источником: списки самого FlyGym, а не
    # наш разбор модели. Сверять разбор модели самим собой бессмысленно.
    fg_act = list(mech.env.fly.actuated_joints)
    fg_adh = list(getattr(mech.env.fly, "adhesion_actuators", []))
    check("встречный: столько же в собственных списках FlyGym",
          len(fg_act) == n_pos and len(fg_adh) == n_adh,
          "fly.actuated_joints %d, fly.adhesion_actuators %d"
          % (len(fg_act), len(fg_adh)))

    act_j = {int(m.actuator_trnid[i][0]) for i in mech.pos_act}
    unact = [mech.j_names[i] for i in range(m.njnt) if i not in act_j]
    groups = {
        "свободный корень": [s for s in unact if s == "0/"],
        "тарсусы 2..5": [s for s in unact if "Tarsus" in s],
        "шея (Head)": [s for s in unact if "Head" in s],
        "антенны": [s for s in unact if any(k in s for k in
                                            ("Pedicel", "Funiculus",
                                             "Arista"))],
    }
    note("неприводимые шарниры", {k: len(v) for k, v in groups.items()})
    check("неприводимых шарниров 46 и все они разобраны по группам",
          len(unact) == 46 and sum(len(v) for v in groups.values()) == 46,
          "неприводимых %d: %s" % (len(unact), ", ".join(
              "%s %d" % (k, len(v)) for k, v in groups.items())))
    check("шейные суставы БЕЗ приводов, пока стабилизация головы выключена",
          len(groups["шея (Head)"]) == 3,
          "шейных без привода %d из 3; head_stab=%s"
          % (len(groups["шея (Head)"]), mech.cfg.head_stab))

    fr = np.array(m.actuator_forcerange[mech.pos_act])
    kp = np.array(m.actuator_gainprm[mech.pos_act])[:, 0]
    note("предел момента позиционного привода",
         {"forcerange": [float(fr[:, 0].min()), float(fr[:, 1].max())],
          "одинаков у всех 42": bool(np.ptp(fr[:, 1]) == 0),
          "kp": float(kp[0]), "kp одинаков": bool(np.ptp(kp) == 0)})
    check("пределы момента одинаковы у всех 42 приводов и равны ±65",
          bool(np.allclose(fr[:, 1], 65.0) and np.allclose(fr[:, 0], -65.0)),
          "forcerange [%g, %g], kp %g" % (fr[0, 0], fr[0, 1], kp[0]))
    adh_g = np.array(m.actuator_gainprm[mech.adh_act])[:, 0]
    check("приводы прилипания односторонние (ctrl ≥ 0) с усилением 40",
          bool(np.allclose(np.array(m.actuator_ctrlrange[mech.adh_act])[:, 0],
                           0.0) and np.allclose(adh_g, 40.0)),
          "ctrlrange[0] %g, gain %g"
          % (m.actuator_ctrlrange[mech.adh_act[0]][0], adh_g[0]))


# ========================================================== 2. крылья


def part2(mech):
    print("\n2. Крылья и жужжальца: привод или только геометрия")
    m = mech.m
    wing_b = [i for i, s in enumerate(mech.b_names)
              if s and ("Wing" in s or "Haltere" in s)]
    note("тела крыльев и жужжалец",
         {mech.b_names[i]: float(m.body_mass[i]) for i in wing_b})
    check("четыре тела есть: LWing, RWing, LHaltere, RHaltere",
          len(wing_b) == 4, ", ".join(mech.b_names[i] for i in wing_b))

    j_on = [mech.j_names[j] for j in range(m.njnt)
            if int(m.jnt_bodyid[j]) in wing_b]
    a_on = [mech.a_names[i] for i in range(m.nu)
            if (m.actuator_trntype[i] == 0
                and int(m.jnt_bodyid[int(m.actuator_trnid[i][0])]) in wing_b)
            or (m.actuator_trntype[i] == 5
                and int(m.actuator_trnid[i][0]) in wing_b)]
    check("на крыльях и жужжальцах НОЛЬ шарниров и НОЛЬ приводов",
          len(j_on) == 0 and len(a_on) == 0,
          "шарниров %d, приводов %d" % (len(j_on), len(a_on)))

    # ВСТРЕЧНЫЙ ЗАМЕР. Отрицательный результат выше ничего не стоит, пока
    # не показано, что тот же запрос умеет найти шарнир и привод там, где
    # они есть. Бедро средней левой ноги — есть.
    leg_b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "0/LMFemur")]
    j_leg = [mech.j_names[j] for j in range(m.njnt)
             if int(m.jnt_bodyid[j]) in leg_b]
    a_leg = [mech.a_names[i] for i in range(m.nu)
             if m.actuator_trntype[i] == 0
             and int(m.jnt_bodyid[int(m.actuator_trnid[i][0])]) in leg_b]
    check("встречный: тот же запрос на 0/LMFemur находит шарниры и приводы",
          len(j_leg) > 0 and len(a_leg) > 0,
          "шарниров %d (%s), приводов %d (%s)"
          % (len(j_leg), ", ".join(j_leg), len(a_leg), ", ".join(a_leg)))
    # ВТОРОЙ НЕЗАВИСИМЫЙ ИСТОЧНИК: сам текст поставляемого MJCF. Разбор
    # собранной модели и разбор её исходника — две разные дороги, и если
    # бы наша сборка теряла шарниры крыльев по пути, расхождение вылезло
    # бы здесь. Проверяется и то, что геометрии крыльев выключены из
    # столкновений прямо в активе (`contype="0" conaffinity="0"`).
    import re

    import flygym_gymnasium as FG

    xml = (Path(FG.__file__).parent / "data" / "mjcf"
           / "neuromechfly_seqik_kinorder_ypr.xml")
    txt = xml.read_text(encoding="utf-8", errors="replace")
    has_joint = [b for b in re.findall(
        r"<body name=\"(?:LWing|RWing|LHaltere|RHaltere)\"(.*?)</body>",
        txt, re.S) if "<joint" in b]
    noncol = len(re.findall(
        r"<geom name=\"(?:LWing|RWing|LHaltere|RHaltere)\"[^>]*"
        r"contype=\"0\"[^>]*conaffinity=\"0\"", txt))
    note("поставляемый MJCF", {
        "файл": str(xml),
        "шарниров на крыльях и жужжальцах в тексте актива": len(has_joint),
        "геометрий крыльев с contype=0 и conaffinity=0": noncol,
        "всего <joint> в файле": txt.count("<joint"),
    })
    check("встречный источник: в самом MJCF у крыльев нет ни одного "
          "<joint>, а их геометрии выключены из столкновений",
          len(has_joint) == 0 and noncol == 4,
          "блоков крыльев с <joint> %d, геометрий contype=0/conaffinity=0 "
          "%d из 4; всего <joint> в файле %d"
          % (len(has_joint), noncol, txt.count("<joint")))
    note("крылья", {
        "шарниров": len(j_on), "приводов": len(a_on),
        "следствие": ("крылья приварены к груди: относительного движения "
                      "нет вовсе, махать нечем"),
    })


# =================================================== 3. аэродинамика


def part3(mech):
    print("\n3. Аэродинамика: есть ли вообще жидкостные силы")
    m = mech.m
    dens, visc = float(m.opt.density), float(m.opt.viscosity)
    wind = np.array(m.opt.wind, dtype=float)
    gfluid = np.array(m.geom_fluid, dtype=float)
    note("опции среды", {"density": dens, "viscosity": visc,
                         "wind": wind.tolist(),
                         "ненулевых geom_fluid": int((gfluid != 0).sum()),
                         "ненулевых body_gravcomp":
                             int((np.array(m.body_gravcomp) != 0).sum())})
    check("плотность, вязкость, ветер и коэффициенты обтекания — нули",
          dens == 0.0 and visc == 0.0 and not wind.any()
          and not gfluid.any(),
          "density %g, viscosity %g, wind %s, geom_fluid ненулевых %d"
          % (dens, visc, wind.tolist(), int((gfluid != 0).sum())))

    # ВСТРЕЧНЫЙ ЗАМЕР, и он здесь главный. «Плотность равна нулю» — это
    # про поле в структуре. Проверяется СЛЕДСТВИЕ: жидкостная сила при
    # заданной скорости. Сравнивать с нулём сам `qfrc_passive` нельзя, и
    # это замерено: при density=0 он равен 0,1309, потому что в него
    # входят пружины и трение шарниров (у суставов ног жёсткость 0,05,
    # у тарсусов 7,5). Поэтому берётся РАЗНОСТЬ двух расчётов на одном и
    # том же состоянии — она и есть вклад среды.
    #
    # Плотность воздуха в единицах модели: масса модели такова, что вес
    # мухи m·g = 10,1 при g = 9810 мм/с², то есть единица массы ≈ грамм,
    # а длина — миллиметр. Воздух 1,2 кг/м³ = 1,2·10⁻⁶ г/мм³.
    env = mech.env
    env.reset(seed=0)
    d = mech.d
    d.qvel[:] = 0.0
    d.qvel[0] = 2000.0   # 2 м/с вдоль x, скорость полёта мухи
    mujoco.mj_forward(m, d)
    passive_off = np.array(d.qfrc_passive, dtype=float).copy()
    saved = m.opt.density
    try:
        m.opt.density = 1.2e-6   # воздух в единицах модели
        mujoco.mj_forward(m, d)
        passive_on = np.array(d.qfrc_passive, dtype=float).copy()
    finally:
        m.opt.density = saved
        mujoco.mj_forward(m, d)
    f_air = float(np.abs(passive_on - passive_off).max())
    note("пассивная сила при 2000 мм/с",
         {"|qfrc_passive| штатно (пружины и трение шарниров)":
             float(np.abs(passive_off).max()),
          "вклад среды при density=1.2e-6 (воздух)": f_air,
          "он же в долях веса мухи": round(f_air / mech.weight, 6)})
    check("встречный: вклад среды появляется, как только плотность "
          "ненулевая, и на штатной модели его нет ровно",
          f_air > 0.0,
          "вклад среды %.6g при воздухе; на штатной модели он равен нулю "
          "по построению (density=0, viscosity=0, geom_fluid=0)" % f_air)
    note("полёт", {
        "вывод": ("подъёмной силы в этой модели не существует: приводов на "
                  "крыльях нет (§2), и жидкостных сил движок не считает "
                  "вовсе (density=0). Даже если бы крылья махали, "
                  "аэродинамическая сила была бы ровно нулевой"),
    })


# ================================= 4. калибровка измерителя силы


def part4(mech):
    print("\n4. Калибровка: вес мухи против реакции пола")
    st = stand(mech)
    # Калибровка меряется в ПОКОЕ и БЕЗ прилипания, и каждое из двух
    # условий обязательно, что замерено:
    #
    # * при ходьбе мгновенная реакция доходит до 132,8 при весе 10,1 —
    #   это не ошибка измерителя, а разгон тела в фазе толчка;
    # * прилипание лап тянет лапу к полу с усилением 40, и пол отвечает
    #   тем же. Сумма нормальных сил тогда равна весу ПЛЮС тяга клея, и
    #   отношение к весу законно больше единицы.
    #
    # Замораживать походную позу тоже нельзя, и это замерено: цели
    # приводов остаются посреди шага, привод продолжает вдавливать лапу
    # в пол, и сумма реакций уходит на 60 093 при весе 10,1. Поэтому
    # калибровка меряется на СВОБОДНОМ теле: `reset`, нулевые цели,
    # клея нет, тело оседает само.
    #
    # Поза при этом НЕ походная: без управления муха садится на коксы
    # (высота груди уходит с 1,1 мм до −0,12 мм). Для калибровки это и
    # нужно — пассивное тело, лежащее на полу, обязано давить на него
    # ровно своим весом. Ни для чего другого эта поза не годится.
    #
    # Сначала замер той самой замороженной позы, чтобы утверждение про
    # 60 093 стояло числом, а не словами в примечании.
    d0 = mech.d
    frozen = np.array(d0.ctrl)
    for i in mech.adh_act:
        frozen[i] = 0.0
    for _ in range(3000):
        d0.ctrl[:] = frozen
        mech.env.physics.step()
    note("замороженная походная поза (НЕ калибровка)",
         {"реакция пола": round(mech.ground_reaction(), 2),
          "вес": round(mech.weight, 4),
          "почему не годится": ("цели приводов остались посреди шага, и "
                                "привод продолжает вдавливать лапу в пол; "
                                "это не покой")})
    mech.reset(seed=0)
    d = mech.d
    for _ in range(3000):
        d.ctrl[:] = 0.0
        mech.env.physics.step()
    v = float(np.abs(np.array(d.qvel[:6])).max())
    react = mech.ground_reaction()
    legs = mech.ground_contacts()[1]
    note("свободное тело на полу", {"масса (единицы модели)": mech.mass,
                                  "вес = m·g": mech.weight,
                                  "высота груди, мм": mech.thorax_z(),
                                  "ног в контакте": legs,
                                  "реакция пола": react,
                                  "|qvel| корня макс": v})
    ratio = react / mech.weight if mech.weight else float("nan")
    check("у свободно осевшего тела реакция пола равна весу мухи",
          0.9 <= ratio <= 1.1,
          "вес %.4f, реакция %.4f, отношение %.4f, остаточная скорость "
          "корня %.4g" % (mech.weight, react, ratio, v))

    # Тяга прилипания — отдельный замер, не провал: средняя за цикл
    # походки реакция обязана быть больше веса ровно на неё.
    r_walk = np.array(st["track"]["react"])
    tail = r_walk[len(r_walk) // 2:]
    note("прилипание лап при ходьбе",
         {"средняя реакция за полцикла": round(float(tail.mean()), 4),
          "вес": round(mech.weight, 4),
          "превышение (тяга клея + разгон)":
              round(float(tail.mean()) - mech.weight, 4),
          "мгновенный максимум": round(float(tail.max()), 4)})

    # ВСТРЕЧНЫЙ ЗАМЕР: подвесить муху. Если измеритель реакции показывает
    # вес и у висящей в воздухе, он показывает не реакцию.
    d = mech.d
    d.qpos[2] += 3.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(mech.m, d)
    for _ in range(2):
        mech.env.physics.step()
    n_air, legs_air = mech.ground_contacts()
    react_air = mech.ground_reaction()
    check("встречный: у поднятой на 3 мм мухи ног в контакте ноль и "
          "реакция ноль",
          legs_air == 0 and abs(react_air) < 1e-6,
          "контактов %d, ног %d, реакция %.6g" % (n_air, legs_air, react_air))
    return st


# ================================ 5. запас момента при ходьбе


def part5(mech, st):
    print("\n5. Запас момента: предел привода уже достигается ходьбой")
    # Проверка здесь не «хватает ли момента», а «сколько его осталось».
    # Ответ: пик при обычной ходьбе ДОХОДИТ до полки ±65, но держится на
    # ней доли процента шагов. Значит потолок момента — настоящий (его
    # касаются), но не хронический (почти всё время запас есть). Оба
    # числа в примечании, и путать их нельзя: первое говорит, что
    # просить у модели «ещё момента» некуда, второе — что обычная ходьба
    # в этот потолок не упирается.
    f = np.abs(np.array(st["track"]["force"]))   # пик по 42 приводам на шаг
    lim = 65.0
    sat_steps = int((f >= lim - 1e-9).sum())
    note("момент привода при ходьбе",
         {"пиковый |момент| за эпизод": round(float(f.max()), 4),
          "предел": lim,
          "шагов физики, где хоть один привод на полке": sat_steps,
          "из шагов": int(len(f)),
          "доля шагов на полке": round(sat_steps / max(len(f), 1), 4),
          "вес мухи": round(mech.weight, 4),
          "предел момента / вес, мм": round(lim / mech.weight, 3)})
    check("пиковый момент при ходьбе ДОХОДИТ до предела привода",
          float(f.max()) >= lim - 1e-9,
          "пик %.4f из %.0f; на полке %d шагов физики из %d (%.1f %%)"
          % (f.max(), lim, sat_steps, len(f),
             100.0 * sat_steps / max(len(f), 1)))
    # Встречный замер: предел должен быть достижим в принципе, иначе
    # «ниже предела» ничего не говорит. Проверяется, что привод выходит
    # на полку при большой ошибке позиции.
    d = mech.d
    base = np.array(d.ctrl)
    for i in mech.pos_act:
        d.ctrl[i] = base[i] + 5.0
    mujoco.mj_forward(mech.m, d)
    sat = float(np.abs(np.array(d.actuator_force[mech.pos_act])).max())
    for i in mech.pos_act:
        d.ctrl[i] = base[i]
    mujoco.mj_forward(mech.m, d)
    check("встречный: при ошибке 5 рад привод выходит на полку ±65",
          abs(sat - lim) < 1e-6, "момент на полке %.4f" % sat)


# ============================ 6. может ли тело оторваться от пола


def part6(mech):
    print("\n6. Отрыв от пола: через нейронный интерфейс и сырыми приводами")

    # --- 6а. через штатный интерфейс: ровно два числа команды ---
    box = mech.env.action_space
    lo, hi = float(np.min(box.low)), float(np.max(box.high))
    note("пространство команды", {"вид": str(box), "низ": lo, "верх": hi,
                                  "размерность": int(box.shape[0])})
    rows = []
    for act in ((hi, hi), (lo, lo), (hi, lo), (1.4, 1.4), (0.1, 0.1)):
        st = stand(mech, action=act, steps=SETTLE_STEPS)
        tr = st["track"]
        z = np.array(tr["z"])
        legs = np.array(tr["legs"])
        air = np.array(tr["air"], dtype=bool)
        tail = slice(len(z) // 2, None)   # вторая половина: походка на ритме
        rows.append({
            "команда": list(act),
            "высота груди мин, мм": round(float(z[tail].min()), 4),
            "высота груди макс, мм": round(float(z[tail].max()), 4),
            "размах высоты, мм": round(float(np.ptp(z[tail])), 4),
            "ног в контакте мин": int(legs[tail].min()),
            "шагов без контакта с полом": int(air[tail].sum()),
        })
        print("   команда %-12s высота %.4f…%.4f мм, ног в контакте %d…%d, "
              "шагов в воздухе %d"
              % (str(list(act)), z[tail].min(), z[tail].max(),
                 legs[tail].min(), legs[tail].max(), int(air[tail].sum())))
    note("отрыв через нейронный интерфейс", rows)
    air = max(r["шагов без контакта с полом"] for r in rows)
    check("через нейронный интерфейс тело от пола НЕ отрывается ни на "
          "одном углу пространства команды",
          air == 0, "шагов без контакта максимум %d из %d"
          % (air, SETTLE_STEPS // 2))

    # --- 6б. сырыми приводами, в обход контроллера походки ---
    #
    # Это уже не интерфейс, а предел механики: что приводы могут
    # физически, если им приказать напрямую. Ступень разгибания задаётся
    # из ПОХОДНОЙ позы, а не из начальной: начальная «stretch» сама по
    # себе не стоячая, и муха из неё оседает на кокcы.
    st = stand(mech)
    z0, ctrl0 = st["z"], np.array(st["ctrl"])
    qpos0, qvel0 = np.array(st["qpos"]), np.array(st["qvel"])
    raw = []
    for pat, keys in (("бёдра+голени", ("Femur", "Tibia")),
                      ("все суставы ног", ("Coxa", "Femur", "Tibia",
                                           "Tarsus1"))):
        for scale in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
            r = raw_push(mech, ctrl0, qpos0, qvel0, z0, keys, scale)
            r.update({"набор": pat, "ступень, рад": scale})
            raw.append(r)
            print("   %-16s %+5.2f рад: подъём %+7.4f мм, в воздухе %6.2f мс, "
                  "v₀ %8s мм/с, баллистика %7s мм, подъём в полёте %7s мм, "
                  "наклон %5s°, %s"
                  % (pat, scale, r["подъём, мм"], r["в воздухе, мс"],
                     r.get("скорость отрыва, мм/с", "—"),
                     r.get("баллистическая оценка v²/2g, мм", "—"),
                     r.get("подъём за первый полёт, мм", "—"),
                     r["наклон к концу, °"], r["итог"]))
    note("отрыв сырыми приводами", raw)
    ok_air = [r for r in raw if r["шагов без контакта"] > 0
              and r["итог"] == "устойчиво"]
    check("сырыми приводами отрыв достижим хотя бы на одной ступени "
          "(иначе отрицательный итог §6а нечем подтвердить)",
          len(ok_air) > 0,
          "устойчивых ступеней с отрывом %d из %d" % (len(ok_air), len(raw)))

    # И вот здесь отрицательный итог становится ОТВЕТОМ ПРО ПРЫЖОК.
    # «Ноль контактов» сам по себе ничего не значит: три ступени с самым
    # долгим отрывом кончаются наклоном 138,7°, 176,2° и 177,7° — муха
    # лежит на спине, и это падение, а не прыжок. Поэтому считаются
    # только те ступени, после которых тело осталось на ногах, и у них
    # подъём В ПОЛЁТЕ сравнивается со СВОЕЙ ЖЕ стоячей высотой.
    upright = [r for r in raw if r["итог"] == "устойчиво"
               and (r["наклон к концу, °"] or 999.0) < 15.0]
    rise_up = max([r.get("подъём за первый полёт, мм", 0.0)
                   for r in upright] or [0.0])
    bal_up = max([r.get("баллистическая оценка v²/2g, мм", 0.0)
                  for r in upright] or [0.0])
    note("прыжок", {
        "стоячая высота груди, мм": round(z0, 4),
        "ступеней, после которых муха осталась на ногах": len(upright),
        "наибольший подъём в полёте у них, мм": round(rise_up, 4),
        "он же в долях стоячей высоты": round(rise_up / z0, 3) if z0 else None,
        "наибольшая баллистическая оценка у них, мм": round(bal_up, 4),
        "скорость, нужная чтобы подняться на стоячую высоту, мм/с":
            round(float(np.sqrt(2.0 * G * z0)), 1) if z0 > 0 else None,
        "вывод": ("прыжка нет. Устойчивая ступень, после которой тело "
                  "осталось на ногах, поднимает грудь в полёте на "
                  "доли стоячей высоты; ступени, дающие больший отрыв, "
                  "опрокидывают муху на спину или ломают интегратор"),
    })
    check("ни одна ступень, после которой муха осталась на ногах, не "
          "поднимает тело даже на его собственную стоячую высоту",
          rise_up < z0,
          "наибольший подъём в полёте %.4f мм при стоячей высоте %.4f мм "
          "(%.0f %%); нужная для этого скорость %.0f мм/с, измеренная "
          "наибольшая у неопрокинутых %.1f мм/с"
          % (rise_up, z0, 100.0 * rise_up / z0, float(np.sqrt(2.0 * G * z0)),
             max([r.get("скорость отрыва, мм/с", 0.0)
                  for r in upright] or [0.0])))
    best = max(raw, key=lambda r: r["подъём, мм"])
    note("самый большой отрыв вообще (с опрокидыванием)", best)


def raw_push(mech, ctrl0, qpos0, qvel0, z0, keys, scale,
             push=250, hold=750):
    """Ступень на выбранные приводы из походной позы. Прилипание снято.

    Прилипание лап снимается нарочно: с ним привод прилипания тянет лапу
    к полу с усилением 40, и измерялась бы не способность оттолкнуться, а
    способность отклеиться. Какое из двух число нужно — зависит от
    вопроса; здесь вопрос «может ли тело оторваться», значит клей надо
    снять и сказать, что он снят.
    """
    d = mech.d
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    mujoco.mj_forward(mech.m, d)
    tgt = ctrl0.copy()
    for i in mech.pos_act:
        if any(k in (mech.a_names[i] or "") for k in keys):
            tgt[i] = ctrl0[i] + scale
    zmax, legs_min, air, react_max = z0, 9, 0, 0.0
    v_off, z_off, z_air_max, landed = None, None, None, False
    was_air = False
    verdict = "устойчиво"
    tilt_max = 0.0

    def tilt_deg():
        """Угол между осью «вверх» груди и мировой вертикалью, градусы.

        Нужен затем, что «ноль контактов с полом» ещё не полёт: тело,
        опрокинутое разгибанием ног, тоже какое-то время ничего не
        касается. Если к концу отрыва муха лежит на боку, это падение, и
        называть его прыжком нельзя.
        """
        R = np.array(mech.d.xmat[mech.thorax]).reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    try:
        for k in range(push + hold):
            c = tgt if k < push else ctrl0
            for i in mech.pos_act:
                d.ctrl[i] = c[i]
            for i in mech.adh_act:
                d.ctrl[i] = 0.0
            mech.env.physics.step()
            z = mech.thorax_z()
            zmax = max(zmax, z)
            ncon, legs = mech.ground_contacts()
            legs_min = min(legs_min, legs)
            if ncon == 0:
                air += 1
                if not was_air:
                    # Момент отрыва: вертикальная скорость корня и
                    # высота. По ним считается баллистическая оценка
                    # подъёма — она и отличает настоящий толчок от
                    # численного выброса.
                    v_off = float(d.qvel[2])
                    z_off = z
                    z_air_max = z
                else:
                    z_air_max = max(z_air_max, z)
                was_air = True
            else:
                if was_air:
                    landed = True
                was_air = False
            react_max = max(react_max, mech.ground_reaction())
            tilt_max = max(tilt_max, tilt_deg())
    except Exception as e:
        verdict = "НЕУСТОЙЧИВО (%s)" % type(e).__name__
    out = {"подъём, мм": round(zmax - z0, 4),
           "ног в контакте мин": int(legs_min),
           "шагов без контакта": int(air),
           "в воздухе, мс": round(air * float(mech.m.opt.timestep) * 1000.0, 2),
           "макс реакция / вес": round(react_max / mech.weight, 3),
           "макс наклон груди, °": round(tilt_max, 1),
           "наклон к концу, °": round(tilt_deg(), 1)
           if verdict == "устойчиво" else None,
           "итог": verdict}
    if v_off is not None:
        bal = v_off * v_off / (2.0 * G) if v_off > 0 else 0.0
        out.update({
            "скорость отрыва, мм/с": round(v_off, 2),
            "подъём за первый полёт, мм": round(z_air_max - z_off, 4),
            "баллистическая оценка v²/2g, мм": round(bal, 4),
            "сел обратно": bool(landed)})
    return out


# ============== 7. инженерный слой выбора действия: назвать и замерить


def part7(mech):
    print("\n7. Инженерный слой между командой и приводами: что он делает")
    from flybrain import gait as Gt

    st = stand(mech, action=(1.0, 1.0), steps=SETTLE_STEPS)
    ctrl = np.array(st["track"]["ctrl"])     # (шагов, 42)
    tail = ctrl[len(ctrl) // 2:]
    # При НЕИЗМЕННОЙ команде позиционные цели всё равно ходят: ритм даёт
    # контроллер, а не мозг. Это и есть размер инженерного слоя, в числах.
    span = np.ptp(tail, axis=0)
    note("слой выбора действия", {
        "имя": "flybrain.gait — вендоренная копия шага HybridTurningController",
        "вид": "CPG на шесть ног плюс два правила (втягивание, спотыкание)",
        "размерность команды": int(mech.env.action_space.shape[0]),
        "каналов приводов": int(mech.m.nu),
        "каналов, недостижимых командой": int(mech.m.nu
                                              - mech.env.action_space.shape[0]),
        "размах позиционной цели при НЕИЗМЕННОЙ команде, рад":
            {"медиана": round(float(np.median(span)), 5),
             "макс": round(float(span.max()), 5),
             "суставов с размахом > 0,01 рад": int((span > 0.01).sum())},
        "вендор": getattr(Gt, "__doc__", "").strip().splitlines()[0]
        if getattr(Gt, "__doc__", None) else "",
    })
    check("ритм шага порождает КОНТРОЛЛЕР, а не команда: при постоянной "
          "команде цели приводов всё равно ходят",
          int((span > 0.01).sum()) >= 30,
          "суставов с размахом > 0,01 рад: %d из 42, макс размах %.4f рад"
          % (int((span > 0.01).sum()), span.max()))
    # Встречный замер: команда обязана менять цели ИЗМЕРИМО, иначе
    # предыдущая проверка доказывала бы, что мозг не влияет вовсе.
    st2 = stand(mech, action=(0.2, 1.4), steps=SETTLE_STEPS)
    c2 = np.array(st2["track"]["ctrl"])[len(ctrl) // 2:]
    diff = float(np.abs(tail.mean(axis=0) - c2.mean(axis=0)).max())
    check("встречный: смена команды меняет средние цели приводов",
          diff > 1e-3,
          "наибольшее изменение средней цели %.5f рад при команде "
          "[1,1] против [0.2,1.4]" % diff)
    note("две степени свободы команды", {
        "что задаёт мозг": "две амплитуды шага, по тройке ног на сторону",
        "чего мозг не задаёт": ("ритм, фазы, последовательность ног, "
                                "втягивание, прилипание лап, позу — всё это "
                                "контроллер походки"),
    })


def main():
    t0 = time.perf_counter()
    print("Механика тела: степени свободы, приводы, пределы")
    mech = Mech()
    print("  модель: nq %d, nv %d, nu %d, njnt %d, nbody %d"
          % (mech.m.nq, mech.m.nv, mech.m.nu, mech.m.njnt, mech.m.nbody))
    part1(mech)
    part2(mech)
    part3(mech)
    st = part4(mech)
    part5(mech, st)
    part6(mech)
    part7(mech)

    rep["итог"] = {"проверок": len(rep["проверки"]),
                   "провалов": len(fails),
                   "по стене, с": round(time.perf_counter() - t0, 1)}
    (OUT / "motor_mech.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d, отчёт %s"
          % (len(rep["проверки"]), len(fails), OUT / "motor_mech.json"))
    if fails:
        print("провалились: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
