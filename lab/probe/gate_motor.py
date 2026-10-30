"""ШЛЮЗ D. Моторная причинность: вмешательство в нисходящие меняет команду.

Что обязан показать шлюз: **изменение или подавление нужной нисходящей
активности измеримо меняет моторную команду, а мимо-целевые вмешательства
того же размера того же эффекта не дают.**

Вторая половина здесь не вежливость, а необходимость. Проверка «под
маской команда умерла» сама по себе доказывает не «команда приходит из
сети», а «маска что-нибудь ломает»: умерла бы она и от маски на
посторонних нейронах, и от маски на половине мозга. Поэтому шлюз D
требует трёх замеров сразу — норма, цель, мимо-цель — и мимо-цель берётся
двух видов: по имени (посторонние нисходящие) и ТОГО ЖЕ РАЗМЕРА
(случайная выборка нисходящих вне декодера ровно той же численности).

Третья часть шлюза — **знак**. «Команда изменилась» слабее, чем
«изменилась в предсказанную сторону»: подавление левого DNa02 и правого
обязаны сдвинуть поворот в ПРОТИВОПОЛОЖНЫЕ стороны. Это предсказание
роли (односторонняя активация DNa02 поворачивает ипсилатерально), и
проверяется оно числом, а не пересказом статьи.

Четвёртая — **тождество**. `DNa01`, `DNa04`, `DNa05` нельзя выбирать по
имени: имя присвоено разным клеткам двумя именующими авторитетами.
`hemibrain=DNa01` даёт пару с `cell_type = DNae001`, `celltype=DNa01` —
пару с `hemibrain_type = VES006`. Обе пары нисходящие и холинергические,
и подмена проходит молча: число совпадает, отпечаток `root_id` — нет.
Поэтому шлюз требует явного столбца и сверяет отпечатки.

Подавление здесь — `hold`, а не `fire`. У `fire` измеренное
последствие: мембрана копит потенциал (за 500 шагов дошла до +1299,5 мВ
при пороге −45) и при снятии вся популяция разряжается синхронно. Это
вычислительный артефакт способа подавления, и мешать его с
нейробиологией нельзя. У `hold` уравнение объявлено: v = v_clamp до
порога, накопления нет, залпа при снятии нет.

Тела здесь нет нарочно: физика MuJoCo стоит порядка 140 мс настенных на
шаг управления, а статистику надо набрать на сотнях шагов. Проверяется
связка «сеть → измеритель → декодер → два числа», и тело получает ровно
эти два числа. Телесная половина — шлюзы E и F.

    cd <рабочее дерево>
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/gate_motor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe._gate import Report, forge, rule  # noqa: E402

OUT = Path(__file__).parent
CONTROL_MS = 20.0
TAU_S = 0.5
STEPS = 150               # шагов управления в замере = 3,0 с
PERIOD_S = 1.5            # период качания асимметрии запаха
BASE_HZ = 75.0            # средняя частота ORN
DEPTH = 0.85              # глубина качания
HOLD_STEPS = 60           # шагов на одно условие в замере знака
DNA02_HZ = 100.0          # прямая подача на DNa02 в замере знака

# Посторонние нисходящие: в декодере их нет. Выборка ТОЧНЫМ столбцом.
OFFTARGET = ("DNp03", "DNp10", "DNp11")


class Rig:
    """Движок, измеритель прямых зондов и подача по каналам."""

    def __init__(self, neurons, weights, device, seed=0):
        import torch

        from flybrain import data as D, motor, sim

        self.torch = torch
        self.n = len(neurons)
        self.device = device
        self.brain_steps = max(1, int(round(CONTROL_MS / sim.DT)))
        self.block_s = self.brain_steps * sim.DT / 1000.0
        self.dt_s = CONTROL_MS / 1000.0
        self.chan = {}
        for name, sel in (
                ("orn_l", "class:olfactory+side:left"),
                ("orn_r", "class:olfactory+side:right"),
                ("dna02_l", "celltype=DNa02+side:left"),
                ("dna02_r", "celltype=DNa02+side:right"),
        ):
            idx = D.select(neurons, sel)
            self.chan[name] = (idx, torch.as_tensor(idx, device=device))
        driven = np.unique(np.concatenate(
            [v[0] for v in self.chan.values()]))
        self.engine = sim.build_engine(weights=weights, neurons=neurons,
                                       device=device, seed=seed,
                                       no_refractory=driven)
        self.readout = motor.DnReadout(neurons, device)
        self.rates = torch.zeros(1, self.n, device=device)

    def drive(self, **hz):
        self.rates.zero_()
        for name, v in hz.items():
            if v:
                self.rates[0, self.chan[name][1]] = float(v)

    def step(self):
        acc = self.torch.zeros(self.n, device=self.device)
        for _ in range(self.brain_steps):
            acc += self.engine.step(self.rates)[0]
        return self.readout.read(acc, self.block_s)

    def hold(self, idx):
        self.engine.set_masks(hold=np.asarray(idx, np.int64))


def warmed(neurons, weights, device, seed=0):
    """Рига с откалиброванным декодером. Калибровка — ДО любой маски."""
    from flybrain import motor

    rig = Rig(neurons, weights, device, seed=seed)
    dec = motor.make_decoder("dn_linear",
                             shape=motor.DecoderShape(dn_tau_s=TAU_S))
    rig.drive()
    warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
    for _ in range(warm):
        dec.observe(rig.step(), rig.dt_s)
    dec.calibrate()
    dec.reset_state()
    return rig, dec


def schedule(neurons, weights, device, mask=None):
    """Расписание асимметрии запаха -> ряд команд. Калибровка до маски."""
    rig, dec = warmed(neurons, weights, device)
    if mask is not None and len(mask):
        rig.hold(mask)
    turn, stim = [], []
    for s in range(STEPS):
        ph = 2.0 * np.pi * (s * CONTROL_MS / 1000.0) / PERIOD_S
        a = DEPTH * float(np.sin(ph))
        hl = BASE_HZ * (1.0 + a)
        hr = BASE_HZ * (1.0 - a)
        rig.drive(orn_l=hl, orn_r=hr)
        cmd = dec.command(rig.step(), rig.dt_s)
        turn.append(cmd.u_turn)
        stim.append(hl - hr)
    t = np.asarray(turn, dtype=float)
    x = np.asarray(stim, dtype=float)
    sd = float(t.std())
    r0 = (float(np.corrcoef(t, x)[0, 1])
          if sd > 1e-12 and x.std() > 1e-12 else None)
    # Корреляция СО СДВИГОМ, и это не подгонка. У декодера объявлена
    # своя постоянная интегрирования `dn_tau_s`: команда
    # отстаёт от раздражителя по построению, и корреляция при нулевом
    # сдвиге занижает зависимость ровно на этот фазовый сдвиг. Поэтому
    # перебираются сдвиги до ТРЁХ объявленных постоянных, и в отчёт
    # идут оба числа — при нуле и наилучшее со сдвигом.
    best_r, best_lag = r0, 0
    if sd > 1e-12 and x.std() > 1e-12:
        max_lag = int(round(3.0 * TAU_S / (CONTROL_MS / 1000.0)))
        for lag in range(0, min(max_lag, len(t) // 3) + 1):
            a = t[lag:] if lag else t
            b = x[:len(x) - lag] if lag else x
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            rr = float(np.corrcoef(a, b)[0, 1])
            if abs(rr) > abs(best_r):
                best_r, best_lag = rr, lag
    del rig
    return {"ряд": [round(v, 6) for v in turn],
            "разброс команды": round(sd, 8),
            "разброс раздражителя, Гц": round(float(x.std()), 4),
            "корреляция при нулевом сдвиге": (round(r0, 4) if r0 is not None
                                              else None),
            "корреляция с раздражителем": (round(best_r, 4)
                                           if best_r is not None else None),
            "наилучший сдвиг, шагов": int(best_lag),
            "наилучший сдвиг, с": round(best_lag * CONTROL_MS / 1000.0, 3),
            "предел перебора сдвига, с": round(3.0 * TAU_S, 3)}


# ----------------------------------------------------------- правила шлюза

def r_follows(m):
    n = m["норма"]
    r = n["корреляция с раздражителем"]
    ok = n["разброс команды"] > 1e-4 and r is not None and abs(r) >= 0.4
    return ok, ("норма: разброс команды %.8f при разбросе раздражителя "
                "%.4f Гц. Корреляция при нулевом сдвиге %s, наилучшая "
                "%s на сдвиге %.2f с (перебор до %.1f с = трёх "
                "объявленных постоянных интегрирования %.1f с). Нужно "
                "|r| ≥ 0,40. Раздражитель — качание асимметрии ORN, "
                "период %.1f с, глубина %.2f при средней %.0f Гц. Сдвиг "
                "допущен потому, что у декодера постоянная объявлена: "
                "команда отстаёт по построению, и нулевой сдвиг занижает "
                "зависимость на фазовый сдвиг фильтра"
                % (n["разброс команды"], n["разброс раздражителя, Гц"],
                   n["корреляция при нулевом сдвиге"],
                   n["корреляция с раздражителем"],
                   n["наилучший сдвиг, с"], n["предел перебора сдвига, с"],
                   TAU_S, PERIOD_S, DEPTH, BASE_HZ))


def r_ontarget(m):
    n, c = m["норма"], m["цель"]
    r = c["корреляция с раздражителем"]
    ok = (c["разброс команды"] <= 0.1 * n["разброс команды"]
          and (r is None or abs(r) <= 0.1))
    return ok, ("подавление `hold` на %d нейронов зондов декодера: "
                "разброс команды %.8f против %.8f в норме (нужно ≤ 0,1 "
                "от нормы), корреляция %s (нужно |r| ≤ 0,10). "
                "Раздражитель при этом меняется РОВНО так же: разброс "
                "%.4f Гц против %.4f в норме"
                % (m["цель: нейронов"], c["разброс команды"],
                   n["разброс команды"], c["корреляция с раздражителем"],
                   c["разброс раздражителя, Гц"],
                   n["разброс раздражителя, Гц"]))


def r_offtarget(m):
    n = m["норма"]
    строки = []
    плохие = []
    for имя in ("мимо-цель по имени", "мимо-цель того же размера"):
        v = m[имя]
        доля = (v["разброс команды"] / n["разброс команды"]
                if n["разброс команды"] > 0 else 0.0)
        строки.append("%s (%d нейронов): разброс %.8f, то есть %.3f от "
                      "нормы" % (имя, m[имя + ": нейронов"],
                                 v["разброс команды"], доля))
        if доля < 0.5:
            плохие.append(имя)
    ok = not плохие
    return ok, ("%s. Нужно ≥ 0,5 от нормы у КАЖДОЙ: иначе первый "
                "контроль доказывал бы лишь, что маска ломает что угодно. "
                "Не выдержали: %s"
                % ("; ".join(строки), ", ".join(плохие) or "нет"))


def r_sign(m):
    s = m["знак"]
    ok = (s["сдвиг при подавлении слева"]
          * s["сдвиг при подавлении справа"] < 0
          and min(abs(s["сдвиг при подавлении слева"]),
                  abs(s["сдвиг при подавлении справа"]))
          >= s["порог сдвига"])
    return ok, ("прямая подача %.0f Гц на DNa02 с двух сторон даёт "
                "cmd_turn %+.6f. Подавление ЛЕВОГО DNa02: %+.6f, сдвиг "
                "%+.6f. Подавление ПРАВОГО: %+.6f, сдвиг %+.6f. Знаки "
                "обязаны быть противоположны, и каждый сдвиг по модулю "
                "не меньше %.6f (это собственный разброс команды в "
                "симметричном условии). Предсказание роли: "
                "односторонняя активация DNa02 поворачивает "
                "ипсилатерально, значит односторонее подавление обязано "
                "сдвинуть поворот в противоположные стороны"
                % (DNA02_HZ, s["симметрично"], s["слева подавлен"],
                   s["сдвиг при подавлении слева"], s["справа подавлен"],
                   s["сдвиг при подавлении справа"], s["порог сдвига"]))


def r_identity(m):
    i = m["тождество"]
    плохие = [k for k, v in i["зонды"].items() if not v["совпало"]]
    спор = i["спор имён DNa01"]
    ok = (not плохие) and спор["отпечатки различны"]
    return ok, ("шесть зондов декодера сверены по отпечатку "
                "отсортированных root_id: %s. Не совпало: %s. "
                "Спор имён: `hemibrain=DNa01` даёт %s (cell_type %s), "
                "`celltype=DNa01` даёт %s (hemibrain_type %s) — "
                "отпечатки различны: %s. Это и есть причина, по которой "
                "шлюз требует СТОЛБЕЦ, а не имя"
                % (", ".join("%s %s" % (k, v["отпечаток"])
                             for k, v in i["зонды"].items()),
                   ", ".join(плохие) or "нет",
                   спор["hemibrain=DNa01"]["отпечаток"],
                   спор["hemibrain=DNa01"]["cell_type"],
                   спор["celltype=DNa01"]["отпечаток"],
                   спор["celltype=DNa01"]["hemibrain_type"],
                   спор["отпечатки различны"]))


# ----------------------------------------------------------------- подлоги

def f_no_follow(m):
    m = dict(m)
    m["норма"] = dict(m["норма"],
                      **{"корреляция с раздражителем": 0.02,
                         "корреляция при нулевом сдвиге": 0.01})
    return m


def f_ontarget_alive(m):
    """Маска на цель команду не изменила: причинности нет."""
    m = dict(m)
    m["цель"] = dict(m["норма"])
    return m


def f_offtarget_dead(m):
    """Мимо-цель ломает так же: значит ломается от любой маски."""
    m = dict(m)
    m["мимо-цель того же размера"] = dict(
        m["мимо-цель того же размера"], **{"разброс команды": 0.0})
    return m


def f_sign_same(m):
    """Сторона вмешательства знака не задаёт."""
    m = dict(m)
    s = dict(m["знак"])
    s["сдвиг при подавлении справа"] = s["сдвиг при подавлении слева"]
    m["знак"] = s
    return m


def f_name_swap(m):
    """Зонд DNa01 подменён парой из другого столбца.

    Число нейронов совпадает (1 на сторону у обеих пар), и размерная
    проверка прошла бы молча. Отличает их только отпечаток root_id.
    """
    m = dict(m)
    i = {k: (dict(v) if isinstance(v, dict) else v)
         for k, v in m["тождество"].items()}
    зонды = {k: dict(v) for k, v in i["зонды"].items()}
    чужой = i["спор имён DNa01"]["celltype=DNa01"]["отпечаток"]
    зонды["DNa01"] = dict(зонды["DNa01"], **{
        "отпечаток": чужой,
        "совпало": чужой == зонды["DNa01"]["ожидался"]})
    i["зонды"] = зонды
    m["тождество"] = i
    return m


def main():
    rep = Report("ШЛЮЗ D. Моторная причинность на нынешнем v2",
                 OUT / "gate_motor.json")
    import torch

    from flybrain import data as D, motor

    neurons = D.load_neurons()
    W = D.load_weights()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- тождество зондов: явный столбец и отпечаток root_id ---
    rows = motor.verify_probes(neurons, strict=False)
    зонды = {}
    for r in rows:
        if r.get("зонд") not in motor.DN_PROBES:
            continue
        зонды[r["зонд"]] = {
            "селектор": r["селектор"],
            "слева": r["слева"], "справа": r["справа"],
            "отпечаток": r["отпечаток"],
            "ожидался": r.get("ожидался отпечаток"),
            "отпечаток сошёлся": r.get("отпечаток сошёлся"),
            "совпало": bool(r.get("совпало", False)),
        }
    спор = {}
    for sel, кол in (("hemibrain=DNa01", "cell_type"),
                     ("celltype=DNa01", "hemibrain_type")):
        idx = D.select(neurons, sel)
        sub = neurons.iloc[idx]
        спор[sel] = {
            "нейронов": int(len(idx)),
            "отпечаток": _fp_ids(neurons, idx),
            кол: sorted(set(str(x) for x in sub[кол]))[:3],
            "сторона": sorted(set(str(x) for x in sub["side"])),
            "надкласс": sorted(set(str(x) for x in sub["super_class"])),
        }
    спор["отпечатки различны"] = (
        спор["hemibrain=DNa01"]["отпечаток"]
        != спор["celltype=DNa01"]["отпечаток"])
    тождество = {"зонды": зонды, "спор имён DNa01": спор}
    rep.measure("тождество зондов", тождество)
    print("  зонды: " + ", ".join(
        "%s %s%s" % (k, v["отпечаток"], "" if v["совпало"] else " НЕ СОВПАЛ")
        for k, v in зонды.items()), flush=True)
    print("  спор имён DNa01: hemibrain %s против celltype %s, различны %s"
          % (спор["hemibrain=DNa01"]["отпечаток"],
             спор["celltype=DNa01"]["отпечаток"],
             спор["отпечатки различны"]), flush=True)

    # --- кого подавляем: цель и две мимо-цели ---
    rig0 = Rig(neurons, W, dev)
    цель = rig0.readout.indices()
    del rig0
    off_name = np.unique(np.concatenate(
        [D.select(neurons, "celltype=%s" % t) for t in OFFTARGET]))
    dn_all = D.select(neurons, "super:descending")
    пул = np.setdiff1d(dn_all, цель)
    rng = np.random.default_rng(0)
    off_size = np.sort(rng.choice(пул, size=len(цель), replace=False))
    rep.measure("кого подавляем", {
        "цель: нейронов": int(len(цель)),
        "цель: отпечаток": _fp_ids(neurons, цель),
        "мимо-цель по имени: типы": list(OFFTARGET),
        "мимо-цель по имени: нейронов": int(len(off_name)),
        "мимо-цель по имени: отпечаток": _fp_ids(neurons, off_name),
        "мимо-цель того же размера: нейронов": int(len(off_size)),
        "мимо-цель того же размера: отпечаток": _fp_ids(neurons, off_size),
        "нисходящих всего": int(len(dn_all)),
        "пул для случайной выборки": int(len(пул)),
        "семантика подавления": "hold (v = v_clamp до порога)",
    })
    print("  цель %d нейронов; мимо-цель по имени %d (%s); мимо-цель того "
          "же размера %d из пула %d"
          % (len(цель), len(off_name), ", ".join(OFFTARGET), len(off_size),
             len(пул)), flush=True)

    # --- три условия на одном расписании ---
    усл = {}
    for имя, маска in (("норма", None), ("цель", цель),
                       ("мимо-цель по имени", off_name),
                       ("мимо-цель того же размера", off_size)):
        усл[имя] = schedule(neurons, W, dev, mask=маска)
        print("  %-26s разброс %.8f, r при нуле %s, лучшее r %s на "
              "%.2f с" % (имя, усл[имя]["разброс команды"],
                          усл[имя]["корреляция при нулевом сдвиге"],
                          усл[имя]["корреляция с раздражителем"],
                          усл[имя]["наилучший сдвиг, с"]), flush=True)
    rep.measure("расписание асимметрии", усл)

    # --- знак: подавление одной стороны DNa02 ---
    знак = {}
    for имя, маска in (
            ("симметрично", None),
            ("слева подавлен", D.select(neurons,
                                        "celltype=DNa02+side:left")),
            ("справа подавлен", D.select(neurons,
                                         "celltype=DNa02+side:right"))):
        rig, dec = warmed(neurons, W, dev)
        if маска is not None and len(маска):
            rig.hold(маска)
        rig.drive(dna02_l=DNA02_HZ, dna02_r=DNA02_HZ)
        vals = []
        for i in range(HOLD_STEPS):
            cmd = dec.command(rig.step(), rig.dt_s)
            if i >= HOLD_STEPS // 2:
                vals.append(cmd.u_turn)
        знак[имя] = round(float(np.mean(vals)), 6)
        знак[имя + ": разброс"] = round(float(np.std(vals)), 6)
        del rig
        print("  %-16s cmd_turn %+.6f ± %.6f"
              % (имя, знак[имя], знак[имя + ": разброс"]), flush=True)
    знак["сдвиг при подавлении слева"] = round(
        знак["слева подавлен"] - знак["симметрично"], 6)
    знак["сдвиг при подавлении справа"] = round(
        знак["справа подавлен"] - знак["симметрично"], 6)
    # Порог сдвига — собственный разброс команды в симметричном условии,
    # а не выбранное число: сдвиг обязан быть больше того, что даёт сама
    # сеть без всякого вмешательства.
    знак["порог сдвига"] = max(знак["симметрично: разброс"], 1e-6)
    знак["подача на DNa02, Гц"] = DNA02_HZ
    rep.measure("знак вмешательства", знак)

    m = {
        "норма": усл["норма"],
        "цель": усл["цель"],
        "цель: нейронов": int(len(цель)),
        "мимо-цель по имени": усл["мимо-цель по имени"],
        "мимо-цель по имени: нейронов": int(len(off_name)),
        "мимо-цель того же размера": усл["мимо-цель того же размера"],
        "мимо-цель того же размера: нейронов": int(len(off_size)),
        "знак": знак,
        "тождество": тождество,
    }
    rep.measure("замер шлюза D", m)

    rep.gate(
        "D", "моторная причинность: вмешательство в нужные нисходящие "
             "измеримо меняет команду, мимо-цель — нет", m,
        rules=(
            rule("команда следует за раздражителем", r_follows,
                 "разброс > 0 и |r| ≥ 0,40 со сдвигом ≤ 3 τ"),
            rule("подавление НУЖНЫХ нисходящих убивает зависимость",
                 r_ontarget, "разброс ≤ 0,1 нормы и |r| ≤ 0,10"),
            rule("мимо-целевое вмешательство того же эффекта НЕ даёт",
                 r_offtarget, "разброс ≥ 0,5 нормы у каждой мимо-цели"),
            rule("сторона вмешательства задаёт знак сдвига команды",
                 r_sign, "знаки противоположны, сдвиг > собственного "
                         "разброса"),
            rule("зонды названы явным столбцом и сверены по root_id",
                 r_identity, "отпечатки совпали; спор имён виден"),
        ),
        forgeries=(
            forge("команда раздражителю не следует", f_no_follow,
                  "команда следует за раздражителем",
                  "|r| = 0,02: ряд команд есть, к раздражителю "
                  "отношения не имеет"),
            forge("маска на цель ничего не изменила", f_ontarget_alive,
                  "подавление НУЖНЫХ нисходящих убивает зависимость",
                  "команда под маской та же, что в норме: значит она "
                  "приходит не из той популяции, которую подавляли"),
            forge("мимо-цель ломает команду так же", f_offtarget_dead,
                  "мимо-целевое вмешательство того же эффекта НЕ даёт",
                  "ровно это и делает первый контроль бессмысленным: "
                  "если команда умирает от любой маски, она не "
                  "доказывает ничего"),
            forge("сторона знака не задаёт", f_sign_same,
                  "сторона вмешательства задаёт знак сдвига команды",
                  "оба односторонних подавления сдвинули поворот в одну "
                  "сторону: роль не подтверждена"),
            forge("зонд DNa01 подменён парой из другого столбца",
                  f_name_swap,
                  "зонды названы явным столбцом и сверены по root_id",
                  "у обеих пар по 1 нейрону на сторону, обе нисходящие "
                  "и холинергические — размерная проверка проходит "
                  "молча, отличает только отпечаток"),
        ))
    return rep.finish()


def _fp_ids(neurons, idx):
    """Отпечаток выборки по её root_id — тем же способом, что verify_probes.

    Отпечаток, а не размер: подмена `celltype=DNp04` под именем `DNp01`
    даёт те же 1/1 на сторону и размерную проверку проходит молча.
    """
    import hashlib

    root = neurons["root_id"].to_numpy(np.int64)
    ids = np.sort(root[np.asarray(idx, dtype=np.int64)].astype(np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
