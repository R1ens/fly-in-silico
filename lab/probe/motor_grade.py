"""Градуальность моторной команды, негативные контроли и зёрна.

Эта проба добавляет к `motor_check.py` то, чего в нём нет, и перемеряет
то, что в нём записано по старому коду.

**Зачем вообще.** `motor_check.py` проверяет дозовую кривую двумя
условиями: размах больше 0,1 и монотонность. Оба проходят у СТУПЕНЬКИ:
ступенька монотонна (нестрого), и размах у неё максимальный. То есть эта
пара проверок не отличает декодер от реле, а реле — это ровно то, чем
декодер и был до правки `norm="rate"`: в записанном `motor_check.json`
стимул 10 Гц даёт `DNa02` 5,26 Гц и сразу `cmd_turn` +0,556, и дальше до
200 Гц команда не меняется ни на единицу последнего знака, хотя
активность зонда растёт в 38 раз.

Поэтому здесь дозовая кривая меряется часто, и у неё считается ТРЕТЬЯ
величина, которой у ступеньки быть не может: сколько различных значений
команда принимает и какую долю размаха даёт середина диапазона. Реле
даёт два значения и ноль в середине; градуальный декодер — много.

**Негативные контроли.** Их три, и ни один не проходит на пустом
механизме:

1. маска ``fire`` на съёмную популяцию: разброс команды → 0, |r| с
   раздражителем ≤ 0,1. Если команда следует за раздражителем при
   заглушённом съёме, она приходит не из сети;
2. маска на ПОСТОРОННИЕ нисходящие: команда обязана ВЫЖИТЬ. Иначе
   первый контроль доказывал бы не «команда из сети», а «маска ломает
   что угодно»;
3. зёрна: то же зерно — побитово тот же ряд команд; другое зерно —
   другой. Иначе «воспроизводимость» означала бы, что команда
   вообще не зависит от состояния сети.

Раздражитель обонятельный и прямой: зрительный путь до LPLC2/DNp01 в
v783 не доходит (`VISION.md` §5), поэтому проверять моторный декодер
зрительным раздражителем нельзя.

    cd <рабочее дерево>
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_grade.py
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
CONTROL_MS = 20.0
TAU_S = 0.5

fails = []
rep = {"проверки": {}, "замеры": {}, "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def fp(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()
    ).hexdigest()[:16]


class Rig:
    """Сеть без тела. Раздражитель задаётся частотами, съём — зондами.

    Тела здесь нет нарочно: статистику надо набирать на сотнях шагов
    управления, а физика MuJoCo стоит порядка 20 с настенных на секунду
    модельного времени. Проверяется связка «сеть → измеритель →
    декодер → два числа», и она от тела не зависит: тело получает ровно
    эти два числа.
    """

    def __init__(self, neurons, weights, device, seed=0, readout_pair=None):
        import torch

        from flybrain import data as D, motor, sim

        self.torch = torch
        self.n = len(neurons)
        self.device = device
        self.brain_steps = max(1, int(round(CONTROL_MS / sim.DT)))
        self.block_s = self.brain_steps * sim.DT / 1000.0
        self.dt_s = CONTROL_MS / 1000.0
        # Раздражители берутся по СТОЛБЦУ, а не легаси-ключом `type=`:
        # для научной выборки объединение трёх столбцов — молчаливое
        # слияние популяций. Для LC4 и LPLC2 они сегодня совпадают
        # (проверено `motor_audit.py`), но полагаться на это незачем.
        self.chan = {}
        for name, sel in (
                ("orn_l", "class:olfactory+side:left"),
                ("orn_r", "class:olfactory+side:right"),
                ("dna02_l", "celltype=DNa02+side:left"),
                ("dna02_r", "celltype=DNa02+side:right"),
                ("mdn_l", "celltype=MDN+side:left"),
        ):
            idx = D.select(neurons, sel)
            self.chan[name] = (idx, torch.as_tensor(idx, device=device))
        driven = np.unique(np.concatenate([v[0] for v in self.chan.values()]))
        self.engine = sim.build_engine(weights=weights, neurons=neurons,
                                       device=device, seed=seed,
                                       no_refractory=driven)
        self.readout = motor.DnReadout(neurons, device,
                                       pairs=readout_pair or {})
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

    def mask_fire(self, idx):
        self.engine.set_masks(fire=np.asarray(idx, np.int64))


def warmed(neurons, weights, device, seed=0, pair=None, decoder="dn_linear",
           weights_path=None):
    """Рига с откалиброванным декодером. Калибровка — ДО любой маски."""
    from flybrain import motor

    rig = Rig(neurons, weights, device, seed=seed, readout_pair=pair or {})
    kw = {"weights_path": weights_path} if decoder == "dn_trained" else {}
    dec = motor.make_decoder(decoder,
                             shape=motor.DecoderShape(dn_tau_s=TAU_S), **kw)
    return rig, dec


# ------------------------------------------- 1. дозовая кривая, часто


def part_dose(neurons, weights, device):
    """Активность зонда -> команда. Главное: СКОЛЬКО значений она даёт."""
    print("\n1. дозовая кривая: прямой стимул на DNa02 слева", flush=True)
    levels = (0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0,
              140.0, 200.0)
    rows = []
    for cond in ("норма", "маска съёма"):
        rig, dec = warmed(neurons, weights, device)
        rig.drive()
        warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
        for _ in range(warm):
            dec.observe(rig.step(), rig.dt_s)
        dec.calibrate()
        dec.reset_state()
        if cond == "маска съёма":
            rig.mask_fire(rig.readout.indices())
        pts = []
        for hz in levels:
            rig.drive(dna02_l=hz)
            n_hold = int(round(3 * TAU_S / rig.dt_s))
            turn, amps, got, zs = [], [], [], []
            for i in range(n_hold):
                rd = rig.step()
                cmd = dec.command(rd, rig.dt_s)
                if i >= n_hold // 2:
                    turn.append(cmd.u_turn)
                    amps.append(cmd.amp_l - cmd.amp_r)
                    got.append(rd["DNa02"][0] - rd["DNa02"][1])
                    zs.append({k: round(v, 4) for k, v in cmd.z.items()})
            pts.append({
                "стимул, Гц": hz,
                "DNa02 Л-П, Гц": round(float(np.mean(got)), 2),
                "cmd_turn": round(float(np.mean(turn)), 6),
                "amp_l - amp_r": round(float(np.mean(amps)), 6),
                "z последнего шага": zs[-1] if zs else {},
            })
            print("  %-12s стимул %6.1f -> DNa02 Л-П %7.2f Гц -> "
                  "cmd_turn %+9.6f" % (cond, hz, pts[-1]["DNa02 Л-П, Гц"],
                                       pts[-1]["cmd_turn"]), flush=True)
        t = np.array([p["cmd_turn"] for p in pts])
        a = np.array([p["DNa02 Л-П, Гц"] for p in pts])
        span = float(t.max() - t.min())
        # Различных значений команды. У реле их два.
        uniq = int(len(np.unique(np.round(t, 6))))
        # Доля размаха, которую даёт СЕРЕДИНА диапазона активности. У
        # ступеньки она равна нулю: все значения либо внизу, либо вверху.
        inner = t[(a > 0.02 * a.max()) & (a < 0.98 * a.max())]
        frac = (float((inner.max() - inner.min()) / span)
                if span > 1e-12 and len(inner) > 1 else 0.0)
        rows.append({"условие": cond, "точки": pts,
                     "размах cmd_turn": round(span, 6),
                     "различных значений команды": uniq,
                     "доля размаха из середины диапазона": round(frac, 4)})
    rep["замеры"]["1. дозовая кривая"] = rows
    norm, mask = rows[0], rows[1]

    check("команда зависит от активности зонда: размах есть",
          norm["размах cmd_turn"] > 0.1,
          "размах %.5f" % norm["размах cmd_turn"])
    # Вот эта проверка и отличает декодер от реле.
    check("кривая ГРАДУАЛЬНА, а не ступенька "
          "(больше 4 различных значений)",
          norm["различных значений команды"] > 4,
          "%d различных значений на %d точках"
          % (norm["различных значений команды"], len(levels)))
    check("середина диапазона активности даёт больше половины размаха",
          norm["доля размаха из середины диапазона"] > 0.5,
          "доля %.3f" % norm["доля размаха из середины диапазона"])
    # Монотонность проверяется по ИЗМЕРЕННОЙ АКТИВНОСТИ зонда, а не по
    # заданному стимулу, и это не придирка. Связка «частота стимула →
    # частота DNa02» у самой сети НЕ монотонна: замер даёт при стимуле
    # 60 Гц активность 50,00 Гц, а при 80 Гц — 44,74 Гц, то есть меньше.
    # Декодер обязан следовать за тем, что измерено, а не за тем, что
    # подано; требовать от него монотонности по стимулу значило бы
    # требовать, чтобы он ИСПРАВЛЯЛ сеть.
    pts = sorted(norm["точки"], key=lambda p: p["DNa02 Л-П, Гц"])
    t_act = [p["cmd_turn"] for p in pts]
    mono = all(t_act[i + 1] >= t_act[i] - 1e-9
               for i in range(len(t_act) - 1))
    check("команда монотонно НЕ УБЫВАЕТ по измеренной активности зонда",
          mono, "по возрастанию DNa02: %s"
          % ", ".join("%+.4f" % v for v in t_act))
    t = [p["cmd_turn"] for p in norm["точки"]]
    mono_stim = all(t[i + 1] >= t[i] - 1e-9 for i in range(len(t) - 1))
    if not mono_stim:
        bad = [(pp["стимул, Гц"], pp["DNa02 Л-П, Гц"])
               for i, pp in enumerate(norm["точки"][1:], 1)
               if pp["cmd_turn"] < norm["точки"][i - 1]["cmd_turn"] - 1e-9]
        note("по ЗАДАННОМУ стимулу кривая не монотонна, и виновата "
             "сеть, а не декодер: у самой связки «стимул → активность "
             "DNa02» есть провалы (%s). Команда при этом монотонна по "
             "активности — то есть следует за измерением, а не за "
             "подачей"
             % "; ".join("стимул %g Гц даёт %g Гц" % b for b in bad))
    # Где она упирается в потолок: probe_ref_hz = 100 Гц объявлен.
    sat = [p["стимул, Гц"] for p in norm["точки"]
           if abs(p["cmd_turn"] - max(t)) < 1e-9]
    note("команда выходит на потолок при стимуле %g Гц и выше: "
         "probe_ref_hz = 100 Гц объявлен, и всё, что даёт разницу "
         "DNa02 больше 100 Гц, обрезается" % (min(sat) if sat else -1))

    check("ВСТРЕЧНЫЙ ЗАМЕР — под маской тот же стимул команду не двигает",
          mask["размах cmd_turn"] < 1e-9,
          "размах %.3g против %.6f в норме"
          % (mask["размах cmd_turn"], norm["размах cmd_turn"]))
    return rows


# ------------------------------- 2. негативные контроли на раздражителе


def sweep(neurons, weights, device, decoder, cond, weights_path=None,
          n_steps=400, seed=0):
    """Расписание асимметрии запаха -> ряд команд. Калибровка до маски."""
    from flybrain import data as D

    pair = ({"__readout__": ("class:ALPN+side:left",
                             "class:ALPN+side:right")}
            if decoder == "legacy" else None)
    rig, dec = warmed(neurons, weights, device, seed=seed, pair=pair,
                      decoder=decoder, weights_path=weights_path)
    base, depth, period_s = 75.0, 0.85, 2.5
    rig.drive(orn_l=base, orn_r=base)
    warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
    for _ in range(warm):
        dec.observe(rig.step(), rig.dt_s)
    calib = dec.calibrate()
    dec.reset_state()

    masked = 0
    if cond == "маска съёма":
        idx = rig.readout.indices()
        if decoder == "legacy":
            il, ir = rig.readout._idx["__readout__"]
            idx = np.unique(np.concatenate(
                [idx, il.cpu().numpy(), ir.cpu().numpy()]))
        rig.mask_fire(idx)
        masked = len(idx)
    elif cond == "маска посторонних":
        # Нисходящие, которых нет ни в одном варианте декодера.
        idx = D.select(neurons, "celltype=DNp03,celltype=DNp10,"
                                "celltype=DNp11")
        rig.mask_fire(idx)
        masked = len(idx)

    turn, stim = [], []
    for i in range(n_steps):
        t = i * CONTROL_MS / 1000.0
        m = depth * math.sin(2 * math.pi * t / period_s)
        l, r = base * (1.0 + m), base * (1.0 - m)
        rig.drive(orn_l=l, orn_r=r)
        cmd = dec.command(rig.step(), rig.dt_s)
        turn.append(cmd.u_turn)
        stim.append((l - r) / (l + r + 1e-9))
    k0 = int(round(TAU_S / (CONTROL_MS / 1000.0)))
    turn, stim = np.array(turn[k0:]), np.array(stim[k0:])

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    r = corr(turn, stim)
    return {
        "декодер": decoder, "условие": cond,
        "нейронов под маской": masked, "шагов в счёте": int(len(turn)),
        "sd команды": round(float(np.std(turn)), 8),
        "sd раздражителя": round(float(np.std(stim)), 6),
        "r(команда, раздражитель)": None if r is None else round(r, 4),
        "отпечаток ряда команд": fp(turn),
        "калибровка": {k: v for k, v in calib.items()
                       if k in ("DNa02", "__readout__")},
    }


def part_neg(neurons, weights, device, trained=None):
    print("\n2. негативные контроли на обонятельном раздражителе",
          flush=True)
    jobs = [("dn_linear", "норма"), ("dn_linear", "маска съёма"),
            ("dn_linear", "маска посторонних"),
            ("legacy", "норма"), ("legacy", "маска съёма")]
    if trained:
        jobs += [("dn_trained", "норма"), ("dn_trained", "маска съёма")]
    rows = []
    for dec_name, cond in jobs:
        t0 = time.perf_counter()
        r = sweep(neurons, weights, device, dec_name, cond,
                  weights_path=trained if dec_name == "dn_trained" else None)
        r["по стене, с"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
        print("  %-11s %-18s sd(команда) %.8f  sd(раздраж.) %.4f  r %-8s "
              "маска %4d  %.0f с"
              % (dec_name, cond, r["sd команды"], r["sd раздражителя"],
                 r["r(команда, раздражитель)"], r["нейронов под маской"],
                 r["по стене, с"]), flush=True)
    rep["замеры"]["2. негативные контроли"] = rows
    by = {(r["декодер"], r["условие"]): r for r in rows}

    for name in ("dn_linear", "legacy", "dn_trained"):
        norm, mask = by.get((name, "норма")), by.get((name, "маска съёма"))
        if not norm or not mask:
            continue
        check("%s: команда в норме шевелится" % name,
              norm["sd команды"] > 1e-4, "sd %.8f" % norm["sd команды"])
        check("%s: НЕГАТИВНЫЙ КОНТРОЛЬ — при маске съёма разброс -> 0"
              % name, mask["sd команды"] < 1e-6,
              "sd %.3g при sd раздражителя %.4f (раздражитель меняется "
              "ровно так же)" % (mask["sd команды"], mask["sd раздражителя"]))
        rm = mask["r(команда, раздражитель)"]
        check("%s: НЕГАТИВНЫЙ КОНТРОЛЬ — при маске съёма |r| <= 0,1" % name,
              rm is None or abs(rm) <= 0.1,
              "r = %s" % ("не определена (команда постоянна)" if rm is None
                          else "%.4f" % rm))
    o, nm = by.get(("dn_linear", "маска посторонних")), by.get(
        ("dn_linear", "норма"))
    if o and nm:
        check("КОНТРОЛЬ К КОНТРОЛЮ — маска ПОСТОРОННИХ нисходящих "
              "команду не убивает", o["sd команды"] > 1e-4,
              "sd %.6f против %.6f в норме; иначе первый контроль "
              "доказывал бы лишь, что маска ломает что угодно"
              % (o["sd команды"], nm["sd команды"]))
    return rows


# ---------------------------------------------- 3. зёрна и побитовость


def part_seeds(neurons, weights, device):
    print("\n3. воспроизводимость по зёрнам", flush=True)
    a = sweep(neurons, weights, device, "dn_linear", "норма", n_steps=150,
              seed=0)
    b = sweep(neurons, weights, device, "dn_linear", "норма", n_steps=150,
              seed=0)
    c = sweep(neurons, weights, device, "dn_linear", "норма", n_steps=150,
              seed=1)
    check("то же зерно -> ПОБИТОВО тот же ряд команд",
          a["отпечаток ряда команд"] == b["отпечаток ряда команд"],
          "отпечаток %s дважды" % a["отпечаток ряда команд"])
    # Без этого встречного замера «воспроизводимость» означала бы, что
    # команда вообще не зависит от состояния сети.
    check("ВСТРЕЧНЫЙ ЗАМЕР — другое зерно даёт ДРУГОЙ ряд",
          a["отпечаток ряда команд"] != c["отпечаток ряда команд"],
          "зерно 0: %s, зерно 1: %s"
          % (a["отпечаток ряда команд"], c["отпечаток ряда команд"]))
    rep["замеры"]["3. зёрна"] = {
        "зерно 0, прогон 1": a["отпечаток ряда команд"],
        "зерно 0, прогон 2": b["отпечаток ряда команд"],
        "зерно 1": c["отпечаток ряда команд"],
        "sd зерно 0": a["sd команды"], "sd зерно 1": c["sd команды"],
        "r зерно 0": a["r(команда, раздражитель)"],
        "r зерно 1": c["r(команда, раздражитель)"],
    }


def main():
    import torch

    from flybrain import data as D

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("устройство: %s, нейронов %d" % (device, len(neurons)), flush=True)

    part_dose(neurons, weights, device)
    trained = OUT / "motor_decoder_v1.npz"
    part_neg(neurons, weights, device,
             trained=str(trained) if trained.exists() else None)
    part_seeds(neurons, weights, device)

    rep["секунд"] = round(time.perf_counter() - t0, 1)
    (OUT / "motor_grade.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d (%.0f с)"
          % (len(rep["проверки"]), len(fails), rep["секунд"]), flush=True)
    for f in fails:
        print("  - " + f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
