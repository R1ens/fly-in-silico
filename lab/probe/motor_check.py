"""Главная проверка моторного пути, и она со ВСТРЕЧНЫМ ЗАМЕРОМ.

Проверяются три разных утверждения, и ни одно из них не проверяется
одним числом.

**А. Прямой зонд видит Giant Fiber, а усреднение — нет.** На детекторы
надвигания подаётся вход, и сравнивается, насколько сдвинулась частота
прямого зонда ``DNp01`` (2 нейрона) и насколько — прежний съём по
полушарию (1299 нейронов). Это не риторика про «растворяется в
среднем», а два числа.

**Б. Моторная команда зависит от активности нисходящих.** Асимметрия
обонятельного входа гоняется расписанием, и считается разброс команды и
её корреляция с раздражителем.

**В. Встречный замер: при маске ``fire`` на съёмную популяцию команда
обязана ПЕРЕСТАТЬ зависеть от раздражителя.** Раздражитель при этом
продолжает меняться ровно так же — это видно по его собственному
разбросу в той же таблице. Если команда следует за раздражителем при
заглушённом съёме, значит она приходит не из сети, и это надо показать
числом: разброс → 0, |r| ≤ 0,1.

И ещё один встречный замер, без которого первый ничего не стоит:
**маска на ПОСТОРОННИЕ нисходящие** (DNp03, DNp10, DNp11 — те, которых в
декодере нет). Если бы команда умирала от любой маски, проверка Б/В
доказывала бы не «команда из сети», а «маска что угодно ломает».

Тело здесь не считается нарочно: физика MuJoCo стоит 21,6 с настенных на
секунду модельного времени, а статистику надо набрать на сотнях шагов
управления. Проверяется связка «сеть → измеритель → декодер → два
числа», и она от тела не зависит: тело получает ровно эти два числа.
Отдельная сверка в теле — `probe/motor_body.py`.

    cd <рабочее дерево>
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_check.py
"""

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
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# ------------------------------------------------------------- движок


class Rig:
    """Сеть без тела: расписание частот на входе, зонды на выходе."""

    def __init__(self, neurons, weights, device, seed=0, readout_pair=None):
        import torch

        from flybrain import data as D, motor, sim

        self.torch = torch
        self.n = len(neurons)
        self.device = device
        self.sim = sim
        self.brain_steps = max(1, int(round(CONTROL_MS / sim.DT)))
        self.block_s = self.brain_steps * sim.DT / 1000.0
        self.dt_s = CONTROL_MS / 1000.0

        self.chan = {}
        for name, sel in (("orn_l", "class:olfactory+side:left"),
                          ("orn_r", "class:olfactory+side:right"),
                          ("loom_l", "type=LPLC2+side:left,type=LC4+side:left"),
                          ("loom_r", "type=LPLC2+side:right,type=LC4+side:right"),
                          # Прямой нейронный стимул на сам рулевой зонд.
                          # Нужен для дозовой кривой: это проверка
                          # связки «активность зонда → команда» без
                          # посредников, и её не обойти.
                          ("dna02_l", "type=DNa02+side:left"),
                          ("dna02_r", "type=DNa02+side:right")):
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


# -------------------------------------------- А. зонд против среднего


def part_a(neurons, weights, device):
    """Видит ли прямой зонд то, чего не видит усреднение по 1299."""
    print("\nА. прямой зонд DNp01 против усреднения по 1299 нисходящим")
    rig = Rig(neurons, weights, device, seed=0)
    n_warm, n_meas = 25, 50

    def run(**hz):
        rig.drive(**hz)
        for _ in range(n_warm):
            rig.step()
        acc = {}
        for _ in range(n_meas):
            rd = rig.step()
            for k, (l, r) in rd.items():
                acc.setdefault(k, []).append((l, r))
        return {k: (float(np.mean([a for a, _ in v])),
                    float(np.mean([b for _, b in v]))) for k, v in acc.items()}

    rest = run()
    loom = run(loom_l=150.0)
    rows = {}
    for k in ("DNp01", "DNp04", "DNa01", "DNa02", "DNp09", "MDN",
              "motor_neck", "DN_all"):
        rows[k] = {
            "покой, Гц Л/П": [round(rest[k][0], 2), round(rest[k][1], 2)],
            "надвигание слева, Гц Л/П": [round(loom[k][0], 2),
                                         round(loom[k][1], 2)],
            "сдвиг слева, Гц": round(loom[k][0] - rest[k][0], 3),
        }
        print("  %-11s покой %7.2f/%7.2f   надвигание %7.2f/%7.2f   "
              "сдвиг слева %+8.2f"
              % (k, rest[k][0], rest[k][1], loom[k][0], loom[k][1],
                 loom[k][0] - rest[k][0]))
    rep["замеры"]["А. зонд против среднего"] = rows

    d_gf = loom["DNp01"][0] - rest["DNp01"][0]
    d_dn = loom["DN_all"][0] - rest["DN_all"][0]
    rep["замеры"]["А. итог"] = {
        "сдвиг прямого зонда DNp01, Гц": round(d_gf, 3),
        "сдвиг прежнего съёма (1299), Гц": round(d_dn, 3),
        "во сколько раз зонд виднее": (round(abs(d_gf / d_dn), 1)
                                       if abs(d_dn) > 1e-9 else None),
    }
    check("прямой зонд DNp01 отвечает на надвигание", d_gf > 20.0,
          "сдвиг %+.2f Гц на 2 нейронах" % d_gf)
    check("прямой зонд сдвинулся заметно сильнее прежнего съёма",
          abs(d_gf) > 10.0 * abs(d_dn) + 1e-9,
          "%.2f Гц против %.2f Гц" % (d_gf, d_dn))
    return rig


# ---------------------------- Б/В. зависимость команды и встречный замер


def sweep(neurons, weights, device, motor_name, cond, weights_path=None,
          readout_sel=None, n_steps=400, seed=0, symmetric=False):
    """Один прогон расписания: условие -> ряд команд и ряд раздражителя.

    ``cond``: ``"норма"``, ``"маска съёма"``, ``"маска посторонних"``.
    Калибровка (перекос и шум каждого зонда) всегда снимается на
    ЦЕЛОЙ сети и до маски: так ставят настоящий опыт — сперва
    измеряют животное, потом выключают звено. Это ещё и строже: перекос
    и шум остаются ненулевыми, и утечка сигнала была бы видна.
    """
    from flybrain import data as D, motor

    pair = None
    if motor_name == "legacy":
        pair = {"__readout__": (readout_sel[0], readout_sel[1])}
    rig = Rig(neurons, weights, device, seed=seed, readout_pair=pair or {})
    shape = motor.DecoderShape(dn_tau_s=TAU_S)
    kw = {"weights_path": weights_path} if motor_name == "dn_trained" else {}
    dec = motor.make_decoder(motor_name, shape=shape, **kw)

    base, depth = 75.0, 0.85
    period_s = 2.5

    # разогрев: симметричный вход, целая сеть
    rig.drive(orn_l=base, orn_r=base)
    warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
    for _ in range(warm):
        dec.observe(rig.step(), rig.dt_s)
    calib = dec.calibrate()
    dec.reset_state()

    if cond == "маска съёма":
        idx = rig.readout.indices()
        if motor_name == "legacy":
            il, ir = rig.readout._idx["__readout__"]
            idx = np.unique(np.concatenate(
                [idx, il.cpu().numpy(), ir.cpu().numpy()]))
        rig.mask_fire(idx)
        masked = len(idx)
    elif cond == "маска посторонних":
        # Нисходящие, которых в декодере нет ни в одном варианте.
        idx = D.select(neurons, "type=DNp03,type=DNp10,type=DNp11")
        rig.mask_fire(idx)
        masked = len(idx)
    else:
        masked = 0

    turn, stim, amp, gf = [], [], [], []
    probe_hz = {}
    for i in range(n_steps):
        t = i * CONTROL_MS / 1000.0
        if symmetric:
            l = r = base
        else:
            m = depth * math.sin(2 * math.pi * t / period_s)
            l, r = base * (1.0 + m), base * (1.0 - m)
        rig.drive(orn_l=l, orn_r=r)
        rd = rig.step()
        cmd = dec.command(rd, rig.dt_s)
        turn.append(cmd.u_turn)
        amp.append(cmd.amp_l - cmd.amp_r)
        stim.append((l - r) / (l + r + 1e-9))
        gf.append(max(rd["DNp01"]))
        for k, (a, b) in rd.items():
            probe_hz.setdefault(k, []).append((a, b))

    turn, stim, amp = np.array(turn), np.array(stim), np.array(amp)
    # Переходный процесс фильтра отбрасывается: первая τ не про сигнал.
    k0 = int(round(TAU_S / (CONTROL_MS / 1000.0)))
    turn, stim, amp = turn[k0:], stim[k0:], amp[k0:]

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "декодер": motor_name,
        "условие": cond,
        "нейронов под маской": masked,
        "шагов в счёте": int(len(turn)),
        "разброс команды (sd cmd_turn)": round(float(np.std(turn)), 6),
        "разброс разницы амплитуд, sd": round(float(np.std(amp)), 6),
        "разброс раздражителя, sd": round(float(np.std(stim)), 6),
        "r(команда, раздражитель)": (None if corr(turn, stim) is None
                                     else round(corr(turn, stim), 4)),
        "r(амплитуды, раздражитель)": (None if corr(amp, stim) is None
                                       else round(corr(amp, stim), 4)),
        "DNp01 средняя, Гц": round(float(np.mean(gf)), 3),
        "зонды, средняя Гц Л/П": {
            k: [round(float(np.mean([a for a, _ in v])), 3),
                round(float(np.mean([b for _, b in v])), 3)]
            for k, v in probe_hz.items()},
        "калибровка зондов": calib,
    }


def part_bv(neurons, weights, device, trained=None):
    print("\nБ/В. зависимость команды от активности и встречный замер")
    rows = []
    jobs = [("dn_linear", "норма", False),
            ("dn_linear", "маска съёма", False),
            ("dn_linear", "маска посторонних", False),
            ("dn_linear", "норма, симметричный вход", True),
            ("legacy", "норма", False),
            ("legacy", "маска съёма", False)]
    if trained is not None:
        jobs += [("dn_trained", "норма", False),
                 ("dn_trained", "маска съёма", False)]
    for name, cond, sym in jobs:
        c = cond.split(",")[0].strip()
        t0 = time.perf_counter()
        r = sweep(neurons, weights, device, name, c,
                  weights_path=trained if name == "dn_trained" else None,
                  readout_sel=("class:ALPN+side:left", "class:ALPN+side:right"),
                  symmetric=sym)
        r["условие"] = cond
        r["по стене, с"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
        print("  %-11s %-26s sd(команда) %.6f  sd(раздраж.) %.4f  "
              "r %s  маска %d нейронов  %.0f с"
              % (name, cond, r["разброс команды (sd cmd_turn)"],
                 r["разброс раздражителя, sd"],
                 r["r(команда, раздражитель)"], r["нейронов под маской"],
                 r["по стене, с"]))
    rep["замеры"]["Б/В. зависимость и встречный замер"] = rows

    by = {(r["декодер"], r["условие"]): r for r in rows}
    for name in ("dn_linear", "legacy", "dn_trained"):
        norm = by.get((name, "норма"))
        mask = by.get((name, "маска съёма"))
        if norm is None or mask is None:
            continue
        # Разброс — это проверка: команда обязана шевелиться. А вот
        # величина r с ОБОНЯТЕЛЬНЫМ раздражителем проверкой быть не
        # может, и ставить её сюда было бы подгонкой под желаемое:
        # доносят ли нисходящие сторону запаха — это ЗАМЕР, и его
        # результат может быть отрицательным. Он и выходит
        # отрицательным у прямых зондов, и так и записан.
        check("%s: команда в норме шевелится" % name,
              norm["разброс команды (sd cmd_turn)"] > 1e-4,
              "sd %.6f" % norm["разброс команды (sd cmd_turn)"])
        rep["замеры"].setdefault("связь с обонятельным раздражителем", {})[
            name] = {
            "r(команда, раздражитель)": norm["r(команда, раздражитель)"],
            "sd команды": norm["разброс команды (sd cmd_turn)"],
            "DNp01 средняя, Гц": norm["DNp01 средняя, Гц"],
        }
        r_m = mask["r(команда, раздражитель)"]
        check("%s: ВСТРЕЧНЫЙ ЗАМЕР — при маске съёма разброс -> 0" % name,
              mask["разброс команды (sd cmd_turn)"] < 1e-6,
              "sd %.3g при sd раздражителя %.4f"
              % (mask["разброс команды (sd cmd_turn)"],
                 mask["разброс раздражителя, sd"]))
        check("%s: ВСТРЕЧНЫЙ ЗАМЕР — при маске съёма |r| <= 0,1" % name,
              r_m is None or abs(r_m) <= 0.1,
              "r = %s" % ("не определена (команда постоянна)"
                          if r_m is None else "%.4f" % r_m))
    other = by.get(("dn_linear", "маска посторонних"))
    norm = by.get(("dn_linear", "норма"))
    if other and norm:
        check("ВСТРЕЧНЫЙ ЗАМЕР к встречному: маска ПОСТОРОННИХ "
              "нисходящих команду не убивает",
              other["разброс команды (sd cmd_turn)"] > 1e-4
              and other["r(команда, раздражитель)"] is not None,
              "sd %.6f против %.6f в норме"
              % (other["разброс команды (sd cmd_turn)"],
                 norm["разброс команды (sd cmd_turn)"]))
    sym = by.get(("dn_linear", "норма, симметричный вход"))
    if sym:
        r_s = sym["r(команда, раздражитель)"]
        check("при симметричном входе команда шумит, но раздражителя нет",
              sym["разброс команды (sd cmd_turn)"] > 1e-5 and r_s is None,
              "sd %.6f, r %s" % (sym["разброс команды (sd cmd_turn)"],
                                 r_s))


def part_g(neurons, weights, device):
    """Дозовая кривая: активность ЗОНДА -> команда, и то же под маской.

    Это и есть прямая проверка требования «моторный выход обязан
    зависеть от вычисленных нейронных сигналов», и она не опирается на
    то, доносит ли обонятельный вход сторону до нисходящих. На сам
    рулевой зонд DNa02 подаётся прямой нейронный стимул возрастающей
    силы, только слева, и смотрится команда.

    Встречный замер здесь в той же таблице: тот же стимул при маске
    ``fire`` на зонды. Стимул на входе тот же, спайков у зонда нет —
    команда обязана остаться на опорном значении.
    """
    from flybrain import motor

    print("\nГ. дозовая кривая: прямой стимул на DNa02 слева -> команда")
    rows = []
    for cond in ("норма", "маска съёма"):
        rig = Rig(neurons, weights, device, seed=0)
        dec = motor.make_decoder("dn_linear",
                                 shape=motor.DecoderShape(dn_tau_s=TAU_S))
        rig.drive()
        warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
        for _ in range(warm):
            dec.observe(rig.step(), rig.dt_s)
        dec.calibrate()
        dec.reset_state()
        if cond == "маска съёма":
            rig.mask_fire(rig.readout.indices())
        pts = []
        for hz in (0.0, 10.0, 25.0, 50.0, 100.0, 200.0):
            rig.drive(dna02_l=hz)
            # Держим стимул несколько τ, потом берём среднее по хвосту:
            # у сглаживания τ = 0,5 с, и мерить сразу значит мерить
            # переходный процесс фильтра.
            n_hold = int(round(3 * TAU_S / rig.dt_s))
            turn, amps, got = [], [], []
            for i in range(n_hold):
                rd = rig.step()
                cmd = dec.command(rd, rig.dt_s)
                if i >= n_hold // 2:
                    turn.append(cmd.u_turn)
                    amps.append(cmd.amp_l - cmd.amp_r)
                    got.append(rd["DNa02"][0])
            pts.append({"стимул, Гц": hz,
                        "DNa02 слева, Гц": round(float(np.mean(got)), 2),
                        "cmd_turn": round(float(np.mean(turn)), 5),
                        "amp_l - amp_r": round(float(np.mean(amps)), 5)})
            print("  %-12s стимул %6.1f Гц -> DNa02 %7.2f Гц -> "
                  "cmd_turn %+8.5f, разница амплитуд %+8.5f"
                  % (cond, hz, pts[-1]["DNa02 слева, Гц"],
                     pts[-1]["cmd_turn"], pts[-1]["amp_l - amp_r"]))
        span = (max(p["cmd_turn"] for p in pts)
                - min(p["cmd_turn"] for p in pts))
        rows.append({"условие": cond, "точки": pts,
                     "размах cmd_turn": round(span, 6)})
    rep["замеры"]["Г. дозовая кривая"] = rows
    norm, mask = rows[0], rows[1]
    check("команда зависит от активности зонда (дозовая кривая)",
          norm["размах cmd_turn"] > 0.1,
          "размах cmd_turn %.5f при стимуле 0…200 Гц"
          % norm["размах cmd_turn"])
    # Монотонность по возрастанию активности зонда — отдельно от размаха:
    # размах могла дать и одна случайная точка.
    t = [p["cmd_turn"] for p in norm["точки"]]
    mono = all(t[i + 1] >= t[i] - 1e-9 for i in range(len(t) - 1)) or \
        all(t[i + 1] <= t[i] + 1e-9 for i in range(len(t) - 1))
    check("дозовая кривая монотонна", mono, "cmd_turn: %s"
          % ", ".join("%+.4f" % v for v in t))
    check("ВСТРЕЧНЫЙ ЗАМЕР — под маской тот же стимул команду не двигает",
          mask["размах cmd_turn"] < 1e-9,
          "размах %.3g против %.5f в норме"
          % (mask["размах cmd_turn"], norm["размах cmd_turn"]))


def main():
    import torch

    from flybrain import data as D, motor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("устройство:", device)

    print("\n0. прямые зонды: существование и размер")
    rows = motor.verify_probes(neurons)
    rep["замеры"]["прямые зонды"] = rows
    for r in rows:
        print("  %-11s %-32s %3s/%-3s  ожидалось %s  %s"
              % (r["зонд"], r["селектор"], r.get("слева"), r.get("справа"),
                 r.get("ожидалось на сторону"),
                 "ok" if r["совпало"] else "ПРОВАЛ"))
    check("все прямые зонды нашлись и сошлись по размеру",
          all(r["совпало"] for r in rows), "%d зондов" % len(rows))

    part_a(neurons, weights, device)
    part_g(neurons, weights, device)

    trained = OUT / "motor_decoder_v1.npz"
    use_trained = None
    if trained.exists():
        # Пригодность файла весов проверяется ДО прогона, иначе проба
        # падала бы посередине. `DnTrainedDecoder` отказывается от файла,
        # не записавшего свою нормировку: веса применились бы в другом
        # масштабе молча. Отказ показывается, а не обходится.
        try:
            motor.make_decoder("dn_trained",
                               shape=motor.DecoderShape(dn_tau_s=TAU_S),
                               weights_path=str(trained))
            use_trained = str(trained)
        except (ValueError, FileNotFoundError) as e:
            print("\n  обученный декодер в проверку НЕ взят: %s" % e)
            rep["замеры"]["dn_trained пропущен"] = str(e)
    part_bv(neurons, weights, device, trained=use_trained)

    (OUT / "motor_check.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
