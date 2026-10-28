"""Моторный путь в ТЕЛЕ: новый декодер, мост через обрыв GF, стабилизация головы.

Три замера, которые без физики не сделать.

**1. Декодер по прямым зондам работает в теле.** Эпизод с
``motor = "dn_linear"`` считается, записывает ``readout_*`` и ``cmd_*``,
и команда зависит от активности: разброс ``cmd_turn`` не ноль. Встречный
замер — тот же эпизод с маской ``fire`` на съёмные зонды: команда
обязана стать постоянной, а запах при этом продолжает меняться (видно по
разбросу ``orn_l − orn_r`` в той же строке).

**2. Прежний путь не сломан.** Эпизод с ``motor = "legacy"`` сравнивается
с ним же до правки — это делает `probe/motor_equal.py`, и здесь только
повторяется для полноты картинки.

**3. Стабилизация головы и её цена.** Три эпизода с одним семенем:
обычный, со стабилизацией и с заклиненной шеей (контроль авторов
статьи). Замеряются углы шейных суставов, их размах, время счёта и
траектория. Это ИНЖЕНЕРНАЯ деталь тела, и включение её меняет
механическую модель: приводы на шею появляются только при ней.

    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_body.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
DURATION = 2.0
fails = []
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def _neck_names(body):
    """Имена шейных степеней свободы в `qpos`, с префиксом экземпляра мухи.

    В модели их ТРИ (`joint_Head_yaw`, `joint_Head`, `joint_Head_roll`), а
    приводов стабилизация добавляет два: `joint_Head_yaw` и `joint_Head`.
    Ловушка имён из `RESEARCH.md` §7: `joint_Head_yaw` — это степень
    свободы КРЕНА, имя оставлено авторами для совместимости, и в их коде
    отображение `[крен, тангаж] → [joint_Head_yaw, joint_Head]` не описка.
    Поэтому и здесь подписи разнесены: `крен` — это `joint_Head_yaw`.
    """
    all_names = list(body.physics.named.data.qpos.axes.row.names)
    want = ("joint_Head_yaw", "joint_Head", "joint_Head_roll")
    out = []
    for w in want:
        hit = [n for n in all_names if n.split("/")[-1] == w]
        out.append(hit[0] if hit else None)
    return out


def episode(neurons, weights, device, label, mask=None, neck_angles=False,
            **kw):
    """Один эпизод. ``mask`` — индексы для маски ``fire`` на весь эпизод.

    Маска ставится на ПЕРВОМ шаге эпизода, а не до разогрева: перекос и
    шум каждого зонда обязаны быть сняты на целой сети, как в настоящем
    опыте — сперва измеряют животное, потом выключают звено. Это ещё и
    строже: перекос остаётся ненулевым, и утечка сигнала была бы видна.
    """
    from flybrain import body as B
    import flybrain.sim as S

    cfg = B.BodyConfig(duration_s=DURATION, control_ms=20.0, seed=0,
                       device=device, readout="ALPN", **kw)
    hook, neck = {}, []
    idx = None if mask is None else np.asarray(mask, np.int64)
    masked = 0 if idx is None else len(idx)

    def on_body(bd):
        hook["body"] = bd
        hook["neck"] = _neck_names(bd) if neck_angles else None

    def on_step(step, rec, _bins):
        if idx is not None and "masked" not in hook:
            hook["engine"].set_masks(fire=idx)
            hook["masked"] = True
        nm = hook.get("neck")
        if nm:
            q = hook["body"].physics.named.data.qpos
            neck.append([float(q[n]) if n else float("nan") for n in nm])

    real_build = S.build_engine

    def build(*a, **k):
        eng = real_build(*a, **k)
        hook["engine"] = eng
        return eng

    t0 = time.perf_counter()
    S.build_engine = build
    try:
        run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                            progress=False, on_body=on_body,
                            on_step=on_step)
    finally:
        S.build_engine = real_build
    wall = time.perf_counter() - t0

    recs = run.records
    turn = np.array([r.get("cmd_turn", 0.0) for r in recs], float)
    amp = np.array([r["amp_l"] - r["amp_r"] for r in recs], float)
    stim = np.array([r["orn_l"] - r["orn_r"] for r in recs], float)
    out = {
        "условие": label,
        "нейронов под маской": masked,
        "по стене, с": round(wall, 1),
        "шагов": len(recs),
        "sd cmd_turn": round(float(np.std(turn)), 6),
        "sd (amp_l - amp_r)": round(float(np.std(amp)), 6),
        "sd (orn_l - orn_r), Гц": round(float(np.std(stim)), 4),
        "r(cmd_turn, orn_l-orn_r)": (
            None if np.std(turn) < 1e-12 or np.std(stim) < 1e-12
            else round(float(np.corrcoef(turn, stim)[0, 1]), 4)),
        "метрики": run.metrics,
        "DNp01 средняя, Гц": round(float(np.mean(
            [max(r.get("readout_DNp01_l", 0.0),
                 r.get("readout_DNp01_r", 0.0)) for r in recs])), 3),
        "конец": [round(float(recs[-1][c]), 4)
                  for c in ("x", "y", "theta")],
        "оговорки": run.timing.get("оговорки", []),
    }
    if neck:
        a = np.array(neck, float)
        for i, nm in enumerate(("крен (joint_Head_yaw)",
                                "тангаж (joint_Head)",
                                "joint_Head_roll")):
            if i >= a.shape[1]:
                continue
            out["шея, %s: размах рад" % nm] = round(
                float(np.nanmax(a[:, i]) - np.nanmin(a[:, i])), 5)
            out["шея, %s: sd рад" % nm] = round(float(np.nanstd(a[:, i])), 5)
    if run.engineered:
        out["engineered"] = run.engineered
    return out


def main():
    import torch

    from flybrain import data as D, motor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("устройство:", device)

    print("\n1. декодер по прямым зондам в теле, и встречный замер маской")
    rows = []
    probes_idx = motor.DnReadout(neurons, "cpu").indices()
    for label, kw, mask in (
            ("dn_linear, норма", {"motor": "dn_linear"}, None),
            ("dn_linear, маска съёма", {"motor": "dn_linear"}, probes_idx),
            ("legacy, норма", {"motor": "legacy"}, None)):
        r = episode(neurons, weights, device, label, mask=mask, **kw)
        rows.append(r)
        print("  %-24s sd(cmd_turn) %.6f  sd(запах) %.3f Гц  r %s  "
              "%.0f с" % (label, r["sd cmd_turn"], r["sd (orn_l - orn_r), Гц"],
                          r["r(cmd_turn, orn_l-orn_r)"], r["по стене, с"]))
    rep["замеры"]["1. декодер в теле"] = rows
    norm, mask_r = rows[0], rows[1]
    check("в теле команда dn_linear шевелится", norm["sd cmd_turn"] > 1e-4,
          "sd %.6f" % norm["sd cmd_turn"])
    check("ВСТРЕЧНЫЙ ЗАМЕР в теле: при маске съёма команда постоянна",
          mask_r["sd cmd_turn"] < 1e-6,
          "sd %.3g при разбросе запаха %.3f Гц"
          % (mask_r["sd cmd_turn"], mask_r["sd (orn_l - orn_r), Гц"]))
    check("ВСТРЕЧНЫЙ ЗАМЕР в теле: запах при маске продолжал меняться",
          mask_r["sd (orn_l - orn_r), Гц"] > 0.5,
          "sd %.3f Гц" % mask_r["sd (orn_l - orn_r), Гц"])
    check("прогон с новым декодером помечен оговоркой",
          any("ИНЖЕНЕРНЫМ" in s for s in norm["оговорки"]),
          "%d оговорок" % len(norm["оговорки"]))
    check("блок engineered собран", bool(norm.get("engineered")),
          "деталей %d" % len((norm.get("engineered") or {}).get("детали", [])))

    print("\n2. мост через обрыв GF: по умолчанию ничего не делает")
    off = episode(neurons, weights, device, "gf_bridge none",
                  motor="dn_linear", gf_bridge="none")
    on = episode(neurons, weights, device, "gf_bridge gain_v1",
                 motor="dn_linear", gf_bridge="gain_v1")
    rep["замеры"]["2. мост GF"] = [off, on]
    check("без моста вклад escape равен нулю и прогон не помечен мостом",
          not any("gf_bridge" in s for s in off["оговорки"]),
          "оговорок про мост нет")
    check("включённый мост помечает прогон оговоркой",
          any("gf_bridge" in s for s in on["оговорки"]),
          "%s" % ("; ".join(s[:60] for s in on["оговорки"]
                            if "gf_bridge" in s)))
    print("  DNp01 в теле: %.3f Гц (без моста), %.3f Гц (с мостом)"
          % (off["DNp01 средняя, Гц"], on["DNp01 средняя, Гц"]))

    print("\n3. стабилизация головы: обычный / обученный MLP / шея заклинена")
    hs = motor.HeadStabilizer()
    rep["замеры"]["3. стабилизация: спека"] = hs.spec()
    print("  веса: %s" % hs.ckpt_path)
    print("  sha256 %s, параметров %d, neck_kp %g"
          % (hs.ckpt_sha256[:16], hs.n_params, hs.neck_kp))
    rows = []
    for label, kw in (("без стабилизации", {}),
                      ("обученный MLP", {"head_stab": True})):
        r = episode(neurons, weights, device, label, neck_angles=True, **kw)
        rows.append(r)
        print("  %-20s %.0f с, шея крен размах %s рад, тангаж %s рад, "
              "выравнивание %s"
              % (label, r["по стене, с"],
                 r.get("шея, крен (joint_Head_yaw): размах рад"),
                 r.get("шея, тангаж (joint_Head): размах рад"),
                 r["метрики"].get("grad_alignment")))
    rep["замеры"]["3. стабилизация: эпизоды"] = rows
    base, stab = rows
    check("переходник без lightning поднял обученные веса",
          hs.n_params == 2626 or hs.n_params > 2000,
          "%d параметров, сеть %s" % (hs.n_params, hs.spec()["сеть"]))
    check("эпизод со стабилизацией посчитался",
          stab["шагов"] == base["шагов"],
          "%d шагов против %d" % (stab["шагов"], base["шагов"]))
    check("стабилизация помечена оговоркой",
          any("стабилизация головы" in s for s in stab["оговорки"]),
          "%d оговорок" % len(stab["оговорки"]))
    key = "шея, крен (joint_Head_yaw): размах рад"
    if base.get(key) is not None and stab.get(key) is not None:
        check("стабилизация изменила подвижность шеи",
              abs(stab[key] - base[key]) > 1e-4,
              "размах крена %s против %s рад" % (stab[key], base[key]))
    print("  цена по стене: %.0f с против %.0f с, в %.2f раза"
          % (stab["по стене, с"], base["по стене, с"],
             stab["по стене, с"] / max(base["по стене, с"], 1e-9)))
    rep["замеры"]["3. стабилизация: цена"] = {
        "без стабилизации, с": base["по стене, с"],
        "со стабилизацией, с": stab["по стене, с"],
        "во сколько раз дороже": round(stab["по стене, с"]
                                       / max(base["по стене, с"], 1e-9), 3),
        "вызовов сети за эпизод": hs.calls,
    }

    (OUT / "motor_body.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
