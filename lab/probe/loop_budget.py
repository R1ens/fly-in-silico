r"""Полный контур по составляющим: физика, мозг, зрение, остальное.

`perf_budget.py` разложил блок «физика + контроллер». Здесь меряется
**весь шаг управления** настоящего эпизода — с коннектомом, со зрением
и с моторным путём, — и раскладывается на четыре доли:

* **физика с контроллером** — сто вызовов `sim.step`;
* **мозг** — шаги движка LIF;
* **зрение** — подача кадра сетчатки в фоторецепторы;
* **остальное** — декодер, запись, обвязка эпизода; считается
  вычитанием, и именно поэтому доли складываются ровно в целое.

Замер делается подменой методов В САМОЙ ПРОБЕ: производственный код не
трогается ни на строку. Приём тот же, что у `retina_repro.py`.

Доли меряются на тех же шагах, то есть одновременно, а не в разных
прогонах: иначе сложились бы числа, которых вместе никогда не было.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\loop_budget.py --зрение да
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_loop_budget_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
rep = {"проверки": {}, "замеры": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--зрение", default="да")
    ap.add_argument("--шагов", type=int, default=20)
    args = ap.parse_args()
    vision = str(getattr(args, "зрение")).lower() in ("да", "yes", "1")

    from flybrain import body as B, data as D, sim as S, vision as V

    # Доли СОБСТВЕННОГО времени: вложенное вычитается. Первая редакция
    # мерила `sim.step` целиком и получила «физика 211,6 мс» — в неё
    # попали и рендер обоих глаз, и сборка наблюдения, и ожидание
    # видеокарты, занятой мозгом. Та же ловушка, что разобрана в
    # `PERF_RT.md` §5: слои надо считать собственным временем, иначе
    # одно и то же время засчитывается дважды.
    часы = {"шаг тела": 0.0, "физика": 0.0, "рендер глаз": 0.0,
            "наблюдение": 0.0, "мозг": 0.0, "подача зрения": 0.0}
    счёт = {k: 0 for k in часы}
    стек = []

    def мерить(ключ, было):
        """Обёртка, считающая СОБСТВЕННОЕ время: вложенное вычитается."""
        def новый(*a, **k):
            стек.append(0.0)
            t = time.perf_counter()
            try:
                return было(*a, **k)
            finally:
                кон_ = time.perf_counter()
                вложено = стек.pop()
                часы[ключ] += (кон_ - t) - вложено
                счёт[ключ] += 1
                if стек:
                    стек[-1] += кон_ - t
        return новый

    def обернуть(объект, имя, ключ):
        было = getattr(объект, имя)
        setattr(объект, имя, мерить(ключ, было))
        return было

    было_движок = S.Engine.step
    S.Engine.step = мерить("мозг", было_движок)
    было_подача = V.RetinaAdapter.feed
    V.RetinaAdapter.feed = мерить("подача зрения", было_подача)

    шаги = []
    метка = {"t": None}

    def on_body(тело):
        # Четыре слоя внутри шага тела, и каждый считает своё время:
        # сам шаг (контроллер походки), шаг физики dm_control, рендер
        # обоих глаз и сборка наблюдения.
        обернуть(тело, "step", "шаг тела")
        обернуть(тело.physics, "step", "физика")
        обернуть(тело.fly, "_update_vision", "рендер глаз")
        обернуть(тело.fly, "get_observation", "наблюдение")

    def on_step(step, rec, bins_row):
        now = time.perf_counter()
        if метка["t"] is not None:
            шаг = {"всего, мс": (now - метка["t"]) * 1000.0}
            шаг.update({k + ", мс": v * 1000.0 for k, v in часы.items()})
            шаги.append(шаг)
        метка["t"] = now
        for k in часы:
            часы[k] = 0.0

    print("прогон: зрение %s, шагов %d"
          % ("включено" if vision else "выключено", args.шагов), flush=True)
    neurons = D.load_neurons()
    weights = D.load_weights(device="cuda")
    cfg = B.BodyConfig(duration_s=args.шагов * 0.02, control_ms=20.0,
                       seed=0, device="cuda", readout="ALPN",
                       vision=vision, vision_hz=100.0)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        sources=[[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]],
                        progress=False, on_body=on_body, on_step=on_step)
    всего = time.perf_counter() - t0
    S.Engine.step = было_движок
    V.RetinaAdapter.feed = было_подача

    # Первые шаги платят за разогрев графов и за первый рендер: их
    # оставлять нельзя, они описывают не установившийся ход.
    живые = шаги[3:]
    check("шагов набралось", len(живые) >= 5, "%d" % len(живые))
    if len(живые) < 5:
        return

    def мед(k):
        return float(np.median([s[k] for s in живые]))

    в = мед("всего, мс")
    слои = ["физика", "рендер глаз", "наблюдение", "шаг тела",
            "мозг", "подача зрения"]
    знач = {k: мед(k + ", мс") for k in слои}
    ост = в - sum(знач.values())
    подписи = {
        "физика": "шаг физики MuJoCo (dm_control)",
        "рендер глаз": "рендер обоих глаз",
        "наблюдение": "сборка наблюдения",
        "шаг тела": "контроллер походки и обвязка FlyGym",
        "мозг": "мозг (LIF)",
        "подача зрения": "подача кадра в фоторецепторы",
    }
    строки = [("шаг управления целиком", в, 100.0)]
    for k in слои:
        строки.append((подписи[k], знач[k], 100.0 * знач[k] / в))
    строки.append(("остальное (декодер, запись, эпизод)",
                   ост, 100.0 * ост / в))
    rep["замеры"]["шаг управления, мс"] = [
        {"что": n, "мс": round(v, 2), "доля, %": round(p, 1)}
        for n, v, p in строки]
    rep["замеры"]["вызовов на шаг"] = dict(счёт)
    rep["замеры"]["доля реального"] = round(20.0 / в, 5)
    rep["замеры"]["всего по стене, с"] = round(всего, 1)
    print("\n  %-38s %8s %7s" % ("что", "мс", "доля"), flush=True)
    for n, v, p in строки:
        print("  %-38s %7.2f %6.1f %%" % (n, v, p), flush=True)
    print("  доля реального: %.4f" % (20.0 / в), flush=True)

    check("остаток положителен", ост > 0.0,
          "%.2f мс — отрицательный остаток значит двойной счёт" % ост)
    check("доля реального сходится с прежним замером",
          0.03 < 20.0 / в < 0.20, "%.4f" % (20.0 / в))
    rep["находки"].append(
        "чтобы взять 0,2× (100 мс на шаг), надо снять %.0f мс из %.0f"
        % (в - 100.0, в))
    print("  " + rep["находки"][-1], flush=True)

    имя = "loop_budget_%s.json" % ("зрение" if vision else "без")
    (OUT / имя).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
