"""Сверка шага LIF: любая правка движка обязана давать те же числа.

Эта проба — шлюз для оптимизаций первого уровня. Она гоняет движок и
записывает отпечатки состояния: мембранное напряжение, проводимость,
кольцо задержек, счётчик рефрактерности и накопленные спайки. Отпечаток
снимается от сырых байтов, поэтому ловит расхождение в последнем бите.

Порядок работы:

    ..\\.venv\\Scripts\\python.exe probe\\lif_equal.py --save эталон
    (правим sim.py)
    ..\\.venv\\Scripts\\python.exe probe\\lif_equal.py --against эталон

Второй запуск сравнивает и падает при любом расхождении. Это уровень 1 из
брифа: совпадение, а не близость.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent


def fp(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def snapshot(steps=2000, batch=1, seed=0, device=None):
    """Прогнать движок и снять отпечатки. Никаких усреднений."""
    import torch

    from flybrain import data as D, sim

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    n = len(neurons)
    idx = D.select(neurons, "class:gustatory+side:left")
    free = torch.as_tensor(np.asarray(idx, np.int64), device=device)

    eng = sim.build_engine(weights=weights, neurons=neurons, device=device,
                           batch=batch, seed=seed, no_refractory=free)
    rates = torch.zeros(batch, n, device=device)
    rates[:, idx] = 200.0

    counts = torch.zeros(n, dtype=torch.int64, device=device)
    # Отпечатки берём не только в конце: расхождение может возникнуть и
    # затухнуть, и тогда по одному финальному снимку его не видно.
    marks = {}
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        sp = eng.step(rates)
        counts += sp.sum(0).to(torch.int64)
        if (step + 1) in (1, 10, 19, 20, 100, 500, 1000, steps):
            marks[step + 1] = {
                "спайков всего": int(counts.sum().item()),
                "v": fp(eng.v.detach().cpu().numpy()),
                "проводимость": fp(eng.conductance.detach().cpu().numpy()),
                "спайки": fp(sp.detach().cpu().numpy()),
            }
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return {
        "устройство": device, "шагов": steps, "партия": batch, "сид": seed,
        "нейронов": int(n), "длина кольца задержек": int(eng.buf_len),
        "мс на шаг": round(wall * 1000.0 / steps, 4),
        "итог": {
            "спайков всего": int(counts.sum().item()),
            "отпечаток счётчика": fp(counts.cpu().numpy()),
            "v": fp(eng.v.detach().cpu().numpy()),
            "проводимость": fp(eng.conductance.detach().cpu().numpy()),
            "кольцо задержек": fp(eng.delay_buffer.detach().cpu().numpy()),
            "рефрактерность": fp(eng.refrac.detach().cpu().numpy()),
            "указатель кольца": int(eng.ptr),
        },
        "вехи": marks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=None, metavar="ИМЯ")
    ap.add_argument("--against", default=None, metavar="ИМЯ")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    got = snapshot(steps=args.steps)
    print("шагов %d, %.4f мс/шаг, спайков %d"
          % (got["шагов"], got["мс на шаг"], got["итог"]["спайков всего"]))
    for k, v in got["итог"].items():
        print("   %-20s %s" % (k, v))

    if args.save:
        p = OUT / ("lif_equal_%s.json" % args.save)
        p.write_text(json.dumps(got, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print("снимок:", p)
        return 0

    if args.against:
        p = OUT / ("lif_equal_%s.json" % args.against)
        ref = json.loads(p.read_text(encoding="utf-8"))
        bad = []
        for k, v in ref["итог"].items():
            if got["итог"].get(k) != v:
                bad.append("итог/%s: было %s, стало %s"
                           % (k, v, got["итог"].get(k)))
        for step, m in ref["вехи"].items():
            cur = got["вехи"].get(int(step)) or got["вехи"].get(step) or {}
            for k, v in m.items():
                if cur.get(k) != v:
                    bad.append("шаг %s / %s: было %s, стало %s"
                               % (step, k, v, cur.get(k)))
        print()
        if bad:
            print("РАСХОЖДЕНИЙ %d:" % len(bad))
            for b in bad[:10]:
                print("  -", b)
            return 1
        print("совпало полностью: %d вех и %d полей итога"
              % (len(ref["вехи"]), len(ref["итог"])))
        print("скорость: было %.4f мс/шаг, стало %.4f мс/шаг — в %.2f раза"
              % (ref["мс на шаг"], got["мс на шаг"],
                 ref["мс на шаг"] / max(got["мс на шаг"], 1e-9)))
        return 0

    print("\nнечего делать: укажите --save ИМЯ или --against ИМЯ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
