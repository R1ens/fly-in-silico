"""Проверка карты мозга: подписи, фокус, режимы, наведение."""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D                                  # noqa: E402
from flybrain import regions as R                               # noqa: E402
from flybrain.ui import brainview, font as F, gpu, theme        # noqa: E402

W, H = 1180, 820
OUT = Path(__file__).parent


def main():
    df = D.load_neurons()
    xyz = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    p = xyz[keep]
    p -= 0.5 * (p.min(0) + p.max(0))
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0

    reg = df["region"].to_numpy()[keep].astype(str)
    names = sorted(set(reg))
    cols = theme.region_colors({n: R.super_region(n) for n in names})
    rgb = np.array([cols[n] for n in names], dtype=np.float32)
    code = np.searchsorted(np.array(names), reg).astype(np.int32)

    full = np.zeros(len(df), dtype=np.float32)
    z = np.load(D.OUT / "demo-sugar" / "bins.npz")
    full[z["neuron_idx"].astype(np.int64)] = z["bins"].max(axis=0).astype(np.float32)
    act = full[keep]
    hi = float(np.percentile(act[act > 0], 99))
    act = np.clip(act / hi, 0, 1).astype(np.float32)

    ti.init(arch=ti.vulkan, offline_cache=True)
    atlas = F.load(13)
    surf = gpu.Surface()
    surf.ensure(W, H)

    bv = brainview.BrainView(p, code, names, rgb)
    bv.set_activity(act)
    vp = (16, 16, W - 32, H - 32)

    def shot(name, note):
        surf.clear(theme.BG)
        dl = gpu.DrawList()
        bv.draw(surf, dl, atlas, vp)
        dl.frame(vp[0] - 1, vp[1] - 1, vp[0] + vp[2] + 1, vp[1] + vp[3] + 1,
                 theme.HAIRLINE)
        dl.text(atlas, note, vp[0] + 10, vp[1] + 8, theme.FAINT)
        surf.draw(dl, atlas)
        ti.sync()
        a = np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]
        from PIL import Image
        Image.fromarray((a * 255).astype(np.uint8)).save(OUT / name)
        print("записан " + name + "  (" + note + ")")

    shot("bv_activity.png", "активность · три четверти")
    bv.set_mode("regions")
    shot("bv_regions.png", "области · три четверти")
    bv.set_focus(names.index("AL"))
    shot("bv_focus_al.png", "фокус на AL (антеннальная доля)")
    bv.set_focus(None)
    bv.next_view()
    shot("bv_front.png", "области · фронт")
    bv.set_mode("activity")
    bv.only_active = True
    shot("bv_only_active.png", "только активные · фронт")

    # наведение: берём точку в центре и смотрим, что вернётся
    bv.only_active = False
    bv.set_mode("activity")
    surf.clear(theme.BG)
    dl = gpu.DrawList()
    bv.draw(surf, dl, atlas, vp)
    t0 = time.perf_counter()
    hits = 0
    for _ in range(50):
        if bv.pick((vp[0] + vp[2] // 2, vp[1] + vp[3] // 2)) is not None:
            hits += 1
    pick_ms = (time.perf_counter() - t0) * 1000 / 50
    print("наведение: " + format(pick_ms, ".2f") + " мс на запрос, попаданий "
          + str(hits) + " из 50")

    # цена кадра карты целиком
    for _ in range(5):
        surf.clear(theme.BG)
        dl = gpu.DrawList()
        bv.draw(surf, dl, atlas, vp)
        surf.draw(dl, atlas)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(40):
        surf.clear(theme.BG)
        dl = gpu.DrawList()
        bv.draw(surf, dl, atlas, vp)
        surf.draw(dl, atlas)
    ti.sync()
    ms = (time.perf_counter() - t0) * 1000 / 40
    print("карта целиком: " + format(ms, ".2f") + " мс ("
          + format(1000 / ms, ".0f") + " кадр/с)")


if __name__ == "__main__":
    main()
