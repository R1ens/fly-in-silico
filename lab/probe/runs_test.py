"""Проверка обзора прогонов без окна: собрать панель и посмотреть глазами.

Окно в этой среде самовольно меняет размер — за один сеанс оно побывало
1945×984 и 257×1824, — поэтому судить о виде по снимку живого окна нельзя.
Здесь панель собирается в заданный размер и ложится в PNG.

Снимок идёт по разу на каждый вид прогона, который нашёлся в `out/`: разбор у
видов разный, и наезд подписей в таблице серии не виден на разборе опыта.
Заодно это замер: сколько стоит кадр обзора и сколько в нём команд.
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui import font as F                                  # noqa: E402
from flybrain.ui import gpu, kit, runs, theme                      # noqa: E402

OUT = Path(__file__).parent
SIZES = ((1945, 984), (1500, 880), (803, 984))


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    ui = kit.Ui(load=F.load)

    st = runs.RunsState()
    st.refresh(force=True)
    print("обход out/: " + format(st.scan_ms, ".1f") + " мс, "
          + str(len(st.entries)) + " записей")
    if not st.entries:
        print("в out/ нечего показывать")
        return

    # По одному прогону на вид, в том порядке, в каком виды стоят в фильтре.
    picks = []
    for i, (label, kind) in enumerate(runs.KINDS):
        if kind is None:
            continue
        found = [e for e in st.entries if e["kind"] == kind]
        if found:
            picks.append((i, kind, found[0]))
    print("виды: " + ", ".join(k + " (" + e["name"] + ")"
                               for _, k, e in picks))

    # Первым снимком — то, с чего панель открывается: фильтр «все», выбран
    # самый свежий прогон. Только в нём у строк списка есть колонка вида.
    picks.insert(0, (0, "all", st.entries[0]))

    from PIL import Image

    for filter_i, kind, entry in picks:
        st.filter_i = filter_i
        st.select(entry)
        for w, h in SIZES:
            surf.ensure(w, h)

            def frame():
                surf.clear(theme.BG)
                ui.begin(w, h)
                # Курсор ставится на вторую строку списка: так на снимке
                # видно и выбранную строку, и ступень фона под курсором.
                ui.input.feed(ui.pad + ui.px(40),
                              ui.pad + ui.row * 4 + ui.row // 2, False)
                runs.panel(ui, st, kit.box(0, 0, w, h),
                           on_open=lambda p: None,
                           on_rerun=lambda e: None,
                           on_rerun_spec=lambda s: None)
                ui.flush(surf)

            for _ in range(4):
                frame()
            ti.sync()
            t0 = time.perf_counter()
            for _ in range(20):
                frame()
            ti.sync()
            ms = (time.perf_counter() - t0) * 1000 / 20

            frame()
            ti.sync()
            a = np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]
            name = OUT / ("runs_" + kind + "_" + str(w) + "x" + str(h)
                          + ".png")
            Image.fromarray((a * 255).astype(np.uint8)).save(name)
            print(kind + "  " + str(w) + "×" + str(h) + ": кадр "
                  + format(ms, ".2f") + " мс (" + format(1000 / ms, ".0f")
                  + " кадр/с), команд " + str(ui.commands())
                  + " (прямоугольников " + str(ui.rect_count())
                  + ", глифов " + str(ui.glyph_count()) + "), масштаб "
                  + format(ui.scale, ".2f"))
            print("  записан " + str(name))


if __name__ == "__main__":
    main()
