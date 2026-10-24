"""Проверка kit.py без окна: собрать кадр и посмотреть его глазами.

Снимок swapchain в этой среде выходит чёрным, зато поле-накопитель читается
напрямую, поэтому демонстрационная панель собирается headless и ложится в PNG.
Прогон идёт в трёх размерах окна — 1500x880, 803x984 и 1280x720, — чтобы было
видно, что метрика считается от площади окна, а не притягивается к краям.

Состояния показаны там, где им место: кнопки, переключатели и ползунки стоят
по три (обычный, под курсором, нажатый), а сегменты, список и бирки
показывают все три состояния внутри одной детали — в них есть и выбранное, и
то, что под курсором, и обычное. Курсор при сборке переставляется прямо в
коробку нужной детали: `Ui` читает ввод в момент вызова, так что одного кадра
хватает на весь набор.
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui import font as F                                 # noqa: E402
from flybrain.ui import gpu, kit, theme                           # noqa: E402

OUT = Path(__file__).parent
SIZES = ((1500, 880), (803, 984), (1280, 720))

REGIONS = (
    "AL(L)", "AL(R)", "LH(L)", "LH(R)", "MB_CA(L)", "MB_CA(R)", "MB_ML(L)",
    "MB_ML(R)", "MB_VL(L)", "EB", "FB", "NO", "PB", "AMMC(L)", "AMMC(R)",
    "SAD(L)", "SAD(R)", "GNG", "CAN(L)", "CAN(R)", "ME(L)", "ME(R)", "LO(L)",
    "LO(R)", "LOP(L)", "LOP(R)", "AOTU(L)", "AOTU(R)", "PLP(L)", "PLP(R)",
    "SLP(L)", "SLP(R)", "SIP(L)", "SIP(R)", "SMP(L)", "SMP(R)", "ATL(L)",
    "ATL(R)", "IB(L)", "IB(R)",
)

LOG = tuple(
    [
        "00,000 с  загружен снимок FlyWire v783",
        "00,000 с  138 625 нейронов, 15 091 983 связи",
        "00,012 с  зонд стороны: левая половина",
        "00,040 с  стимул «сахар» на ALPN, 150 Гц",
    ]
    + ["%06.3f с  бин %d: активных %d, пик %d Гц"
       % (0.05 + i * 0.01, i, 7200 + (i * 173) % 2600, 240 + (i * 37) % 90)
       for i in range(26)]
)


def camera_image(w=128, h=96):
    """Снимок с камеры арены: что-то похожее на данные, а не плашка."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = (np.sin(xx / 9.0) * np.cos(yy / 7.0) * 0.5 + 0.5) * 0.7
    f += 0.3 * np.exp(-(((xx - w * 0.62) ** 2 + (yy - h * 0.4) ** 2) / 320.0))
    f = np.clip(f, 0, 1)
    mg = np.array(theme.MAGMA, dtype=np.float32)
    t = f * (len(mg) - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(mg) - 1)
    a = (t - i0)[:, :, None]
    rgb = mg[i0] * (1 - a) + mg[i1] * a
    return np.ascontiguousarray((rgb * 255).astype(np.uint8))


def demo_series():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 6.0, 240)
    rate = (40 + 22 * np.sin(t * 2.1) + 9 * np.sin(t * 5.3)
            + rng.normal(0, 2.4, t.size))
    volt = -58 + 7 * np.sin(t * 3.7) + rng.normal(0, 0.9, t.size)
    volt[::37] += 34.0
    ang = np.cumsum(rng.normal(0, 0.22, 900))
    step = 0.9 + 0.3 * np.cos(np.linspace(0, 12, 900))
    path = np.cumsum(np.stack([np.cos(ang) * step, np.sin(ang) * step], 1), 0)
    return rate, volt, path


# ---------------------------------------------------------------- ригование

def nowhere(ui):
    ui.input.x = ui.input.y = -1.0
    ui.input.down = False
    ui.active = None


def hover(ui, b):
    ui.input.x, ui.input.y = b.cx, b.cy
    ui.input.down = False
    ui.active = None


def hover_at(ui, x, y):
    ui.input.x, ui.input.y = float(x), float(y)
    ui.input.down = False
    ui.active = None


def held(ui, key, b):
    """Деталь под нажатой кнопкой: тот же путь, которым ходит захват мышью."""
    ui.input.x, ui.input.y = b.cx, b.cy
    ui.input.down = True
    ui.active = key


# ---------------------------------------------------------------- панель

def panel(ui, w, h, img, rate, volt, path):
    # Фон окна кладёт surf.clear своим ядром: прямоугольник во весь экран
    # накрыл бы все детали, а внутри одного слоя порядок не определён.
    dl = ui.dl
    pad, gap = ui.pad, ui.gap
    full = kit.Box(0, 0, w, h)
    foot, full = full.cut_bottom(ui.foot, 0)
    head, body = full.inset(pad, pad).cut_top(ui.line() + ui.line(True), gap)

    # ---- заголовок: подпись, число и сегменты в одну полосу
    seg_w = min(ui.px(300), head.w // 2)
    seg, title = head.cut_right(seg_w, gap)
    ui.label("Симулятор мозга мухи", title.x0, title.y0)
    ui.label("FlyWire v783 · прогон demo-sugar · бин 10 мс",
             title.x0, title.y0 + ui.line(), theme.DIM, small=True)
    seg = kit.Box(seg.x0, seg.y0, seg.x1, seg.y0 + ui.row)
    hover_at(ui, seg.x0 + seg.w * 0.5, seg.cy)      # средняя доля под курсором
    ui.state["view"] = ui.segmented(
        "view", seg, ("активность", "области", "путь"), ui.state["view"])
    nowhere(ui)
    dl.hline(head.x0, head.x1, head.y1 + gap // 2, theme.HAIRLINE)

    # ---- две колонки: рейка сборки и показания
    rail_w = min(ui.rail, int(body.w * 0.42))
    rail, main = body.cut_left(rail_w, pad)
    dl.vline(rail.x1 + pad // 2, rail.y0, rail.y1, theme.HAIRLINE)

    # ================================================== левая рейка
    col = kit.Column(rail, gap)
    r = col.take(ui.line(True), ui.px(4))
    ui.label("сборка прогона", r.x0, r.y0, theme.FAINT, small=True)

    # кнопки: обычная, под курсором, нажатая
    r = col.take(ui.row)
    a, b, c = r.cols(3, ui.px(6))
    nowhere(ui)
    ui.button("b1", a, "шаг")
    hover(ui, b)
    ui.button("b2", b, "шаг")
    held(ui, "b3", c)
    ui.button("b3", c, "шаг")
    nowhere(ui)

    # кнопки по роли: главная, опасная, отключённая
    r = col.take(ui.row)
    a, b, c = r.cols(3, ui.px(6))
    ui.button("run", a, "пуск", kind="primary")
    ui.button("stop", b, "стоп", kind="danger")
    ui.button("back", c, "назад", enabled=False)

    # переключатели: выключен, под курсором, включён
    for i, (key, text, st) in enumerate((("t1", "сетка", False),
                                         ("t2", "оси", False),
                                         ("t3", "связи", True))):
        r = col.take(ui.row, ui.px(4))
        if i == 1:
            hover(ui, r)
        else:
            nowhere(ui)
        ui.state[key] = ui.toggle(key, r, text, ui.state.get(key, st))
    nowhere(ui)

    # ползунки: обычный, под курсором, под тащащей мышью
    sl_h = ui.small.gh + ui.px(14)
    for i, (key, text, lo, hi, step, fmt) in enumerate(
            (("частота", "частота, Гц", 10.0, 300.0, 5.0, "{:.0f}"),
             ("длительность", "длительность, с", 0.1, 4.0, 0.1, None),
             ("усиление", "усиление", 0.0, 2.0, None, None))):
        r = col.take(sl_h, ui.px(6))
        if i == 1:
            hover(ui, r)
        elif i == 2:
            held(ui, key, r)
        else:
            nowhere(ui)
        ui.state[key] = ui.slider(key, r, text, ui.state[key], lo, hi,
                                  step=step, fmt=fmt)
    nowhere(ui)

    # бирки: выбранные, под курсором, обычные
    chip_h = ui.px(22)
    r = col.take(chip_h * 2 + ui.px(6), ui.px(8))
    hover_at(ui, r.x0 + ui.px(30), r.y0 + chip_h // 2)
    ui.state["chips"] = ui.chips("chips", r,
                                 ("зрение", "вкус", "обоняние", "усики",
                                  "ноги", "крылья"),
                                 ui.state["chips"])
    nowhere(ui)

    # список: выбранная строка, строка под курсором, обычные
    r = col.take(ui.line(True), ui.px(4))
    ui.label("области, фильтр «a»", r.x0, r.y0, theme.FAINT, small=True)
    lst = col.rest
    hover_at(ui, lst.cx, lst.y0 + ui.row * 2 + ui.row // 2)
    ui.state["region"] = ui.list_box("regions", lst, REGIONS,
                                     ui.state["region"], filter_text="a")
    nowhere(ui)

    # ================================================== правая колонка
    col = kit.Column(main, gap)

    # показания числами
    r = col.take(ui.font.gh + ui.px(4))
    for cell, (lab, val) in zip(r.cols(3, ui.px(14)),
                                (("время", "640 мс"), ("активных", 9110),
                                 ("пик, Гц", 312))):
        ui.value(lab, val, cell.x0, cell.y0, cell.w)
    dl.hline(r.x0, r.x1, r.y1 + ui.px(3), theme.HAIRLINE)
    col.skip(ui.px(4))

    # меры, искровые графики и журнал забирают постоянную высоту, остаток
    # уходит картинке с ареной: в высоком окне растёт именно она
    log_h = max(ui.px(104), ui.line(True) * 5)
    spark_h = ui.px(40)
    meter_h = ui.small.gh + ui.px(9)
    keep = log_h + gap + 2 * (spark_h + gap) + 3 * (meter_h + ui.px(4)) + gap
    # Арене — остаток, но не больше разумного: в высоком окне лишнее уходит
    # журналу, а не превращается в пустую коробку вокруг картинки
    view = col.take(min(ui.px(300), max(ui.px(150), col.left - keep)))
    left, right = view.cols(2, gap)
    ui.image_box(left, img, label="камера арены")
    ui.track(right, path, theme.magma_at(0.62), closed=False)
    dl.text(ui.small, "путь в арене", right.x0 + ui.px(6), right.y0 + ui.px(6),
            theme.FAINT)

    for lab, val, col_, txt in (("активность", 0.72, theme.ACCENT, "72 %"),
                                ("покрытие", 0.41, theme.magma_at(0.55),
                                 "41 %"),
                                ("сходимость", 0.93, theme.OK, "0,93")):
        ui.meter(col.take(meter_h, ui.px(4)), val, col_, label=lab,
                 value_text=txt)
    col.skip(ui.px(4))

    for lab, s in (("частота ALPN", rate), ("мембрана", volt)):
        r = col.take(spark_h)
        lw = ui.small.width("частота ALPN") + ui.px(10)
        name, plot = r.cut_left(lw, 0)
        ui.label(lab, name.x0, name.y0 + (name.h - ui.small.gh) // 2,
                 theme.DIM, small=True)
        ui.spark(plot, s, theme.magma_at(0.7))

    ui.log_view("log", col.rest, LOG)

    # ---- подвал
    dl.hline(0, w, foot.y0, theme.HAIRLINE)
    ui.label("vulkan · масштаб " + format(ui.scale, ".2f") + " · "
             + str(w) + "×" + str(h), pad, foot.y0 + ui.px(5), theme.DIM,
             small=True)
    ui.label("тащить · колесо · стрелки · Esc",
             w - pad - ui.small.width("тащить · колесо · стрелки · Esc"),
             foot.y0 + ui.px(5), theme.FAINT, small=True)


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    img = camera_image()
    rate, volt, path = demo_series()
    surf = gpu.Surface()
    ui = kit.Ui(load=F.load)
    # Состояние деталей живёт у вызывающего кода, как и в окне.
    ui.state = {"view": 1, "частота": 150.0, "длительность": 0.8,
                "усиление": 1.0, "chips": {"вкус", "обоняние"},
                "region": 2, "t1": False, "t2": False, "t3": True}

    from PIL import Image

    for w, h in SIZES:
        surf.ensure(w, h)

        def frame():
            surf.clear(theme.BG)
            ui.begin(w, h)
            panel(ui, w, h, img, rate, volt, path)
            ui.flush(surf)

        for _ in range(5):
            frame()
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(30):
            frame()
        ti.sync()
        ms = (time.perf_counter() - t0) * 1000 / 30

        frame()
        ti.sync()
        a = np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]
        name = OUT / ("kit_" + str(w) + "x" + str(h) + ".png")
        Image.fromarray((a * 255).astype(np.uint8)).save(name)
        print(str(w) + "×" + str(h) + ": кадр " + format(ms, ".2f")
              + " мс (" + format(1000 / ms, ".0f") + " кадр/с), команд "
              + str(ui.commands())
              + " (прямоугольников " + str(ui.rect_count()) + ", глифов "
              + str(ui.glyph_count()) + ", картинок " + str(len(ui.dl.images))
              + "), масштаб " + format(ui.scale, ".2f")
              + ", кегль " + str(ui.font.size) + "/" + str(ui.small.size))
        print("  записан " + str(name))


if __name__ == "__main__":
    main()
