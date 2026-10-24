"""Раскладка окна: где что лежит при любом размере.

Отдельный модуль, потому что размер окна здесь не просто меняется, а меняется
сам и до нелепого: в замерах то же самое окно побывало 1500×880, 803×984,
755×984 и 257×1824. Прибитая к пикселям раскладка в таком окне рассыпается, а
просто масштабирование не спасает — в полосе шириной 257 пикселей рейка
шириной 300 съедает всё.

Поэтому ширины считаются долями от окна с ограничителями, а при нехватке
места части раскладки отключаются по порогам: сперва уходит колонка
показаний, потом рейка. Карта мозга не уходит никогда — она и есть окно.
"""

from dataclasses import dataclass, field

from . import theme

__all__ = ["Layout", "compute"]


@dataclass
class Layout:
    w: int
    h: int
    scale: float
    pad: int

    rail: tuple = ()          # левая рейка: сборка прогона
    brain: tuple = ()         # карта мозга
    meters: tuple = ()        # правая колонка: показания
    strip: tuple = ()         # нижняя полоса: тело или арена
    foot: tuple = ()          # строка состояния

    show_rail: bool = True
    show_meters: bool = True
    show_strip: bool = False
    rail_over: bool = False   # рейка лежит поверх карты, а не рядом
    notes: list = field(default_factory=list)

    def has(self, name):
        r = getattr(self, name)
        return bool(r) and r[2] > 0 and r[3] > 0


def compute(w, h, scale, strip=False, rail_open=True):
    """Разложить окно.

    `strip` — показывать ли нижнюю полосу (вид тела или арена).
    `rail_open` — раскрыта ли рейка, когда она в режиме наложения.
    """
    pad = max(6, int(round(theme.PAD * scale)))
    foot_h = max(18, int(round(theme.FOOT_H * scale)))
    lay = Layout(w=w, h=h, scale=scale, pad=pad)

    rail_w = int(min(max(0.21 * w, 240 * scale), 360 * scale))
    met_w = int(min(max(0.19 * w, 210 * scale), 320 * scale))
    min_brain = int(420 * scale)

    # Колонка показаний уходит первой: её содержимое дублируется в отчёте,
    # а карта без места становится бесполезной.
    if w - rail_w - met_w < min_brain:
        lay.show_meters = False
        met_w = 0
        lay.notes.append("показания убраны: не хватает ширины")

    # Рейка уходит второй и становится выдвижной поверх карты.
    if w - rail_w < min_brain:
        lay.rail_over = True
        rail_w = int(min(max(0.72 * w, 200 * scale), 340 * scale))
        if not rail_open:
            lay.show_rail = False
        lay.notes.append("рейка поверх карты: окно слишком узкое")

    body_y = h - foot_h
    strip_h = 0
    if strip:
        strip_h = int(min(max(0.26 * h, 170 * scale), 340 * scale))
        if body_y - strip_h < int(260 * scale):
            strip_h = max(0, body_y - int(260 * scale))
        lay.show_strip = strip_h > int(60 * scale)
        if not lay.show_strip:
            strip_h = 0
            lay.notes.append("полоса тела убрана: не хватает высоты")

    lay.foot = (0, body_y, w, foot_h)

    if lay.rail_over:
        bx0 = 0
        bx1 = w
    else:
        bx0 = rail_w if lay.show_rail else 0
        bx1 = w - met_w if lay.show_meters else w

    lay.rail = (0, 0, rail_w, body_y) if lay.show_rail else ()
    lay.meters = ((w - met_w, 0, met_w, body_y) if lay.show_meters else ())

    brain_h = body_y - strip_h
    lay.brain = (bx0 + pad, pad,
                 max(0, bx1 - bx0 - pad * 2),
                 max(0, brain_h - pad * 2))
    lay.strip = ((bx0 + pad, brain_h, max(0, bx1 - bx0 - pad * 2),
                  max(0, strip_h - pad)) if lay.show_strip else ())
    return lay
