"""Атлас глифов как артефакт, а не как кэш рантайма.

Набор символов задан списком явно, чтобы дырки в покрытии всплывали при
печении, а не в окне на показе. Атлас печётся командой `fly font` и ложится в
`cache/`; окно только читает готовый файл.

Раскладка атласа: один горизонтальный ремень, `atlas[x, y]` — альфа 0..255,
где `y = 0` это верх глифа (так отдаёт PIL). Перевод в координаты кадра, где
начало внизу слева, делает ядро в `gpu.py`, а не этот модуль.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import data as D

__all__ = ["GLYPHS", "BLANKS", "ROLES", "Atlas", "bake", "load",
           "available_fonts", "role_path", "match_size"]

# Латиница, кириллица и ровно те знаки, которые нужны показаниям приборов.
GLYPHS = (
    "".join(chr(c) for c in range(32, 127))
    + "".join(chr(c) for c in range(0x410, 0x450))
    + "\u0401\u0451"                      # Ё ё
    + "\u00b0\u00b1\u00d7\u00b7"          # ° ± × ·
    + "\u2014\u2013\u2026"                # — – …
    + "\u2192\u2190\u2191\u2193"          # стрелки
    + "\u2265\u2264\u2248"                # ≥ ≤ ≈
    + "\u2500\u2502"                      # ─ │
    + "\u25b2\u25bc\u25a0\u25cf"          # ▲ ▼ ■ ●
    + "\u00ab\u00bb"                      # « »
    # Пробелы и минус, которых не хватало. Тонкий пробел стоит между числом
    # и единицей в наших же отчётах и в выводе локальных моделей, и без него
    # «410 нейронов» печаталось как «410?нейронов».
    + "\u2009\u00a0\u202f"                # тонкий, неразрывный, узкий неразрывный
    + "\u2212"                            # минус, а не дефис
)

# Пустые по начертанию символы: по ним нельзя судить о дырке в гарнитуре.
BLANKS = frozenset(" \u00a0\u2009\u202f")

# Роли, а не одна гарнитура на всё. Набирать интерфейс целиком моноширинным
# было ошибкой: без пропорционального текста и без разницы в насыщенности
# окно читается как дамп терминала, сколько ни выравнивай.
#
# Corbel — гуманистическая антиква с полной кириллицей и собственным
# характером, не системная по умолчанию.
#
# Числа набираются Segoe UI, а не моноширинной гарнитурой. У Segoe цифры
# табличные — колонка показаний не дёргается от кадра к кадру, — но буквы
# пропорциональные, и строка выходит плотной. Моноширинная на этом месте
# давала разрывы: запятая занимала ту же клетку, что цифра, и «0,8 с»
# читалось как «0 , 8  с». Моноширинная осталась там, где она несёт
# смысл: селекторы и имена типов.
ROLES = {
    "ui":     (r"C:\Windows\Fonts\corbel.ttf",
               r"C:\Windows\Fonts\segoeui.ttf"),
    "strong": (r"C:\Windows\Fonts\corbelb.ttf",
               r"C:\Windows\Fonts\segoeuib.ttf"),
    "num":    (r"C:\Windows\Fonts\segoeui.ttf",
               r"C:\Windows\Fonts\calibri.ttf"),
    "mono":   (r"C:\Windows\Fonts\CascadiaMono.ttf",
               r"C:\Windows\Fonts\consola.ttf"),
}

FONT_CANDIDATES = ROLES["ui"] + ROLES["num"]


def role_path(role):
    for cand in ROLES.get(role, ROLES["ui"]):
        if Path(cand).exists():
            return cand
    found = available_fonts()
    if not found:
        raise RuntimeError("не найдено ни одной гарнитуры")
    return found[0]


@dataclass
class Atlas:
    """Готовый ремень глифов и его метрики."""

    pixels: np.ndarray          # uint8 (total_w, gh), x-major
    gx: np.ndarray              # int32 (n_glyphs,)  начало глифа в ремне
    gw: np.ndarray              # int32 (n_glyphs,)  ширина глифа
    gh: int                     # высота строки в пикселях
    index: dict                 # символ -> номер глифа
    size: int                   # кегль, которым печаталось
    font: str                   # имя файла гарнитуры
    missing: tuple              # символы, которых в гарнитуре не оказалось
    role: str = "ui"            # роль: текст, заголовок, числа

    @property
    def max_gw(self):
        return int(self.gw.max())

    def glyph(self, ch):
        return self.index.get(ch, self.index.get("?", 0))

    def width(self, text):
        """Ширина строки в пикселях."""
        gw, idx = self.gw, self.index
        fallback = idx.get("?", 0)
        return int(sum(gw[idx.get(ch, fallback)] for ch in text))

    def fit(self, text, max_px):
        """Усечь строку под ширину, поставив многоточие."""
        if self.width(text) <= max_px:
            return text
        dots = "\u2026"
        budget = max_px - self.width(dots)
        if budget <= 0:
            return dots
        out, used = [], 0
        for ch in text:
            w = int(self.gw[self.glyph(ch)])
            if used + w > budget:
                break
            out.append(ch)
            used += w
        return "".join(out) + dots


def available_fonts():
    return [p for p in FONT_CANDIDATES if Path(p).exists()]


def _cache_path(font_path, size):
    key = hashlib.sha1(
        (Path(font_path).name + "|" + str(size) + "|"
         + hashlib.sha1(GLYPHS.encode("utf-8")).hexdigest()[:12]).encode("utf-8")
    ).hexdigest()[:12]
    return D.CACHE / ("font-" + Path(font_path).stem + "-" + str(size)
                      + "-" + key + ".npz")


def bake(size=14, font_path=None, force=False, role="ui"):
    """Испечь атлас и положить в cache/. Возвращает Atlas."""
    from PIL import Image, ImageDraw, ImageFont

    if font_path is None:
        font_path = role_path(role)

    out = _cache_path(font_path, size)
    if out.exists() and not force:
        return _read(out)

    font = ImageFont.truetype(font_path, size)
    chars = list(dict.fromkeys(GLYPHS))

    # Ширина глифа — это отступ гарнитуры, а не ширина его чернил. Раньше
    # здесь стояло «ширина чернил плюс два пикселя», и для пропорциональной
    # гарнитуры это разваливает весь набор: боковые полуапроши игнорируются,
    # а лишние два пикселя прибавляются к каждой букве. Текст выходил
    # разреженным, будто набран вразрядку.
    widths = [max(1, int(round(font.getlength(ch)))) for ch in chars]

    # Общая базовая линия через метрики гарнитуры: тогда строки разных
    # начертаний стоят на одной линии, а не каждая по своим чернилам.
    ascent, descent = font.getmetrics()
    gh = ascent + descent
    ink_top = min((font.getbbox(ch)[1] for ch in chars), default=0)
    if ink_top < 0:                       # выносные выше линии шрифта
        gh -= ink_top
    total = int(sum(widths))

    img = Image.new("L", (total, gh), 0)
    draw = ImageDraw.Draw(img)
    gx, x = [], 0
    for ch, w in zip(chars, widths):
        gx.append(x)
        # Пробелы не рисуем вовсе, только отступ. Иначе гарнитура, в
        # которой нет тонкого пробела, кладёт на его место `.notdef` —
        # и в строке «62 %» между числом и знаком встаёт пустой
        # прямоугольник. Шириной пробел при этом остаётся верным:
        # `getlength` её знает и без глифа.
        if ch not in BLANKS:
            draw.text((x, max(0, -ink_top)), ch, fill=255, font=font)
        x += w

    pixels = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).T)
    gx = np.array(gx, dtype=np.int32)
    gw = np.array(widths, dtype=np.int32)

    # Пустой прямоугольник вместо глифа — это дырка в покрытии гарнитуры.
    # Пустой прямоугольник вместо глифа — дырка в гарнитуре. Пробелы из
    # проверки исключены: у них и должно быть пусто.
    missing = tuple(
        ch for ch, a, w in zip(chars, gx, gw)
        if ch not in BLANKS and int(pixels[a:a + w, :].max()) == 0)

    D.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pixels=pixels, gx=gx, gw=gw,
        gh=np.int32(gh), size=np.int32(size),
        chars=np.array(chars, dtype="U1"),
        font=np.array(Path(font_path).name),
        role=np.array(role),
        missing=np.array(list(missing), dtype="U1"))
    return _read(out)


def _read(path):
    z = np.load(path, allow_pickle=False)
    chars = [str(c) for c in z["chars"]]
    return Atlas(
        pixels=np.ascontiguousarray(z["pixels"]),
        gx=np.ascontiguousarray(z["gx"]),
        gw=np.ascontiguousarray(z["gw"]),
        gh=int(z["gh"]),
        index={c: i for i, c in enumerate(chars)},
        size=int(z["size"]),
        font=str(z["font"]),
        role=str(z["role"]) if "role" in z else "ui",
        missing=tuple(str(c) for c in z["missing"]),
    )


def load(size=14, font_path=None, role="ui"):
    """Взять атлас нужной роли и кегля, испечь при первом обращении."""
    return bake(size=size, font_path=font_path, force=False, role=role)


def match_size(size, role="num", against="ui", probe="Н"):
    """Кегль роли, при котором цифра по росту совпадает с текстовой.

    Равняем по высоте прописной, а не цифры: у текстовой гарнитуры цифры
    могут быть строчными, и тогда моноширинная подбирается вдвое мельче
    текста, с которым стоит в одной строке.

    Совпадения по высоте строки тоже мало: у моноширинных гарнитур выносные
    элементы крупнее, и при равной строке цифры выходят заметно мельче
    текста. Сравнивать надо то, что видит глаз, — рост самой цифры.
    """
    from PIL import ImageFont

    base = ImageFont.truetype(role_path(against), size)
    bb = base.getbbox(probe)
    want = bb[3] - bb[1]
    path = role_path(role)
    best, best_err = size, 1e9
    for cand in range(max(7, size - 6), size + 7):
        f = ImageFont.truetype(path, cand)
        b = f.getbbox(probe)
        err = abs((b[3] - b[1]) - want)
        if err < best_err:
            best, best_err = cand, err
    return best
