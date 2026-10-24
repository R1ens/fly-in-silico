"""Подбор палитры областей с замером различимости.

HSV обманывает: два цвета с разным тоном и одинаковой «яркостью» могут
оказаться неразличимы. Поэтому палитра строится в OkLCh, где расстояние
примерно соответствует тому, насколько цвета различает глаз, и проверяется
числом, а не на взгляд.

Задача: 36 нейропилей, собранных в 12 супер-областей. Семейство должно
читаться с одного взгляда (общий тон), области внутри семейства — различаться,
и ни одна пара во всей палитре не должна сливаться.
"""

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D                                  # noqa: E402
from flybrain import regions as R                               # noqa: E402


# ---------------------------------------------------------------- OkLab

def oklab_to_linear(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def linear_to_srgb(c):
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (max(c, 0.0) ** (1 / 2.4)) - 0.055


def lch_to_rgb(L, C, h_deg):
    a = C * math.cos(math.radians(h_deg))
    b = C * math.sin(math.radians(h_deg))
    lin = oklab_to_linear(L, a, b)
    return tuple(min(1.0, max(0.0, linear_to_srgb(v))) for v in lin)


def in_gamut(L, C, h_deg, tol=1e-4):
    a = C * math.cos(math.radians(h_deg))
    b = C * math.sin(math.radians(h_deg))
    return all(-tol <= v <= 1 + tol for v in oklab_to_linear(L, a, b))


def lab_of(L, C, h_deg):
    return (L, C * math.cos(math.radians(h_deg)), C * math.sin(math.radians(h_deg)))


def dist(p, q):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))


# ---------------------------------------------------------------- раскладка

# Тон на супер-область. Разведены по кругу намеренно, и порядок не случайный:
# соседние по тону семейства разнесены по светлоте (см. L_BAND ниже).
# Тон распределён по кругу с запасом там, где в семействе много областей:
# оптика, вентролатеральная группа и центральный комплекс держат по пять-шесть
# нейропилей каждая, и им нужно место.
HUES = {
    "optic":            255.0,   # синий   — 96 495 нейронов, самая крупная масса
    "ventrolateral":    205.0,   # бирюза
    "central complex":  155.0,   # зелёный
    "olfactory":        142.0,   # зелёный
    "mushroom body":     85.0,   # охра
    "superior":          55.0,   # оранжевый
    "inferior":          28.0,   # кирпич
    "gnathal (SEZ)":      5.0,   # красный
    "ventromedial":     335.0,   # розовый
    "periesophageal":   305.0,   # лиловый, приглушённый
    "outside brain":    280.0,   # холодный серый
    "other":            220.0,   # нейтральный, почти без цвета
}

# Светлота: центр полосы и её наименьший размах. Семейства, соседние по тону,
# намеренно стоят в разных полосах — иначе близкие тона сливаются при похожей
# светлоте, и как раз на этом прежняя палитра и спотыкалась.
L_BAND = {
    "optic":            (0.56, 0.22),
    "ventrolateral":    (0.70, 0.16),
    "central complex":  (0.54, 0.18),
    "olfactory":        (0.70, 0.12),
    "mushroom body":    (0.84, 0.10),
    "superior":         (0.60, 0.14),
    "inferior":         (0.47, 0.12),
    "gnathal (SEZ)":    (0.66, 0.14),
    "ventromedial":     (0.48, 0.12),
    "periesophageal":   (0.85, 0.06),
    "outside brain":    (0.40, 0.10),
    "other":            (0.64, 0.04),
}

CHROMA = {
    "optic":            0.105,
    "ventrolateral":    0.085,
    "central complex":  0.090,
    "olfactory":        0.110,
    "mushroom body":    0.105,
    "superior":         0.115,
    "inferior":         0.095,
    "gnathal (SEZ)":    0.120,
    "ventromedial":     0.095,
    "periesophageal":   0.055,
    "outside brain":    0.022,
    "other":            0.008,
}


def build(region_to_super):
    families = {}
    for region, sup in sorted(region_to_super.items()):
        families.setdefault(sup if sup in HUES else "other", []).append(region)

    out = {}
    for sup, regions in families.items():
        h = HUES[sup]
        lc, span = L_BAND[sup]
        c0 = CHROMA[sup]
        k = len(regions)
        # Полоса светлоты растягивается по числу областей в семействе: шесть
        # областей центрального комплекса в узкой полосе сливались.
        span = max(span, min(0.34, 0.085 * (k - 1)))
        for i, region in enumerate(regions):
            t = 0.0 if k == 1 else (i / (k - 1)) * 2 - 1      # -1 .. +1
            # Ниже 0.33 цвет пропадает на графитовом фоне, выше 0.88 слепит.
            L = min(0.88, max(0.33, lc + t * span * 0.5))
            # Насыщенность падает к светлому краю (иначе светлые вылетают из
            # охвата) и заодно работает второй осью различия внутри семейства.
            C = c0 * (1.0 - 0.34 * t)
            # Небольшой разброс по тону: семейство ещё читается, а соседние
            # члены расходятся заметнее.
            hh = h + t * 7.0
            while not in_gamut(L, C, hh) and C > 0.005:
                C *= 0.93
            out[region] = (L, C, hh)
    return out


def main():
    df = D.load_neurons()
    names = sorted({str(r) for r in df["region"].to_numpy()})
    r2s = {n: R.super_region(n) for n in names}
    cnt = df["region"].value_counts()

    lch = build(r2s)
    labs = {n: lab_of(*lch[n]) for n in names}
    rgbs = {n: lch_to_rgb(*lch[n]) for n in names}

    pairs = sorted((dist(labs[a], labs[b]), a, b)
                   for a, b in itertools.combinations(names, 2))
    print("областей: " + str(len(names)))
    print("минимальное расстояние: " + format(pairs[0][0], ".3f")
          + "   (было 0.042 на прежней палитре)")
    print("ближайшие пары:")
    for d, a, b in pairs[:10]:
        same = "в одном семействе" if r2s[a] == r2s[b] else "РАЗНЫЕ семейства"
        print("  %.3f  %-6s %-17s %-6s %-17s %s"
              % (d, a, r2s[a], b, r2s[b], same))

    cross = [p for p in pairs if r2s[p[1]] != r2s[p[2]]]
    inner = [p for p in pairs if r2s[p[1]] == r2s[p[2]]]
    print("\nминимум между семействами: " + format(cross[0][0], ".3f")
          + "  (" + cross[0][1] + " / " + cross[0][2] + ")")
    print("минимум внутри семейства:  " + format(inner[0][0], ".3f")
          + "  (" + inner[0][1] + " / " + inner[0][2] + ")")

    def hx(c):
        return "#%02x%02x%02x" % tuple(round(v * 255) for v in c)

    print("\nпо супер-областям:")
    bysup = {}
    for n in names:
        bysup.setdefault(r2s[n], []).append(n)
    for s in sorted(bysup, key=lambda k: -sum(int(cnt.get(n, 0)) for n in bysup[k])):
        tot = sum(int(cnt.get(n, 0)) for n in bysup[s])
        print("  %-18s %2d обл. %7d  %s" % (s, len(bysup[s]), tot,
              "  ".join(hx(rgbs[n]) + " " + n for n in bysup[s])))

    # картинка-образец, чтобы посмотреть глазами
    from PIL import Image, ImageDraw

    cw, ch, gap = 150, 34, 4
    cols_n = 3
    rows = (len(names) + cols_n - 1) // cols_n
    img = Image.new("RGB", (cols_n * (cw + gap) + gap, rows * (ch + gap) + gap),
                    (14, 16, 18))
    dr = ImageDraw.Draw(img)
    order = sorted(names, key=lambda n: (r2s[n], n))
    for i, n in enumerate(order):
        r, c = divmod(i, cols_n)
        x, y = gap + c * (cw + gap), gap + r * (ch + gap)
        dr.rectangle([x, y, x + cw, y + ch],
                     fill=tuple(round(v * 255) for v in rgbs[n]))
        L = lch[n][0]
        dr.text((x + 8, y + 10), n + "  " + r2s[n][:9],
                fill=(0, 0, 0) if L > 0.62 else (255, 255, 255))
    out = Path(__file__).parent / "palette.png"
    img.save(out)
    print("\nобразец: " + str(out))

    print("\n--- для theme.py ---")
    print("REGION_LCH = {")
    for n in order:
        L, C, h = lch[n]
        print('    "%s": (%.4f, %.4f, %.1f),' % (n, L, C, h))
    print("}")


if __name__ == "__main__":
    main()
