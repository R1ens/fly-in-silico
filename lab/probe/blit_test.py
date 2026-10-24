"""Откуда берутся цветные разводы на кадре тела.

На живом снимке пол арены вышел бирюзовым с розовой штриховкой, хотя
MuJoCo рисует серую клетку. Виноват либо перенос кадра в окно, либо сам
кадр. Здесь проверяется перенос: берётся заведомо правильная картинка,
кладётся в кадр теми же вызовами, что и живой кадр, и сравнивается с
исходником по каналам.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\blit_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent


def main():
    from PIL import Image

    from flybrain.desk.shell import Offscreen

    src_path = OUT / "body_camera_right.png"
    src = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    print("исходник %s %s" % (src_path.name, src.shape))

    sh = Offscreen((900, 700))
    for scale, tag in ((1.0, "один к одному"), (0.5, "вдвое меньше")):
        ctx = sh.begin()
        h, w = src.shape[0], src.shape[1]
        dw, dh = int(w * scale), int(h * scale)
        ctx.dl.image(src, 20, 20, dw, dh)
        sh.end()
        a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
        got = (a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
        cut = got[20:20 + dh, 20:20 + dw]
        Image.fromarray(cut).save(OUT / ("blit_" + str(scale) + ".png"))
        if scale == 1.0:
            d = np.abs(cut.astype(np.int16) - src.astype(np.int16))
            print("  %s: наибольшее расхождение по каналам %s, среднее %.2f"
                  % (tag, d.max(axis=(0, 1)).tolist(), d.mean()))
        # «серость»: у серого пикселя каналы совпадают. Если перенос
        # сдвигает каналы, серый пол перестаёт быть серым.
        def greyness(img):
            f = img.astype(np.int16)
            return float(np.abs(f - f.mean(axis=2, keepdims=True)
                                ).mean())
        print("  %s: разброс каналов в исходнике %.2f, в кадре %.2f"
              % (tag, greyness(src), greyness(cut)))
    sh.close()


if __name__ == "__main__":
    main()
