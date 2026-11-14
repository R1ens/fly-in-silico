r"""Насколько трясётся камера в ГОТОВОЙ записи.

Замер по самому ролику, а не по тому, что мы думаем о камере. Берётся
кусок сцены и считается среднее отличие соседних кадров: пол в сцене
клетчатый и контрастный, поэтому покачивание камеры даёт разницу по
всему кадру, а ход самой мухи — только там, где она. Число безразмерное,
0..255 по яркости, и смысл имеет РАЗНИЦА между прогонами, а не оно само.

Отдельно печатается доля кадра, изменившаяся больше чем на 8 уровней:
при тряске это почти весь кадр, при спокойной камере — пятно вокруг
мухи.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\promo_steady.py ..\promo\fly-in-silico.mp4 4.5 8.5
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

КОРЕНЬ = Path(__file__).resolve().parents[2]
FF = (КОРЕНЬ / ".venv/Lib/site-packages/imageio_ffmpeg/binaries"
      / "ffmpeg-win-x86_64-v7.1.exe")
Ш, В = 480, 270


def кадры(файл, от, до):
    cmd = [str(FF), "-v", "error", "-ss", str(от), "-i", str(файл),
           "-t", str(до - от), "-vf", "scale=%d:%d" % (Ш, В),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    сырое = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    n = len(сырое) // (Ш * В)
    return np.frombuffer(сырое[:n * Ш * В], dtype=np.uint8).reshape(n, В, Ш)


def main():
    файл = Path(sys.argv[1])
    от = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5
    до = float(sys.argv[3]) if len(sys.argv) > 3 else 8.5
    к = кадры(файл, от, до).astype(np.int16)
    if len(к) < 3:
        print("мало кадров: %d" % len(к))
        return 1
    d = np.abs(np.diff(к, axis=0))
    среднее = d.mean(axis=(1, 2))
    доля = (d > 8).mean(axis=(1, 2))
    print("%s  %.1f..%.1f с, кадров %d" % (файл.name, от, до, len(к)))
    print("  отличие соседних кадров: среднее %.2f, медиана %.2f, "
          "наибольшее %.2f" % (среднее.mean(), np.median(среднее),
                               среднее.max()))
    print("  изменилось больше 8 уровней: %.1f %% кадра (медиана)"
          % (np.median(доля) * 100.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
