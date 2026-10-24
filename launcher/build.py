"""Сборка FlyBrain.exe. Научный стек внутрь не попадает — и это проверяется.

Собирается только пускатель: стандартная библиотека плюс свой
интерпретатор от PyInstaller. torch, CUDA, Taichi, MuJoCo, numpy, pandas и
коннектом остаются там, где лежат, — в окружении проекта и в `upstream/`.
Чтобы это не держалось на честном слове, большие модули перечислены в
`--exclude-module`, а после сборки exe проверяется на их имена.

Каталог `out/` с научными результатами сборка не трогает вовсе: ни читает,
ни пишет, ни копирует. Временные файлы PyInstaller уходят в `launcher/build`.

    cd D:\\fly-brain && .venv\\Scripts\\python.exe launcher\\build.py
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE = ROOT / "release"
WORK = HERE / "build"

# Всё, чему в пускателе делать нечего. Список не для красоты: без него
# любая будущая правка, случайно затащившая numpy, раздула бы exe в сотни
# мегабайт и утянула бы внутрь кусок научного стека.
HEAVY = (
    "torch", "taichi", "mujoco", "flygym", "flygym_gymnasium",
    "numpy", "pandas", "pyarrow", "scipy", "matplotlib", "PIL", "cv2",
    "tqdm", "IPython", "pytest", "setuptools", "pip", "tkinter",
    "flybrain",
)


def build(name, console, src):
    """Один exe. Возвращает путь к нему."""
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--onefile", "--clean",
        "--console" if console else "--windowed",
        "--name", name,
        "--distpath", str(RELEASE),
        "--workpath", str(WORK),
        "--specpath", str(WORK),
    ]
    for mod in HEAVY:
        args += ["--exclude-module", mod]
    args.append(str(src))
    print("  " + " ".join(args[2:8]) + " …")
    t0 = time.perf_counter()
    p = subprocess.run(args, cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    took = time.perf_counter() - t0
    if p.returncode != 0:
        print((p.stdout or "")[-2000:])
        print((p.stderr or "")[-2000:])
        raise SystemExit("PyInstaller вернул %d" % p.returncode)
    exe = RELEASE / (name + ".exe")
    print("  собран %s: %.1f МБ за %.0f с"
          % (exe.name, exe.stat().st_size / 1e6, took))
    return exe


def check_clean(exe):
    """Убедиться, что внутрь не затащило научный стек.

    Ищем имена модулей прямо в байтах: PyInstaller кладёт список
    упакованного в оглавление архива, и torch оттуда видно невооружённым
    глазом. Проверка грубая, зато её нельзя обмануть настройкой.
    """
    blob = exe.read_bytes()
    found = []
    for mod in ("torch", "taichi", "mujoco", "numpy", "pandas", "pyarrow",
                "scipy"):
        if (mod + ".pyd").encode() in blob or (b"\\" + mod.encode()
                                               + b"\\__init__") in blob:
            found.append(mod)
    return found


def main():
    src = HERE / "flybrain_launcher.py"
    if not src.exists():
        raise SystemExit("нет " + str(src))

    # Корень проекта зашивается запасным вариантом: он нужен лишь тогда,
    # когда exe унесли из проекта. Это первая местная сборка, и знать
    # свой дом ей не зазорно.
    text = src.read_text(encoding="utf-8")
    stamped = HERE / "_launcher_stamped.py"
    stamped.write_text(text.replace('BUILD_ROOT = r""',
                                    'BUILD_ROOT = r"%s"' % ROOT),
                       encoding="utf-8")
    console_src = HERE / "_launcher_console.py"
    shutil.copyfile(stamped, console_src)

    RELEASE.mkdir(parents=True, exist_ok=True)
    print("каталог выпуска:", RELEASE)
    made = []
    try:
        made.append(build("FlyBrain", False, stamped))
        made.append(build("FlyBrain-console", True, console_src))
    finally:
        for f in (stamped, console_src):
            f.unlink(missing_ok=True)

    print()
    for exe in made:
        dirty = check_clean(exe)
        print("  %-22s %6.1f МБ  sha256 %s  научный стек: %s"
              % (exe.name, exe.stat().st_size / 1e6,
                 hashlib.sha256(exe.read_bytes()).hexdigest()[:16],
                 ", ".join(dirty) if dirty else "не найден"))
        if dirty:
            raise SystemExit("в %s затащило: %s" % (exe.name,
                                                    ", ".join(dirty)))
    shutil.rmtree(WORK, ignore_errors=True)
    print("\nготово. Временные файлы сборки убраны, out/ не тронут.")


if __name__ == "__main__":
    main()
