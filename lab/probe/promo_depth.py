r"""Заставка по карте глубины: НАШИ ЖЕ кадры через ComfyUI.

Ничего не генерируется заново. На вход идёт кусок уже снятого ролика —
муха в MuJoCo, — и из него делается карта глубины. То есть это не
выдуманная муха и не «нейросеть нарисовала», а обработка собственной
записи: исходник остаётся в кадре узнаваемым.

Почему Video-Depth-Anything, а не покадровый Depth Anything v2. Покадровая
глубина на видео мерцает: у соседних кадров независимые оценки, и ровный
пол начинает дышать. `VideoDepthAnythingProcess` держит временную
согласованность, ради неё модель и сделана.

Палитра `inferno` выбрана не для красоты вообще, а ради связи с остальным
роликом: активность мозга в окне красится магмой, и заставка попадает в
тот же ряд, вместо того чтобы быть куском из другого фильма.

ComfyUI должен быть уже поднят:
    D:\Comfy_HVideo_1\ComfyUI_windows_portable\run_nvidia_gpu.bat

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\promo_depth.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

СЕРВЕР = "http://127.0.0.1:8188"
КОРЕНЬ = Path(__file__).resolve().parents[2]
PROMO = КОРЕНЬ / "promo"
COMFY = Path(r"D:\Comfy_HVideo_1\ComfyUI_windows_portable\ComfyUI")
ВЫХОД = COMFY / "output"
FF = (КОРЕНЬ / ".venv/Lib/site-packages/imageio_ffmpeg/binaries"
      / "ffmpeg-win-x86_64-v7.1.exe")

# Исходник — отдельный короткий прогон одного вида
# (`promo.py --заставка`), а не кусок ролика: в ролике сцена тела идёт
# двумя видами сразу, и каждая половина вертикальная. Прогон тот же
# самый: та же песочница, то же тело, та же сглаженная камера.
ОТ_СЕК, ДЛИНА_СЕК, FPS = 0.2, 3.0, 50


def позвать(путь, данные=None):
    тело = None if данные is None else json.dumps(данные).encode("utf-8")
    req = urllib.request.Request(
        СЕРВЕР + путь, data=тело,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def жив():
    try:
        позвать("/system_stats")
        return True
    except (urllib.error.URLError, OSError):
        return False


def нарезать(исходник, куда):
    """Вырезать кусок сцены «тело» и ОБРЕЗАТЬ всё, что не сцена.

    Обрезка здесь не косметика. Первый прогон шёл по кадру целиком, и
    модель глубины честно приняла за геометрию то, что видела: панель
    интерфейса стала ближним планом, рамка зоны — ступенькой в
    пространстве, а муха — крошечным пятном в середине ямы. Карта
    глубины была правильной картой НЕ ТОГО.

    Поэтому в кадре остаётся только отрисовка MuJoCo: сверху срезается
    панель кнопок, снизу — строка состояния, по бокам — до 16:9. Выход
    растягивается обратно до 1920×1080, то есть заставка идёт с лёгким
    наездом на муху, и это ей только на пользу.
    """
    окно = "crop=1440:810:190:155,scale=1920:1080:flags=lanczos"
    cmd = [str(FF), "-v", "error", "-y", "-ss", str(ОТ_СЕК),
           "-i", str(исходник), "-t", str(ДЛИНА_СЕК), "-vf", окно,
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
           str(куда)]
    subprocess.run(cmd, check=True)
    return куда


def граф(исходник, кадров):
    """API-формат: {"<id>": {class_type, inputs}}, связи — ["<id>", слот]."""
    return {
        "1": {"class_type": "VHS_LoadVideoPath", "inputs": {
            "video": str(исходник), "force_rate": 0,
            "custom_width": 0, "custom_height": 0,
            "frame_load_cap": кадров, "skip_first_frames": 0,
            "select_every_nth": 1}},
        # vitl — самая крупная из трёх, и она здесь уместна: три секунды
        # считаются один раз, а заставка идёт первой, её и разглядывают.
        "2": {"class_type": "LoadVideoDepthAnythingModel", "inputs": {
            "model": "video_depth_anything_vitl.pth"}},
        "3": {"class_type": "VideoDepthAnythingProcess", "inputs": {
            "vda_model": ["2", 0], "images": ["1", 0],
            "input_size": 518, "max_res": 1280, "precision": "fp16"}},
        "4": {"class_type": "VideoDepthAnythingOutput", "inputs": {
            "depths": ["3", 0], "colormap": "inferno"}},
        "5": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["4", 0], "frame_rate": float(FPS),
            "loop_count": 0, "filename_prefix": "flybrain_depth",
            # pix_fmt / crf / save_metadata / trim_to_audio — это не
            # вольные имена, а параметры КОНКРЕТНОГО формата: узел
            # объявляет их в `formats["video/h264-mp4"]`, и подсмотреть
            # их можно только в /object_info.
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16,
            "save_metadata": False, "trim_to_audio": False,
            "pingpong": False, "save_output": True}},
    }


def ждать(pid, предел=1800):
    край = time.time() + предел
    while time.time() < край:
        h = позвать("/history/" + pid)
        if pid in h:
            зап = h[pid]
            st = зап.get("status", {})
            if st.get("status_str") == "error" or not st.get("completed", True):
                print("  ComfyUI вернул ошибку:", flush=True)
                for m in st.get("messages", [])[-6:]:
                    print("   ", m, flush=True)
                return None
            return зап
        time.sleep(2.0)
    print("  не дождались за %d с" % предел, flush=True)
    return None


def смешать(исходник, глубина, цель=None):
    """Глубина поверх исходника мягким светом.

    Одна голая карта глубины — ровный градиент: пол уходит вдаль
    гладким пятном, и фактуры в кадре не остаётся вовсе. Поэтому она
    кладётся на собственный кадр режимом `softlight`: силуэт мухи и
    клетка пола читаются, а глубина красит.

    Шаг этот раньше делался руками отдельной командой, и в скриптах его
    не было — то есть повторить сборку по репозиторию было нельзя.
    Теперь он здесь.

    `scale` до смешивания обязателен: выход модели идёт 1280×720, а
    `blend` отказывается работать с разными размерами.
    """
    цель = цель or (PROMO / "заставка.mp4")
    фильтр = ("[1:v]scale=1920:1080:flags=lanczos[d];"
              "[0:v][d]blend=all_mode=softlight[v]")
    cmd = [str(FF), "-v", "error", "-y", "-i", str(исходник),
           "-i", str(глубина), "-filter_complex", фильтр, "-map", "[v]",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           str(цель)]
    subprocess.run(cmd, check=True)
    return цель


def main():
    if not жив():
        print("ComfyUI не отвечает на " + СЕРВЕР)
        print("Запустите run_nvidia_gpu.bat и повторите.")
        return 1
    # Исходник можно задать: пока идёт пересъёмка, основной файл
    # переписывается на ходу, и читать его нельзя.
    исходник = (Path(sys.argv[1]) if len(sys.argv) > 1
                else PROMO / "_заставка-исходник.mp4")
    if not исходник.exists():
        print("нет исходника " + str(исходник))
        return 1

    кусок = PROMO / "_глубина-исходник.mp4"
    нарезать(исходник, кусок)
    кадров = int(ДЛИНА_СЕК * FPS)
    print("  нарезано %.1f с от %.1f с, %d кадров"
          % (ДЛИНА_СЕК, ОТ_СЕК, кадров), flush=True)

    cid = str(uuid.uuid4())
    ответ = позвать("/prompt", {"prompt": граф(кусок, кадров),
                                "client_id": cid})
    pid = ответ.get("prompt_id")
    if not pid:
        print("ComfyUI не принял задание:", ответ)
        return 1
    print("  задание %s, считается..." % pid[:8], flush=True)
    зап = ждать(pid)
    if зап is None:
        return 1

    файлы = []
    for узел, out in (зап.get("outputs") or {}).items():
        for ключ in ("gifs", "videos", "images"):
            for z in out.get(ключ, []) or []:
                p = ВЫХОД / (z.get("subfolder") or "") / z["filename"]
                if p.exists():
                    файлы.append(p)
    if not файлы:
        print("  ComfyUI досчитал, но файла на выходе нет")
        return 1
    готов = max(файлы, key=lambda p: p.stat().st_mtime)
    цель = PROMO / "заставка-глубина.mp4"
    цель.write_bytes(готов.read_bytes())
    print("  глубина: %s (%.1f МБ)"
          % (цель.name, цель.stat().st_size / (1 << 20)), flush=True)
    готовая = смешать(кусок, цель)
    print("  заставка: %s (%.1f МБ)"
          % (готовая.name, готовая.stat().st_size / (1 << 20)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
