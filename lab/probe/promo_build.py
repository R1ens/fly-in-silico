r"""Сборка готового ролика: заставка + запись окна + подложка.

Три части, и у каждой своё происхождение, которое важно не путать:

* `fly-in-silico.mp4` — ЗАПИСЬ настоящего окна, кадр привязан к
  модельному времени (`promo.py --видео`). Ничего не добавлено;
* `заставка.mp4` — те же кадры, пропущенные через Video-Depth-Anything
  в ComfyUI и смешанные с исходником (`promo_depth.py`). Обработка
  собственной записи, а не выдуманная картинка;
* `подложка.flac` — ACE-Step в ComfyUI (`promo_music.py`). Единственное
  место, где ничего не утверждается о мухе.

На выходе две версии. Своя частота ролика — 50 кадров в секунду, и она
не выбрана, а следует из модели: один кадр — один шаг управления, 20 мс.
Версия на 30 выбрасывает два кадра из пяти и идёт с рывками; она здесь
только потому, что её попросили.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\promo_build.py
"""

import json
import subprocess
import sys
from pathlib import Path

КОРЕНЬ = Path(__file__).resolve().parents[2]
PROMO = КОРЕНЬ / "promo"
FF = (КОРЕНЬ / ".venv/Lib/site-packages/imageio_ffmpeg/binaries"
      / "ffmpeg-win-x86_64-v7.1.exe")

ЗАСТАВКА = PROMO / "заставка.mp4"
ЗАПИСЬ = PROMO / "fly-in-silico.mp4"
ЗВУК = PROMO / "подложка.flac"


def длина(путь):
    out = subprocess.run(
        [str(FF), "-hide_banner", "-i", str(путь)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    for строка in (out.stderr or "").splitlines():
        if "Duration:" in строка:
            ч, м, с = строка.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(ч) * 3600 + int(м) * 60 + float(с)
    return 0.0


def собрать(цель, fps, всего):
    """Склейка и звук одним проходом: промежуточных файлов не остаётся."""
    # Звук: ровный уровень, вход и выход плавные. -16 LUFS — тихо для
    # музыки и громко для фона; здесь фон и есть музыка, говорить
    # поверх неё некому.
    звук = ("[2:a]atrim=0:%0.3f,aformat=sample_rates=48000:"
            "channel_layouts=stereo,loudnorm=I=-16:TP=-1.5:LRA=11,"
            "afade=t=in:st=0:d=1.2,afade=t=out:st=%0.3f:d=2.0[a]"
            % (всего, max(0.0, всего - 2.0)))
    фильтр = ("[0:v]settb=1/90000,fps=%d[v0];"
              "[1:v]settb=1/90000,fps=%d[v1];"
              "[v0][v1]concat=n=2:v=1:a=0[v];%s" % (fps, fps, звук))
    cmd = [str(FF), "-v", "error", "-y",
           "-i", str(ЗАСТАВКА), "-i", str(ЗАПИСЬ), "-i", str(ЗВУК),
           "-filter_complex", фильтр, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-crf", "20", "-preset", "slow",
           # `-ar` ПОСЛЕ loudnorm и обязательно: фильтр работает на
           # своей внутренней частоте и отдаёт 192 кГц, а кодек брал её
           # как есть и писал 96 кГц в звуковой дорожке к ролику.
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
           "-b:a", "192k",
           "-movflags", "+faststart", str(цель)]
    subprocess.run(cmd, check=True)
    return цель


def main():
    нет = [p.name for p in (ЗАСТАВКА, ЗАПИСЬ, ЗВУК) if not p.exists()]
    if нет:
        print("не хватает: " + ", ".join(нет))
        return 1
    всего = длина(ЗАСТАВКА) + длина(ЗАПИСЬ)
    отчёт = {"части": {}, "выход": {}}
    for p in (ЗАСТАВКА, ЗАПИСЬ, ЗВУК):
        отчёт["части"][p.name] = round(длина(p), 2)
    print("  заставка %.1f с + запись %.1f с = %.1f с"
          % (длина(ЗАСТАВКА), длина(ЗАПИСЬ), всего), flush=True)

    for fps in (50, 30):
        цель = PROMO / ("fly-in-silico-%dfps.mp4" % fps)
        собрать(цель, fps, всего)
        мб = цель.stat().st_size / (1 << 20)
        отчёт["выход"][цель.name] = {"кадров в секунду": fps,
                                     "с": round(длина(цель), 2),
                                     "МБ": round(мб, 1)}
        print("  %s — %.1f с, %.1f МБ" % (цель.name, длина(цель), мб),
              flush=True)
    (PROMO / "promo_build.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
