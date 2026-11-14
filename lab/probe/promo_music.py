r"""Музыкальная подложка под ролик: ACE-Step через ComfyUI.

Единственное в этой затее, что ничего не утверждает о мухе. Картинка
обязана быть записью настоящего окна, а подложка — это подложка: её
задача не мешать и держать тридцать секунд.

Отсюда и теги. Никакой мелодии, никакого ритма с ударными: под запись
прибора нужен ровный гул, в котором ничего не происходит. Вокал
запрещён прямо — и тегом, и пустым `[instrumental]` в месте слов.

ComfyUI должен быть уже поднят, веса — на месте:
    models/checkpoints/ace_step_v1_3.5b.safetensors  (7,7 ГБ)

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\promo_music.py [секунд] [зерно]
"""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

СЕРВЕР = "http://127.0.0.1:8188"
КОРЕНЬ = Path(__file__).resolve().parents[2]
PROMO = КОРЕНЬ / "promo"
ВЫХОД = Path(r"D:\Comfy_HVideo_1\ComfyUI_windows_portable\ComfyUI\output")

ТЕГИ = ("ambient, dark ambient, instrumental, no vocals, no drums, "
        "slow, minimal, sustained synth pad, low drone, sparse, "
        "cinematic, scientific documentary, calm, cold, spacious, "
        "subtle analog texture, 60 bpm")
НЕ_НАДО = ("vocals, singing, voice, choir, drums, percussion, beat, "
           "guitar, pop, edm, dubstep, distortion, applause, speech")


def позвать(путь, данные=None):
    тело = None if данные is None else json.dumps(данные).encode("utf-8")
    req = urllib.request.Request(
        СЕРВЕР + путь, data=тело,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def граф(секунд, зерно):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ace_step_v1_3.5b.safetensors"}},
        # Слова пустые: `[instrumental]` — это принятая у ACE-Step
        # пометка «здесь не поют», а не текст песни.
        "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": ТЕГИ,
            "lyrics": "[instrumental]", "lyrics_strength": 1.0}},
        "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {
            "clip": ["1", 1], "tags": НЕ_НАДО,
            "lyrics": "", "lyrics_strength": 1.0}},
        "4": {"class_type": "EmptyAceStepLatentAudio", "inputs": {
            "seconds": float(секунд), "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": int(зерно), "steps": 50,
            "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "denoise": 1.0}},
        "6": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveAudio", "inputs": {
            "audio": ["6", 0], "filename_prefix": "audio/flybrain_bed"}},
    }


def ждать(pid, предел=2400):
    край = time.time() + предел
    while time.time() < край:
        h = позвать("/history/" + pid)
        if pid in h:
            зап = h[pid]
            st = зап.get("status", {})
            if st.get("status_str") == "error":
                print("  ComfyUI вернул ошибку:", flush=True)
                for m in st.get("messages", [])[-8:]:
                    print("   ", m, flush=True)
                return None
            return зап
        time.sleep(3.0)
    print("  не дождались за %d с" % предел, flush=True)
    return None


def main():
    секунд = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
    зерно = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    try:
        позвать("/system_stats")
    except (urllib.error.URLError, OSError):
        print("ComfyUI не отвечает на " + СЕРВЕР)
        return 1

    cid = str(uuid.uuid4())
    ответ = позвать("/prompt", {"prompt": граф(секунд, зерно),
                                "client_id": cid})
    pid = ответ.get("prompt_id")
    if not pid:
        print("ComfyUI не принял задание:", ответ)
        return 1
    print("  задание %s, %.0f с музыки, зерно %d..."
          % (pid[:8], секунд, зерно), flush=True)
    зап = ждать(pid)
    if зап is None:
        return 1

    файлы = []
    for _узел, out in (зап.get("outputs") or {}).items():
        for z in (out.get("audio") or []):
            p = ВЫХОД / (z.get("subfolder") or "") / z["filename"]
            if p.exists():
                файлы.append(p)
    if not файлы:
        print("  досчитал, но звука на выходе нет")
        return 1
    готов = max(файлы, key=lambda p: p.stat().st_mtime)
    цель = PROMO / ("подложка%s" % готов.suffix)
    цель.write_bytes(готов.read_bytes())
    print("  готово: %s (%.1f МБ)"
          % (цель.name, цель.stat().st_size / (1 << 20)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
