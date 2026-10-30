"""Воспроизведение записанного зрения: побитово между свежими процессами.

Что уже установлено замером
---------------------------

* эпизод БЕЗ зрения воспроизводим побитово и в процессе, и между
  процессами (`probe/vision_repro.py`, 3 зерна из 3 на обоих уровнях);
* эпизод СО ЖИВЫМ зрением — только в процессе (между процессами 2 из 3);
* первая расходящаяся ступень названа и это **сырые пиксели**
  `physics.render`: отпечаток выхода видеокарты отличается между
  свежими процессами уже на шаге 0, при побитово совпавшем всём
  остальном (`probe/retina_repro.py`).

Отсюда задача этой пробы: показать, что **воспроизведение записанной
сенсорики от видеокарты не зависит**, то есть даёт побитовое равенство
между свежими процессами. И показать это рядом с живой дорогой, в одном
прогоне, чтобы разницу нельзя было приписать условиям замера.

Две воспроизводимости, которые нельзя путать
--------------------------------------------

``живой рендер``
    воспроизводимости между процессами НЕТ, и это свойство видеокарты;

``воспроизведение из записи``
    воспроизводимость обязана быть, потому что рендер в ней не
    участвует вовсе.

Проба проверяет обе и печатает их рядом.

Встречные замеры, без которых проба ничего не стоит
---------------------------------------------------

Сверка записи обязана УМЕТЬ провалиться, иначе «сверено 15 из 15» —
пустые слова. Проверяется, что воспроизведение отказывает:

* на **чужом шаге управления** — кадры пришлись бы на другое модельное
  время;
* на **короткой записи** — прогон просит шаг, которого в ленте нет;
* на **испорченном кадре** — отпечаток манифеста не сходится.

Каждый отказ обязан быть громким и называть причину, а не подставлять
ближайший кадр. Причина опознаётся по **ASCII-метке** в тексте отказа
(``[TAPE-STEP]``, ``[TAPE-END]``, ``[TAPE-HASH]``), а не по русским
словам: кодировка вывода дочернего процесса читателю не подчинена, и
сверка по кириллице проваливается на исправном отказе. Поймано
прогоном.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_tape.py
    ..\\.venv\\Scripts\\python.exe probe\\vision_tape.py --шагов 20 --зерно 1

Коды: 0 — всё сошлось; 1 — есть провал.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

fails = []
rep = {"проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  — " + text)
    rep.setdefault("оговорки", []).append(text)


def отпечаток_эпизода(run):
    """Отпечаток по траектории, растру и счёту спайков вместе."""
    recs = run.records
    поля = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
            "amp_l", "amp_r", "spikes")
    t = np.array([[r.get(k, np.nan) for k in поля] for r in recs],
                 dtype=np.float64)
    bins = np.asarray(run.bins.cpu().numpy() if hasattr(run.bins, "cpu")
                      else run.bins)
    counts = np.asarray(run.counts.cpu().numpy()
                        if hasattr(run.counts, "cpu") else run.counts)
    h = hashlib.sha256()
    for a in (t, bins, counts):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def эпизод(зерно, шагов, лента=None, роль=None, control_ms=20.0):
    """Эпизод со зрением: живой, с записью или из записи."""
    import torch
    from flybrain import body as B, data as D, objects as O

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=dev)
    kw = dict(duration_s=шагов * control_ms / 1000.0, control_ms=control_ms,
              seed=зерно, device=dev, readout="ALPN", vision=True,
              vis_objects=1, vision_hz=100.0,
              vis_track=O.preset("loom_dark"))
    if роль == "запись":
        kw["vision_record"] = str(лента)
    elif роль == "воспроизведение":
        kw["vision_replay"] = str(лента)
    run = B.run_episode(cfg=B.BodyConfig(**kw), neurons=neurons,
                        weights=weights, progress=False)
    return отпечаток_эпизода(run), (run.timing or {}).get(
        "зрение", {}).get("лента")


def в_процессе(зерно, шагов, лента=None, роль=None, control_ms=20.0):
    """Тот же эпизод отдельным процессом. Метка ASCII нарочно."""
    код = (
        "import sys; sys.path.insert(0, '.')\n"
        "import probe._isolate\n"
        "from probe.vision_tape import эпизод\n"
        "fp, _ = эпизод(%d, %d, %r, %r, %r)\n"
        "print('FP=' + fp)\n"
        % (int(зерно), int(шагов),
           None if лента is None else str(лента), роль, float(control_ms)))
    корень = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-I", "-c", код], cwd=str(корень),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600)
    for строка in (r.stdout or "").splitlines():
        if "FP=" in строка:
            return строка.split("FP=", 1)[1].strip()
    return ("нет метки FP=; код возврата %s; хвост stderr: %s"
            % (r.returncode, (r.stderr or "")[-300:]))


def отказ(лента, зерно, шагов, control_ms=20.0):
    """Текст отказа воспроизведения, либо None если отказа не было."""
    код = (
        "import sys; sys.path.insert(0, '.')\n"
        "import probe._isolate\n"
        "from probe.vision_tape import эпизод\n"
        "try:\n"
        "    эпизод(%d, %d, %r, 'воспроизведение', %r)\n"
        "    print('FP=НЕТ ОТКАЗА')\n"
        "except BaseException as e:\n"
        "    print('FP=' + type(e).__name__ + ': ' "
        "+ str(e).replace(chr(10), ' ')[:160])\n"
        % (int(зерно), int(шагов), str(лента), float(control_ms)))
    корень = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-I", "-c", код], cwd=str(корень),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600)
    for строка in (r.stdout or "").splitlines():
        if "FP=" in строка:
            т = строка.split("FP=", 1)[1].strip()
            return None if т == "НЕТ ОТКАЗА" else т
    return "проба не получила ответа; хвост stderr: %s" % (
        (r.stderr or "")[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--зерно", type=int, default=1,
                    help="зерно. По умолчанию 1 — то, на котором живой "
                         "рендер и разошёлся между процессами")
    ap.add_argument("--шагов", type=int, default=20)
    a = ap.parse_args()

    из = OUT.parent / "bench" / "лента-зрения"
    из.mkdir(parents=True, exist_ok=True)
    лента = из / ("зерно%d.npz" % a.зерно)
    for п in (лента, лента.with_suffix(".json")):
        if п.exists():
            п.unlink()

    print("— воспроизведение записанного зрения —")
    print("  зерно %d, шагов управления %d" % (a.зерно, a.шагов))
    rep["зерно"] = a.зерно
    rep["шагов"] = a.шагов

    # --- 1. запись, отдельным процессом ------------------------------
    зап = в_процессе(a.зерно, a.шагов, лента, "запись")
    check("запись сделана отдельным процессом",
          лента.exists() and not зап.startswith("нет метки"),
          зап if зап.startswith("нет метки") else "эпизод %s" % зап)
    if not лента.exists():
        print("\nбез записи сравнивать нечего")
        return 1
    ман = json.loads(лента.with_suffix(".json").read_text(encoding="utf-8"))
    rep["манифест"] = {k: ман[k] for k in
                       ("версия", "шагов", "кадров непустых", "форма кадра",
                        "тип", "шаг управления, мс",
                        "частота обновления зрения, Гц", "отпечаток ленты")}
    check("манифест называет форму, тип, число шагов и отпечаток ленты",
          ман["шагов"] == a.шагов and ман["отпечаток ленты"]
          and ман["форма кадра"] and ман["тип"],
          "шагов %d, форма %s, тип %s, отпечаток %s"
          % (ман["шагов"], ман["форма кадра"], ман["тип"],
             ман["отпечаток ленты"]))
    check("в манифесте есть модельное время и отпечаток КАЖДОГО шага",
          len(ман["модельное время, мс"]) == a.шагов
          and len(ман["отпечаток каждого шага"]) == a.шагов,
          "%d отметок времени, %d отпечатков"
          % (len(ман["модельное время, мс"]),
             len(ман["отпечаток каждого шага"])))

    # --- 2. ЖИВАЯ дорога между процессами: встречный замер -----------
    ж1 = в_процессе(a.зерно, a.шагов)
    ж2 = в_процессе(a.зерно, a.шагов)
    rep["живой рендер"] = {"первый": ж1, "второй": ж2}
    живое_совпало = (ж1 == ж2)
    note("живой рендер между процессами: %s против %s — %s"
         % (ж1, ж2, "совпало" if живое_совпало else "РАЗОШЛОСЬ"))

    # --- 3. ВОСПРОИЗВЕДЕНИЕ между процессами: главное утверждение ----
    в1 = в_процессе(a.зерно, a.шагов, лента, "воспроизведение")
    в2 = в_процессе(a.зерно, a.шагов, лента, "воспроизведение")
    rep["воспроизведение"] = {"первый": в1, "второй": в2}
    check("ВОСПРОИЗВЕДЕНИЕ записи совпало побитово между свежими "
          "процессами", в1 == в2 and not в1.startswith("нет метки"),
          "%s против %s" % (в1, в2))
    if живое_совпало:
        note("на этом зерне и этой длине живой рендер тоже совпал, "
             "поэтому сравнение двух дорог здесь ничего не различает — "
             "расхождение живого пути замерено отдельно "
             "(probe/vision_repro.py, probe/retina_repro.py)")
    else:
        check("и это ОТЛИЧАЕТ его от живого рендера, который разошёлся",
              в1 == в2 and not живое_совпало,
              "воспроизведение %s = %s, живое %s != %s"
              % (в1, в2, ж1, ж2))

    # --- 4. сверка записи УМЕЕТ провалиться --------------------------
    о = отказ(лента, a.зерно, a.шагов, control_ms=10.0)
    check("воспроизведение отказывает на ЧУЖОМ шаге управления",
          о is not None and "[TAPE-STEP]" in str(о), о)

    о = отказ(лента, a.зерно, a.шагов + 5)
    check("воспроизведение отказывает на КОРОТКОЙ записи",
          о is not None and "[TAPE-END]" in str(о), о)

    порча = из / ("порча%d.npz" % a.зерно)
    shutil.copy(лента, порча)
    shutil.copy(лента.with_suffix(".json"), порча.with_suffix(".json"))
    d = dict(np.load(порча))
    л = d["лента"]
    if л.size:
        л = л.copy()
        # Портится ОДИН элемент одного кадра: проверка обязана ловить
        # порчу, а не только обрезание.
        плоско = л.reshape(-1)
        плоско[0] = np.float32(плоско[0] + 1.0)
        d["лента"] = л
        np.savez_compressed(порча, **d)
        о = отказ(порча, a.зерно, a.шагов)
        check("воспроизведение отказывает на ИСПОРЧЕННОМ кадре",
              о is not None and "[TAPE-HASH]" in str(о), о)
    else:
        check("воспроизведение отказывает на ИСПОРЧЕННОМ кадре", False,
              "лента пуста, портить нечего")

    (OUT / "vision_tape.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    if fails:
        print("ПРОВАЛЫ: %s" % ", ".join(fails))
        return 1
    print("воспроизведение записи детерминировано, и сверка умеет "
          "провалиться")
    return 0


if __name__ == "__main__":
    sys.exit(main())
