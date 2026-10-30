"""Где именно расходится эпизод со зрением: ступень за ступенью.

Что уже замерено и чего ещё нет
-------------------------------

`probe/vision_repro.py` показал: эпизод БЕЗ зрения воспроизводим побитово
и в одном процессе, и между процессами (3 зерна из 3 на обоих уровнях), а
СО зрением — в процессе да (3 из 3), между процессами нет (2 из 3, зерно
1 разошлось). Отсюда был сделан ВЫВОД, что причина в рендере: это
единственный вход, зависящий от процесса.

**Вывод — не замер.** Эта проба его проверяет: отпечатки берутся на
каждой ступени ОТДЕЛЬНО, и первое расхождение называет причину.

Ступени и где они перехватываются
---------------------------------

Производственный код не меняется ни на строку: всё перехватывается
подменой на месте, из пробы.

1. **сырые пиксели** — `physics.render(...)` в `Fly._update_vision`. Это
   выход видеокарты, до любой обработки. Перехват: подмена метода
   `render` у объекта `physics`, полученного через `on_body`;
2. **рыбий глаз** — `retina.correct_fisheye(raw_img)`, numba на
   процессоре. Берётся из `fly._curr_raw_visual_input` (для этого
   `render_raw_vision` поднимается в `on_body`);
3. **омматидии** — `retina.raw_image_to_hex_pxls(fish_img)`, numba.
   Это `obs["vision"]`, то есть `fly._curr_visual_input`;
4. **частоты фоторецепторов** — `vision.RetinaAdapter.hz`, наш пересчёт
   яркости в герцы;
5. **вход сети** — весь вектор `rates` после записи всех каналов;
6. **спайки** — строка растра этого шага управления;
7. **команда** — `amp_l`, `amp_r`, `dn_l`, `dn_r` из записи шага;
8. **физика** — `qpos` тела после шага.

Отпечаток каждой ступени — свой, и сравниваются они по шагам. Первая
ступень, на которой отпечатки двух процессов разошлись, И НОМЕР ШАГА —
это и есть ответ: если разошлись сырые пиксели, причина в рендере; если
пиксели совпали, а частоты нет — в пересчёте; если совпало всё до
спайков, причина ниже по пути, в состоянии или генераторе случайных
чисел.

Встречный замер
---------------

Без него проба ничего не стоила бы: **то же сравнение без зрения** должно
сойтись на всех ступенях. Тогда «ступень X разошлась» означает свойство
зрения, а не свойство замера.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\retina_repro.py
    ..\\.venv\\Scripts\\python.exe probe\\retina_repro.py --шагов 20 --зерно 1

Коды: 0 — расхождений нет или причина названа; 1 — проба не смогла
сравнить.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

# Ступени в порядке пути. Порядок важен: ответом считается ПЕРВАЯ
# разошедшаяся, и список задаёт, что значит «первая».
СТУПЕНИ = ("сырые пиксели", "рыбий глаз", "омматидии",
           "частоты фоторецепторов", "вход сети", "спайки", "команда",
           "физика")

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


def отп(a):
    if a is None:
        return None
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def записать(путь, зрение, зерно, шагов):
    """Один эпизод с отпечатками всех ступеней по шагам."""
    import torch
    from flybrain import body as B, data as D, objects as O, vision as V

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=dev)

    # Накопитель: на каждом шаге управления — отпечаток каждой ступени.
    по_шагам = []
    текущий = {}

    # --- 1. сырые пиксели: подмена `render` у объекта physics ---------
    держатель = {}

    def on_body(body):
        держатель["body"] = body
        if зрение:
            # Чтобы ступень «рыбий глаз» вообще появилась: без этого
            # FlyGym её не сохраняет.
            body.fly.render_raw_vision = True
            ориг = body.physics.render

            def рендер(*a, **kw):
                кадр = ориг(*a, **kw)
                # Кадров два на обновление (левый и правый глаз),
                # поэтому отпечаток накапливается по обоим в порядке
                # вызова, а не перезаписывается.
                текущий.setdefault("_пиксели", []).append(отп(кадр))
                return кадр

            body.physics.render = рендер

    # --- 2-5. ступени зрения: подмена метода класса -------------------
    ориг_feed = V.RetinaAdapter.feed

    def feed(self, rates, frame, contrast=1.0):
        текущий["омматидии"] = отп(np.asarray(frame))
        текущий["частоты фоторецепторов"] = отп(
            self.hz(frame, contrast=contrast))
        свод = ориг_feed(self, rates, frame, contrast=contrast)
        # Вход сети берётся ПОСЛЕ записи фоторецепторов, но до прямых
        # стимулов и вмешательств: так ступень отвечает именно за
        # зрительную подачу.
        текущий["вход сети"] = отп(rates.detach().cpu().numpy())
        return свод

    if зрение:
        V.RetinaAdapter.feed = feed

    def on_step(step, rec, row_cpu):
        body = держатель.get("body")
        строка = {"шаг": int(step)}
        пикс = текущий.pop("_пиксели", None)
        строка["сырые пиксели"] = (отп(np.array(
            [int(h, 16) for h in пикс], dtype=np.uint64)) if пикс else None)
        строка["число кадров"] = len(пикс) if пикс else 0
        сырое = (getattr(body.fly, "_curr_raw_visual_input", None)
                 if body is not None else None)
        строка["рыбий глаз"] = отп(сырое) if сырое is not None else None
        for к in ("омматидии", "частоты фоторецепторов", "вход сети"):
            строка[к] = текущий.pop(к, None)
        строка["спайки"] = отп(row_cpu) if row_cpu is not None else None
        строка["команда"] = отп(np.array(
            [rec.get(k, np.nan) for k in ("amp_l", "amp_r", "dn_l", "dn_r")],
            dtype=np.float64))
        строка["физика"] = (отп(np.asarray(body.physics.data.qpos))
                            if body is not None else None)
        по_шагам.append(строка)

    kw = dict(duration_s=шагов * 0.02, control_ms=20.0, seed=зерно,
              device=dev, readout="ALPN")
    if зрение:
        kw.update(vision=True, vis_objects=1, vision_hz=100.0,
                  vis_track=O.preset("loom_dark"))
    try:
        B.run_episode(cfg=B.BodyConfig(**kw), neurons=neurons,
                      weights=weights, on_body=on_body, on_step=on_step,
                      progress=False)
    finally:
        if зрение:
            V.RetinaAdapter.feed = ориг_feed

    Path(путь).write_text(
        json.dumps({"зрение": bool(зрение), "зерно": зерно,
                    "шагов": шагов, "по шагам": по_шагам},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("FP=записано %d шагов" % len(по_шагам))


def в_процессе(путь, зрение, зерно, шагов):
    """Запустить запись отдельным процессом. Метка ASCII нарочно."""
    корень = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-I", str(Path(__file__)),
         "--запись", str(путь), "--зерно", str(зерно),
         "--шагов", str(шагов)] + ([] if зрение else ["--без-зрения"]),
        cwd=str(корень), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=3600)
    if not Path(путь).exists():
        return ("запись не появилась; код возврата %s; хвост stderr: %s"
                % (r.returncode, (r.stderr or "")[-300:]))
    return None


def сравнить(a, b):
    """Первая разошедшаяся ступень и номер шага, либо None."""
    ша, шб = a["по шагам"], b["по шагам"]
    if len(ша) != len(шб):
        return ("число шагов", -1, len(ша), len(шб))
    for i, (x, y) in enumerate(zip(ша, шб)):
        for к in СТУПЕНИ:
            if x.get(к) != y.get(к):
                return (к, i, x.get(к), y.get(к))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--запись", default=None,
                    help="внутренний режим: записать отпечатки в файл")
    ap.add_argument("--зерно", type=int, default=1,
                    help="зерно. По умолчанию 1 — то, на котором "
                         "`vision_repro` и нашёл расхождение")
    ap.add_argument("--шагов", type=int, default=30)
    ap.add_argument("--без-зрения", action="store_true", dest="без_зрения")
    a = ap.parse_args()

    if a.запись:
        записать(a.запись, not a.без_зрения, a.зерно, a.шагов)
        return 0

    из = OUT.parent / "bench" / "ступени"
    из.mkdir(parents=True, exist_ok=True)
    print("— где расходится эпизод со зрением: ступень за ступенью —")
    print("  зерно %d, шагов управления %d" % (a.зерно, a.шагов))
    rep["зерно"] = a.зерно
    rep["шагов"] = a.шагов

    for зрение, имя in ((False, "без зрения"), (True, "со зрением")):
        пути = []
        плохо = None
        for n in (1, 2):
            путь = из / ("%s-%d-%d.json"
                         % ("v" if зрение else "nov", a.зерно, n))
            if путь.exists():
                путь.unlink()
            беда = в_процессе(путь, зрение, a.зерно, a.шагов)
            if беда:
                плохо = беда
                break
            пути.append(путь)
        if плохо:
            check("%s: оба процесса записали отпечатки" % имя, False, плохо)
            continue
        check("%s: оба процесса записали отпечатки" % имя, True,
              "%d шагов" % a.шагов)
        д = [json.loads(p.read_text(encoding="utf-8")) for p in пути]
        # Проверка, что мера вообще чувствительна: ступени обязаны
        # МЕНЯТЬСЯ между шагами. Иначе «совпало» значило бы, что
        # отпечатки постоянны.
        меняется = [к for к in СТУПЕНИ
                    if len({ш.get(к) for ш in д[0]["по шагам"]}) > 1]
        check("%s: отпечатки ступеней меняются между шагами" % имя,
              len(меняется) >= 4,
              "меняются: %s" % ", ".join(меняется))
        расх = сравнить(д[0], д[1])
        rep[имя] = {"расхождение": (None if расх is None else
                                    {"ступень": расх[0], "шаг": расх[1],
                                     "первый": расх[2], "второй": расх[3]})}
        if расх is None:
            check("%s: все ступени совпали между процессами" % имя, True,
                  "%d шагов, %d ступеней" % (a.шагов, len(СТУПЕНИ)))
        else:
            ступень, шаг, x, y = расх
            check("%s: все ступени совпали между процессами" % имя, False,
                  "ПЕРВОЕ расхождение — ступень «%s» на шаге %d: %s "
                  "против %s" % (ступень, шаг, x, y))
            note("%s: первая разошедшаяся ступень — «%s», шаг %d. Всё, "
                 "что выше по пути, совпало побитово." % (имя, ступень, шаг))

    (OUT / "retina_repro.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    без = rep.get("без зрения", {}).get("расхождение")
    со = rep.get("со зрением", {}).get("расхождение")
    if без is None and со is not None:
        print("ПРИЧИНА НАЗВАНА: без зрения расхождений нет, со зрением "
              "первая разошедшаяся ступень — «%s» на шаге %d"
              % (со["ступень"], со["шаг"]))
        return 0
    if без is None and со is None:
        print("расхождений нет ни со зрением, ни без: на этом зерне и "
              "этой длине оно не воспроизвелось")
        return 0
    print("проба не смогла сравнить: см. провалы выше")
    return 1


if __name__ == "__main__":
    sys.exit(main())
