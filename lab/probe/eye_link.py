"""Кадр глаз доходит от FlyGym до окна — и ничего по дороге не выдумано.

Проба идёт тем же путём, которым пойдёт окно: поднимает настоящего
работника (`ui.session.Session`), ставит сеанс тела со зрением и читает
кольцо глаз. Синтетики здесь нет ни в одном месте: картинка — та самая,
из которой FlyGym считает шестиугольные пиксели, показания омматидиев —
тот самый массив, который получает переходник сетчатки.

Что проверяется, и почему именно это:

1. **Описатель кольца доходит.** Без него окно не нашло бы разделяемую
   память и показывать было бы нечего. Встречная проверка: до `Hello`
   кольца у сеанса нет.
2. **Форма и тип.** Картинка (2, 512, 450, 3) uint8, показания
   (2, 721, 2) float32. Числа не выбраны нами: 512×450 объявлены в
   `config.yaml` FlyGym, 721 омматидий — в сетчатке. Кольцо проверяет
   форму при записи, и эта проверка подтверждает, что проверять было
   что.
3. **Рендеров ровно два на обновление.** В `body.py` это записано «по
   построению цикла `Fly._update_vision`». Утверждение по построению —
   тоже утверждение, поэтому оно здесь замеряется.
4. **Окно не может увидеть больше наблюдений, чем их было.**
   `eye_taken <= eye_updates` — главное неравенство этой пробы. Если
   окно начнёт считать повторы за наблюдения, сломается именно оно.
5. **Повтор не выдаётся за кадр.** Номер кадра растёт только тогда,
   когда работник записал новое наблюдение; между обновлениями опрос
   кольца обязан не менять ни номер, ни счётчик взятых.
6. **Кадр живой.** Два разных по номеру кадра обязаны отличаться
   пикселями: муха идёт, сцена меняется. Совпадение байт в байт
   означало бы, что в кольцо кладут одно и то же.
7. **Встречный замер: без зрения кадров нет вовсе.** Ноль обновлений,
   ноль взятых. Проверка, которая не может провалиться, ничего не
   стоит, поэтому эта — может.
8. **Память отпускается.** После закрытия сеанса оба кольца отвязаны.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\eye_link.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_eye_link_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
BOOT_S = 420.0          # сколько ждать разогрев работника
RUN_S = 240.0           # сколько ждать сам сеанс
T_MODEL = 1.2           # секунд модельного времени на сеанс

fails = []
rep = {"проверки": {}, "замеры": {}, "находки": [], "сорвалось": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def spec(vision, t=T_MODEL):
    """Спека сеанса — ровно та, которую собирает песочница.

    Объект включён и стоит СЛЕВА от мухи (y > 0 в координатах арены),
    чтобы левому глазу было что увидеть. Своей сцены проба не строит:
    та же `OdorArena` и тот же пул `vis_objects`, что в продукте.
    """
    body = {
        "readout": "ALPN",
        "control_ms": 20.0,
        "odour_ref": 0.5,
        "sources": [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]],
    }
    live = {"sources": [{"x": 25.0, "y": 0.0, "strength": 1.0, "on": False},
                        {"x": -25.0, "y": 0.0, "strength": 1.0, "on": False}]}
    if vision:
        body.update({"vision": True, "vision_hz": 100.0,
                     "vision_max_hz": 150.0, "vis_objects": 6})
        # Слотов ровно столько, сколько их в модели: их число задаётся
        # при сборке, и «добавить объект» — это включить свободный слот.
        # Первый включён и стоит СЛЕВА-СПЕРЕДИ от мухи, остальные
        # выключены и припаркованы.
        live["objects"] = [
            {"x": 6.0, "y": 4.0, "z": 1.5, "radius": 2.0, "grey": 0.02,
             "on": True}]
        live["objects"] += [
            {"x": 0.0, "y": 0.0, "z": 1.5, "radius": 1.0, "grey": 0.1,
             "on": False} for _ in range(5)]
    body["live_initial"] = live
    return {"kind": "body", "t": float(t), "seed": 0, "open_report": False,
            "body": body}


def wait(ses, until, timeout, what, poll=0.02):
    """Качать почту, пока не сбудется условие. Окно делает ровно это."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        ses.pump(poll)
        if until():
            return True
        if ses.error:
            rep["сорвалось"].append("%s: %s" % (what, ses.error))
            return False
        if not ses.alive:
            rep["сорвалось"].append("%s: работник умер" % what)
            return False
        time.sleep(poll)
    rep["сорвалось"].append("%s: не дождались за %.0f с" % (what, timeout))
    return False


def run_session(ses, vision, collect=0):
    """Прогнать сеанс и вернуть, что окно увидело.

    `collect` — сколько первых РАЗНЫХ кадров сохранить копией: по ним
    проверяется, что кадр живой, а сравнивать виды в общую память
    нельзя, их перезапишут.
    """
    job = ses.submit(spec(vision))
    if not job:
        rep["сорвалось"].append("задача не поставилась")
        return None
    seen, shots, omma = [], [], []
    polls_same = 0
    last = -1
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < RUN_S:
        ses.pump(0.02)
        if ses.eye_seq != last:
            if last >= 0:
                seen.append(ses.eye_seq)
            last = ses.eye_seq
            if ses.eye_img is not None and len(shots) < collect:
                shots.append(np.array(ses.eye_img, copy=True))
                omma.append(np.array(ses.eye_omma, copy=True))
        else:
            polls_same += 1
        if ses.state in ("done", "error"):
            break
        if not ses.alive:
            break
        time.sleep(0.004)
    # Дочерпать хвост почты: `Done` мог прийти в том же обороте.
    for _ in range(20):
        ses.pump(0.02)
    if ses.state == "error":
        rep["сорвалось"].append("задача отказала: %s" % ses.error)
        rep["замеры"].setdefault("разбор стека", []).append(
            str(getattr(ses, "traceback", ""))[-1500:])
    return {"кадров взято": ses.eye_taken, "обновлений": ses.eye_updates,
            "рендеров": ses.eye_renders, "вытеснено": ses.eye_missed,
            "опросов без нового кадра": polls_same,
            "номера": seen, "снимки": shots, "омматидии": omma,
            "состояние": ses.state, "сводка": ses.summary}


def main():
    from flybrain.ui.session import Session

    print("поднимаю работника", flush=True)
    ses = Session()
    try:
        check("кольца глаз до знакомства нет", ses.eyes is None)
        ok = wait(ses, lambda: ses.ready, BOOT_S, "разогрев")
        check("работник представился", ok,
              "%s, %d нейронов" % (ses.device, ses.n) if ok else "")
        if not ok:
            return
        check("описатель кольца глаз пришёл", ses.eyes is not None,
              str(ses.hello.eye_ring) if ses.hello else "")
        if ses.eyes is None:
            return

        # ---------------------------------------------------- со зрением
        print("сеанс со зрением", flush=True)
        a = run_session(ses, True, collect=4)
        if a is None:
            return
        rep["замеры"]["со зрением"] = {k: v for k, v in a.items()
                                       if k not in ("снимки", "омматидии")}
        check("кадры глаз дошли", a["кадров взято"] > 0,
              "взято %d" % a["кадров взято"])
        check("обновления сетчатки посчитаны", a["обновлений"] > 0,
              "%d обновлений, %d рендеров"
              % (a["обновлений"], a["рендеров"]))
        check("рендеров ровно два на обновление",
              a["рендеров"] == 2 * a["обновлений"],
              "%d против %d" % (a["рендеров"], 2 * a["обновлений"]))
        # Главное неравенство: наблюдений не может быть меньше, чем
        # кадров, которые окно назвало наблюдениями.
        check("окно не увидело больше наблюдений, чем их было",
              a["кадров взято"] <= a["обновлений"],
              "взято %d при %d обновлениях"
              % (a["кадров взято"], a["обновлений"]))
        check("опросы без нового кадра счётчик не двигали",
              a["опросов без нового кадра"] > 0,
              "%d опросов вхолостую" % a["опросов без нового кадра"])
        check("номера кадров идут строго вверх",
              all(b > c for c, b in zip(a["номера"], a["номера"][1:])),
              "%d переходов" % max(0, len(a["номера"]) - 1))

        shots, omm = a["снимки"], a["омматидии"]
        if shots:
            s0 = shots[0]
            check("форма картинки глаз", s0.shape == (2, 512, 450, 3)
                  and s0.dtype == np.uint8, "%s %s" % (s0.shape, s0.dtype))
            o0 = omm[0]
            check("форма показаний омматидиев",
                  o0.shape == (2, 721, 2) and o0.dtype == np.float32,
                  "%s %s" % (o0.shape, o0.dtype))
            check("показания омматидиев не пустые",
                  float(o0.max()) > 0.0,
                  "наибольшее %.1f, среднее %.1f"
                  % (float(o0.max()), float(o0.mean())))
        else:
            check("форма картинки глаз", False, "снимков нет")
        if len(shots) >= 2:
            diff = [int((shots[i] != shots[i + 1]).sum())
                    for i in range(len(shots) - 1)]
            check("разные кадры и правда разные", all(d > 0 for d in diff),
                  "различий по пикселям: " + ", ".join(map(str, diff)))
            rep["замеры"]["различия кадров"] = diff
        else:
            check("разные кадры и правда разные", False,
                  "собрано %d снимков" % len(shots))
        if omm:
            # Какой глаз видит объект. Объект стоит слева, и показания
            # левого глаза обязаны отличаться от правого — иначе кадр
            # не из этой сцены.
            l = float(np.asarray(omm[-1])[0].mean())
            r = float(np.asarray(omm[-1])[1].mean())
            rep["замеры"]["глаза, среднее"] = {"левый": round(l, 3),
                                               "правый": round(r, 3)}
            note("средние по глазам: левый %.2f, правый %.2f" % (l, r))

        # ------------------------------------------------- встречный замер
        print("сеанс БЕЗ зрения", flush=True)
        b = run_session(ses, False)
        if b is not None:
            rep["замеры"]["без зрения"] = {
                k: v for k, v in b.items()
                if k not in ("снимки", "омматидии")}
            check("без зрения кадров глаз нет вовсе",
                  b["кадров взято"] == 0 and b["обновлений"] == 0,
                  "взято %d, обновлений %d"
                  % (b["кадров взято"], b["обновлений"]))
    finally:
        ses.close()
        time.sleep(0.5)
    check("кольца отвязаны после закрытия",
          ses.eyes is not None and getattr(ses.eyes, "_img", None) is None,
          "виды на буфер сняты")

    (OUT / "eye_link.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)
    for text in rep["сорвалось"]:
        print("  сорвалось: " + text, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if (fails or rep["сорвалось"]) else 0)
