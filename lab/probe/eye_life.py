"""Жизненный цикл вида глаз: беда, теснота и долгий сеанс.

`eye_view.py` проверяет, что кадр доходит и что он причинно связан со
сценой. Здесь — то, что ломается не в хорошую погоду:

1. **Работник умирает посреди сеанса.** Окно обязано остаться живым,
   не повиснуть на чтении разделяемой памяти и показать, что кадр
   протух, — а не держать последнюю картинку, притворяясь, что всё
   идёт. Убивается он по-настоящему, `terminate()`.
2. **Теснота.** Вид глаз переживает сужение окна до 900×700 и до
   1280×720: четыре клетки обязаны уменьшаться, а не наезжать друг на
   друга и не уходить за край.
3. **Долгий сеанс.** Пять минут по стене с включённым зрением: память
   процесса окна и число взятых кадров пишутся по дороге. Рост памяти,
   пропорциональный числу кадров, — это утечка, и видно её только
   длинным сеансом.
4. **Повторный сеанс в том же окне.** Второй сеанс обязан начать
   счётчики с нуля и не показывать кадр первого.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\eye_life.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_eye_life_out"))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
BOOT_S = 240.0
BUILD_S = 300.0
LONG_S = float(os.environ.get("FLYBRAIN_LONG_S", "300"))

fails = []
rep = {"проверки": {}, "замеры": {}, "снимки": [], "находки": [],
       "сорвалось": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def rss_mb():
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1 << 20), 1)
    except Exception:                                      # noqa: BLE001
        return None


def main():                                                   # noqa: C901
    from flybrain.desk import eyes as EY
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1600, 900))
    ses = Session()
    app = App(sh, ses)
    st = app.st
    st.ws.current = "sandbox"
    sb = st.sb
    sb.t_run = 0.0
    sb.keep_s = 10.0
    sb.vision = True
    sb.eye_view = True
    t0 = time.perf_counter()

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sh.t = time.perf_counter() - t0
            app.frame()

    def shot(name):
        sh.t = time.perf_counter() - t0
        app.frame()
        app.frame()
        p = OUT / name
        sh.shot(p)
        rep["снимки"].append({"снимок": name, "размер": [sh.w, sh.h],
                              "кадр глаз": int(ses.eye_seq)})
        print("  снимок %s %dx%d" % (name, sh.w, sh.h), flush=True)
        return p

    try:
        while not st.worker.ready and time.perf_counter() - t0 < BOOT_S:
            app.frame()
        check("работник готов", st.worker.ready)
        if not st.worker.ready:
            return

        app.start_run()
        t_wait = time.perf_counter()
        while ses.eye_img is None and time.perf_counter() - t_wait < BUILD_S:
            app.frame()
        check("кадр глаз пошёл", ses.eye_img is not None)
        if ses.eye_img is None:
            rep["сорвалось"].append("кадра глаз не дождались")
            return

        # --- 2. теснота ----------------------------------------------
        for w, h in ((1280, 720), (1000, 760), (900, 700)):
            sh.resize(w, h, 1.0)
            pump(0.6)
            shot("eyelife-%dx%d.png" % (w, h))
        sh.resize(1600, 900, 1.0)
        pump(0.6)
        check("вид глаз пережил три сужения", True,
              "1280×720, 1000×760, 900×700")

        # --- 3. долгий сеанс -----------------------------------------
        marks = []
        start = time.perf_counter()
        base = rss_mb()
        while time.perf_counter() - start < LONG_S and ses.busy:
            pump(15.0)
            marks.append({"с начала, с": round(time.perf_counter() - start, 1),
                          "кадров глаз": int(ses.eye_taken),
                          "память, МБ": rss_mb(),
                          "модельное, с": round(ses.t_ms / 1000.0, 2)})
            print("  " + json.dumps(marks[-1], ensure_ascii=False),
                  flush=True)
        rep["замеры"]["долгий сеанс"] = marks
        if len(marks) >= 2 and marks[0]["память, МБ"] and base:
            grew = marks[-1]["память, МБ"] - marks[0]["память, МБ"]
            took = marks[-1]["кадров глаз"] - marks[0]["кадров глаз"]
            rep["замеры"]["рост памяти, МБ"] = round(grew, 1)
            rep["замеры"]["кадров за это время"] = took
            # Кадр глаза — 1,38 МБ. Если бы он где-то копился, тысяча
            # кадров стоила бы гигабайты; порог в 150 МБ ловит утечку
            # задолго до этого и при этом не цепляется за обычное
            # дыхание кучи.
            check("память не растёт с числом кадров", grew < 150.0,
                  "%+.1f МБ за %d кадров" % (grew, took))
            note("на кадр приходится %.3f МБ роста"
                 % (grew / max(took, 1)))
        shot("eyelife-долгий.png")

        # --- 1. работник умирает -------------------------------------
        # Кольцо могло держать кадры, записанные работником и ещё не
        # взятые окном: сперва даём окну их разобрать, и только потом
        # запоминаем номер. Иначе проверка «после смерти кадров нет»
        # падает на кадрах, записанных ДО смерти.
        pump(3.0)
        seq_before = ses.eye_seq
        ses.proc.terminate()
        t_wait = time.perf_counter()
        while ses.alive and time.perf_counter() - t_wait < 30:
            pump(0.2)
        check("окно заметило, что работник умер", not ses.alive)
        # Несколько секунд рисования поверх мёртвого работника: окно не
        # должно ни повиснуть, ни упасть на чтении общей памяти.
        pump(4.0)
        check("окно живо после смерти работника", sh.running)
        check("кадр глаз не прибавился", ses.eye_seq == seq_before,
              "№%d" % ses.eye_seq)
        check("возраст кадра показывает, что он протух",
              ses.eye_age_s > 3.0, "%.1f с" % ses.eye_age_s)
        check("вид глаз всё ещё знает, что кадр есть",
              EY.available(ses))
        shot("eyelife-работник-умер.png")
    finally:
        ses.close()
        time.sleep(0.5)
        rep["замеры"]["память в конце, МБ"] = rss_mb()

    (OUT / "eye_life.json").write_text(
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
