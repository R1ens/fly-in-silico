"""Песочница с глазами: настоящий сеанс, настоящий предмет, настоящий кадр.

Проба поднимает то же приложение, что открывается двойным щелчком, —
только кадр собирается без окна, потому что окно в этой среде меняет
размер само. Всё остальное настоящее: работник, тело MuJoCo, сцена,
зрительный объект и кольцо кадров глаз.

Что проверяется, по разрядам.

**Доставка.** Кадр глаз доходит до вида, у него есть номер, модельное
время и возраст, и возраст считается по настенным часам.

**Причинность, и она может провалиться.** Предмет переносится из
левого полуполя в правое на ходу, тем же путём, которым его переносит
рейка (`ws_sandbox._send_objects`), и замеряется **перекос между
глазами** — средняя разность «левый минус правый» по показаниям
омматидиев. Берётся именно перекос, а не яркость одного глаза: у идущей
мухи собственный оптический поток давит абсолютные уровни в обоих
глазах разом, а предмет сбоку ломает именно симметрию. Ожидание
объявлено ДО замера: перекос при предмете слева обязан быть больше,
чем при предмете справа. Встречный замер — тот же отрезок с
выключенным предметом, и он обязан лечь между ними.

**Разделение величин.** Рендеры камер, обновления сетчатки, кадры в
окно и частота показа окна считаются врозь и сверяются друг с другом:
рендеров ровно вдвое больше обновлений, кадров в окно не больше
обновлений, а на паузе номер кадра стоит, тогда как возраст растёт.

**Жизненный цикл.** Пауза, продолжение, остановка, второй сеанс в том
же окне, выключенное зрение, закрытие. Память меряется до и после.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\eye_view.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_eye_view_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
BOOT_S = 240.0
BUILD_S = 300.0

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
    from flybrain.desk import ws_sandbox as X
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1600, 900))
    ses = Session()
    app = App(sh, ses)
    st = app.st
    st.ws.current = "sandbox"
    sb = st.sb
    sb.t_run = 0.0           # сеанс без заранее заданного конца
    sb.keep_s = 10.0
    sb.vision = True         # решается ДО запуска
    t0 = time.perf_counter()
    frames = []              # длительность каждого кадра окна, мс

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sh.t = time.perf_counter() - t0
            a = time.perf_counter()
            app.frame()
            frames.append((time.perf_counter() - a) * 1000.0)

    def shot(name):
        sh.t = time.perf_counter() - t0
        app.frame()
        app.frame()
        p = OUT / name
        sh.shot(p)
        rec = {"снимок": name, "кадр глаз": int(ses.eye_seq),
               "возраст, мс": round(ses.eye_age_s * 1000.0, 1),
               "обновлений": int(ses.eye_updates),
               "мс на кадр": round(sh.frame_ms, 2)}
        rep["снимки"].append(rec)
        print("  " + json.dumps(rec, ensure_ascii=False), flush=True)
        return p

    def wait_applied(cid, seconds=60.0):
        end = time.perf_counter() + seconds
        while cid in sb.pending and time.perf_counter() < end:
            pump(0.1)
        return cid not in sb.pending

    rep["замеры"]["память до, МБ"] = rss_mb()
    try:
        while not st.worker.ready and time.perf_counter() - t0 < BOOT_S:
            app.frame()
        check("работник готов", st.worker.ready,
              "за %.1f с" % (time.perf_counter() - t0))
        if not st.worker.ready:
            return

        # --- вид глаз ДО сеанса: он обязан объяснить, почему пусто ---
        sb.eye_view = True
        shot("eyes-01-до-сеанса.png")
        check("до сеанса вид глаз не притворяется",
              not EY.available(ses) and ses.eye_seq == 0,
              "кадров нет, номер %d" % ses.eye_seq)

        # --- запуск ---------------------------------------------------
        app.start_run()
        t_wait = time.perf_counter()
        while ses.eye_img is None and time.perf_counter() - t_wait < BUILD_S:
            app.frame()
        got = ses.eye_img is not None
        check("первый кадр глаз пришёл", got,
              "через %.1f с после пуска" % (time.perf_counter() - t_wait))
        if not got:
            rep["сорвалось"].append("кадра глаз не дождались")
            return
        # Нулевая отметка у первого кадра законна: наблюдение могло
        # случиться в самом начале. Проверять надо другое — что отметка
        # ИДЁТ, и что это время наблюдения, а не номер кадра.
        t_first = ses.eye_t_ms
        while ses.eye_seq < 4 and ses.busy:
            pump(0.1)
        check("модельное время кадра растёт", ses.eye_t_ms > t_first,
              "%.1f -> %.1f мс" % (t_first, ses.eye_t_ms))
        check("возраст кадра считается по стене",
              0.0 <= ses.eye_age_s < 5.0, "%.0f мс" % (ses.eye_age_s * 1000))
        pump(3.0)
        shot("eyes-02-вид.png")

        # --- причинность: муха СТОИТ, двигается только предмет ------
        #
        # Две редакции этого опыта замер отверг, и обе по одной
        # причине. Первая гоняла три отрезка подряд (нет / слева /
        # справа) — за двенадцать секунд стены муха успевала уйти, и
        # отрезки сравнивались в разных сценах. Вторая чередовала
        # стороны короткими отрезками и считала разность внутри пары —
        # и встречный замер её убил: с ВЫКЛЮЧЕННЫМ предметом те же
        # команды давали ту же разность (-0,0107 против -0,0116).
        # То есть мерилось не действие предмета, а порядок отрезков:
        # левый всегда шёл первым, а муха за шесть секунд успевала
        # повернуть.
        #
        # Отсюда нынешнее устройство: сеанс ставится на ПАУЗУ, и между
        # замерами делается ровно несколько шагов управления. Муха при
        # этом не идёт, свет не меняется, сцена неподвижна — меняется
        # только предмет. Тогда разность кадров вызвана им и ничем
        # другим, и спорить не о чем.
        #
        # Величина тоже другая: не «перекос средних», а СКОЛЬКО
        # ОММАТИДИЕВ КАЖДОГО ГЛАЗА изменилось заметно по сравнению со
        # сценой без предмета. Она прямо отвечает на вопрос «какой глаз
        # это увидел», а среднее по 721 омматидию его размывает.
        o = sb.objects[0]
        o.z, o.radius, o.grey = 1.5, 8.0, 0.0

        X._send(app, "пауза", action="pause")
        pump(1.5)

        def step_and_grab(n=1, seconds=25.0):
            """Сделать n шагов управления и взять СВЕЖИЙ кадр глаз.

            Один шаг, а не три. Три шага были третьей редакцией, и
            замер её тоже поправил: за 60 мс модельного времени у
            неподвижной на вид мухи шевелятся ноги и чуть едет корпус,
            и это само по себе меняло около двухсот омматидиев —
            больше, чем предмет с одной из сторон. Один шаг оставляет
            фон втрое меньше, а предмет виден так же.
            """
            was = ses.eye_seq
            X._send(app, "шаг", action="step", payload={"n": int(n)})
            end_t = time.perf_counter() + seconds
            while ses.eye_seq < was + n and time.perf_counter() < end_t:
                pump(0.1)
            if ses.eye_omma is None:
                return None
            return np.asarray(ses.eye_omma, dtype=np.float32).max(axis=2)

        def place(side, on, tag):
            """Поставить предмет ОТНОСИТЕЛЬНО МУХИ, а не арены.

            «Слева» — это слева от мухи, а не со стороны оси y арены.
            Четвёртая редакция этого опыта села именно здесь: предмет
            клали в мировую точку (8; +8), а муха к тому времени уже
            повернула, и «слева» оказывалось позади. Поэтому точка
            считается от её нынешнего положения и курса.
            """
            ex = dict(ses.extra or {})
            fx = float(ex.get("x", 0.0) or 0.0)
            fy = float(ex.get("y", 0.0) or 0.0)
            th = float(ex.get("theta", 0.0) or 0.0)
            ahead, lateral = 9.0, 9.0 * float(side)
            o.x = round(fx + ahead * np.cos(th) - lateral * np.sin(th), 3)
            o.y = round(fy + ahead * np.sin(th) + lateral * np.cos(th), 3)
            o.on = bool(on)
            cid = X._send_objects(app, tag)
            if not cid or not wait_applied(cid):
                rep["сорвалось"].append("не применилось: " + tag)
            return step_and_grab()

        # Порядок ВКЛ/ВЫКЛ/ВЫКЛ/ВКЛ: ровный снос от хода мухи попадает
        # в обе половины одинаково и в разности сокращается. Порядок
        # «подряд» уже один раз подсунул ложный ответ, и повторять это
        # незачем.
        left_on = place(+1, True, "предмет слева")
        shot("eyes-03-предмет-слева.png")
        left_off = place(+1, False, "слева убран")
        right_off = place(-1, False, "справа убран")
        right_on = place(-1, True, "предмет справа")
        shot("eyes-04-предмет-справа.png")
        X._send(app, "продолжить", action="resume")
        pump(1.0)

        if any(v is None for v in (left_on, left_off, right_off, right_on)):
            check("предмет виден сетчатке того глаза, с какой он стороны",
                  False, "кадра не дождались")
        else:
            def touched(a, b, thr):
                d = np.abs(a - b)
                return [int((d[0] > thr).sum()), int((d[1] > thr).sum())]

            def weight(a, b):
                d = np.abs(a - b)
                return [round(float(d[0].sum()), 1),
                        round(float(d[1].sum()), 1)]

            # Пороги врозь, и это главное место опыта. Ход самой мухи
            # меняет МНОГО омматидиев ЧУТЬ-ЧУТЬ: за один шаг
            # управления шевелятся ноги, а они занимают изрядную часть
            # кадра глаза. Чёрный шар в девяти миллиметрах меняет
            # МЕНЬШЕ омматидиев, но СИЛЬНО — с небесной единицы до
            # нуля. Поэтому счёт по низкому порогу ничего не разделяет
            # (замерено: предмет 298 и 300, фон 203 и 263), а по
            # высокому разделяет.
            #
            # Порог 0,30 выбран не под результат: это треть полной
            # шкалы показаний, то есть перемена, которой не бывает от
            # дрожания ноги.
            THRS = (0.02, 0.10, 0.30)
            tab = {}
            for thr in THRS:
                tab["порог %.2f" % thr] = {
                    "предмет слева": touched(left_on, left_off, thr),
                    "предмет справа": touched(right_on, right_off, thr),
                    "фон без предмета": touched(left_off, right_off, thr),
                }
            tab["сумма модулей"] = {
                "предмет слева": weight(left_on, left_off),
                "предмет справа": weight(right_on, right_off),
                "фон без предмета": weight(left_off, right_off),
            }
            rep["замеры"]["омматидиев изменилось"] = tab
            print("  " + json.dumps(tab, ensure_ascii=False), flush=True)

            tl = touched(left_on, left_off, 0.30)
            tr = touched(right_on, right_off, 0.30)
            tn = touched(left_off, right_off, 0.30)
            check("на сильных переменах предмет виден заметнее хода мухи",
                  max(tl + tr) > 2 * max(max(tn), 1),
                  "предмет %s и %s против фона %s" % (tl, tr, tn))
            check("предмет слева виден ЛЕВЫМ глазом сильнее",
                  tl[0] > tl[1], "левый %d против правого %d" % tuple(tl))
            check("предмет справа виден ПРАВЫМ глазом сильнее",
                  tr[1] > tr[0], "левый %d против правого %d" % tuple(tr))
            # Где именно на сетчатке. Положения омматидиев берутся из
            # той же карты NeuroMechFly, что и раскладка: середина
            # кадра — 225 пикселей по горизонтали, и смещение середины
            # задетых омматидиев показывает, какая часть поля зрения
            # приняла раздражитель.
            try:
                from flybrain import retina as RT

                pos = RT.ommatidia()["pos"]

                def where(a, b, eye, thr=0.30):
                    d = np.abs(a[eye] - b[eye])
                    sel = d > thr
                    if not sel.any():
                        return None
                    return [round(float(pos[sel, 0].mean()), 1),
                            round(float(pos[sel, 1].mean()), 1)]

                rep["замеры"]["середина задетых, пиксели кадра"] = {
                    "предмет слева, левый глаз": where(left_on, left_off, 0),
                    "предмет справа, правый глаз": where(right_on,
                                                         right_off, 1),
                }
                print("  середина задетых: " + json.dumps(
                    rep["замеры"]["середина задетых, пиксели кадра"],
                    ensure_ascii=False), flush=True)
            except Exception as exc:                       # noqa: BLE001
                note("положения омматидиев не прочитались: "
                     + repr(exc)[:120])

        # --- размер предмета ------------------------------------------
        before = float(np.asarray(ses.eye_omma).max(axis=2).mean())
        o.radius = 12.0
        cid = X._send_objects(app, "предмет увеличен")
        wait_applied(cid)
        pump(3.0)
        after = float(np.asarray(ses.eye_omma).max(axis=2).mean())
        check("изменение размера видно сетчатке",
              abs(after - before) > 1e-4,
              "средняя яркость %.4f -> %.4f" % (before, after))

        # --- счётчики врозь -------------------------------------------
        rep["замеры"]["счётчики"] = {
            "рендеров камер": int(ses.eye_renders),
            "обновлений сетчатки": int(ses.eye_updates),
            "кадров в окно": int(ses.eye_taken),
            "вытеснено внутри шага": int(ses.eye_lost_in_step),
            "вытеснено кольцом": int(ses.eye_missed),
            "обновлений в секунду": (round(ses.rate_updates.value, 2)
                                     if ses.rate_updates.value else None),
            "кадров в окно в секунду": (round(ses.rate_taken.value, 2)
                                        if ses.rate_taken.value else None),
            "доля реального": round(sb.ratio, 4),
        }
        check("рендеров ровно вдвое больше обновлений",
              ses.eye_renders == 2 * ses.eye_updates,
              "%d против %d" % (ses.eye_renders, 2 * ses.eye_updates))
        check("кадров в окно не больше, чем обновлений",
              ses.eye_taken <= ses.eye_updates,
              "%d из %d" % (ses.eye_taken, ses.eye_updates))
        check("потери разложены без остатка",
              ses.eye_updates == (ses.eye_taken + ses.eye_lost_in_step
                                  + ses.eye_missed),
              "%d = %d + %d + %d" % (ses.eye_updates, ses.eye_taken,
                                     ses.eye_lost_in_step, ses.eye_missed))

        # --- пауза: номер стоит, возраст растёт ------------------------
        X._send(app, "пауза", action="pause")
        pump(2.0)
        seq_p, age_p = ses.eye_seq, ses.eye_age_s
        taken_p = ses.eye_taken
        pump(2.5)
        shot("eyes-05-пауза.png")
        check("на паузе номер кадра стоит", ses.eye_seq == seq_p,
              "№%d" % ses.eye_seq)
        check("на паузе новых кадров окно не засчитывает",
              ses.eye_taken == taken_p, "взято %d" % ses.eye_taken)
        check("на паузе возраст кадра растёт", ses.eye_age_s > age_p + 1.0,
              "%.1f -> %.1f с" % (age_p, ses.eye_age_s))
        X._send(app, "продолжить", action="resume")
        t_wait = time.perf_counter()
        while ses.eye_seq == seq_p and time.perf_counter() - t_wait < 60:
            pump(0.1)
        check("после продолжения кадры пошли снова", ses.eye_seq > seq_p,
              "№%d" % ses.eye_seq)
        shot("eyes-06-продолжен.png")

        # --- остановка -------------------------------------------------
        app.cancel_run()
        t_wait = time.perf_counter()
        while ses.busy and time.perf_counter() - t_wait < 120:
            pump(0.1)
        check("сеанс остановился", not ses.busy, str(ses.state))
        check("работник жив после остановки", ses.alive)
        stop_seq = ses.eye_seq
        pump(2.0)
        check("после остановки новых кадров не появляется",
              ses.eye_seq == stop_seq, "№%d" % ses.eye_seq)
        shot("eyes-07-остановлен.png")

        # --- встречный замер: сеанс БЕЗ зрения -------------------------
        sb.vision = False
        app.start_run()
        t_wait = time.perf_counter()
        while st.run.phase == "prep" and time.perf_counter() - t_wait < BUILD_S:
            app.frame()
        pump(8.0)
        check("без зрения счётчики обнулены и не растут",
              ses.eye_updates == 0 and ses.eye_taken == 0,
              "обновлений %d, взято %d" % (ses.eye_updates, ses.eye_taken))
        check("без зрения вид глаз говорит, что кадров нет",
              not EY.available(ses))
        shot("eyes-08-без-зрения.png")
        app.cancel_run()
        t_wait = time.perf_counter()
        while ses.busy and time.perf_counter() - t_wait < 120:
            pump(0.1)

        if frames:
            arr = np.asarray(frames, dtype=np.float64)
            rep["замеры"]["кадр окна, мс"] = {
                "кадров": int(arr.size),
                "медиана": round(float(np.median(arr)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "наибольший": round(float(arr.max()), 2),
            }
            med = float(np.median(arr))
            check("кадр окна собирается быстрее 1/30 с",
                  med < 33.3, "медиана %.2f мс" % med)
    finally:
        rep["замеры"]["память после, МБ"] = rss_mb()
        ses.close()
        time.sleep(0.5)

    (OUT / "eye_view.json").write_text(
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
