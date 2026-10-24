"""Пункты 2, 4, 6: камера, кадрирование и жизненный цикл задачи — живьём.

Работник настоящий, эпизод в теле настоящий, кадры MuJoCo приходят через
общую память. Проверяется не намерение, а результат: после каждого
действия с камерой сверяется и состояние `app.st.camera("body")`, и
содержимое `ses.body_frame` (среднее абсолютное отличие по пикселям).
Работник применяет камеру на шаге управления, поэтому после каждого
действия кадры прокачиваются в цикле `app.frame()` до прихода новых.

Кадр тела — вид в разделяемую память, а не копия, поэтому «до» всегда
копируется.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\bodyfinal_live.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
FRAME_AR = 960.0 / 720.0
T_RUN = 1.6                    # модельное время прогона 1, с
SIZES = ((1280, 720, 1.0), (1500, 880, 1.0), (1920, 1080, 1.0))


def warm_share(frame):
    """Доля «тёплых» пикселей: R − B > 12. Так ищется муха на кадре."""
    if frame is None:
        return None
    a = np.asarray(frame, dtype=np.int16)
    return float(((a[:, :, 0] - a[:, :, 2]) > 12).mean())


def mad(a, b):
    if a is None or b is None:
        return None
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main():
    from flybrain.desk import layout as L
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.desk.tokens import S
    from flybrain.ui.session import Session
    from flybrain import spec as SP

    rep = {}

    # --- 0. что уходит работнику из окна: проверка схемы, без работника ---
    sent = {"kind": "body", "t": 0.4, "seed": 0,
            "readout": "DNa02", "no_open": True}
    got = SP.from_dict(sent)
    rep["спека окна"] = {
        "окно посылает": sent,
        "spec.body.readout после разбора": got.body.readout,
        "readout доехал": got.body.readout == "DNa02",
        "no_open доехал": "no_open" in {f for f in got.to_dict()},
    }
    print("0. спека, которую посылает окно")
    for k, v in rep["спека окна"].items():
        print("   %-34s %s" % (k, v))

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.t_run = T_RUN
    t00 = time.perf_counter()

    def frame():
        sh.t = time.perf_counter() - t00
        app.frame()

    def pump(sec):
        end = time.perf_counter() + sec
        while time.perf_counter() < end:
            frame()

    def wait(cond, timeout, label=""):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            frame()
            if cond():
                return time.perf_counter() - t0
        print("   !! истекло ожидание %s за %.0f с" % (label, timeout))
        return None

    def snap():
        f = ses.body_frame
        return None if f is None else np.array(f, copy=True)

    def cam_state():
        c = app.st.camera("body")
        return {"view": c.view, "yaw": round(c.yaw, 3),
                "pitch": round(c.pitch, 3), "dist": round(c.dist, 4),
                "free": c.free}

    def park():
        sh.inp.set_mouse(-4000, -4000, False)

    try:
        # ------------------------------------------------ 1. подъём работника
        t = wait(lambda: app.st.worker.ready, 180, "подъёма работника")
        rep["подъём работника, с"] = None if t is None else round(t, 2)
        rep["работник"] = {"устройство": app.st.worker.device,
                           "нейронов": app.st.worker.n_neurons,
                           "популяций на выход": len(app.st.worker.readouts)}
        print("\n1. работник готов за %s с: %s"
              % (rep["подъём работника, с"],
                 json.dumps(rep["работник"], ensure_ascii=False)))

        # --------------------------------------------- 2. прогон 1 и камера
        print("\n2. прогон 1: подготовка и кадры")
        lifecycle = []

        def mark(tag):
            r = app.st.run
            rec = {"метка": tag, "фаза": r.phase, "этап": r.stage,
                   "задача": r.job_id, "t_sim": round(r.t_sim, 3),
                   "t_wall": round(r.t_wall, 1), "прогресс": round(r.progress, 3),
                   "путь": r.out_dir, "кадров тела": int(ses.body_seq),
                   "кадров активности": int(ses.frames_seen),
                   "точек пути": len(app.st.track),
                   "показаний в extra": len(ses.extra or {}),
                   "состояние сеанса": ses.state,
                   "работник жив": bool(ses.alive)}
            lifecycle.append(rec)
            print("   " + json.dumps(rec, ensure_ascii=False))
            return rec

        mark("до пуска")
        t_start = time.perf_counter()
        app.start_run()
        pump(0.5)
        mark("сразу после пуска")
        job1 = app.st.run.job_id
        seq_before_run1 = int(ses.body_seq)
        t_first = wait(lambda: ses.body_seq > seq_before_run1, 240,
                       "первого кадра тела")
        rep["первый кадр тела после пуска, с"] = (None if t_first is None
                                                  else round(t_first, 1))
        print("   первый кадр тела через %s с после пуска"
              % rep["первый кадр тела после пуска, с"])
        mark("первый кадр пришёл")

        # --- камера: вращение, колесо, четыре вида, сброс ---
        print("\n   камера: действие -> состояние и кадр")
        cams = []

        def cam_case(name, action, need_seq=2, timeout=60):
            f0, s0, c0 = snap(), int(ses.body_seq), cam_state()
            action()
            pump(0.1)
            dt = wait(lambda: ses.body_seq >= s0 + need_seq, timeout, name)
            f1, c1 = snap(), cam_state()
            rec = {"действие": name, "камера до": c0, "камера после": c1,
                   "камера изменилась": c0 != c1,
                   "кадров пришло": int(ses.body_seq) - s0,
                   "отличие кадров": None if mad(f0, f1) is None
                   else round(mad(f0, f1), 3),
                   "ожидание, с": None if dt is None else round(dt, 1)}
            cams.append(rec)
            print("     %-26s камера %s, кадров %d, отличие %s, %s с"
                  % (name, "да" if rec["камера изменилась"] else "НЕТ",
                     rec["кадров пришло"], rec["отличие кадров"],
                     rec["ожидание, с"]))
            return rec

        vw = sh.ctx.rect_of("body/view")

        def drag(dx, dy):
            """Протяжка по вьюпорту: курсор уже внутри, затем нажатие."""
            x, y = vw.cx, vw.cy
            sh.inp.set_mouse(x, y, False)
            frame()
            sh.inp.set_mouse(x, y, True)
            frame()
            n = 10
            for i in range(1, n + 1):
                sh.inp.set_mouse(x + dx * i // n, y + dy * i // n, True)
                frame()
            sh.inp.set_mouse(x + dx, y + dy, False)
            frame()
            park()
            frame()

        cam_case("протяжка мышью +120,+40", lambda: drag(120, 40))
        # Та же протяжка, но курсор перед нажатием был вне вида: так в окне
        # бывает при входе курсора во вьюпорт за один кадр.
        def jump_drag():
            park()
            frame()
            sh.inp.set_mouse(vw.cx, vw.cy, True)
            frame()
            sh.inp.set_mouse(vw.cx + 20, vw.cy, True)
            frame()
            sh.inp.set_mouse(vw.cx + 20, vw.cy, False)
            frame()
            park()
            frame()

        c_before = cam_state()
        jump_drag()
        rep["скачок при входе курсора"] = {
            "до": c_before, "после": cam_state(),
            "прирост yaw": round(app.st.camera("body").yaw - c_before["yaw"], 1),
            "протяжка в кадре, px": 20}
        print("     скачок при входе курсора: прирост yaw %s при протяжке 20 px"
              % rep["скачок при входе курсора"]["прирост yaw"])

        app.st.camera("body").reset()

        def wheel(v):
            sh.inp.set_mouse(vw.cx, vw.cy, False)
            frame()
            sh.inp.wheel = float(v)
            frame()
            park()
            frame()

        cam_case("колесо +3", lambda: wheel(3))
        cam_case("колесо -3", lambda: wheel(-3))

        def toolbar(wid):
            park()
            frame()
            r = sh.ctx.rect_of(wid)
            if r is None:
                print("     !! нет детали " + wid)
                return
            sh.inp.set_mouse(r.cx, r.cy, False)
            frame()
            sh.inp.set_mouse(r.cx, r.cy, True)
            frame()
            sh.inp.set_mouse(r.cx, r.cy, False)
            frame()
            park()
            frame()

        for i, nm in enumerate(("сзади", "сбоку", "сверху", "спереди")):
            cam_case("вид %s (кнопка %d)" % (nm, i),
                     lambda i=i: toolbar("body/tb/view/%d" % i))
        cam_case("приблизить (кнопка)",
                 lambda: toolbar("body/tb/zoom_in"))
        cam_case("сброс вида (кнопка)", lambda: toolbar("body/tb/reset"))
        rep["камера"] = cams

        # ------------------------------------------- 3. кадрирование мухи
        print("\n3. кадрирование мухи")
        shots = []
        dock = app.st.ws.dock()

        def framing(tag, w, h, dpi, hide=(), maxi=False):
            for z in ("inspector", "secondary", "timeline"):
                setattr(dock, z + "_open", z not in hide)
            dock.maximized = bool(maxi)
            sh.resize(w, h, dpi)
            park()
            frame()
            frame()
            lay = L.compute(sh.w, sh.h, sh.sc, dock)
            centre = lay.centre.inset(sh.sc.px(S.xs))
            view = centre.inset(sh.sc.px(S.sm))
            box = view.fit_aspect(FRAME_AR)
            f = snap()
            fly = warm_share(f)
            rec = {"случай": tag, "размер": [w, h], "dpi": dpi,
                   "спрятано": list(hide), "развёрнут": bool(maxi),
                   "центр": [centre.w, centre.h],
                   "кадр в окне": [box.w, box.h],
                   "доля кадра в центре": round(
                       box.w * box.h / max(1, centre.w * centre.h), 4),
                   "доля мухи в кадре": None if fly is None else round(fly, 5),
                   "доля мухи в центре": None if fly is None else round(
                       fly * box.w * box.h / max(1, centre.w * centre.h), 5),
                   "муха не меньше 3 %": None if fly is None else fly >= 0.03}
            shots.append(rec)
            print("   %-26s центр %dx%d, кадр %dx%d (%.1f %% центра), "
                  "муха %.2f %% кадра %s"
                  % (tag, centre.w, centre.h, box.w, box.h,
                     rec["доля кадра в центре"] * 100,
                     (fly or 0) * 100,
                     "" if rec["муха не меньше 3 %"] else "НИЖЕ ПОРОГА"))
            sh.shot(OUT / ("bodyfinal_" + tag + ".png"))
            return rec

        for (w, h, dpi) in SIZES:
            framing("fit_%dx%d" % (w, h), w, h, dpi)
        framing("fit_1500x880_без_боковых", 1500, 880, 1.0,
                hide=("inspector", "secondary"))
        framing("fit_1920x1080_развёрнут", 1920, 1080, 1.0, maxi=True)
        framing("fit_1280x720_только_центр", 1280, 720, 1.0,
                hide=("inspector", "secondary", "timeline"))
        rep["кадрирование"] = shots
        for z in ("inspector", "secondary", "timeline"):
            setattr(dock, z + "_open", True)
        dock.maximized = False
        sh.resize(1500, 880, 1.0)
        frame()

        # ------------------------------------------- 4. отмена и конец
        print("\n4. отмена прогона 1")
        f_last_run1 = snap()
        seq_run1 = int(ses.body_seq)
        extra_run1 = dict(ses.extra or {})
        mark("перед отменой")
        t_cancel = time.perf_counter()
        app.cancel_run()
        pump(0.3)
        mark("сразу после «стоп»")
        dt = wait(lambda: app.st.run.phase in ("done", "fail"), 180,
                  "конца после отмены")
        rep["отмена из фазы run, с"] = None if dt is None else round(dt, 1)
        print("   конец через %s с после «стоп»" % rep["отмена из фазы run, с"])
        done1 = mark("конец прогона 1")
        rep["сводка прогона 1"] = (dict(ses.summary)
                                   if isinstance(ses.summary, dict) else None)
        if rep["сводка прогона 1"]:
            s1 = rep["сводка прогона 1"]
            print("   сводка: " + json.dumps(
                {k: s1[k] for k in s1 if k not in ("bench",)},
                ensure_ascii=False)[:600])

        # ------------------------------- 5. повторный запуск: не старое ли
        print("\n5. повторный запуск: что показывает интерфейс в фазе prep")
        app.st.t_run = 0.4
        app.start_run()
        pump(0.4)
        prep2 = mark("prep прогона 2")
        f_prep2 = snap()
        rep["повторный запуск"] = {
            "номер задачи 1": job1,
            "номер задачи 2": app.st.run.job_id,
            "номер сменился": app.st.run.job_id != job1,
            "кадров тела на конце прогона 1": seq_run1,
            "кадров тела в prep прогона 2": int(ses.body_seq),
            "счётчик кадров тела сброшен": int(ses.body_seq) == 0,
            "отличие кадра от последнего кадра прогона 1":
                None if mad(f_last_run1, f_prep2) is None
                else round(mad(f_last_run1, f_prep2), 4),
            "кадр тот же самый": (mad(f_last_run1, f_prep2) == 0.0
                                  if f_prep2 is not None else None),
            "кадр вообще есть": f_prep2 is not None,
            "показаний в extra прогона 1": len(extra_run1),
            "показаний в extra в prep прогона 2": len(ses.extra or {}),
            "кадров активности в prep прогона 2": int(ses.frames_seen),
            "точек пути в prep прогона 2": len(app.st.track),
            "путь прогона 1 (out_dir)": done1["путь"],
            "путь в prep прогона 2": app.st.run.out_dir,
            "подпись под видом обещает обновление":
                not app.st.run.busy,
        }
        for k, v in rep["повторный запуск"].items():
            print("   %-50s %s" % (k, v))
        sh.shot(OUT / "bodyfinal_prep2.png")

        print("\n6. отмена из фазы prep")
        t_c2 = time.perf_counter()
        app.cancel_run()
        pump(0.3)
        dt2 = wait(lambda: app.st.run.phase in ("done", "fail"), 300,
                   "конца после отмены из prep")
        rep["отмена из фазы prep, с"] = None if dt2 is None else round(dt2, 1)
        print("   конец через %s с после «стоп» в подготовке"
              % rep["отмена из фазы prep, с"])
        mark("конец прогона 2")

        # ------------------------------------------------- 7. отказ
        print("\n7. отказ: несуществующая популяция на выход")
        bad = {"kind": "body", "t": 0.3, "seed": 0,
               "body": {"readout": "НЕТ-ТАКОЙ-ПОПУЛЯЦИИ"}}
        jid = ses.submit(bad)
        rep["отказ"] = {"задача принята": jid is not None, "спека": bad}
        pump(0.5)
        t_f = time.perf_counter()
        wait(lambda: app.st.run.phase == "fail" or not ses.alive, 180,
             "отказа")
        rep["отказ"].update({
            "ждали, с": round(time.perf_counter() - t_f, 1),
            "фаза окна": app.st.run.phase,
            "этап окна": app.st.run.stage,
            "состояние сеанса": ses.state,
            "ошибка сеанса": (str(ses.error)[:300] if ses.error else None),
            "работник жив": bool(ses.alive),
            "последние строки журнала": [t for _ts, _lv, t in app.st.log[-6:]],
        })
        for k, v in rep["отказ"].items():
            if k != "спека":
                print("   %-30s %s" % (k, v))
        pump(1.0)
        rep["отказ"]["фаза окна через секунду"] = app.st.run.phase
        rep["отказ"]["работник жив через секунду"] = bool(ses.alive)
        sh.shot(OUT / "bodyfinal_fail.png")
        rep["жизненный цикл"] = lifecycle
    finally:
        try:
            ses.close()
        except Exception as exc:                               # noqa: BLE001
            rep["закрытие сеанса"] = repr(exc)[:200]
        (OUT / "bodyfinal_live.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nотчёт: probe/bodyfinal_live.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
