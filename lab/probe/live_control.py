"""Управление идущим расчётом: пауза, шаг, продолжение, остановка.

Проверяется не замысел, а поведение настоящего работника: команды уходят
в ту же очередь, что и задачи, разбираются на границе шага управления и
подтверждаются МОДЕЛЬНЫМ временем, в которое их применили, — а не фактом
отправки.

Что именно проверяется:

* пауза действительно останавливает модельное время, а окно при этом
  живо и продолжает получать кадры состояния;
* одиночный шаг двигает ровно на один шаг управления;
* продолжение возобновляет счёт с того же места;
* подтверждение приходит на каждую команду и несёт модельное время
  применения, которое не раньше времени подачи;
* команда «на будущее модельное время» применяется не раньше него;
* остановка доводит прогон до конца без потери результата;
* команды не теряются и не переставляются.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\live_control.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "подтверждения": [], "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    from flybrain import ipc
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("работник поднялся", ses.ready,
          "за %.1f с" % (time.perf_counter() - t0))
    if not ses.ready:
        return 1

    applied = []

    def pump(seconds, note=""):
        """Крутить приём и собирать подтверждения. Возвращает худший кадр."""
        end = time.perf_counter() + seconds
        worst = 0.0
        while time.perf_counter() < end:
            a = time.perf_counter()
            ses.pump(dt=0.02)
            worst = max(worst, (time.perf_counter() - a) * 1000.0)
            for rec in ses.take_applied():
                applied.append(rec)
                rep["подтверждения"].append({
                    "команда": rec.cmd_id, "действие": rec.action,
                    "модельное, мс": rec.t_model_ms,
                    "по стене, с": rec.wall_s, "исход": rec.outcome,
                    "почему": rec.why})
            time.sleep(0.01)
        return worst

    # Арена: шаг управления 20 мс, физики нет — реакция быстрая, и видно
    # именно работу очереди, а не ожидание MuJoCo.
    spec = {"kind": "arena", "t": 6.0, "seed": 0, "open_report": False,
            "arena": {"probes": False}}
    job = ses.submit(spec)
    check("прогон запущен", bool(job), job)

    t_start = time.perf_counter()
    while ses.state not in ("running",) and time.perf_counter() - t_start < 120:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("счёт пошёл", ses.state == "running",
          "состояние %s за %.1f с" % (ses.state, time.perf_counter() - t_start))

    pump(1.0)
    # Шаг управления берётся у самого прогона, а не задаётся числом:
    # у арены он равен шагу записи, и у разных видов он разный.
    control_ms = float(ses.bin_ms)
    rep["замеры"]["шаг управления, мс"] = control_ms

    sent_at = ses.t_ms
    ses.control("pause")
    worst = pump(1.5)
    pause_rec = next((a for a in applied if a.action == "pause"), None)
    check("пауза подтверждена модельным временем", pause_rec is not None,
          "подана на %.1f мс, применена на %s"
          % (sent_at, pause_rec.t_model_ms if pause_rec else "—"))
    # Команда применяется на ближайшей границе, а начатый шаг успевает
    # дойти до конца. Задержка не скрывается: она измеряется.
    lag = (pause_rec.t_model_ms - sent_at) if pause_rec else -1.0
    rep["замеры"]["задержка подача->применение, мс"] = lag
    check("задержка применения — не больше одного шага управления",
          0.0 <= lag <= control_ms + 1e-6,
          "%.1f мс при шаге %.1f мс" % (lag, control_ms))

    frozen = ses.t_ms
    worst = max(worst, pump(1.5))
    check("на паузе модельное время стоит", abs(ses.t_ms - frozen) < 1e-9,
          "%.1f мс и через полторы секунды те же %.1f мс"
          % (frozen, ses.t_ms))
    check("окно живо на паузе", worst < 200.0,
          "худший кадр приёма %.0f мс" % worst)
    rep["замеры"]["худший кадр на паузе, мс"] = round(worst, 1)

    paused_at = ses.t_ms
    ses.control("step")
    pump(1.0)
    one = ses.t_ms
    check("одиночный шаг двигает ровно на шаг управления",
          abs((one - paused_at) - control_ms) < 1e-6,
          "с %.1f до %.1f мс при шаге %.1f" % (paused_at, one, control_ms))

    ses.control("step", {"n": 3})
    pump(1.0)
    three = ses.t_ms
    check("три шага двигают на три шага",
          abs((three - one) - 3 * control_ms) < 1e-6,
          "с %.1f до %.1f мс" % (one, three))

    # команда на будущее модельное время
    future = three + 500.0
    ses.control("pause", at_t_ms=future)
    ses.control("resume")
    pump(3.0)
    late = [a for a in applied if a.action == "pause"
            and a.t_model_ms >= future - 1e-6]
    check("команда на будущее применена не раньше срока", bool(late),
          "просили с %.0f мс, применили в %s"
          % (future, [a.t_model_ms for a in late] or "не применили"))
    check("после срока модель встала",
          ses.state == "running" and abs(ses.t_ms - future) < 3 * control_ms,
          "модельное %.1f мс против заказанных %.1f" % (ses.t_ms, future))

    ses.control("resume")
    pump(1.0)
    moved = ses.t_ms
    check("продолжение возобновило счёт", moved > future,
          "%.1f -> %.1f мс" % (future, moved))

    ses.control("stop")
    t1 = time.perf_counter()
    while ses.busy and time.perf_counter() - t1 < 60:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    stopped = time.perf_counter() - t1
    check("остановка довела прогон до конца", not ses.busy,
          "за %.1f с" % stopped)

    # --- что пришло обратно ---
    ids = [a.cmd_id for a in applied]
    check("подтверждение пришло на каждую команду",
          len(ids) == len(set(ids)) and len(ids) >= 6,
          "команд подтверждено %d: %s" % (len(ids), ", ".join(ids)))
    order_ok = all(applied[i].t_model_ms <= applied[i + 1].t_model_ms
                   for i in range(len(applied) - 1))
    check("подтверждения не переставлены по модельному времени", order_ok,
          " ".join("%.0f" % a.t_model_ms for a in applied))
    check("все исходы названы",
          all(a.outcome in ("applied", "rejected", "cancelled", "error")
              for a in applied),
          ", ".join(sorted({a.outcome for a in applied})))

    ses.close()
    (OUT / "live_control.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
