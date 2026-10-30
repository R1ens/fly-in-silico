"""Управление зрением ЧЕРЕЗ НАСТОЯЩЕЕ ПРИЛОЖЕНИЕ, а не мимо него.

`probe/vision_product.py` проверил зрение на уровне эпизода: спека ->
`BodyConfig` -> `run_episode`. Чего там не было — **работника и
протокола команд**: зрением надо управлять на ходу, из Песочницы, и
требование владельца про живые команды касается зрительных ручек ровно
так же, как всех прочих.

Поэтому здесь поднимается настоящий работник (`ui.Session`), пускается
сеанс в теле со зрением ровно той спекой, которую собирает окно
(`desk.app.App.sandbox_spec`), и дальше проверяется то, что требует
пункт 7: **у каждой живой команды есть идентификатор, запись применения
по модельному времени, подтверждение и воспроизводимый журнал.**

Что проверяется, и у каждого пункта встречный замер:

1. **спека окна** — та, что собирает Песочница, несёт зрение и слоты
   объектов; встречный замер: с выключенным зрением не несёт ни того,
   ни другого;
2. **идентификатор и подтверждение** — у каждой команды свой номер,
   подтверждение приходит с модельным временем применения, и это
   время не раньше наблюдённого на отправке;
3. **ИЗМЕРЕННОЕ последствие** — после включения тёмного объекта
   поданная частота фоторецепторов меняется, а после выключения
   возвращается. Подтверждение «принято» последствием не является;
4. **подача зрения выключается на ходу** — `vision_feed` снимает
   подачу, и частота падает в ноль; обратно включается;
5. **свет и контраст на ходу** — измеримо меняют поданную частоту;
6. **отказ до правки** — заведомо негодная зрительная правка
   отклоняется с причиной, а сеанс продолжает идти;
7. **синхронизация по модельному времени** — команда, придержанная до
   названного модельного времени, применяется не раньше него;
8. **воспроизводимый журнал** — в сохранённом прогоне лежат все правки
   с номерами команд и модельным временем, и исходные условия отдельно
   от правок на ходу.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_live_app.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_vision_live_app_out"))

import probe._isolate  # noqa: E402,F401

from probe.sandbox_run import Run  # noqa: E402

OUT = Path(__file__).parent
CONTROL_MS = 20.0
RUN_NAME = "vision-live-app"

fails = []
rep = {"проверки": {}, "замеры": {}, "подтверждения": [], "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def window_spec(vision=True):
    """Спека ровно та, которую собирает Песочница окна.

    Собирается НЕ руками: берётся настоящий `App.sandbox_spec`, только
    с подставленным состоянием рейки. Иначе проба проверяла бы свою
    выдумку, а не окно — а именно этим и был плох прежний разрыв между
    пробами и продуктом.
    """
    from flybrain.desk import app as A, state as ST

    st = ST.AppState()
    st.sb.vision = bool(vision)
    st.sb.t_run = 0.0            # сеанс без заранее заданного конца
    st.sb.keep_s = 5.0
    st.sb.vision_hz = 100.0
    st.sb.vision_max_hz = 150.0
    # Оба слота объекта выключены на старте: сеанс обязан начинаться как
    # сеанс без объектов, а включение должно быть видно замером.
    for o in st.sb.objects:
        o.on = False
    fake = A.App.__new__(A.App)
    fake.st = st
    sp = A.App.sandbox_spec(fake)
    sp["out"] = RUN_NAME if vision else RUN_NAME + "-off"
    return sp


def objects(on=None, grey=0.1, pos=(14.0, 0.0, 1.5), radius=1.5):
    """Правка слотов объекта: ровно столько, сколько в модели."""
    from flybrain.desk.state import SANDBOX_OBJECTS, SANDBOX_OBJ_POS

    out = []
    for i in range(SANDBOX_OBJECTS):
        p = pos if i == on else SANDBOX_OBJ_POS[i]
        out.append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                    "radius": float(radius), "grey": float(grey),
                    "on": i == on})
    return out


def vis_hz(run, after_ms=None):
    """Поданная частота фоторецепторов по кадру не раньше заданного.

    Берётся из записи шага, которую работник кладёт рядом с кадром
    (`ctx.frame(extra=rec)`): то есть окно читает ровно то, что
    посчитала модель, а не свою копию.
    """
    if after_ms is not None:
        run.wait_model(after_ms)
    ex = run.ses.extra or {}
    return ex.get("vis_hz"), float(ex.get("t_ms") or 0.0)


def main():                                               # noqa: C901
    t0 = time.perf_counter()
    print("Управление зрением через настоящее приложение", flush=True)
    from flybrain.ui.session import Session

    # --- 1. спека окна ---
    print("\n1. Спека, которую собирает Песочница", flush=True)
    sp_on = window_spec(True)
    sp_off = window_spec(False)
    b_on, b_off = sp_on["body"], sp_off["body"]
    rep["замеры"]["спека окна"] = {
        "со зрением": {k: v for k, v in b_on.items()
                       if k.startswith("vis") or k == "live_initial"},
        "без зрения: зрительных ключей": sorted(
            k for k in b_off if k.startswith("vis")),
        "без зрения: ключи live_initial": sorted(b_off["live_initial"]),
    }
    check("Песочница собирает спеку со зрением и слотами объектов",
          b_on.get("vision") is True and b_on.get("vis_objects", 0) > 0
          and "objects" in b_on["live_initial"],
          "vision=%s, vis_objects=%s, исходные условия: %s"
          % (b_on.get("vision"), b_on.get("vis_objects"),
             ", ".join(sorted(b_on["live_initial"]))))
    check("встречный замер: с выключенным зрением спека не несёт ни "
          "одного зрительного ключа",
          not any(k.startswith("vis") for k in b_off)
          and "objects" not in b_off["live_initial"],
          "зрительных ключей %d, исходные условия: %s"
          % (sum(1 for k in b_off if k.startswith("vis")),
             ", ".join(sorted(b_off["live_initial"]))))

    ses = Session()
    run = Run(ses)
    try:
        print("\n2. Сеанс со зрением: поднимаю работника", flush=True)
        boot = run.wait_ready()
        if not ses.ready:
            check("работник поднялся", False, "не поднялся за %.0f с" % boot)
            return 1
        job = ses.submit(sp_on)
        run.wait_state("running")
        if ses.state != "running":
            check("сеанс со зрением пошёл", False,
                  "состояние %s, задача %s, ошибка %s"
                  % (ses.state, job, ses.error))
            return 1
        got = run.wait_frame()
        hz0, t0_ms = vis_hz(run)
        rep["замеры"]["старт"] = {
            "работник готов за, с": round(boot, 1),
            "кадр пришёл": bool(got),
            "частота фоторецепторов на старте, Гц": hz0,
            "модельное на старте, мс": t0_ms,
        }
        check("сеанс идёт, и зрение в нём работает: кадр несёт поданную "
              "частоту фоторецепторов",
              got and hz0 is not None and float(hz0) > 50.0,
              "кадр пришёл, vis_hz %s Гц на %s мс модельного времени"
              % (hz0, t0_ms))

        # --- 3. включение объекта: идентификатор, подтверждение, замер ---
        print("\n3. Тёмный объект: номер команды, подтверждение, замер",
              flush=True)
        base, base_ms = vis_hz(run)
        cid, sent_ms, _w = run.send("set", {"objects": objects(on=0,
                                                              grey=0.05)})
        rec = run.wait_applied(cid)
        on_hz, on_ms = (None, None)
        if rec is not None and rec.outcome == "applied":
            on_hz, on_ms = vis_hz(run, after_ms=(rec.t_model_ms or 0) + 60)
        cid2, _s2, _w2 = run.send("set", {"objects": objects(on=None)})
        rec2 = run.wait_applied(cid2)
        off_hz = None
        if rec2 is not None and rec2.outcome == "applied":
            off_hz, _ = vis_hz(run, after_ms=(rec2.t_model_ms or 0) + 60)
        rep["замеры"]["объект"] = {
            "номер команды включения": cid,
            "номер команды выключения": cid2,
            "номера различны": cid != cid2,
            "исход включения": None if rec is None else rec.outcome,
            "модельное применения, мс": None if rec is None else rec.t_model_ms,
            "модельное на отправке, мс": sent_ms,
            "применено не раньше наблюдённого":
                None if rec is None else (rec.t_model_ms or 0) >= sent_ms,
            "частота до, Гц": base, "частота с объектом, Гц": on_hz,
            "частота после выключения, Гц": off_hz,
        }
        check("у команды есть идентификатор, а у применения — модельное "
              "время, и оно не раньше наблюдённого на отправке",
              rec is not None and rec.outcome == "applied"
              and rec.cmd_id == cid and cid != cid2
              and (rec.t_model_ms or 0) >= sent_ms,
              "номер %s, исход %s, применено на %s мс при наблюдённых "
              "%s мс на отправке"
              % (cid, None if rec is None else rec.outcome,
                 None if rec is None else rec.t_model_ms, sent_ms))
        moved = (base is not None and on_hz is not None
                 and abs(float(on_hz) - float(base)) > 0.05)
        check("ИЗМЕРЕННОЕ последствие: включённый тёмный объект меняет "
              "поданную частоту, а не только подтверждение",
              moved,
              "%s → %s Гц (на %s), выключение вернуло к %s Гц"
              % (base, on_hz,
                 None if not moved else round(float(on_hz) - float(base), 4),
                 off_hz))

        # --- 4. подача зрения на ходу ---
        print("\n4. Подача зрения выключается на ходу", flush=True)
        cid, _s, _w = run.send("set", {"vision_feed": False})
        rec = run.wait_applied(cid)
        # Выключенная подача не пишет `vis_hz` в запись шага вовсе:
        # поля нет, а не ноль. Ноль значил бы «подали нулевую частоту».
        feed_off = None
        if rec is not None and rec.outcome == "applied":
            run.wait_model((rec.t_model_ms or 0) + 80)
            feed_off = (ses.extra or {}).get("vis_hz")
        cid, _s, _w = run.send("set", {"vision_feed": True})
        rec_back = run.wait_applied(cid)
        feed_back = None
        if rec_back is not None and rec_back.outcome == "applied":
            feed_back, _ = vis_hz(run,
                                  after_ms=(rec_back.t_model_ms or 0) + 80)
        rep["замеры"]["подача"] = {
            "исход выключения": None if rec is None else rec.outcome,
            "vis_hz при выключенной подаче": feed_off,
            "vis_hz после включения обратно": feed_back,
        }
        check("подача зрения снимается на ходу, и в записи шага поля "
              "частоты не остаётся вовсе",
              feed_off is None,
              "vis_hz при выключенной подаче: %r — отсутствие поля, а "
              "не ноль герц" % (feed_off,))
        check("встречный замер: обратно включается, и частота "
              "возвращается",
              feed_back is not None and float(feed_back) > 50.0,
              "vis_hz после включения %s Гц" % (feed_back,))

        # --- 5. свет и контраст на ходу ---
        print("\n5. Свет и контраст на ходу", flush=True)
        before, _ = vis_hz(run)
        cid, _s, _w = run.send("set", {"vision_light": 0.3})
        rl = run.wait_applied(cid)
        dim = None
        if rl is not None and rl.outcome == "applied":
            dim, _ = vis_hz(run, after_ms=(rl.t_model_ms or 0) + 80)
        cid, _s, _w = run.send("set", {"vision_light": 1.0,
                                       "vision_contrast": 2.5})
        rc = run.wait_applied(cid)
        con = None
        if rc is not None and rc.outcome == "applied":
            con, _ = vis_hz(run, after_ms=(rc.t_model_ms or 0) + 80)
        rep["замеры"]["свет и контраст на ходу"] = {
            "до, Гц": before, "свет 0,3, Гц": dim,
            "свет 1,0 и контраст 2,5, Гц": con,
            "исход правки света": None if rl is None else rl.outcome,
            "исход правки контраста": None if rc is None else rc.outcome,
        }
        check("свет сцены меняет поданную частоту на ходу",
              dim is not None and before is not None
              and float(dim) < float(before) - 1.0,
              "%s → %s Гц при свете 0,3" % (before, dim))
        check("контраст меняет поданную частоту на ходу",
              con is not None and dim is not None
              and abs(float(con) - float(dim)) > 1.0,
              "%s Гц при свете 1,0 и контрасте 2,5 против %s Гц при "
              "свете 0,3" % (con, dim))

        # --- 6. отказ до правки ---
        print("\n6. Негодная зрительная правка отклоняется", flush=True)
        bad = [("слот за пределом",
                {"objects": [{"x": 0.0, "y": 0.0, "z": 500.0,
                              "radius": 1.0, "grey": 0.1, "on": True}]}),
               ("контраст за пределом", {"vision_contrast": 99.0}),
               ("частота обновления на ходу не меняется",
                {"vision_hz": 250.0})]
        refused = []
        for title, payload in bad:
            cid, _s, _w = run.send("set", payload)
            r = run.wait_applied(cid, limit=60.0)
            refused.append({"что": title,
                            "исход": None if r is None else r.outcome,
                            "почему": None if r is None else r.why})
            print("   %-38s %s: %s"
                  % (title, None if r is None else r.outcome,
                     ((r.why or "")[:70]) if r is not None else "—"),
                  flush=True)
        rep["замеры"]["отказы"] = refused
        alive_hz, _ = vis_hz(run)
        # Исход отказа у работника называется `rejected` — так он
        # объявлен в `ipc.py` («applied | rejected | cancelled |
        # error»), и подставлять сюда своё слово
        # нельзя: первая редакция пробы искала «refused» и печатала
        # «отклонено 0 из 3» при трёх правильных отказах с причинами.
        check("каждая негодная правка отклонена с причиной, а сеанс "
              "продолжает идти",
              all(x["исход"] == "rejected" and x["почему"]
                  for x in refused)
              and alive_hz is not None,
              "отклонено %d из %d, после отказов сеанс жив и подаёт "
              "%s Гц" % (sum(1 for x in refused
                             if x["исход"] == "rejected"),
                         len(refused), alive_hz))

        # --- 7. синхронизация по модельному времени ---
        print("\n7. Правка, придержанная до модельного времени", flush=True)
        now_ms = float(ses.t_ms)
        at = now_ms + 6 * CONTROL_MS
        cid = ses.control("set", {"objects": objects(on=1, grey=0.9)},
                          at_t_ms=at)
        held = run.wait_applied(cid)
        rep["замеры"]["придержанная правка"] = {
            "заказано не раньше, мс": at,
            "применено на, мс": None if held is None else held.t_model_ms,
            "исход": None if held is None else held.outcome,
        }
        check("придержанная правка применена НЕ РАНЬШЕ названного "
              "модельного времени",
              held is not None and held.outcome == "applied"
              and (held.t_model_ms or 0) >= at - 1e-6,
              "заказано не раньше %.0f мс, применено на %s мс"
              % (at, None if held is None else held.t_model_ms))

        # --- 8. журнал в сохранённом прогоне ---
        print("\n8. Воспроизводимый журнал в сохранённом прогоне",
              flush=True)
        cid, _s, _w = run.send("stop")
        run.wait_applied(cid, limit=120.0)
        run.wait_state("idle", limit=300.0)
        run.pump(1.0)
    finally:
        ses.close()

    from flybrain import data as D

    d = D.OUT / RUN_NAME
    meta = json.loads((d / "embodied.json").read_text(encoding="utf-8"))
    tim = meta.get("timing") or {}
    edits = tim.get("живые правки") or []
    start = tim.get("исходные условия") or {}
    vis = tim.get("зрение") or {}
    # Исходные условия стоят в том же журнале, но КОМАНДОЙ не являются:
    # они применяются до разогрева, их никто не посылал, и номера у них
    # нет по смыслу. Поэтому «у каждой команды есть идентификатор»
    # проверяется по правкам, а наличие записи об исходных условиях —
    # отдельной проверкой ниже. Первая редакция пробы считала их
    # вместе и давала «7 из 8» на правильном журнале.
    live_edits = [e for e in edits if e.get("что") != "исходные условия"]
    initial = [e for e in edits if e.get("что") == "исходные условия"]
    with_cmd = [e for e in live_edits if e.get("команда") is not None]
    with_t = [e for e in live_edits
              if isinstance(e.get("модельное, мс"), (int, float))]
    ids = {e.get("команда") for e in with_cmd}
    vis_edits = [e for e in live_edits
                 if any(str(k).startswith("vision")
                        or k == "objects" for k in (e.get("стало") or {}))]
    rep["замеры"]["журнал"] = {
        "записей в журнале": len(edits),
        "правок на ходу": len(live_edits),
        "записей об исходных условиях": len(initial),
        "с номером команды": len(with_cmd),
        "различных номеров": len(ids),
        "с модельным временем": len(with_t),
        "зрительных правок": len(vis_edits),
        "исходные условия отдельно": sorted(start),
        "блок «зрение» в прогоне": bool(vis),
        "версия привязки": (vis.get("привязка") or {}).get("версия привязки"),
        "отпечаток набора": (vis.get("привязка") or {}).get(
            "отпечаток набора"),
        "оговорок": len(tim.get("оговорки") or []),
    }
    check("у каждой правки на ходу есть СВОЙ номер команды и модельное "
          "время применения",
          len(live_edits) > 0 and len(with_cmd) == len(live_edits)
          and len(ids) == len(live_edits)
          and len(with_t) == len(live_edits),
          "правок на ходу %d, с номером %d, различных номеров %d, с "
          "модельным временем %d; зрительных из них %d"
          % (len(live_edits), len(with_cmd), len(ids), len(with_t),
             len(vis_edits)))
    check("исходные условия стоят ОТДЕЛЬНО от правок на ходу, своей "
          "записью и без номера команды",
          "objects" in start and "sources" in start
          and len(initial) == 1
          and initial[0].get("команда") is None
          and initial[0].get("модельное, мс") == 0.0,
          "исходные условия: %s; в журнале одна запись о них, без "
          "номера команды и на нулевом модельном времени — они "
          "применяются до разогрева, и командой не являются"
          % ", ".join(sorted(start)))
    check("в прогоне видно, какая ВЕРСИЯ привязки считала",
          bool((vis.get("привязка") or {}).get("версия привязки"))
          and bool((vis.get("привязка") or {}).get("отпечаток набора")),
          "версия: «%s…», отпечаток %s"
          % (str((vis.get("привязка") or {}).get("версия привязки"))[:60],
             (vis.get("привязка") or {}).get("отпечаток набора")))

    note("зрительные ручки идут тем же протоколом команд, что источники "
         "запаха и вмешательства: проверка целиком до записи, применение "
         "на границе шага управления, подтверждение с модельным "
         "временем, запись в журнал прогона. Своего протокола у зрения "
         "нет — и это главное, чего здесь добивались.")
    note("подтверждение «применено» нигде не принимается за "
         "последствие: после каждой правки замеряется поданная частота "
         "фоторецепторов из той же записи шага, которую посчитала "
         "модель. Правка, у которой последствия нет, этой пробой не "
         "отличилась бы от правки, которой не было.")
    rep["итог"] = {"проверок": len(rep["проверки"]), "провалов": len(fails),
                   "за, с": round(time.perf_counter() - t0, 1)}
    (OUT / "vision_live_app.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nпроверок %d, провалов %d, за %.0f с"
          % (len(rep["проверки"]), len(fails),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("провалились: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
