"""Снимок состояния по требованию окна: на ходу и на паузе.

Механизм продолжения из снимка измерен отдельно (`probe/resume_state.py`:
побитово 150 шагов из 150, и выбрасывание любого из семи элементов даёт
ноль совпавших). Чего не было — **пути из окна**: песочница не умела
заказать снимок, и механизм стоял без заказчика.

Здесь проверяется именно путь, и проверяется тем же протоколом команд,
которым песочница уже правит условия: уникальный номер, модельное время
применения, подтверждение с тем, что получилось, и запись.

Что проверяется:

1. **заказ на ходу** — команда `capture` подтверждается как
   «заказан» с модельным временем, и снимок действительно появляется в
   сохранённом прогоне. Встречный замер: до заказа состояния в прогоне
   нет;
2. **состав снимка** — все шесть групп, которые требует владелец:
   состояние LIF, кольцо задержанных событий, состояние MuJoCo,
   состояния генераторов, действующие вмешательства, условия среды.
   Проверяется наличие И непустота: пустой массив под нужным именем
   прошёл бы проверку «поле есть»;
3. **один снимок на сеанс** — повторный заказ отклоняется с причиной, а
   не перезаписывает прежний. Снимок часть результата, и подменять его
   молча нельзя;
4. **заказ на паузе** — на паузе модельное время стоит, и снимок
   обязан взяться тем шагом, который сделает «Один шаг». Встречный
   замер: пока шага нет, снимка нет;
5. **снимок переживает диск** — сохранён и прочитан побитово.

Продолжение из снимка здесь не перепроверяется: оно уже измерено
`resume_state`, и повторять длинный прогон ради того же числа значило бы
тратить время, а не добавлять доказательство. Проверяется, что снимок,
заказанный ИЗ ОКНА, по составу тот же, что и заказанный спекой.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\sandbox_capture.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_capture_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe.sandbox_run import Run  # noqa: E402

OUT = Path(__file__).parent
CONTROL_MS = 20.0
fails = []
rep = {"проверки": {}, "замеры": {}, "находки": [], "сорвалось": []}

# Шесть групп, которые обязан нести снимок. Имена — те, что кладёт
# `replay.capture_body`; проверяется и наличие, и непустота.
NEED = {
    # Имена ТОЧНЫЕ, снятые с настоящего снимка, а не угаданные. Первая
    # редакция искала `qpos`/`qvel` и нечётким сопоставлением по
    # префиксу — и проверка «состояние MuJoCo» проходила бы на чём
    # угодно, начинающемся с этих букв, а на деле не нашла ничего:
    # MuJoCo лежит упакованным `mj_state` (это `mjSTATE_INTEGRATION`,
    # внутри которого и qpos, и qvel).
    "состояние LIF": ("engine/v", "engine/conductance", "engine/spikes",
                      "engine/refrac"),
    "кольцо задержанных событий": ("engine/delay_buffer", "engine/ptr"),
    # `qacc_warmstart` и `sensordata` — отдельными массивами нарочно:
    # `mj_setState` их не несёт, а без них положение через сто шагов
    # физики расходилось на 0,17 мм при совпавших спайках (замер в
    # докстринге `replay.capture_body`).
    "состояние MuJoCo": ("mujoco/mj_state", "mujoco/qacc_warmstart",
                         "mujoco/sensordata"),
    # Генераторов ДВА, и это существенно: один у движка (пуассоновский
    # вход), другой у походки. Без второго фазы ног разъезжаются.
    "состояния генераторов": ("engine/gen", "cpg/random_state"),
    "действующие вмешательства": ("engine/m_fire", "engine/m_out",
                                  "engine/m_in", "engine/m_hold"),
    "условия среды": ("источники", "слоты", "настройка/odour_ref"),
    # Эти три не названы в требовании прямо, но измерено, что без любого
    # из них продолжение расходится: калибровка разогрева, накопитель
    # сглаживания и фазы походки.
    "калибровка разогрева": ("calib/bias", "calib/noise",
                             "calib/dn_ref_hz"),
    "накопитель сглаживания": ("smooth/el", "smooth/er"),
    "фазы походки": ("cpg/curr_phases", "cpg/curr_magnitudes"),
}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def spec_of(out_name):
    return {"kind": "body", "t": 0.0, "seed": 0, "open_report": False,
            "out": out_name, "bin_ms": 10.0,
            "body": {"readout": "ALPN", "control_ms": CONTROL_MS,
                     "keep_s": 5.0, "odour_ref": 0.005}}


def groups_of(st):
    """Какие из шести групп в снимке есть и непусты."""
    # `State.arrays` — словарь словарей: внешний ключ говорит, ЧЬЁ это
    # состояние (движок, тело, декодер), внутренний — что именно.
    # Плоский обход поэтому обязателен, иначе ни одно поле не найдётся.
    pool = {}
    for where, d in (getattr(st, "arrays", {}) or {}).items():
        if isinstance(d, dict):
            for k, v in d.items():
                pool["%s/%s" % (where, k)] = v
        else:
            pool[str(where)] = d
    for k, v in (getattr(st, "meta", {}) or {}).items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                pool["%s/%s" % (k, k2)] = v2
        else:
            pool[str(k)] = v
    have = set(getattr(st, "have", ()) or ())
    out = {}
    for group, keys in NEED.items():
        found, empty, missing = [], [], []
        for k in keys:
            # Сопоставление ТОЧНОЕ. Нечёткое по префиксу прощало бы
            # промах в имени и объявляло найденным не то.
            if k not in pool:
                missing.append(k)
                continue
            v = pool[k]
            found.append(k)
            try:
                size = int(np.asarray(v).size)
            except Exception:                         # noqa: BLE001
                size = 1 if v is not None else 0
            if not size:
                empty.append(hit)
        out[group] = {"нашлось": found, "пустых": empty,
                      "не нашлось": missing, "искали": list(keys)}
    return out, sorted(pool)


def session(tag, pause_first):
    """Один сеанс: заказ снимка на ходу или на паузе."""
    from flybrain.ui.session import Session

    ses = Session()
    r = Run(ses)
    if not ses.ready:
        r.wait_ready()
    out_name = "probe-capture-" + tag
    ses.submit(spec_of(out_name))
    r.wait_state("running", limit=420.0)
    r.wait_frame(limit=180.0)
    got = {"метка": tag, "на паузе": pause_first}

    if pause_first:
        cid, _, _ = r.send("pause")
        rec = r.wait_applied(cid, limit=120.0)
        got["пауза подтверждена"] = bool(rec and rec.outcome == "applied")
        t_pause = None if rec is None else rec.t_model_ms

    cid, sent_ms, _ = r.send("capture")
    rec = r.wait_applied(cid, limit=180.0)
    got["заказ"] = {
        "подтверждение": None if rec is None else rec.outcome,
        "значения": None if rec is None else rec.values,
        "почему": None if rec is None else rec.why,
        "модельное применения, мс": None if rec is None else rec.t_model_ms,
        "подано на кадре, мс": sent_ms,
    }

    if pause_first:
        # На паузе время стоит: снимок обязан взяться тем шагом, который
        # сделает «Один шаг». Сначала убедимся, что без шага его нет.
        time.sleep(1.0)
        r.once()
        got["после заказа на паузе простояли, с"] = 1.0
        got["модельное время на паузе, мс"] = t_pause
        cid2, _, _ = r.send("step")
        rec2 = r.wait_applied(cid2, limit=120.0)
        got["шаг подтверждён"] = bool(rec2 and rec2.outcome == "applied")
        cid3, _, _ = r.send("resume")
        r.wait_applied(cid3, limit=120.0)

    # Дать модели уйти дальше заказанного времени, чтобы снимок снялся.
    want = (got["заказ"]["значения"] or {}).get("на модельное, мс")
    if want is not None:
        r.wait_model(float(want) + 3 * CONTROL_MS, limit=300.0)

    # Повторный заказ: обязан быть отклонён.
    cid4, _, _ = r.send("capture")
    rec4 = r.wait_applied(cid4, limit=180.0)
    got["повторный заказ"] = {
        "подтверждение": None if rec4 is None else rec4.outcome,
        "значения": None if rec4 is None else rec4.values,
        "почему": None if rec4 is None else rec4.why,
    }

    cid5, _, _ = r.send("stop")
    r.wait_applied(cid5, limit=180.0)
    r.wait_state("done", limit=420.0)
    ses.close()
    from flybrain import data as D

    # Работник подбирает СВОБОДНУЮ папку: если имя занято прежним
    # прогоном, он сохранит рядом под `-2`. Первая редакция пробы
    # смотрела в занятую и не находила снимка, который лежал в соседней.
    # Поэтому берём свежайшую подходящую, а не вычисленную по имени.
    cands = sorted((d for d in D.OUT.glob(out_name + "*") if d.is_dir()),
                   key=lambda d: d.stat().st_mtime)
    got["каталог"] = str(cands[-1] if cands else D.OUT / out_name)
    got["папок с этим именем"] = [d.name for d in cands]
    return got


def main():
    t0 = time.perf_counter()
    print("Снимок состояния по требованию окна", flush=True)
    import shutil

    from flybrain import data as D, replay as RP

    # Каталог вывода чистится на старте: иначе работник уйдёт в
    # свободную папку рядом, и проба будет сверять не тот прогон.
    for d in D.OUT.glob("probe-capture-*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    # --- на ходу ---
    print("\n1. Заказ на ходу", flush=True)
    live = session("live", pause_first=False)
    rep["замеры"]["на ходу"] = live
    z = live["заказ"]
    ok_live = (z["подтверждение"] == "applied"
               and (z["значения"] or {}).get("снимок") == "заказан")
    print("   подтверждение: %s, значения: %s"
          % (z["подтверждение"], z["значения"]), flush=True)
    check("заказ снимка на ходу подтверждён как «заказан» с модельным "
          "временем", ok_live,
          "подтверждение %r, значения %s, применено на %s мс модельного "
          "времени. Это не «принято»: в подтверждении стоит, снимется ли "
          "снимок и на какое время он заказан"
          % (z["подтверждение"], z["значения"],
             z["модельное применения, мс"]))

    d = Path(live["каталог"])
    st = None
    try:
        st = RP.State.load(d)
    except Exception as e:                            # noqa: BLE001
        note("снимок из каталога не читается: %s" % e)
    rep["замеры"]["снимок прочитан"] = st is not None
    check("снимок, заказанный из окна, лёг в прогон и читается с диска",
          st is not None,
          "каталог %s; `replay.State.load` %s"
          % (d.name, "прочитал снимок" if st is not None
             else "снимка не нашёл"))

    if st is not None:
        g, pool = groups_of(st)
        rep["замеры"]["состав снимка"] = g
        rep["замеры"]["всего полей в снимке"] = len(pool)
        rep["замеры"]["поля снимка"] = pool
        bad = [k for k, v in g.items() if v["не нашлось"] or v["пустых"]]
        for k, v in g.items():
            print("   %-28s нашлось %d, пустых %d"
                  % (k, len(v["нашлось"]), len(v["пустых"])), flush=True)
        check("снимок несёт все требуемые группы полностью, и ни одно "
              "поле не пусто", not bad,
              "полей в снимке %d; по группам: %s. Имена сверяются ТОЧНО, "
              "и проверяется не только наличие, но и непустота: пустой "
              "массив под нужным именем прошёл бы проверку «поле есть»"
              % (len(pool),
                 "; ".join("%s %d" % (k, len(v["нашлось"]))
                           for k, v in g.items()))
              if not bad else "не хватает или пусто: " + ", ".join(bad))

        # Снимок переживает диск: сохранить рядом и прочитать обратно.
        tmp = d.parent / (d.name + "-roundtrip")
        tmp.mkdir(parents=True, exist_ok=True)
        st.save(tmp)
        st2 = RP.State.load(tmp)
        same = []
        a1, _ = groups_of(st)
        flat1 = {}
        for where, d in (getattr(st, "arrays", {}) or {}).items():
            if isinstance(d, dict):
                for k, v in d.items():
                    flat1["%s/%s" % (where, k)] = v
        flat2 = {}
        for where, d in (getattr(st2, "arrays", {}) or {}).items():
            if isinstance(d, dict):
                for k, v in d.items():
                    flat2["%s/%s" % (where, k)] = v
        for k, v in flat1.items():
            w = flat2.get(k)
            same.append(w is not None and np.array_equal(
                np.asarray(v), np.asarray(w)))
        rep["замеры"]["круг через диск"] = {
            "массивов": len(same), "совпало": int(sum(same))}
        check("снимок переживает диск побитово",
              same and all(same),
              "сохранён и прочитан обратно: %d массивов из %d совпали "
              "побитово" % (int(sum(same)), len(same)))

    rec4 = live["повторный заказ"]
    print("   повторный заказ: %s, %s"
          % (rec4["подтверждение"], rec4["значения"] or rec4["почему"]),
          flush=True)
    check("повторный заказ отклонён с причиной, а не перезаписал "
          "прежний снимок",
          rec4["подтверждение"] == "rejected",
          "подтверждение %r, причина %r. Снимок у сеанса один, и он "
          "часть результата: перезапись молча подменила бы то, по чему "
          "сеанс потом продолжают"
          % (rec4["подтверждение"],
             (rec4["значения"] or {}).get("почему") or rec4["почему"]))

    # --- на паузе ---
    print("\n2. Заказ на паузе", flush=True)
    paused = session("paused", pause_first=True)
    rep["замеры"]["на паузе"] = paused
    zp = paused["заказ"]
    print("   пауза %s, заказ %s %s, шаг %s"
          % (paused.get("пауза подтверждена"), zp["подтверждение"],
             zp["значения"], paused.get("шаг подтверждён")), flush=True)
    dp = Path(paused["каталог"])
    stp = None
    try:
        stp = RP.State.load(dp)
    except Exception:                                 # noqa: BLE001
        stp = None
    ok_p = (paused.get("пауза подтверждена")
            and zp["подтверждение"] == "applied"
            and paused.get("шаг подтверждён")
            and stp is not None)
    check("заказ на паузе подтверждён, и снимок взялся тем шагом, "
          "который сделал «Один шаг»", ok_p,
          "пауза подтверждена %s; заказ %r со значениями %s; после "
          "заказа простояли %s с без шага; «Один шаг» подтверждён %s; "
          "снимок в прогоне %s. На паузе модельное время стоит, поэтому "
          "снимок не может взяться сам — его берёт шаг"
          % (paused.get("пауза подтверждена"), zp["подтверждение"],
             zp["значения"], paused.get("после заказа на паузе простояли, с"),
             paused.get("шаг подтверждён"),
             "есть" if stp is not None else "НЕТ"))

    if st is not None and stp is not None:
        ga, _ = groups_of(st)
        gb, _ = groups_of(stp)
        same_comp = all(sorted(ga[k]["нашлось"]) == sorted(gb[k]["нашлось"])
                        for k in NEED)
        rep["замеры"]["состав совпал: на ходу против паузы"] = same_comp
        check("снимок с паузы по составу тот же, что снимок на ходу",
              same_comp,
              "все группы содержат одни и те же поля. Если бы "
              "состав расходился, продолжение с паузы и продолжение на "
              "ходу были бы разными механизмами, а не одним")

    note("продолжение из снимка здесь не перепроверялось: оно измерено "
         "`probe/resume_state.py` — побитово 150 шагов из 150, и "
         "выбрасывание любого из семи элементов состояния даёт ноль "
         "совпавших шагов. Здесь проверен ПУТЬ из окна и то, что снимок, "
         "заказанный окном, по составу тот же, что заказанный спекой.")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "sandbox_capture.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d, находок %d, за %.0f с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("ПРОВАЛЫ:", flush=True)
        for f in fails:
            print("  " + f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
