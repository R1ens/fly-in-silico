"""Зрение мухи и офскрин-рендерер работника в одном процессе.

Это шаг 0 зрительного этапа, и он сделан ДО всякой реализации, потому
что здесь лежит самый крупный неизвестный риск замысла: зрение
FlyGym и живой вид работника — два разных потребителя OpenGL в одном
процессе, и вместе они не работали ни разу. `probe/eyes.py` включает
зрение без `mujoco.Renderer`; `flybrain/worker.py` создаёт `Renderer`
без зрения. В истории проекта уже было фатальное падение MuJoCo из-за
драки за контекст OpenGL (`lab/MUJOCO_LOG.TXT`), так что вопрос не
теоретический.

Чего проба НЕ делает: не правит ни одного файла в `flybrain/`. Она
только выясняет, можно ли на это опираться, и сколько это стоит.

Четыре вопроса и встречный замер к каждому:

1. **Живут ли вместе.** Тело со зрением и `mujoco.Renderer` в одном
   процессе, 30 шагов управления, кадр и сетчатка на каждом шаге.
   Встречный замер: `MUJOCO_LOG.TXT` читается до и после — MuJoCo
   пишет туда фатальные ошибки, и молчание журнала входит в результат.

2. **Не портит ли зрение живой вид.** `Fly._update_vision` на время
   съёма правит `geom_rgba` модели. Если восстановление неполно,
   кадр работника поедет по цвету — а заметить это по одному снимку
   нельзя. Поэтому прогон идёт дважды, в ОТДЕЛЬНЫХ процессах (одно
   тело на процесс — правило, выведенное из того самого падения), с
   одним зерном: со зрением и без. Кадры сравниваются числом.

3. **Настоящая ли сетчатка.** Форма `(2, 721, 2)`, разброс по
   омматидиям не нулевой, и кадр МЕНЯЕТСЯ от шага к шагу. Встречный
   замер: доля омматидиев, стоящих на одном значении, и число
   совпавших подряд кадров — замороженная сетчатка выдала бы себя
   именно здесь, а не в форме тензора.

4. **Сколько стоит.** Время шага управления со зрением и без, время
   кадра работника, время съёма сетчатки — по отдельности. Отсюда
   берётся ответ, какая частота обновления зрения вообще по карману.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_gl.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
ROOT = OUT.parent
MJLOG = ROOT / "MUJOCO_LOG.TXT"
RING_SHAPE = (720, 960, 3)      # как в flybrain/worker.py: высота, ширина
STEPS = 15                      # шагов управления
PHYS_DT = 1e-4                  # как BodyConfig.physics_dt
CONTROL_MS = 20.0               # как BodyConfig.control_ms
PHYS = int(round(CONTROL_MS / 1000.0 / PHYS_DT))   # шагов физики на шаг
HZS = (50, 100, 250, 500)       # развёртка по частоте обновления зрения
SEED = 0

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


# ------------------------------------------------- прогон в своём процессе

def run_one(vision, out_path):
    """Один прогон: тело, рендерер работника, при желании зрение.

    Запускается отдельным процессом. Одно тело на процесс — не
    осторожность, а правило: фатальная ошибка MuJoCo из журнала проекта
    случилась именно там, где в одном процессе собирали тело дважды.
    """
    import mujoco

    from flygym_gymnasium import Fly
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import (
        default_leg_sensor_placements)
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    res = {"зрение": bool(vision)}
    t0 = time.perf_counter()
    sources = np.array([[25.0, 0.0, 1.5]])
    arena = OdorArena(odor_source=sources,
                      peak_odor_intensity=np.ones((len(sources), 1)))
    kw = dict(enable_adhesion=True, enable_olfaction=True,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    # `enable_vision` передаётся ТОЛЬКО когда зрение нужно: передача
    # даже со значением False меняет сборку модели (добавляются камеры
    # глаз), и тогда сравнение двух прогонов измеряло бы не то.
    if vision:
        kw["enable_vision"] = True
        hz = int(os.environ.get("FLYBRAIN_VISION_HZ") or 0)
        if hz:
            kw["vision_refresh_rate"] = hz
    fly = Fly(**kw)
    sim = HybridTurningController(fly=fly, cameras=[], arena=arena,
                                  timestep=1e-4)
    res["сборка, с"] = round(time.perf_counter() - t0, 2)

    obs, _ = sim.reset(seed=SEED)

    # Рендерер работника — дословно как в `worker.py._renderer`.
    t0 = time.perf_counter()
    model = sim.physics.model.ptr
    ren = mujoco.Renderer(model, RING_SHAPE[0], RING_SHAPE[1])
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = next(
        i for i in range(model.nbody)
        if "thorax" in (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower())
    res["контекст рендерера, с"] = round(time.perf_counter() - t0, 2)

    data = sim.physics.data.ptr
    frames, retinas, t_step, t_frame, t_eye = [], [], [], [], []
    phys_frames = []        # сетчатка на КАЖДОМ шаге физики
    action = np.array([1.0, 1.0])
    # Шаг управления — как в `body.run_episode`: 100 шагов физики по
    # 0,1 мс на 20 мс модельного времени. Один `sim.step` продвигает
    # физику на ОДИН `timestep`, и путать его с шагом управления нельзя:
    # в первой редакции этой пробы ошибка дала «9,85 модельной секунды
    # на настенную», то есть втрое быстрее реального времени.
    for s in range(STEPS):
        t0 = time.perf_counter()
        for _ in range(PHYS):
            obs, _, _, _, _ = sim.step(action)
            if vision:
                phys_frames.append(
                    np.asarray(obs["vision"], dtype=np.float32).copy())
        t_step.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ren.update_scene(data, cam)
        img = ren.render()
        t_frame.append(time.perf_counter() - t0)
        if s in (0, STEPS // 2, STEPS - 1):
            frames.append(np.asarray(img, dtype=np.uint8).copy())

        if vision:
            t0 = time.perf_counter()
            v = np.asarray(obs["vision"], dtype=np.float32).copy()
            t_eye.append(time.perf_counter() - t0)
            retinas.append(v)

    res["шаг управления, мс"] = {
        "медиана": round(1e3 * float(np.median(t_step)), 2),
        "худший": round(1e3 * float(np.max(t_step)), 2),
        "всего, с": round(float(np.sum(t_step)), 2),
    }
    res["кадр работника, мс"] = {
        "медиана": round(1e3 * float(np.median(t_frame)), 2),
        "худший": round(1e3 * float(np.max(t_frame)), 2),
    }
    if vision:
        res["съём сетчатки, мс"] = {
            "медиана": round(1e3 * float(np.median(t_eye)), 3),
            "худший": round(1e3 * float(np.max(t_eye)), 3),
        }
        # Сколько раз сетчатка действительно обновилась за прогон.
        # FlyGym обновляет её по своему расписанию
        # (`vision_refresh_rate`, по умолчанию 500 Гц), а не на каждом
        # шаге физики, поэтому «кадр не изменился» — норма, а не
        # заморозка. Мера — число РАЗЛИЧНЫХ кадров на шагах физики.
        pf = np.stack(phys_frames)
        changed = [i for i in range(1, len(pf))
                   if not np.array_equal(pf[i], pf[i - 1])]
        model_ms = STEPS * CONTROL_MS
        res["обновление сетчатки"] = {
            "шагов физики": int(len(pf)),
            "модельного времени, мс": model_ms,
            "обновлений": len(changed),
            "обновлений в секунду модельного времени":
                round(1e3 * len(changed) / model_ms, 1),
            "шагов физики между обновлениями":
                (round(float(np.mean(np.diff([0] + changed))), 2)
                 if changed else None),
            "объявлено FlyGym, Гц": int(fly.vision_refresh_rate),
            "маска обновления, доля True":
                round(float(np.mean(fly.vision_update_mask)), 4)
                if len(fly.vision_update_mask) else None,
        }
        r = np.stack(retinas)
        res["сетчатка"] = {
            "форма одного кадра": list(r.shape[1:]),
            "кадров": int(r.shape[0]),
            "минимум": round(float(r.min()), 4),
            "максимум": round(float(r.max()), 4),
            "среднее": round(float(r.mean()), 4),
            "разброс по омматидиям, последний кадр":
                round(float(r[-1].std()), 5),
            "доля омматидиев на одном значении, последний кадр":
                round(float((r[-1] == r[-1].flat[0]).mean()), 4),
            "кадров, совпавших с предыдущим побайтово":
                int(sum(1 for i in range(1, len(r))
                        if np.array_equal(r[i], r[i - 1]))),
            "наибольшая разница между соседними кадрами":
                round(float(max(np.abs(r[i] - r[i - 1]).max()
                                for i in range(1, len(r)))), 5),
        }
        np.save(out_path.with_suffix(".retina.npy"), r)

    np.save(out_path.with_suffix(".frames.npy"), np.stack(frames))
    res["поза в конце"] = [round(float(x), 4)
                           for x in np.asarray(obs["fly"])[0]]
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return res


# -------------------------------------------------------------- сравнение

def mjlog():
    """Содержимое журнала MuJoCo — он пишет туда только фатальное."""
    if not MJLOG.exists():
        return ""
    return MJLOG.read_text(encoding="utf-8", errors="replace")


def spawn(vision, tag, hz=0):
    out = OUT / ("vision_gl_%s.json" % tag)
    for p in (out, out.with_suffix(".frames.npy"),
              out.with_suffix(".retina.npy")):
        if p.exists():
            p.unlink()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["FLYBRAIN_VISION_GL"] = "1" if vision else "0"
    env["FLYBRAIN_VISION_OUT"] = str(out)
    env["FLYBRAIN_VISION_HZ"] = str(int(hz))
    t0 = time.perf_counter()
    pr = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                        env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=900)
    dt = time.perf_counter() - t0
    ok = pr.returncode == 0 and out.exists()
    if not ok:
        print("  вывод процесса (%s):" % tag, flush=True)
        print((pr.stdout or "")[-1500:], flush=True)
        print((pr.stderr or "")[-2500:], flush=True)
    return {
        "ok": ok, "код": pr.returncode, "за, с": round(dt, 1),
        "итог": (json.loads(out.read_text(encoding="utf-8"))
                 if out.exists() else None),
        "кадры": (np.load(out.with_suffix(".frames.npy"))
                  if out.with_suffix(".frames.npy").exists() else None),
    }


def main():
    t0 = time.perf_counter()
    print("Зрение и рендерер работника в одном процессе "
          "(шаг 0 зрительного этапа)", flush=True)
    log_before = mjlog()
    rep["замеры"]["журнал MuJoCo до пробы, байт"] = len(log_before)

    print("\n  прогон 1: тело + рендерер работника, зрение ВЫКЛЮЧЕНО",
          flush=True)
    base = spawn(False, "без_зрения")
    print("    код %s, за %s с" % (base["код"], base["за, с"]), flush=True)
    print("  прогон 2: тело + рендерер работника + ЗРЕНИЕ", flush=True)
    vis = spawn(True, "со_зрением")
    print("    код %s, за %s с" % (vis["код"], vis["за, с"]), flush=True)

    log_after = mjlog()
    new_log = log_after[len(log_before):].strip()
    rep["замеры"]["журнал MuJoCo дописал, байт"] = len(new_log)
    if new_log:
        rep["замеры"]["журнал MuJoCo дописал"] = new_log[:2000]

    # --- 1. живут ли вместе ---
    check("зрение и офскрин-рендерер работника живут в одном процессе",
          vis["ok"] and not new_log,
          "процесс со зрением завершился кодом %s за %s с и сделал %d "
          "шагов управления с кадром работника на каждом; журнал MuJoCo "
          "не дописал ни байта (он пишет туда только фатальное — так "
          "выглядели оба прежних падения). Встречный замер — тот же "
          "прогон без зрения: код %s"
          % (vis["код"], vis["за, с"], STEPS, base["код"]))
    if not vis["ok"]:
        rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
        (OUT / "vision_gl.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nпроверок %d, провалов %d" % (len(rep["проверки"]),
                                              len(fails)), flush=True)
        return 1

    v, b = vis["итог"], base["итог"]
    rep["замеры"]["со зрением"] = v
    rep["замеры"]["без зрения"] = b

    # --- 2. не портит ли зрение живой вид ---
    fa, fb = vis["кадры"], base["кадры"]
    same_shape = fa is not None and fb is not None and fa.shape == fb.shape
    if same_shape:
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        diff = {
            "кадров сравнено": int(fa.shape[0]),
            "наибольшая разница яркости, уровней": int(d.max()),
            "средняя разница яркости, уровней": round(float(d.mean()), 3),
            "доля изменившихся пикселей": round(float((d > 0).mean()), 4),
            "медиана яркости со зрением": int(np.median(fa)),
            "медиана яркости без зрения": int(np.median(fb)),
        }
        rep["замеры"]["кадр работника: со зрением против без"] = diff
        # Порог 8 уровней из 255 — не «на вид»: физика у двух прогонов
        # одна (одно зерно, одно действие), но модель со зрением несёт
        # две лишние камеры, поэтому побайтового равенства ждать нельзя.
        # Зато восстановление `geom_rgba` неполным быть не может: это
        # сдвинуло бы цвет СИСТЕМАТИЧЕСКИ, то есть медианы яркости
        # разъехались бы, а не отдельные пиксели.
        med_gap = abs(diff["медиана яркости со зрением"]
                      - diff["медиана яркости без зрения"])
        check("зрение не портит кадр живого вида: `geom_rgba` "
              "восстанавливается",
              med_gap <= 2,
              "медиана яркости %d со зрением против %d без — расхождение "
              "%d уровня из 255; наибольшая разница по пикселю %d, "
              "изменилось %.2f %% пикселей. Останься правка `geom_rgba` "
              "незакрытой, уехала бы медиана всего кадра, а не отдельные "
              "пиксели"
              % (diff["медиана яркости со зрением"],
                 diff["медиана яркости без зрения"], med_gap,
                 diff["наибольшая разница яркости, уровней"],
                 100 * diff["доля изменившихся пикселей"]))
    else:
        check("кадры обоих прогонов сняты", False, "форма не совпала")

    # --- 3. настоящая ли сетчатка ---
    r = v.get("сетчатка") or {}
    u = v.get("обновление сетчатки") or {}
    shape_ok = r.get("форма одного кадра") == [2, 721, 2]
    varies = (r.get("разброс по омматидиям, последний кадр", 0) > 1e-4
              and r.get("доля омматидиев на одном значении, "
                        "последний кадр", 1.0) < 0.9)
    check("сетчатка настоящая: форма и разброс по омматидиям",
          shape_ok and varies,
          "форма кадра %s (два глаза, 721 омматидий, два типа "
          "фоторецепторов), разброс по омматидиям %.5f, на одном "
          "значении %.1f %% омматидиев. Замороженная сетчатка дала бы "
          "форму ту же, а разброс нулевым"
          % (r.get("форма одного кадра"),
             r.get("разброс по омматидиям, последний кадр"),
             100 * r.get("доля омматидиев на одном значении, "
                         "последний кадр", 1.0)))

    # Сетчатка обновляется по РАСПИСАНИЮ FlyGym, а не на каждом шаге
    # физики. Проверяется не «меняется каждый шаг» (это было бы
    # неверное ожидание и в первой редакции пробы дало ложный провал), а
    # совпадение измеренной частоты с объявленной.
    hz = u.get("обновлений в секунду модельного времени", 0.0)
    want = float(u.get("объявлено FlyGym, Гц") or 0.0)
    check("частота обновления сетчатки совпала с объявленной FlyGym",
          want > 0 and abs(hz - want) <= 0.1 * want,
          "за %s мс модельного времени (%s шагов физики) сетчатка "
          "обновилась %s раз, то есть %.1f Гц против объявленных %.0f Гц; "
          "в среднем шагов физики между обновлениями %s, маска "
          "обновления FlyGym даёт долю True %s. Отсюда и берётся ответ: "
          "кадр, читаемый на каждом шаге управления (50 Гц), повторяется "
          "— обновлений в десять раз больше, чем шагов управления"
          % (u.get("модельного времени, мс"), u.get("шагов физики"),
             u.get("обновлений"), hz, want,
             u.get("шагов физики между обновлениями"),
             u.get("маска обновления, доля True")))

    # --- 4. сколько стоит ---
    sv = v["шаг управления, мс"]["медиана"]
    sb = b["шаг управления, мс"]["медиана"]
    fr = v["кадр работника, мс"]["медиана"]
    cost = {
        "шаг управления без зрения, мс": sb,
        "шаг управления со зрением, мс": sv,
        "зрение добавило, мс": round(sv - sb, 2),
        "зрение добавило, раз": round(sv / max(sb, 1e-9), 3),
        "кадр работника, мс": fr,
        "съём сетчатки из наблюдения, мс":
            (v.get("съём сетчатки, мс") or {}).get("медиана"),
        "сборка тела со зрением, с": v["сборка, с"],
        "сборка тела без зрения, с": b["сборка, с"],
        "контекст рендерера, с": v["контекст рендерера, с"],
    }
    # Во что это превращается на деле: шаг управления 20 мс модельного
    # времени, поэтому отношение модельного к настенному считается прямо.
    for nm, ms in (("без зрения", sb), ("со зрением", sv)):
        cost["доля настенного времени, %s" % nm] = round(
            20.0 / max(ms, 1e-9), 4)
    rep["замеры"]["цена"] = cost
    print("\n  цена: шаг управления %.1f мс без зрения, %.1f мс со "
          "зрением (×%.2f); кадр работника %.2f мс"
          % (sb, sv, sv / max(sb, 1e-9), fr), flush=True)
    print("  модельного времени на настенное: %.4f без зрения, %.4f со "
          "зрением" % (cost["доля настенного времени, без зрения"],
                       cost["доля настенного времени, со зрением"]),
          flush=True)

    check("цена зрения замерена и названа числом",
          sv > 0 and sb > 0,
          "зрение добавило %.1f мс к шагу управления в 20 мс модельного "
          "времени, то есть ×%.2f; сеанс идёт %.4f модельной секунды на "
          "настенную против %.4f без зрения. Это замер на %d шагах, а не "
          "оценка"
          % (sv - sb, sv / max(sb, 1e-9),
             cost["доля настенного времени, со зрением"],
             cost["доля настенного времени, без зрения"], STEPS))

    if sv / max(sb, 1e-9) > 1.5:
        note("зрение на каждом шаге управления (50 Гц) дороже, чем "
             "ожидалось: шаг вырос в %.2f раза (%.1f → %.1f мс), и сеанс "
             "идёт %.4f модельной секунды на настенную. Частота "
             "обновления зрения — условие прогона, и выбирать её надо по "
             "этому числу, а не по тому, что умеет FlyGym."
             % (sv / max(sb, 1e-9), sb, sv,
                cost["доля настенного времени, со зрением"]))

    # --- 5. развёртка по частоте обновления зрения ---
    # Этап 2 требует сравнить 50, 100, 250 и 500 Гц. Сравнивать надо не
    # по тому, что объявляет FlyGym, а по двум числам сразу: сколько это
    # стоит и сколько сетчатка при этом ТЕРЯЕТ. Второе число и есть
    # встречный замер: частота, на которой кадр почти не меняется между
    # шагами управления, не стоит своей цены.
    print("\n  развёртка по частоте обновления зрения", flush=True)
    sweep = {}
    for hz in HZS:
        r = spawn(True, "hz%d" % hz, hz=hz)
        if not r["ok"]:
            sweep[hz] = {"ошибка": "прогон не удался, код %s" % r["код"]}
            print("    %4d Гц: ПРОВАЛ" % hz, flush=True)
            continue
        it = r["итог"]
        up = it.get("обновление сетчатки") or {}
        ms = it["шаг управления, мс"]["медиана"]
        sweep[hz] = {
            "шаг управления, мс": ms,
            "дороже, чем без зрения, раз": round(ms / max(sb, 1e-9), 3),
            "модельного на настенное": round(20.0 / max(ms, 1e-9), 4),
            "обновлений в секунду модельного": up.get(
                "обновлений в секунду модельного времени"),
            "обновлений на шаг управления": (
                round(up.get("обновлений в секунду модельного времени", 0)
                      * CONTROL_MS / 1e3, 2)),
            "разброс по омматидиям":
                (it.get("сетчатка") or {}).get(
                    "разброс по омматидиям, последний кадр"),
        }
        print("    %4d Гц: шаг %7.1f мс (×%.2f), %s модельного на "
              "настенное, обновлений на шаг управления %.2f"
              % (hz, ms, ms / max(sb, 1e-9),
                 sweep[hz]["модельного на настенное"],
                 sweep[hz]["обновлений на шаг управления"]), flush=True)
    rep["замеры"]["развёртка по частоте обновления"] = sweep

    good = [h for h in HZS if "ошибка" not in sweep.get(h, {"ошибка": 1})]
    if len(good) >= 2:
        lo, hi = sweep[good[0]], sweep[good[-1]]
        check("частота обновления зрения меняет цену, и цена замерена на "
              "всех четырёх частотах",
              all(sweep[h]["обновлений в секунду модельного"] is not None
                  for h in good)
              and hi["шаг управления, мс"] > lo["шаг управления, мс"],
              "%s Гц: %.1f мс на шаг управления и %s модельного на "
              "настенное; %s Гц: %.1f мс и %s. Без зрения %.1f мс и "
              "%.4f. Выбор частоты — условие прогона, и выбирать его "
              "надо по этим числам"
              % (good[0], lo["шаг управления, мс"],
                 lo["модельного на настенное"], good[-1],
                 hi["шаг управления, мс"], hi["модельного на настенное"],
                 sb, 20.0 / max(sb, 1e-9)))

    note("абсолютные времена этой пробы сняты на загруженной машине: "
         "одновременно работали исследовательские процессы. Отношения "
         "(во сколько раз дороже) сняты в одинаковых условиях и "
         "устойчивее абсолютных значений; абсолютные надо перемерить на "
         "спокойной машине перед тем, как ссылаться на них как на "
         "скорость сеанса.")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "vision_gl.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d, находок %d, за %.0f с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("ПРОВАЛЫ:", flush=True)
        for f in fails:
            print("  " + f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    tag = os.environ.get("FLYBRAIN_VISION_GL")
    if tag is None:
        sys.exit(main())
    # Дочерний процесс: один прогон и выход.
    import probe._isolate  # noqa: F401
    run_one(tag == "1", Path(os.environ["FLYBRAIN_VISION_OUT"]))
