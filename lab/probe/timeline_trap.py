"""Ловушка на выдуманные состояния: три формы дырки и неполная запись.

Эта проба проверяет не «работает ли таймлайн», а **нельзя ли выдать
догадку за измерение**. Поэтому главное в ней — не вопросы к целой
записи, а вопросы к ПОРЧЕННОЙ.

Как устроена ловушка. Из копии настоящей записи удаляется каждый второй
кадр, и спрашиваются **именно удалённые** моменты модельного времени.
Критерий нарочно не «близко к правде»: удалённый кадр лежит между двумя
оставшимися на шаге 20 мс, и линейная интерполяция соседей отличается от
правды на уровне шума — то есть выдумывающая система прошла бы проверку
«близко» с запасом. Критерий такой:

* ответ обязан быть «нет данных» с причиной «между кадрами»;
* провал, если ответ — число, и тем более провал, если это число
  побитово равно интерполяции соседей или ближе к ней, чем к правде.

Встречный замер на сам критерий — класс :class:`Liar`: читатель, который
честно интерполирует. Тот же критерий обязан его **забраковать**, и
рядом печатается, насколько интерполяция близка к правде: именно это
число показывает, почему «близко» критерием быть не может.

Второй, дешёвый капкан: растр целочислен по построению (``int16``), и
при рефрактерном периоде 2,2 мс с шагом управления 20 мс нейрон не может
выдать больше ``⌊20/2,2⌋ + 1 = 10`` спайков за шаг. Дробный или
превышающий предел бин — признак выдумки, а не округления.

Третий: продолжение из НЕПОЛНОГО состояния обязано отказываться, а не
делать вид. Проверяется отказом и снятием отказа (``allow_incomplete``),
чтобы ветка отказа была живой, а не мёртвой.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\timeline_trap.py
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
CONTROL_MS = 20.0
# Шагов в записи. Нечётное нарочно: тогда при удалении каждого второго
# кадра последний кадр остаётся на месте, и удалённые моменты лежат
# ВНУТРИ окна записи — иначе отказ пришёл бы по причине «позже конца
# записи», и ловушка проверяла бы не то, что нужно.
N_STEPS = 61

fails = []
rep = {"проверки": {}, "замеры": {}, "ловушка": {}, "встречный замер": {},
       "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------- прогон


def make_run(device):
    """Настоящий короткий эпизод арены с таймлайном и событиями."""
    from flybrain import data as D, loop, replay as RP, world

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    cfg = loop.EpisodeConfig(duration_s=N_STEPS * CONTROL_MS / 1000.0,
                             control_ms=CONTROL_MS, seed=0, device=device,
                             comment=False)
    params = world.BodyParams()
    arena = world.default_arena(0)
    probe_idx = {"съём L": np.asarray(D.select(neurons, params.motor_left),
                                     np.int64),
                 "съём R": np.asarray(D.select(neurons, params.motor_right),
                                      np.int64)}
    tape = RP.Tape(kind="arena", control_ms=CONTROL_MS, keep_s=20.0,
                   probe_idx=probe_idx, mode="пересчёт",
                   capture_at_ms=10 * CONTROL_MS, meta={"семя": 0})

    # События ставятся во время прогона, по модельному времени, а не
    # дописываются потом: отметка времени события — это то, что проверка
    # и сверяет.
    marks = (3, 11, 29, 47)

    def on_step(step, rec, bins_row, commentator=None):
        if step in marks:
            tape.event(rec["t_ms"], "среда", "правка условий №%d" % step,
                       detail="событие поставлено на границе шага "
                              "управления",
                       payload={"было": None, "стало": {"шаг": step}})

    ep = loop.run_episode(arena=arena, config=cfg, body_params=params,
                          neurons=neurons, weights=weights, progress=False,
                          on_step=on_step, tape=tape)
    where = Path(probe._isolate.OUT) / "_trap_run"
    if where.exists():
        shutil.rmtree(where)
    loop.save(ep, where)
    return ep, tape.tl, where, marks


# ------------------------------------------------- отметки времени событий


def part_events(tl, where, marks):
    """Модельное время события сохранено и совпадает при чтении, побитово."""
    from flybrain import timeline as TL

    back = TL.load(where)
    a = [e.get("модельное, мс") for e in tl.events]
    b = [e.get("модельное, мс") for e in back.events]
    rep["замеры"]["событий в записи"] = len(a)
    rep["замеры"]["отметки событий, мс"] = a
    check("событий записано столько, сколько поставлено",
          len(a) == len(marks) and len(b) == len(a),
          "поставлено %d, в записи %d, после чтения %d"
          % (len(marks), len(a), len(b)))
    # Побитово: сравниваются не значения «на глаз», а сырые байты
    # двойной точности.
    same = all(np.float64(x).tobytes() == np.float64(y).tobytes()
               for x, y in zip(a, b))
    check("отметки времени событий после чтения побитово те же", same,
          "%d отметок, отпечаток %s"
          % (len(a), fp(np.asarray(a, np.float64))))
    want = [m * CONTROL_MS for m in marks]
    check("отметки стоят ровно на границах шагов управления",
          [float(x) for x in b] == want,
          "ожидалось %s, в записи %s" % (want, [float(x) for x in b]))

    # Честно про квантование. `Timeline.event` округляет отметку до трёх
    # знаков (наносекунда модельного времени). Для границ шага это
    # обратимо побитово — и это замер, а не обещание.
    exact = sum(1 for k in range(100000)
                if round(k * CONTROL_MS, 3) == k * CONTROL_MS)
    check("округление отметки до трёх знаков на границах шага обратимо "
          "побитово",
          exact == 100000, "проверено %d границ, совпало %d"
          % (100000, exact))
    odd = 1.0 / 3.0
    rep["находки"].append(
        "отметка события округляется до трёх знаков (наносекунда "
        "модельного времени): для границ шага управления это обратимо "
        "побитово, но событие, поставленное на %r мс, в записи станет "
        "%r мс. Это объявленное квантование, а не потеря — на оси "
        "модельного времени событий между шагами не бывает."
        % (odd, round(odd, 3)))


# ------------------------------------------- точность воспроизведения


def part_exact(tl, where):
    """В сохранённый момент ответ побитово равен записи. Со встречным замером."""
    from flybrain import replay as RP, timeline as TL

    back = TL.load(where)
    check("ряды после чтения побитово те же",
          fp(tl.rows) == fp(back.rows),
          "отпечаток %s, строк %d, каналов %d"
          % (fp(back.rows), back.rows.shape[0], back.rows.shape[1]))
    check("ось модельного времени после чтения побитово та же",
          fp(tl.t_ms) == fp(back.t_ms), "отпечаток %s" % fp(back.t_ms))
    check("растр зондов после чтения побитово тот же",
          fp(tl.raster) == fp(back.raster),
          "отпечаток %s, %s" % (fp(back.raster), back.raster.shape))

    # Поканально, через тот самый путь, которым отвечает окно.
    view = RP.View.open(where)
    n_ask = n_bad = 0
    for i, t in enumerate(tl.t_ms):
        if not np.isfinite(t):
            continue
        for c in tl.chans:
            want = tl.rows[i, tl.col[c.key]]
            got, hole = view.value(float(t), c.key)
            n_ask += 1
            if hole is not None:
                if np.isfinite(want):
                    n_bad += 1
                continue
            if np.float32(got).tobytes() != np.float32(want).tobytes():
                n_bad += 1
    check("каждый записанный момент отвечается побитово тем, что записано",
          n_bad == 0, "вопросов %d, расхождений %d" % (n_ask, n_bad))
    rep["замеры"]["сверено ответов"] = n_ask

    # Встречный замер на сам сравниватель: портим один байт и смотрим,
    # ловит ли он. Без этого «расхождений 0» печаталось бы и на пустой
    # проверке.
    spoiled = Path(probe._isolate.OUT) / "_trap_spoiled"
    if spoiled.exists():
        shutil.rmtree(spoiled)
    spoiled.mkdir(parents=True)
    z = np.load(where / "timeline.npz", allow_pickle=False)
    rows = z["rows"].copy()
    i0, j0 = 5, 0
    old = float(rows[i0, j0])
    rows[i0, j0] = np.float32(np.nextafter(old, old + 1.0, dtype=np.float32)
                              if np.isfinite(old) else 1.0)
    np.savez_compressed(spoiled / "timeline.npz", rows=rows,
                        t_ms=z["t_ms"], raster=z["raster"],
                        probe_hz=z["probe_hz"], probe_idx=z["probe_idx"])
    shutil.copy(where / "timeline.json", spoiled / "timeline.json")
    bad = TL.load(spoiled)
    caught = fp(bad.rows) != fp(back.rows)
    key = tl.chans[j0].key
    got, _ = bad.value(float(bad.t_ms[i0]), key)
    rep["встречный замер"]["испорчен один младший бит одного значения"] = {
        "было": old, "стало": got, "поймано": bool(caught)}
    check("встречный замер: сравниватель ловит порчу одного младшего бита",
          caught and np.float32(got).tobytes() != np.float32(old).tobytes(),
          "канал %s: было %r, стало %r" % (key, old, got))


# ------------------------------------- ловушка на выдуманные состояния


class Liar:
    """Выдумывающий читатель: на дырке честно интерполирует соседей.

    Нужен одному — встречному замеру на сам критерий. Если критерий
    этого читателя не бракует, значит он не критерий, а украшение.
    """

    def __init__(self, tl):
        self.tl = tl

    def value(self, t_ms, channel):
        t = float(t_ms)
        times = self.tl.t_ms
        good = np.isfinite(times)
        tt = times[good]
        col = self.tl.col[channel]
        vals = self.tl.rows[good, col]
        hit = np.nonzero(np.abs(tt - t) <= 1e-6)[0]
        if len(hit):
            return float(vals[hit[0]]), None
        j = int(np.searchsorted(tt, t))
        if j <= 0 or j >= len(tt):
            return None, "вне окна"
        w = (t - tt[j - 1]) / (tt[j] - tt[j - 1])
        return float(vals[j - 1] + w * (vals[j] - vals[j - 1])), None


def thin(src, dst):
    """Копия записи без каждого второго кадра. Удалённые моменты возвращаются."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    z = np.load(src / "timeline.npz", allow_pickle=False)
    t = z["t_ms"]
    n = len(t)
    keep = np.arange(0, n, 2)
    gone = np.arange(1, n, 2)
    np.savez_compressed(
        dst / "timeline.npz", rows=z["rows"][keep], t_ms=t[keep],
        raster=z["raster"][keep], probe_hz=z["probe_hz"][keep],
        probe_idx=z["probe_idx"])
    sch = json.loads((src / "timeline.json").read_text(encoding="utf-8"))
    sch["шагов записано"] = int(len(keep))
    sch["запись с, мс"] = float(t[keep][0])
    sch["запись до, мс"] = float(t[keep][-1])
    sch["журнал"] = dict(sch.get("журнал") or {},
                         порча="из копии записи удалён каждый второй кадр: "
                               "проверка ловушки на выдуманные состояния")
    (dst / "timeline.json").write_text(
        json.dumps(sch, ensure_ascii=False, indent=1), encoding="utf-8")
    return t[gone], t[keep]


def judge(answer, hole, truth, interp):
    """Критерий. Возвращает ``(прошло, как именно)``.

    Один и тот же судья применяется и к настоящему читателю, и к
    выдумывающему: иначе «прошло» ничего не значит.
    """
    if answer is None and hole is not None:
        return True, "нет данных"
    if answer is None:
        return False, "ни числа, ни причины"
    a = np.float32(answer)
    if a.tobytes() == np.float32(interp).tobytes():
        return False, "число побитово равно интерполяции соседей"
    d_truth = abs(float(a) - float(truth))
    d_interp = abs(float(a) - float(interp))
    if d_interp < d_truth:
        return False, ("число ближе к интерполяции (%.6g) чем к правде "
                       "(%.6g)" % (d_interp, d_truth))
    return False, "ответом на удалённый кадр дано число %.6g" % float(a)


def part_trap(tl, where):
    """Удаляем каждый второй кадр и спрашиваем именно удалённые моменты."""
    from flybrain import replay as RP, timeline as TL

    thinned = Path(probe._isolate.OUT) / "_trap_thinned"
    gone, kept = thin(where, thinned)
    rep["ловушка"]["кадров в записи"] = int(len(tl.t_ms))
    rep["ловушка"]["кадров удалено"] = int(len(gone))
    rep["ловушка"]["кадров осталось"] = int(len(kept))
    print("    кадров %d, удалено %d, спрашиваем именно удалённые"
          % (len(tl.t_ms), len(gone)))

    full = TL.load(where)
    holed = TL.load(thinned)
    view = RP.View(holed)
    liar = Liar(holed)

    # Каналы берём не все: нужны те, что меняются гладко — именно на них
    # интерполяция и выглядит правдой.
    keys = [c.key for c in holed.chans
            if c.key in ("x", "y", "theta", "dn_l", "dn_r", "asym", "v",
                         "omega", "spikes")]
    rep["ловушка"]["каналы"] = keys

    asked = 0
    ok_hole = 0
    why = {}
    liar_caught = 0
    liar_passed = 0
    per_chan = {k: {"инт": [], "шаг": []} for k in keys}
    for t in gone:
        t = float(t)
        for k in keys:
            i = int(np.nonzero(np.abs(full.t_ms - t) <= 1e-6)[0][0])
            truth = float(full.rows[i, full.col[k]])
            iv, _ = liar.value(t, k)
            if iv is None:
                continue
            asked += 1
            got, hole = view.value(t, k)
            good, how = judge(got, hole, truth, iv)
            tag = how if not good else ("нет данных: %s" % hole.why)
            why[tag] = why.get(tag, 0) + 1
            if good:
                ok_hole += 1
            # Встречный замер: тот же судья на выдумывающем читателе.
            lg, lhow = judge(iv, None, truth, iv)
            liar_passed += int(lg)
            liar_caught += int(not lg)
            per_chan[k]["инт"].append(abs(iv - truth))
            # Шаг между соседями — масштаб, с которым надо сравнивать
            # ошибку интерполяции.
            j = int(np.searchsorted(holed.t_ms, t))
            if 0 < j < len(holed.t_ms):
                per_chan[k]["шаг"].append(
                    abs(float(holed.rows[j, holed.col[k]])
                        - float(holed.rows[j - 1, holed.col[k]])))

    rep["ловушка"]["вопросов"] = asked
    rep["ловушка"]["ответов «нет данных»"] = ok_hole
    rep["ловушка"]["разбор ответов"] = why
    check("на каждый удалённый кадр ответ — «нет данных», ни одного числа",
          asked > 0 and ok_hole == asked,
          "вопросов %d, «нет данных» %d, чисел %d"
          % (asked, ok_hole, asked - ok_hole))
    only_gap = all("между кадрами" in k for k in why)
    check("причина отказа названа верно: «между кадрами»",
          only_gap, "разбор: %s" % why)

    # Нейрон: тот же вопрос про растр.
    nrn = int(holed.probe_idx[0]) if len(holed.probe_idx) else -1
    nb = 0
    for t in gone[:20]:
        v, hole = view.neuron(float(t), nrn)
        nb += int(hole is None)
    check("растр одного нейрона на удалённом кадре тоже «нет данных»",
          nb == 0, "спрошено %d моментов, чисел дано %d"
          % (len(gone[:20]), nb))

    # --- встречный замер на критерий ---
    ratios = {}
    for k, d in per_chan.items():
        mi = float(np.mean(d["инт"])) if d["инт"] else 0.0
        ms = float(np.mean(d["шаг"])) if d["шаг"] else 0.0
        ratios[k] = {"ошибка интерполяции": mi,
                     "шаг между соседними кадрами": ms,
                     "доля от шага, %": (100.0 * mi / ms) if ms else None}
    all_i = [v for d in per_chan.values() for v in d["инт"]]
    all_s = [v for d in per_chan.values() for v in d["шаг"]]
    m_interp = float(np.mean(all_i)) if all_i else 0.0
    m_step = float(np.mean(all_s)) if all_s else 0.0
    rep["встречный замер"]["выдумывающий читатель (Liar)"] = {
        "вопросов": asked, "забраковано критерием": liar_caught,
        "пропущено критерием": liar_passed,
        "средняя ошибка интерполяции": m_interp,
        "средний шаг между соседними кадрами": m_step,
        "доля ошибки от шага": (m_interp / m_step) if m_step else None,
        "по каналам": ratios,
    }
    check("встречный замер: тот же критерий бракует выдумывающего читателя "
          "на каждом вопросе",
          asked > 0 and liar_caught == asked and liar_passed == 0,
          "забраковано %d из %d" % (liar_caught, asked))
    smooth = {k: v["доля от шага, %"] for k, v in ratios.items()
              if v["доля от шага, %"] is not None}
    best = min(smooth, key=smooth.get) if smooth else None
    for k in sorted(smooth, key=smooth.get):
        print("      %-10s интерполяция ошибается на %5.1f %% шага между "
              "кадрами" % (k, smooth[k]))
    check("критерий «близко к правде» на этих данных прошёл бы: "
          "интерполяция отличается от правды меньше, чем соседние кадры "
          "друг от друга",
          m_step > 0 and m_interp < m_step,
          "в среднем по каналам %.1f %% шага; на гладком канале «%s» — "
          "%.2f %% шага (ошибка %.6g при шаге %.6g)"
          % (100.0 * m_interp / m_step if m_step else 0.0, best,
             smooth.get(best, 0.0), ratios[best]["ошибка интерполяции"],
             ratios[best]["шаг между соседними кадрами"]))
    rep["находки"].append(
        "на гладком канале «%s» интерполяция соседей ошибается на %.2f %% "
        "шага между соседними кадрами: именно поэтому критерием не может "
        "быть «близко к правде» — выдумывающая система прошла бы его с "
        "запасом, и отличить её от честной можно только требованием "
        "«нет данных»." % (best, smooth.get(best, 0.0)))

    # Моменты, которых нет по другим причинам: кольцо и конец записи.
    lo, hi = holed.window
    forms = {}
    for t, want in (((lo + hi) / 2.0 + 1.0, "между кадрами"),
                    (lo - 10 * CONTROL_MS, "раньше начала кольца"),
                    (hi + 10 * CONTROL_MS, "позже конца записи")):
        v, hole = view.value(float(t), keys[0])
        forms[want] = None if hole is None else hole.why
    rep["ловушка"]["формы дырки"] = forms
    check("три формы дырки называются каждая своей причиной",
          all(k == v for k, v in forms.items()), str(forms))

    absent = [c for c in TL.channels_of(holed.kind) if c.absent]
    v, hole = view.value(float(kept[0]), absent[0].key) if absent else (0, 1)
    check("объявленный, но неподключённый канал отвечает «канал не "
          "подключён», а не нулём",
          bool(absent) and hole is not None
          and hole.why == "канал не подключён",
          "каналов объявлено без подключения: %d" % len(absent))
    return holed


# ------------------------------------------- растр: целые и предел


def part_raster(tl, holed):
    """Растр целочислен, и бин не больше ⌊20/2,2⌋+1 = 10 спайков."""
    ceiling = tl.raster_ceiling()
    rep["замеры"]["предел растра, спайков на нейрон за шаг"] = ceiling
    check("предел растра посчитан из рефрактерного периода, а не назначен",
          ceiling == int(CONTROL_MS // 2.2) + 1 == 10,
          "⌊%g/2,2⌋+1 = %d" % (CONTROL_MS, ceiling))
    check("растр хранится целыми (int16)", tl.raster.dtype == np.int16,
          "dtype %s" % tl.raster.dtype)
    mx = int(tl.raster.max()) if tl.raster.size else 0
    rep["замеры"]["наибольший бин в растре"] = mx
    check("ни один бин не превышает предела", mx <= ceiling,
          "наибольший %d при пределе %d, бинов %d"
          % (mx, ceiling, tl.raster.size))
    frac = int(np.count_nonzero(tl.raster != tl.raster.astype(np.int64)))
    check("дробных бинов нет ни одного", frac == 0,
          "дробных %d из %d" % (frac, tl.raster.size))

    # Встречный замер на обе половины капкана. Дробное значение в int16
    # записать НЕЛЬЗЯ — оно усекается, и это и есть защита; а превышение
    # предела записать можно, и проверка обязана его поймать.
    a = np.zeros(3, np.int16)
    a[0] = np.int16(3.7)
    rep["встречный замер"]["дробное значение в растре"] = {
        "записали": 3.7, "легло": int(a[0]),
        "почему": "растр int16: дробное усекается при записи, выдумать "
                  "дробный бин нечем"}
    check("встречный замер: дробное значение в растр не ложится",
          int(a[0]) == 3, "записали 3,7 — легло %d" % int(a[0]))
    spoiled = tl.raster.copy()
    if spoiled.size:
        spoiled[0, 0] = np.int16(ceiling + 1)
    caught = int(spoiled.max()) > ceiling
    rep["встречный замер"]["бин выше предела"] = {
        "подложено": ceiling + 1, "поймано": bool(caught)}
    check("встречный замер: проверка предела ловит подложенный бин %d"
          % (ceiling + 1), caught,
          "наибольший стал %d при пределе %d" % (int(spoiled.max()), ceiling))


# --------------------------------------------- предел памяти записи


def part_budget():
    """Предел объявлен и соблюдается: кольцо не растёт со временем сеанса.

    Проверяется не арифметикой из объявления, а настоящей записью: в
    таймлайн с окном 10 шагов пишется 400 шагов, и после этого
    измеряется, сколько памяти он занимает и что в нём лежит.
    Неограниченного хранилища не должно быть ни одного — ни рядов, ни
    растра, ни событий.
    """
    from flybrain import timeline as TL

    keep_s = 10 * CONTROL_MS / 1000.0
    idx = {"зонд": np.arange(1000, dtype=np.int64)}
    tl = TL.Timeline(kind="arena", control_ms=CONTROL_MS, keep_s=keep_s,
                     probe_idx=idx, max_events=50)
    n_put = 400
    counts = np.zeros(138639, np.int16)
    for i in range(n_put):
        counts[1000:1100] = (i % 4)
        tl.put(i * CONTROL_MS, {"x": float(i), "dn_l": float(i) / 7.0},
               counts=counts, lif_calls=200, wall_ms=1.0)
        tl.event(i * CONTROL_MS, "среда", "событие %d" % i)
    bud = tl.budget()
    tl.close()

    held = (tl.rows.nbytes + tl.t_ms.nbytes + tl.raster.nbytes
            + tl.probe_hz.nbytes)
    want = bud["всего, Б"] - bud["события, Б"]
    rep["замеры"]["предел памяти"] = {
        "окно, шагов": tl.keep_steps, "записано шагов": n_put,
        "строк осталось": int(tl.rows.shape[0]),
        "занято массивами, Б": int(held),
        "объявлено (без событий), Б": int(want),
        "плотный растр был бы, МБ": bud["плотный растр был бы, МБ"],
        "событий в кольце": len(tl.events), "предел событий": tl.max_events,
    }
    check("кольцо рядов не растёт со временем сеанса",
          tl.rows.shape[0] == tl.keep_steps == 10,
          "записано %d шагов, строк осталось %d при окне %d"
          % (n_put, tl.rows.shape[0], tl.keep_steps))
    check("занятая память равна объявленному пределу", held == want,
          "занято %d Б, объявлено %d Б" % (held, want))
    check("кольцо провёрнуто по времени: в записи последние шаги, по "
          "возрастанию",
          list(tl.t_ms) == sorted(tl.t_ms)
          and float(tl.t_ms[-1]) == (n_put - 1) * CONTROL_MS
          and float(tl.t_ms[0]) == (n_put - tl.keep_steps) * CONTROL_MS,
          "с %g по %g мс" % (tl.t_ms[0], tl.t_ms[-1]))
    hole = tl.value(0.0, "x")[1]
    check("вытесненный кольцом момент отвечает «раньше начала кольца», а "
          "не нулём",
          hole is not None and hole.why == "раньше начала кольца",
          "спрошено 0 мс при окне с %g мс" % tl.t_ms[0])
    check("кольцо событий не растёт: предел соблюдён",
          len(tl.events) == tl.max_events,
          "поставлено %d событий, в кольце %d при пределе %d"
          % (n_put, len(tl.events), tl.max_events))

    # Встречный замер на объявленный предел: он мал не на словах. Рядом
    # считается, сколько стоила бы плотная запись всего мозга на том же
    # окне — 277 КБ на шаг.
    dense = tl.keep_steps * 138639 * 2
    rep["замеры"]["предел памяти"]["во сколько раз дешевле плотного"] = round(
        dense / max(held, 1), 1)
    check("запись дешевле плотного растра всего мозга в разы, и это "
          "посчитано, а не обещано",
          dense > held * 10,
          "занято %.1f КБ против %.1f КБ у плотного — в %.0f раз"
          % (held / 1024.0, dense / 1024.0, dense / max(held, 1)))

    # Популяция, не умещающаяся в растр, отклоняется с объяснением, а не
    # обрезается молча: по обрезку нельзя сказать, чья это активность.
    big = {"большая": np.arange(TL.MAX_PROBE_NEURONS + 1, dtype=np.int64)}
    how = "прошло молча"
    try:
        TL.Timeline(kind="arena", control_ms=CONTROL_MS, keep_s=keep_s,
                    probe_idx=big)
    except ValueError as exc:
        how = str(exc)[:160]
    rep["встречный замер"]["зонд больше предела"] = how
    check("зонд, не умещающийся в растр, отклоняется с объяснением",
          how != "прошло молча" and str(TL.MAX_PROBE_NEURONS) in how,
          "предел %d нейронов; ответ: %s" % (TL.MAX_PROBE_NEURONS, how[:80]))

    # Сколько популяций съёма в предел не влезают — числом, точными
    # селекторами. Прежнее примечание в `timeline.py` называло «самую
    # крупную популяцию — 6400 нейронов T4»: ни то, ни другое не верно,
    # и оба раза из-за ловушки селектора (`RESEARCH.md` §9).
    from flybrain import data as D, world

    neurons = D.load_neurons()
    cands = world.probe_candidates(neurons)
    sizes = {}
    for name, (ql, qr) in cands.items():
        sizes[name] = len(D.select(neurons, ql)) + len(D.select(neurons, qr))
    over = {k: v for k, v in sizes.items() if v > TL.MAX_PROBE_NEURONS}
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:6]
    rep["замеры"]["популяции съёма"] = {
        "кандидатов": len(sizes), "крупнее предела": len(over),
        "крупнейшие": [list(kv) for kv in top],
        "нисходящие (super:descending)": sizes.get("DN")
        or (len(D.select(neurons, "super:descending+side:left"))
            + len(D.select(neurons, "super:descending+side:right"))),
        "type:T4 свободно": len(D.select(neurons, "type:T4")),
        "type=T4 точно": len(D.select(neurons, "type=T4")),
    }
    check("в предел растра влезает НЕ любая популяция съёма, и это "
          "посчитано точными селекторами",
          len(over) >= 1 and top[0][1] > TL.MAX_PROBE_NEURONS,
          "кандидатов %d, крупнее предела %d, крупнейший %s — %d нейронов"
          % (len(sizes), len(over), top[0][0], top[0][1]))
    check("прежнее число «6400 нейронов T4» — ловушка селектора: точный "
          "селектор даёт ноль",
          len(D.select(neurons, "type=T4")) == 0
          and len(D.select(neurons, "type:T4")) == 6400
          and "T4" not in sizes,
          "type:T4 свободно %d, type=T4 точно %d, среди кандидатов съёма "
          "T4 нет" % (len(D.select(neurons, "type:T4")),
                      len(D.select(neurons, "type=T4"))))

    # Таймлайна без объявленного окна записи не бывает вовсе: окно и
    # есть предел, и без него запись растёт по строке на шаг без конца.
    how2 = "прошло молча"
    try:
        TL.Timeline(kind="arena", control_ms=CONTROL_MS, keep_s=0)
    except ValueError as exc:
        how2 = str(exc)[:120]
    rep["встречный замер"]["таймлайн без окна записи"] = how2
    check("таймлайна без объявленного окна записи не бывает",
          how2 != "прошло молча", how2[:80])


# ------------------------------------- неполное состояние: отказ


def part_refuse(where, device):
    """Продолжение из неполного состояния отказывается, а не делает вид."""
    from flybrain import data as D, replay as RP

    if not (where / "state.json").exists():
        check("снимок состояния лежит рядом с прогоном", False,
              "state.json в папке нет")
        return
    st = RP.State.load(where)
    neurons = D.load_neurons()
    check("целый снимок полон", st.complete(),
          "элементов %d, не хватает: %s"
          % (len(st.have), ", ".join(st.missing()) or "ничего"))

    # Неполная ЗАПИСЬ — это не `drop` (у него есть разрешение
    # продолжать, потому что элемент выброшен нарочно, замером). Это
    # снимок, у которого элемента не было: ровно то, что получилось бы
    # из оборванной или урезанной записи.
    cases = {}
    for part in ("engine", "motor", "calib", "pose", "env"):
        cut = RP.State(st.kind, st.t_ms,
                       {k: v for k, v in st.arrays.items() if k != part},
                       dict(st.meta),
                       [k for k in st.have if k != part])
        try:
            RP.restore_arena(cut, neurons, weights=None, device="cpu")
            cases[part] = "ПРОДОЛЖИЛ"
        except ValueError as exc:
            cases[part] = "отказ: " + str(exc)[:120]
        except Exception as exc:                              # noqa: BLE001
            cases[part] = "упал не отказом: %s: %s" % (type(exc).__name__,
                                                       str(exc)[:80])
    rep["замеры"]["неполное состояние"] = cases
    refused = [k for k, v in cases.items() if v.startswith("отказ")]
    check("продолжение из неполного состояния отказывается по каждому "
          "обязательному элементу",
          len(refused) == len(cases),
          "отказов %d из %d: %s" % (len(refused), len(cases),
                                    ", ".join(sorted(cases))))
    for k, v in cases.items():
        if not v.startswith("отказ"):
            rep["находки"].append(
                "выбрасывание «%s» из записи продолжение не остановило: %s"
                % (k, v))

    # Встречный замер: ветка отказа живая, а не мёртвая. Со снятым
    # запретом тот же вызов доходит до конца и отдаёт то, чем
    # продолжают. Устройство берётся то же, на котором снято состояние:
    # состояние генератора движка снято с него, и на другом оно не
    # ставится.
    cut = RP.State(st.kind, st.t_ms,
                   {k: v for k, v in st.arrays.items() if k != "motor"},
                   dict(st.meta), [k for k in st.have if k != "motor"])
    how = ""
    try:
        out = RP.restore_arena(cut, neurons, weights=None, device=device,
                               allow_incomplete=True)
        how = ("дошёл до конца, отдано %d частей, в отчёте «не хватало»: %s"
               % (len(out), out[-1].get("не хватало в записи")))
    except ValueError as exc:
        how = ("всё равно отказ: " + str(exc)[:100]) if \
            "продолжение из состояния невозможно" in str(exc) else \
            "другая ошибка значения: " + str(exc)[:100]
    except Exception as exc:                                  # noqa: BLE001
        how = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    rep["встречный замер"]["снятие запрета (allow_incomplete)"] = how
    check("встречный замер: с allow_incomplete тот же вызов проходит мимо "
          "отказа — ветка отказа живая",
          how.startswith("дошёл до конца"), how)


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep["замеры"]["устройство"] = device
    rep["замеры"]["шагов в записи"] = N_STEPS
    print("— настоящий короткий эпизод с таймлайном —")
    t0 = time.perf_counter()
    ep, tl, where, marks = make_run(device)
    rep["замеры"]["прогон, с"] = round(time.perf_counter() - t0, 2)
    print("  записано шагов %d, файлы в %s" % (len(ep.records), where))

    print("\n— отметки времени событий —")
    part_events(tl, where, marks)
    print("\n— точность воспроизведения в сохранённый момент —")
    part_exact(tl, where)
    print("\n— ловушка: удалён каждый второй кадр —")
    holed = part_trap(tl, where)
    print("\n— растр: целые и предел —")
    part_raster(tl, holed)
    print("\n— предел памяти записи —")
    part_budget()
    print("\n— неполное состояние: отказ —")
    part_refuse(where, device)

    (OUT / "timeline_trap.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    for n in rep["находки"]:
        print("  замечание: " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
