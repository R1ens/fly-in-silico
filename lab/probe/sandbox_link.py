"""Связка работник — MuJoCo — окно: правка доходит до настоящей модели.

Песочница стоит на утверждении, которое легко подделать картинкой:
«источник переместился». Поэтому здесь проверяется не вид, а три вещи,
которые обязаны совпасть, и каждая живёт в своём месте:

* ``arena.odor_source`` плюс кэш ``_odor_source_repeated`` — то, по чему
  считается яркость на антеннах;
* ``physics.data.mocap_pos`` — то, где маркер стоит в сцене MuJoCo;
* массив ``sources`` — то, по чему эпизод считает расстояние, вкус и
  совпадение курса с направлением на источник.

Разойдись любая пара — и окно показывало бы одно, а муха чуяла другое.
Кэш тут главный подвох: ``OdorArena.get_olfaction`` читает не
``odor_source``, а слепок, собранный в конструкторе.

Первая половина пробы работает в одном процессе: только там видно сами
объекты модели. Вторая гоняет настоящий сеанс через работника и смотрит
на то, что доходит до окна, — кадры MuJoCo, подтверждения с модельным
временем, скорость относительно настенной.

Шесть исправленных дефектов проверяются только в первой половине: через
работника до этих мест не достать. Это модульный массив источников,
который правка слота портила на весь процесс; смешанная правка «слоты и
стимул», переписывавшая слоты до отказа; слот, включённый с нулевой
силой и читавшийся выключенным; прямая стимуляция, складывавшаяся с
сенсорным каналом выше объявленного предела; молчаливый пропуск
ненайденного маркера; и яркость насыщения со сглаживанием выхода,
которые эпизод читал один раз до цикла. У каждой проверки рядом стоит
замер встречного предположения — сколько показало бы прежнее поведение,
чтобы видно было, что проверка не пустая.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sandbox_link.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "подтверждения": []}

SLOTS = np.array([[25.0, 12.0, 1.5], [38.0, -16.0, 1.5],
                  [0.0, 30.0, 1.5], [-20.0, -5.0, 1.5]])


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# --------------------------------------------- половина первая: модель

def part_model():                                             # noqa: C901
    from flybrain import body as B, data as D, live

    cfg = B.BodyConfig(duration_s=0.1, control_ms=20.0, seed=0, device="cpu")
    sources = SLOTS.copy()
    sim, _ = B._build_body(cfg, sources)
    sim.reset(seed=0)
    neurons = D.load_neurons()
    st = live.BodyLive()
    st.attach(sim, sources, cfg, neurons, "cpu")

    check("слоты источника нашлись в модели", st.slots() == 4,
          "слотов %d" % st.slots())
    check("маркеры найдены по имени, а не по счёту",
          len(st._mocap) == 4 and len(st._geom) == 4,
          "mocap %s" % sorted(st._mocap))

    def sensed(pos):
        """Яркость в точке — так, как её считает сама арена."""
        p = np.array([[pos[0], pos[1], 1.5]], dtype=float)
        return float(sim.arena.get_olfaction(p)[0, 0])

    # --- перемещение слота ---
    probe_at = (10.0, 10.0)
    before = sensed(probe_at)
    st.apply({"sources": [
        {"x": 12.0, "y": 11.0, "strength": 1.0, "on": True},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    after = sensed(probe_at)
    ar = sim.arena
    row = st._mocap[0]
    mp = sim.physics.data.mocap_pos[row]
    agree = (abs(float(ar.odor_source[0, 0]) - 12.0) < 1e-9
             and abs(float(mp[0]) - 12.0) < 1e-9
             and abs(float(sources[0, 0]) - 12.0) < 1e-9
             and abs(float(ar.odor_source[0, 1]) - 11.0) < 1e-9
             and abs(float(mp[1]) - 11.0) < 1e-9
             and abs(float(sources[0, 1]) - 11.0) < 1e-9)
    check("положение совпало в модели, в сцене и в массиве эпизода", agree,
          "odor_source %s, mocap_pos %s, sources %s"
          % (ar.odor_source[0, :2], mp[:2], sources[0, :2]))

    # Кэш — главный подвох: без пересборки яркость считалась бы по
    # прежнему положению, и проверка «совпало» ничего бы не значила.
    d = float(np.linalg.norm(np.array([12.0, 11.0, 1.5])
                             - np.array([probe_at[0], probe_at[1], 1.5])))
    want = float(ar.diffuse_func(d))
    check("яркость считается по новому положению, а не по кэшу",
          abs(after - want) < 1e-9,
          "ариф. %.8g, арена %.8g (было %.8g)" % (want, after, before))
    check("перемещение изменило яркость", abs(after - before) > 1e-9,
          "%.8g -> %.8g" % (before, after))

    # --- выключение слота ---
    near = (12.0, 11.5)
    on_val = sensed(near)
    st.apply({"sources": [
        {"x": 12.0, "y": 11.0, "strength": 1.0, "on": False},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    off_val = sensed(near)
    check("выключенный слот не чуется совсем", off_val == 0.0,
          "рядом с ним было %.6g, стало %.6g" % (on_val, off_val))
    check("маркер выключенного слота приглушён",
          abs(float(sim.physics.model.geom_rgba[st._geom[0], 3])
              - live.BodyLive.OFF_ALPHA) < 1e-6,
          "альфа %.3f" % float(sim.physics.model.geom_rgba[st._geom[0], 3]))
    check("маска включённых слотов пуста", not any(st.active_sources()),
          str(st.active_sources()))
    check("заказанная сила выключенного слота не потерялась",
          st._read_sources()[0]["strength"] == 1.0,
          "показана сила %s" % st._read_sources()[0]["strength"])

    # --- включение возвращает ровно то, что было заказано ---
    st.apply({"sources": [
        {"x": 12.0, "y": 11.0, "strength": 2.5, "on": True},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    back = sensed(near)
    check("включение вернуло яркость с заказанной силой",
          abs(back - on_val * 2.5) < 1e-9,
          "%.6g против %.6g × 2,5" % (back, on_val))

    # --- число слотов на ходу не меняется ---
    try:
        st.apply({"sources": [{"x": 0.0, "y": 0.0}]})
        check("нельзя добавить слот на ходу", False, "правка прошла")
    except ValueError as e:
        check("нельзя добавить слот на ходу", "слотов в модели 4" in str(e),
              str(e)[:80])

    # --- прямая стимуляция по DSL ---
    import torch

    got = st.apply({"stim": [{"query": "region:AL+side:left",
                              "rate_hz": 120.0}]})
    rates = torch.zeros(1, len(neurons))
    st.add_drive(rates)
    nz = int((rates > 0).sum())
    check("стимул по DSL нашёл нейронов и попал в частоты",
          got["stim"][0]["нейронов"] == nz and nz > 100,
          "нейронов %d, ненулевых в rates %d" % (got["stim"][0]["нейронов"],
                                                 nz))
    check("частота стимула та, что заказана",
          abs(float(rates.max()) - 120.0) < 1e-6,
          "максимум %.1f Гц" % float(rates.max()))
    for bad, why in ((
            [{"query": "НЕТ-ТАКОЙ-ПОПУЛЯЦИИ", "rate_hz": 50.0}],
            "не нашёл"),
            ([{"query": "region:AL", "rate_hz": 1e6}], "допустимо")):
        was = st.snapshot()["stim"]
        try:
            st.apply({"stim": bad})
            check("отказ стимула: %s" % why, False, "правка прошла")
        except (ValueError, KeyError) as e:
            check("отказ стимула: %s" % why,
                  why in str(e) and st.snapshot()["stim"] == was,
                  str(e)[:70])
    st.apply({"stim": []})
    check("пустой список стимулов снимает подачу", st._drive is None,
          "стимулов %d" % len(st._stims))

    # --- слот, включённый с нулевой силой, остаётся включённым ---
    #
    # Включённость выводилась из `strength > 0`. Слот с силой ноль
    # читался выключенным: тумблер в окне стоял «включён», панель
    # показывала «выключен», и ни одного признака расхождения не было
    # нигде. Поэтому признак хранится отдельно от силы, и проверяются
    # оба следствия: включённый слот с нулевой силой не пахнет, но
    # остаётся включённым, а выключенный не теряет заказанную силу.
    st.apply({"sources": [
        {"x": 12.0, "y": 11.0, "strength": 0.0, "on": True},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    zero = st._read_sources()[0]
    check("слот с нулевой силой читается включённым",
          zero["on"] is True and st.active_sources()[0] is True,
          "панель on %s, маска %s, сила %s"
          % (zero["on"], st.active_sources()[0], zero["strength"]))
    check("включённый слот с нулевой силой не пахнет", sensed(near) == 0.0,
          "яркость рядом %.6g при силе %s (включён, но не пахнет)"
          % (sensed(near), zero["strength"]))
    st.apply({"sources": [
        {"x": 12.0, "y": 11.0, "strength": 2.0, "on": False},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    off = st._read_sources()[0]
    check("выключенный слот при силе 2 читается выключенным",
          off["on"] is False and st.active_sources()[0] is False,
          "панель on %s, маска %s" % (off["on"], st.active_sources()[0]))
    check("сила 2 выключенного слота не потерялась по дороге",
          off["strength"] == 2.0
          and float(ar.peak_odor_intensity[0, 0]) == 0.0,
          "показана %s, в модели %s"
          % (off["strength"], float(ar.peak_odor_intensity[0, 0])))

    # --- либо всё, либо ничего: слоты и стимул в одной правке ---
    #
    # Разбор запроса DSL — единственное в правке, что может отказать
    # после того, как слоты уже переписаны в модель. Раньше он и стоял
    # на этапе записи: подтверждение говорило «отклонено», а источник
    # уже уехал — и в яркость, и в сцену, и в массив эпизода. Снимок
    # берётся со всех шести мест сразу, а порядок ключей перебирается:
    # запись шла по порядку словаря.
    #
    # Слепок чувствителен, и это замерено отдельно: годная правка слотов
    # меняет все шесть мест, и прежний порядок (`_write_sources`, а потом
    # отказ разбора) показал бы ровно их же — шесть из шести.
    def model_state():
        """Все места, где живёт состояние слотов, одним слепком."""
        return {
            "snapshot()": st.snapshot(),
            "arena.odor_source": ar.odor_source.copy(),
            "arena.peak_odor_intensity": ar.peak_odor_intensity.copy(),
            "кэш _odor_source_repeated": ar._odor_source_repeated.copy(),
            "массив sources": sources.copy(),
            "physics.data.mocap_pos": sim.physics.data.mocap_pos.copy(),
        }

    def moved_places(was):
        now = model_state()
        out = []
        for k, v in was.items():
            ok = (bool(np.array_equal(v, now[k]))
                  if isinstance(v, np.ndarray) else v == now[k])
            if not ok:
                out.append(k)
        return out

    far = [{"x": 100.0 - i, "y": -100.0 + i, "strength": 7.0, "on": True}
           for i in range(st.slots())]
    bad_stim = [{"query": "НЕТТАКОЙ", "rate_hz": 100.0}]
    for tag, payload in (("сначала sources",
                          {"sources": far, "stim": bad_stim}),
                         ("сначала stim",
                          {"stim": bad_stim, "sources": far})):
        was = model_state()
        why = ""
        try:
            st.apply(payload)
        except (ValueError, KeyError) as e:
            why = str(e)
        gone = moved_places(was)
        check("смешанная правка отклонена целиком: %s" % tag,
              bool(why) and not gone,
              ("ни одно из %d мест не тронуто; отказ: %s"
               % (len(was), why[:55])) if why and not gone
              else ("изменилось: %s" % ", ".join(gone) if gone
                    else "правка прошла без отказа"))

    # --- предел прямой стимуляции держится поверх сенсорного канала ---
    #
    # `add_drive` складывал тензоры, и стимул на 1000 Гц, наведённый на
    # те же нейроны, что обонятельный канал, давал 1150 Гц — выше
    # объявленного предела, и нигде это не было видно. Селектор берётся
    # прямо из `world.SENSORS`, чтобы пересечение было полным, а порядок
    # повторяет `drive_brain`: ноль, сенсорные каналы, прямые стимулы.
    # Замер сложения на том же тензоре даёт 1150 Гц — проверка не пустая.
    from flybrain import world as W

    i_left = D.select(neurons, W.SENSORS["odour_left"])
    i_right = D.select(neurons, W.SENSORS["odour_right"])
    got_s = st.apply({"stim": [{"query": W.SENSORS["odour_left"],
                                "rate_hz": live.MAX_STIM_HZ}]})
    t_left, t_right = torch.as_tensor(i_left), torch.as_tensor(i_right)
    rates = torch.zeros(1, len(neurons))
    rates.zero_()               # порядок `drive_brain`, шаг за шагом
    rates[0, t_left] = cfg.orn_max_hz
    rates[0, t_right] = cfg.orn_max_hz
    sens_hz = float(rates[0, t_left].max())
    st.add_drive(rates)
    peak = float(rates.max())
    check("прямой стимул не вышел за предел поверх сенсорного канала",
          peak <= live.MAX_STIM_HZ + 1e-6 and sens_hz > 0.0,
          "максимум %.1f Гц при пределе %.1f; сложение дало бы %.1f "
          "(стимул на %d нейронов, сенсорных на них %.1f Гц)"
          % (peak, live.MAX_STIM_HZ, live.MAX_STIM_HZ + sens_hz,
             got_s["stim"][0]["нейронов"], sens_hz))
    only_right = np.setdiff1d(i_right, i_left)
    t_only = torch.as_tensor(only_right)
    check("сенсорная частота вне стимула осталась прежней",
          len(only_right) > 0
          and float(rates[0, t_only].min()) == cfg.orn_max_hz
          and float(rates[0, t_only].max()) == cfg.orn_max_hz,
          "нейронов мимо стимула %d, у всех %.1f Гц"
          % (len(only_right),
             float(rates[0, t_only].max()) if len(only_right) else -1.0))
    st.apply({"stim": []})

    # --- маркеры источника обязаны найтись ---
    #
    # Настоящую модель при этом не ломаем. Состояние привязывается к
    # подставному телу: физика та же, настоящая, а арена у него своя, и
    # слотов в ней больше, чем маркеров в модели. Отказ обязан быть
    # громким — раньше правка применялась наполовину молча: яркость и
    # массив эпизода переезжали, а маркер в сцене оставался на месте.
    from types import SimpleNamespace

    was = model_state()
    fake = SimpleNamespace(
        physics=sim.physics,
        arena=SimpleNamespace(odor_source=np.zeros((st.slots() + 2, 3))))
    st_bad = live.BodyLive()
    try:
        st_bad.attach(fake, None, cfg, neurons, "cpu")
        check("недостача маркеров источника — громкий отказ", False,
              "привязка прошла молча, слотов %d" % st_bad.slots())
    except RuntimeError as e:
        msg = str(e)
        check("недостача маркеров источника — громкий отказ",
              "нет слотов" in msg and "[4, 5]" in msg,
              msg[:95])
    check("подставное тело не задело настоящую модель",
          not moved_places(was) and len(st._mocap) == 4,
          "мест изменилось %d, маркеров по-прежнему %d"
          % (len(moved_places(was)), len(st._mocap)))

    check("журнал несёт исходные условия и правки",
          st.log and all({"что", "было", "стало"} <= set(r) for r in st.log),
          "записей %d" % len(st.log))
    sim.close()

    # Своё тело и настоящие эпизоды — после того, как это закрыто.
    part_default_sources(neurons)
    part_odour_ref(neurons)


def part_default_sources(neurons):
    """Модульный массив источников не портится переносом слота.

    `body.run_episode` в ветке ``sources=None`` брал сам
    `body.DEFAULT_SOURCES` без копии, а `BodyLive._write_sources` пишет в
    этот массив на месте. После одного переноса слота КАЖДЫЙ следующий
    эпизод в том же процессе работника собирался бы с переехавшими
    источниками — молча и невидимо по файлам прогона.

    Тело здесь своё: слотов в нём столько, сколько строк в модульном
    массиве, а не четыре, как в остальной пробе. Замер прежнего
    поведения: привязка того же состояния прямо к `DEFAULT_SOURCES`, без
    копии, переносит слоты в сам модуль — он становится
    ``[[-100, 90, 1.5], [-99, 89, 1.5]]``. Так что проверка не пустая.

    Через настоящий `run_episode` в ветке ``sources=None`` то же самое
    проверяет `part_odour_ref`: там эпизоды всё равно считаются.
    """
    from flybrain import body as B, live

    want = B.DEFAULT_SOURCES.copy()
    # Ровно то же выражение, которым берёт источники `run_episode`,
    # когда их не задали (см. начало `body.run_episode`).
    arg = None
    src = (B.DEFAULT_SOURCES.copy() if arg is None
           else np.array(arg, dtype=float))
    check("массив эпизода — отдельная память, а не сам модуль",
          src is not B.DEFAULT_SOURCES
          and not np.shares_memory(src, B.DEFAULT_SOURCES),
          "общая память %s" % bool(np.shares_memory(src,
                                                    B.DEFAULT_SOURCES)))

    cfg = B.BodyConfig(duration_s=0.1, control_ms=20.0, seed=0, device="cpu")
    sim, _ = B._build_body(cfg, src)
    sim.reset(seed=0)
    st = live.BodyLive()
    st.attach(sim, src, cfg, neurons, "cpu")
    st.apply({"sources": [
        {"x": -100.0 + i, "y": 90.0 - i, "strength": 1.0, "on": True}
        for i in range(st.slots())]})
    check("перенос слота записался в массив эпизода",
          abs(float(src[0, 0]) + 100.0) < 1e-9
          and abs(float(src[0, 1]) - 90.0) < 1e-9,
          "слотов %d, массив эпизода %s"
          % (st.slots(), src[:, :2].tolist()))
    check("модульный массив источников не изменился ни на бит",
          B.DEFAULT_SOURCES.tobytes() == want.tobytes(),
          "было %s, стало %s" % (want.tolist(),
                                 B.DEFAULT_SOURCES.tolist()))
    sim.close()


def part_odour_ref(neurons):                                  # noqa: C901
    """Яркость насыщения и сглаживание доходят до расчёта каждым шагом.

    `odour_ref` и `dn_tau_s` подтверждались как применённые, но
    `run_episode` читал их один раз до цикла шагов: живая правка
    попадала в журнал прогона, ничего при этом не меняя. Доказать
    обратное можно только настоящим эпизодом, поэтому здесь считаются
    два прогона по 0,6 с модельного времени с одним зерном. Счёт дорогой
    (около минуты), поэтому он здесь один, а спрашивается с него три
    вещи сразу:

    * **величина вообще доходит до расчёта** — два эпизода с разной
      заказанной яркостью насыщения дают разные частоты рецепторов уже
      на первом шаге, где наблюдение у обоих ещё одно и то же;
    * **она перечитывается на каждом шаге, а не один раз до цикла** —
      в первом эпизоде обе величины переправляются живьём посередине
      (на границе шага, из `on_step`), и дальше запись обязана считаться
      по новым. Этого двумя эпизодами с разной настройкой не поймать:
      прочитанная один раз до цикла, она тоже была бы разной;
    * **модульный массив источников не портится** настоящим эпизодом в
      ветке ``sources=None`` — исходные условия переносят оба слота.
    """
    import torch

    from flybrain import body as B, data as D, live as L

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = D.load_weights(device=device)
    orn_max = B.BodyConfig().orn_max_hz
    tau0, tau1 = 0.5, 0.05      # сглаживание до и после живой правки
    ref0, ref1 = 0.005, 0.05    # яркость насыщения до и после неё
    cut = 10                    # на каком шаге правка вступает в силу
    dt_s = 0.02
    print("  два эпизода по 0,6 с модельного времени на %s …" % device,
          flush=True)

    # Заодно — тот же модульный массив, но уже через настоящий
    # `run_episode` в ветке `sources=None`: исходные условия переносят
    # оба слота, и перенос обязан уйти в массив эпизода, а не в модуль.
    # Своего тела для этого не нужно, эпизоды всё равно считаются.
    want = B.DEFAULT_SOURCES.copy()
    moved = [{"x": 26.0, "y": 12.0, "strength": 1.0, "on": True},
             {"x": 39.0, "y": -16.0, "strength": 1.0, "on": True}]

    def episode(ref, switch=False):
        cfg = B.BodyConfig(duration_s=0.6, control_ms=20.0, seed=0,
                           device=device, readout="ALPN", odour_ref=ref,
                           dn_tau_s=tau0)
        st = L.BodyLive(initial={"sources": list(moved)})
        done = []

        def on_step(step, rec, row):
            # Правка подаётся по окончании шага `cut - 1`: со шага `cut`
            # запись обязана считаться по новым величинам.
            if switch and step == cut - 1:
                done.append(st.apply({"odour_ref": ref1,
                                      "dn_tau_s": tau1}))

        t0 = time.perf_counter()
        run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                            progress=False, live=st,
                            on_step=on_step if switch else None)
        return {
            "по стене, с": round(time.perf_counter() - t0, 2),
            "orn_l": [float(r["orn_l"]) for r in run.records],
            "orn_r": [float(r["orn_r"]) for r in run.records],
            "odour_l": [float(r["odour_l"]) for r in run.records],
            "odour_r": [float(r["odour_r"]) for r in run.records],
            "dn_l": [float(r["dn_l"]) for r in run.records],
            "dn_r": [float(r["dn_r"]) for r in run.records],
            "asym": [float(r["asym"]) for r in run.records],
            "насыщение": run.calib.get("odour_ref"),
            "перекос": run.calib.get("asym_bias"),
            "правка": done[0] if done else None,
            "источники эпизода": run.sources.copy(),
        }

    a, b = episode(ref0, switch=True), episode(ref1)
    rep["замеры"]["эпизоды яркости насыщения, с"] = [a["по стене, с"],
                                                     b["по стене, с"]]

    # --- модульный массив: настоящий эпизод, ветка `sources=None` ---
    check("перенос слота в настоящем эпизоде ушёл в массив эпизода",
          abs(float(a["источники эпизода"][0, 0]) - 26.0) < 1e-9
          and abs(float(b["источники эпизода"][1, 0]) - 39.0) < 1e-9,
          "источники эпизодов %s и %s"
          % (a["источники эпизода"][:, 0].tolist(),
             b["источники эпизода"][:, 0].tolist()))
    check("настоящий эпизод не испортил модульный массив",
          B.DEFAULT_SOURCES.tobytes() == want.tobytes(),
          "после двух эпизодов %s" % B.DEFAULT_SOURCES.tolist())

    # --- величина доходит до расчёта ---
    rep["замеры"]["частоты рецепторов на первом шаге, Гц"] = {
        "odour_ref %g" % ref0: a["orn_l"][0] if a["orn_l"] else None,
        "odour_ref %g" % ref1: b["orn_l"][0] if b["orn_l"] else None,
    }
    check("яркость насыщения дошла до частот рецепторов",
          bool(a["orn_l"]) and a["orn_l"] != b["orn_l"]
          and a["orn_r"] != b["orn_r"],
          "на первом шаге %.1f против %.1f Гц, записано насыщение %s и %s"
          % (a["orn_l"][0], b["orn_l"][0], a["насыщение"], b["насыщение"]))
    # Расходятся именно частоты, а не наблюдение: яркость на антеннах на
    # первом шаге у обоих прогонов одна и та же, и каждая частота сходится
    # с арифметикой по своей заказанной яркости насыщения.
    ar_a = orn_max * min(1.0, a["odour_l"][0] / ref0) if a["odour_l"] else -1.0
    ar_b = orn_max * min(1.0, b["odour_l"][0] / ref1) if b["odour_l"] else -1.0
    check("частоты первого шага сходятся с арифметикой по своей яркости",
          bool(a["odour_l"]) and a["odour_l"][0] == b["odour_l"][0] > 0.0
          and abs(a["orn_l"][0] - ar_a) < 0.1
          and abs(b["orn_l"][0] - ar_b) < 0.1,
          "яркость %.7g у обоих; %.1f против ариф. %.1f и %.1f против %.1f"
          % (a["odour_l"][0], a["orn_l"][0], ar_a, b["orn_l"][0], ar_b))

    # --- она же перечитывается на каждом шаге ---
    #
    # Проверяется арифметикой по самой записи: частота рецепторов — это
    # `orn_max_hz * min(1, яркость / насыщение)`, и яркость с частотой
    # стоят в записи рядом. Если живая правка дошла, то до шага `cut`
    # запись сходится со старым насыщением, а с него — с новым.
    def ref_misfit(rec, refs):
        worst = 0.0
        for i, (oi, hz) in enumerate(zip(rec["odour_l"], rec["orn_l"])):
            worst = max(worst, abs(orn_max * min(1.0, oi / refs[i]) - hz))
        for i, (oi, hz) in enumerate(zip(rec["odour_r"], rec["orn_r"])):
            worst = max(worst, abs(orn_max * min(1.0, oi / refs[i]) - hz))
        return worst

    n = len(a["orn_l"])
    live_refs = [ref0 if i < cut else ref1 for i in range(n)]
    stale_refs = [ref0] * n
    miss_live = ref_misfit(a, live_refs)
    miss_stale = ref_misfit(a, stale_refs)
    rep["замеры"]["яркость насыщения: невязка записи, Гц"] = {
        "с живой правкой": round(miss_live, 4),
        "если бы читалась один раз": round(miss_stale, 4),
    }
    check("живая правка яркости насыщения дошла до частот рецепторов",
          n > cut and a["правка"] is not None
          # Невязка в 0,05 Гц — это округление записи: частота пишется с
          # одним знаком после запятой. Чтение один раз до цикла даёт
          # невязку в тысячу раз больше, так что порог ни к чему не
          # подогнан.
          and miss_live < 0.08 and miss_stale > 1.0,
          "невязка записи %.3f Гц против %.3f Гц, если бы насыщение "
          "читалось один раз до цикла (правка на шаге %d из %d, стало %s)"
          % (miss_live, miss_stale, cut, n,
             (a["правка"] or {}).get("odour_ref")))

    # То же для сглаживания выхода. Его в записи нет прямо, но есть обе
    # величины, из которых оно считается: `asym` — это перекос двух
    # экспоненциальных средних от `dn_l` и `dn_r` с шагом `k =
    # exp(-dt / tau)`, минус перекос разогрева. Пересчитываем запись по
    # двум предположениям: tau переправлена живьём и tau прежняя.
    def asym_misfit(rec, taus):
        el = er = None
        bias = float(rec["перекос"] or 0.0)
        worst = 0.0
        for i, (dl, dr) in enumerate(zip(rec["dn_l"], rec["dn_r"])):
            k = math.exp(-dt_s / max(taus[i], 1e-6))
            el = dl if el is None else k * el + (1 - k) * dl
            er = dr if er is None else k * er + (1 - k) * dr
            got = (el - er) / (el + er + 1e-9) - bias
            worst = max(worst, abs(got - rec["asym"][i]))
        return worst

    live_taus = [tau0 if i < cut else tau1 for i in range(n)]
    tau_live = asym_misfit(a, live_taus)
    tau_stale = asym_misfit(a, [tau0] * n)
    rep["замеры"]["сглаживание: невязка перекоса"] = {
        "с живой правкой": round(tau_live, 7),
        "если бы читалось один раз": round(tau_stale, 7),
    }
    check("живая правка сглаживания дошла до расчёта перекоса",
          tau_live < 1e-4 and tau_stale > 10.0 * max(tau_live, 1e-9),
          "невязка перекоса %.2g против %.2g, если бы tau читалась один "
          "раз (tau %g -> %g на шаге %d, стало %s)"
          % (tau_live, tau_stale, tau0, tau1, cut,
             (a["правка"] or {}).get("dn_tau_s")))


# ------------------------------------------- половина вторая: работник

def part_worker():                                            # noqa: C901
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 240:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("работник поднялся", ses.ready,
          "за %.1f с" % (time.perf_counter() - t0))
    if not ses.ready:
        return

    applied, worst, arrived = [], [0.0], {}

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            a = time.perf_counter()
            ses.pump(dt=0.02)
            worst[0] = max(worst[0], (time.perf_counter() - a) * 1000.0)
            for rec in ses.take_applied():
                # Время прихода запоминается здесь, а не после ожидания:
                # первая версия мерила длину самого ожидания и показала
                # ровно его, пять секунд.
                applied.append(rec)
                arrived[rec.cmd_id] = time.perf_counter()
                rep["подтверждения"].append(
                    {"действие": rec.action, "модельное, мс": rec.t_model_ms,
                     "по стене, с": rec.wall_s, "исход": rec.outcome,
                     "почему": rec.why})
            time.sleep(0.01)

    # Сеанс без заданного конца, с кольцом записи и с числовой яркостью
    # насыщения: источники начинают выключенными, и измеренная по началу
    # яркость была бы нулевой.
    spec = {"kind": "body", "t": 0.0, "seed": 0, "open_report": False,
            "out": "probe-sandbox-link", "bin_ms": 10.0,
            # Кольцо намеренно короткое: сеанс живёт около полутора
            # секунд модельного времени, и при окне в двадцать секунд
            # кольцо не провернулось бы — проверять было бы нечего.
            "body": {"readout": "ALPN", "control_ms": 20.0, "keep_s": 0.6,
                     "odour_ref": 0.005,
                     "sources": SLOTS.tolist(),
                     "live_initial": {"sources": [
                         {"x": 25.0, "y": 12.0, "strength": 1.0, "on": False},
                         {"x": 38.0, "y": -16.0, "strength": 1.0,
                          "on": False},
                         {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
                         {"x": -20.0, "y": -5.0, "strength": 1.0,
                          "on": False},
                     ]}}}
    ses.submit(spec)
    t0 = time.perf_counter()
    while ses.state != "running" and time.perf_counter() - t0 < 420:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("сеанс пошёл", ses.state == "running",
          "%s за %.1f с" % (ses.state, time.perf_counter() - t0))
    if ses.state != "running":
        ses.close()
        return

    # --- кадры MuJoCo доходят до окна и меняются ---
    pump(3.0)
    seq1, fr1 = ses.body_seq, None
    if ses.body_frame is not None:
        fr1 = np.array(ses.body_frame, copy=True)
    pump(4.0)
    seq2 = ses.body_seq
    fr2 = (np.array(ses.body_frame, copy=True)
           if ses.body_frame is not None else None)
    check("кадры MuJoCo доходят до окна", fr1 is not None and seq2 > seq1,
          "кадров %d -> %d" % (seq1, seq2))
    if fr1 is not None and fr2 is not None:
        same = bool(np.array_equal(fr1, fr2))
        check("кадр меняется, а не стоит картинкой", not same,
              "форма %s, средняя разница %.2f"
              % (fr1.shape,
                 float(np.abs(fr1.astype(float) - fr2.astype(float)).mean())))

    # --- выключенные источники: нечего чуять, цели нет ---
    ex = dict(ses.extra or {})
    check("без включённых источников вход рецепторов нулевой",
          float(ex.get("odour_l", 1)) == 0.0
          and float(ex.get("odour_r", 1)) == 0.0,
          "слева %s, справа %s" % (ex.get("odour_l"), ex.get("odour_r")))
    check("без включённых источников расстояние не выдумывается",
          float(ex.get("dist", 0)) == -1.0 and int(ex.get("source", 0)) == -1,
          "dist %s, source %s" % (ex.get("dist"), ex.get("source")))

    # --- включаем источник слева ---
    sent_ms, sent_wall = ses.t_ms, time.perf_counter()
    ses.control("set", {"sources": [
        {"x": 25.0, "y": 12.0, "strength": 1.0, "on": True},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    pump(5.0)
    ok_set = [a for a in applied if a.action == "set"
              and a.outcome == "applied"]
    check("включение источника подтверждено работником", bool(ok_set),
          "подтверждений %d" % len(ok_set))
    if ok_set:
        lag_ms = ok_set[0].t_model_ms - sent_ms
        lag_wall = ok_set[0].wall_s
        rep["замеры"]["задержка правки, мс модельного"] = lag_ms
        rep["замеры"]["шаг управления, мс"] = float(ses.bin_ms)
        check("правка применена на границе шага управления",
              abs((ok_set[0].t_model_ms / 20.0)
                  - round(ok_set[0].t_model_ms / 20.0)) < 1e-6,
              "%.1f мс кратно 20" % ok_set[0].t_model_ms)
        wall = round(arrived.get(ok_set[0].cmd_id, time.perf_counter())
                     - sent_wall, 3)
        rep["замеры"]["задержка правки, с по стене"] = wall
        # Две разные величины, и путать их нельзя: модельная задержка —
        # это сколько модельного времени прошло между подачей и
        # применением, настенная — сколько прождал человек. При скорости
        # около 0,06 от настенной один шаг управления в 20 мс модельного
        # времени занимает по стене около трети секунды.
        check("задержка применения названа числом", lag_ms >= 0.0,
              "%.0f мс модельного, %.2f с по стене (от подачи до "
              "подтверждения в окне)" % (lag_ms, wall))
        _ = lag_wall
    ex2 = dict(ses.extra or {})
    check("включение источника изменило вход рецепторов",
          float(ex2.get("odour_l", 0)) > 0.0
          and float(ex2.get("odour_r", 0)) > 0.0,
          "слева %s, справа %s" % (ex2.get("odour_l"), ex2.get("odour_r")))
    check("цель появилась", float(ex2.get("dist", -1)) > 0.0
          and int(ex2.get("source", -1)) == 0,
          "dist %s мм, слот %s" % (ex2.get("dist"), ex2.get("source")))

    # --- скорость сеанса: честное отношение к настенному ---
    t_a, w_a = ses.t_ms, time.perf_counter()
    pump(12.0)
    t_b, w_b = ses.t_ms, time.perf_counter()
    ratio = ((t_b - t_a) / 1000.0) / max(w_b - w_a, 1e-9)
    rep["замеры"]["скорость к настенной"] = round(ratio, 4)
    check("скорость сеанса измерена, а не объявлена", ratio > 0.0,
          "%.3f от настенного времени (%.1f с модельного за %.1f с)"
          % (ratio, (t_b - t_a) / 1000.0, w_b - w_a))

    # --- пауза, один шаг, продолжение ---
    ses.control("pause")
    pump(4.0)
    frozen = ses.t_ms
    pump(3.0)
    check("на паузе модельное время стоит", abs(ses.t_ms - frozen) < 1e-9,
          "%.0f мс и через три секунды те же %.0f" % (frozen, ses.t_ms))
    check("окно живо на паузе", worst[0] < 300.0,
          "худший кадр приёма %.0f мс" % worst[0])

    # На паузе правка готовится и применяется на следующем шаге — это и
    # есть требование «подготовить изменение и применить при шаге».
    ses.control("set", {"sources": [
        {"x": -25.0, "y": 12.0, "strength": 1.0, "on": True},
        {"x": 38.0, "y": -16.0, "strength": 1.0, "on": False},
        {"x": 0.0, "y": 30.0, "strength": 1.0, "on": False},
        {"x": -20.0, "y": -5.0, "strength": 1.0, "on": False},
    ]})
    pump(2.0)
    on_pause = [a for a in applied if a.action == "set"][-1]
    before_step = ses.t_ms
    ses.control("step")
    pump(6.0)
    moved = ses.t_ms - before_step
    check("одиночный шаг двигает ровно на шаг управления",
          abs(moved - 20.0) < 1e-6,
          "с %.0f на %.0f мс" % (before_step, ses.t_ms))
    check("правка, поданная на паузе, применена",
          on_pause.outcome == "applied",
          "исход %s на %.0f мс" % (on_pause.outcome, on_pause.t_model_ms))

    ses.control("resume")
    pump(6.0)
    check("продолжение возобновило сеанс", ses.t_ms > before_step + 20.0,
          "%.0f -> %.0f мс" % (before_step, ses.t_ms))

    # --- остановка и сохранение ---
    ses.control("stop")
    t1 = time.perf_counter()
    while ses.busy and time.perf_counter() - t1 < 240:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("остановка довела сеанс до конца", not ses.busy,
          "за %.1f с, состояние %s" % (time.perf_counter() - t1, ses.state))
    t1 = time.perf_counter()
    while ses.summary is None and ses.error is None \
            and time.perf_counter() - t1 < 60:
        pump(0.2)

    path = (ses.summary or {}).get("path") or next(
        (p for k, p in ses.artifacts if k == "run"), None)
    out = Path(path) if path else None
    check("сеанс сохранён", out is not None and out.exists(),
          str(out) if out else "пути нет; ошибка %s" % ses.error)
    if out and out.exists():
        meta = json.loads((out / "embodied.json").read_text(encoding="utf-8"))
        tm = meta.get("timing") or {}
        rep["замеры"]["timing сохранённого"] = {
            k: v for k, v in tm.items() if k != "оговорки"}
        check("сохранённый сеанс несёт исходные условия",
              bool(tm.get("исходные условия")),
              str(list((tm.get("исходные условия") or {}))))
        edits = tm.get("живые правки") or []
        check("сохранённый сеанс несёт журнал правок с их видом",
              len(edits) >= 3
              and any(e.get("что") == "исходные условия" for e in edits),
              "записей %d, видов %s"
              % (len(edits), sorted({e.get("что") for e in edits})))
        check("запись активности — окно, и это сказано",
              abs(float(tm.get("окно записи, с") or 0.0) - 0.6) < 0.05
              and float(tm.get("запись с, мс") or 0.0) > 0.0
              and any("окно" in s for s in (tm.get("оговорки") or [])),
              "окно %s с, с %s мс"
              % (tm.get("окно записи, с"), tm.get("запись с, мс")))
        check("сеанс без конца помечен как таковой",
              any("без заранее заданного конца" in s
                  for s in (tm.get("оговорки") or [])),
              "оговорок %d" % len(tm.get("оговорки") or []))
        import numpy as _np

        z = _np.load(out / "bins.npz")
        rep["замеры"]["строк записи"] = int(z["bins"].shape[0])

    ses.close()


def main():
    print("— модель: правка доходит до MuJoCo —")
    part_model()
    print("— работник: сеанс, кадры, подтверждения —")
    part_worker()
    (OUT / "sandbox_link.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
