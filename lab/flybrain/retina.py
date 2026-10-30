"""Привязка омматидиев тела к фоторецепторам коннектома.

**Это инженерная конструкция, а не восстановленная анатомия.** Проба
`probe/retinotopy.py` показала замерами, что ретинотопия из координат
сом v783 НЕ восстанавливается: пучков по шесть нет (расстояние до
шестого соседа больше, чем до пятого, в 1,082 раза против 5,425 на
искусственных пучках), ближайший R7 лежит в 2,4 шага решётки, а по
направлению взгляда — дальше, чем до соседнего узла. Значит никакая
привязка не может быть «настоящей», и эта — тоже.

Что она делает и чего не делает:

* **делает** — раскладывает 721 омматидий тела по листу сом так, что
  соседние омматидии достаются соседним нейронам. Это единственное
  свойство, которое нужно обнаружению движения, и единственное, которое
  здесь обеспечено;
* **не делает** — не утверждает, что конкретный нейрон принадлежит
  конкретному омматидию. Такого утверждения данные не поддерживают.

Правило объявлено целиком и состоит из четырёх шагов.

1. **Положения омматидиев** берутся из `ommatidia_id_map.npy`
   NeuroMechFly — это настоящие данные, а не выдумка: 721 омматидий по
   230…243 пикселя, правильная гексагональная решётка с шагом 16,41 px,
   все шесть соседей на одном расстоянии. Положение омматидия — центр
   тяжести его пикселей.
2. **Положения нейронов** — проекция координат сомы на собственную
   плоскость популяции (две главные компоненты). Основание:
   две главные оси держат 96,8…99,5 % дисперсии, то есть сомы лежат
   листом. Третья ось отбрасывается, и это осознанная потеря: толщина
   листа 0,94 шага решётки слева и 2,24 справа.
3. **Совмещение** двух облаков — одно из двух объявленных:
   ``"whiten"`` приводит каждую ось к единичной дисперсии (сохраняет
   соседство, искажает пропорции) и ``"rigid"`` масштабирует обе оси
   одним числом (сохраняет пропорции, хуже покрывает края).
4. **Назначение** — каждому нейрону ближайший омматидий в совмещённой
   плоскости.

**Неустранимая неоднозначность, и её нельзя замолчать.** Главные
компоненты определены с точностью до знака, а оси можно поменять
местами. Отсюда **восемь** разных совмещений, и данные не выбирают
между ними: в них нет ни одного поля, связывающего сому с направлением
взгляда. Зеркальное совмещение переворачивает направление движения в
поле зрения. Значит **знак направленной избирательности задаёт наше
соглашение, а не коннектом**, и соглашение названо здесь:
``ALIGN_DEFAULT``. Последствие замерено отдельно — см.
`probe/retina_map.py`.

Встречный замер к любому утверждению про зрение — ``shuffle=True``:
перестановка назначения, **сохраняющая число нейронов на омматидий**.
Суммарный вход при этом тот же до последнего разряда, разрушена только
пространственная структура. Если отклик сети не отличается между
настоящей привязкой и перемешанной, значит структурой сеть не
пользуется вовсе, и говорить о пространственном зрении нельзя.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Типы фоторецепторов и сколько их анатомически на один омматидий.
# Числа не подогнаны: в v783 на один R7 приходится 6,05 нейрона R1-6
# слева и 5,82 справа (`probe/retinotopy.py`).
PR_TYPES = {"R1-6": 6, "R7": 1, "R8": 1}

# Соглашение о совмещении: (знак первой оси, знак второй оси, менять ли
# оси местами). Данные между восемью вариантами не выбирают, см.
# докстринг модуля. Выбран тождественный — не потому, что он вернее, а
# потому, что он должен быть объявлен один.
ALIGN_DEFAULT = (1, 1, False)
ALIGNMENTS = tuple((sx, sy, sw) for sw in (False, True)
                   for sx in (1, -1) for sy in (1, -1))

MODES = ("whiten", "rigid")

# Способ назначения. `nearest` — каждому нейрону ближайший омматидий,
# и это самое простое правило; но оно игнорирует анатомию, и проба это
# поймала: 668 нейронов R7 ложатся всего на 372 омматидия по 1,8 на
# каждый, хотя анатомически R7 ровно один на омматидий. `capacity`
# вносит анатомическое число как ОГРАНИЧЕНИЕ: у омматидия не больше
# `PR_TYPES[kind]` мест, и назначение идёт по возрастанию расстояния.
ASSIGNS = ("capacity", "nearest")
CAND = 24          # сколько ближайших омматидиев рассматривать на нейрон


def _vision_dir() -> Path:
    import flygym_gymnasium as fg

    return Path(fg.__file__).parent / "data" / "vision"


_OMMA = None


def ommatidia():
    """Положения 721 омматидия в пикселях и маска бледных.

    Читается карта NeuroMechFly, а не строится своя: это те же данные,
    по которым тело считает `obs["vision"]`, и расходиться им нельзя.
    Результат кэшируется: карта 512×450 разбирается за десятые доли
    секунды, но зовётся на каждый сеанс.
    """
    global _OMMA
    if _OMMA is not None:
        return _OMMA
    d = _vision_dir()
    m = np.load(d / "ommatidia_id_map.npy")
    pale = np.load(d / "pale_mask.npy").astype(bool)
    ids = np.unique(m)
    ids = ids[ids > 0]
    if len(ids) != len(pale):
        raise ValueError(
            "в карте омматидиев %d кружков, а в маске бледных %d — "
            "наборы разъехались" % (len(ids), len(pale)))
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    pos = np.empty((len(ids), 2), dtype=float)
    px = np.empty(len(ids), dtype=np.int64)
    for i, k in enumerate(ids):
        sel = m == k
        pos[i] = (xx[sel].mean(), yy[sel].mean())
        px[i] = int(sel.sum())
    _OMMA = {"pos": pos, "pale": pale, "пикселей": px,
             "кадр": tuple(m.shape)}
    return _OMMA


def plane(xyz):
    """Координаты в собственной плоскости облака: две главные оси.

    Возвращает также третью координату — не для привязки, а чтобы
    толщина листа была измерима, а не предполагалась.
    """
    c = np.asarray(xyz, dtype=float)
    c = c - c.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    uv = c @ vt[:2].T
    w = c @ vt[2]
    return uv, w, (s ** 2 / (s ** 2).sum())


def _fit(uv, mode):
    """Привести облако к безразмерным координатам объявленным способом."""
    uv = uv - uv.mean(0)
    sd = uv.std(0)
    if mode == "whiten":
        return uv / np.maximum(sd, 1e-12)
    if mode == "rigid":
        # Один множитель на обе оси: пропорции облака сохраняются.
        # Берётся среднеквадратичный радиус, а не размах: размах зависит
        # от единичных выбросов, а здесь нужна устойчивая мера.
        r = float(np.sqrt((uv ** 2).sum(1).mean()))
        return uv / max(r, 1e-12)
    raise ValueError("неизвестный способ совмещения: %r" % (mode,))


def _align(uv, align):
    sx, sy, swap = align
    out = uv.copy()
    if swap:
        out = out[:, ::-1]
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def _assign_nearest(me, op):
    from scipy.spatial import cKDTree

    return cKDTree(op).query(me, k=1)[1].astype(np.int32)


def _assign_capacity(me, op, cap):
    """Назначение с анатомическим ограничением на место.

    Правило объявлено целиком и детерминировано. Для каждого нейрона
    берутся ``CAND`` ближайших омматидиев; все пары «нейрон —
    омматидий» сортируются по расстоянию; пары разбираются по порядку, и
    пара принимается, если нейрон ещё не занят и у омматидия осталось
    место. Нейроны, которым в пределах ``CAND`` кандидатов места не
    нашлось, получают ближайший омматидий **без** ограничения — иначе
    часть популяции осталась бы без входа вовсе, а это хуже, чем
    переполненный омматидий. Сколько таких нейронов, записывается в
    сводку: это цена правила, и она не прячется.

    Почему не точная задача о назначении: 4044 нейрона против 4326 мест
    дают матрицу 17,5 млн элементов, и точный алгоритм на ней считается
    куда дольше, чем стоит этот выигрыш. Жадный разбор по возрастанию
    расстояния даёт тот же порядок величины, и он воспроизводим.
    """
    from scipy.spatial import cKDTree

    k = min(CAND, len(op))
    d, who = cKDTree(op).query(me, k=k)
    n = len(me)
    rows = np.repeat(np.arange(n), k)
    cols = who.ravel().astype(np.int64)
    dist = d.ravel()
    # Устойчивый порядок: при равных расстояниях решает номер нейрона, а
    # не порядок в памяти, иначе отпечаток зависел бы от сборки numpy.
    order = np.lexsort((cols, rows, dist))
    free = np.full(len(op), int(cap), dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    left = n
    for p in order:
        if left == 0:
            break
        i = rows[p]
        if out[i] >= 0:
            continue
        j = cols[p]
        if free[j] <= 0:
            continue
        out[i] = j
        free[j] -= 1
        left -= 1
    spill = int((out < 0).sum())
    if spill:
        miss = out < 0
        out[miss] = _assign_nearest(me[miss], op)
    return out.astype(np.int32), spill


@dataclass
class Map:
    """Готовая привязка одной популяции одного глаза.

    ``omma`` — номер омматидия (0…720) для каждого нейрона популяции;
    ``neurons`` — индексы нейронов в таблице, в том же порядке.
    """

    side: str
    kind: str
    mode: str
    assign: str
    align: tuple
    shuffled: bool
    neurons: np.ndarray
    omma: np.ndarray
    counts: np.ndarray            # нейронов на каждый из 721 омматидия
    толщина_листа: float
    доля_дисперсии: tuple
    сверх_места: int = 0
    # Имя версии подгонки: `своя` — прежняя, у каждой популяции своя
    # плоскость и нормировка; `общая` — одна на все три популяции глаза.
    # Умолчание прежнее нарочно: все сохранённые прогоны посчитаны им, и
    # у них этого поля в файле нет вовсе. Стоит последним, потому что
    # поля со значением по умолчанию не могут идти перед полями без
    # него — поймано сборкой.
    fit: str = "своя"

    def fingerprint(self) -> str:
        """Отпечаток привязки: по нему прогон опознаётся однозначно."""
        h = hashlib.sha256()
        for part in (self.side.encode(), self.kind.encode(),
                     self.mode.encode(), self.assign.encode(),
                     self.fit.encode(), repr(self.align).encode(),
                     b"1" if self.shuffled else b"0",
                     np.ascontiguousarray(self.neurons, dtype=np.int64)
                     .tobytes(),
                     np.ascontiguousarray(self.omma, dtype=np.int32)
                     .tobytes()):
            h.update(part)
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        c = self.counts
        want = PR_TYPES[self.kind]
        return {
            "сторона": self.side, "тип": self.kind,
            "способ": self.mode, "назначение": self.assign,
            "подгонка": self.fit,
            "совмещение": list(self.align),
            "перемешано": bool(self.shuffled),
            "нейронов сверх места": int(self.сверх_места),
            "нейронов": int(len(self.neurons)),
            "омматидиев всего": int(len(c)),
            "омматидиев занято": int((c > 0).sum()),
            "омматидиев пусто": int((c == 0).sum()),
            "на омматидий: анатомически": want,
            "на омматидий: среднее по занятым":
                round(float(c[c > 0].mean()) if (c > 0).any() else 0.0, 3),
            "на омматидий: медиана": float(np.median(c)),
            "на омматидий: СО": round(float(c.std()), 3),
            "на омматидий: наибольшее": int(c.max()),
            "доля омматидиев с анатомическим числом":
                round(float((c == want).mean()), 4),
            "толщина листа, мкм": round(self.толщина_листа, 2),
            "доля дисперсии по осям":
                [round(float(v), 5) for v in self.доля_дисперсии],
            "отпечаток": self.fingerprint(),
        }


# Имена версий подгонки. В прогон уезжает имя, а не признак: по
# сохранённому файлу должно быть видно, какая привязка считала.
FIT_OWN = "своя"          # прежняя: своя плоскость и нормировка у каждой
FIT_JOINT = "общая"       # совместная: одна на все три популяции глаза
FITS = (FIT_OWN, FIT_JOINT)


def _joint_frame(neurons, side, data=None):
    """Одна плоскость и одна нормировка на ВСЕ фоторецепторы глаза.

    Считается по объединённому облаку R1-6, R7 и R8 одной стороны.
    Возвращает всё, что нужно, чтобы привести любую из популяций в те же
    безразмерные координаты: центр в трёхмерии, две главные оси, третью
    ось (для измеримой толщины листа), центр и масштаб в плоскости.

    Масштаб берётся по ОБЪЕДИНЁННОМУ облаку, а не по каждой популяции:
    в этом и смысл — у трёх популяций одного глаза преобразование одно,
    поэтому их взаимное положение сохраняется.
    """
    from . import data as D

    src = neurons if data is None else data
    части = []
    for k in PR_TYPES:
        idx = D.select(src, "type=%s+side:%s" % (k, side))
        if not len(idx):
            raise ValueError("пустая популяция %s на стороне %s" % (k, side))
        части.append(neurons.iloc[idx][["x_um", "y_um", "z_um"]]
                     .to_numpy(float))
    все = np.concatenate(части)
    c3 = все.mean(0)
    _, s3, vt = np.linalg.svd(все - c3, full_matrices=False)
    uv = (все - c3) @ vt[:2].T
    c2 = uv.mean(0)
    uv = uv - c2
    sd = uv.std(0)
    r = float(np.sqrt((uv ** 2).sum(1).mean()))
    return {"центр": c3, "оси": vt[:2], "третья": vt[2],
            "центр в плоскости": c2, "отклонения": sd,
            "радиус": r, "доли дисперсии": (s3 ** 2 / (s3 ** 2).sum()),
            "нейронов в облаке": int(len(все))}


def _joint_apply(xyz, рама, mode):
    """Привести облако популяции в ОБЩИЕ безразмерные координаты."""
    c = np.asarray(xyz, dtype=float) - рама["центр"]
    uv = c @ рама["оси"].T - рама["центр в плоскости"]
    w = c @ рама["третья"]
    if mode == "whiten":
        return uv / np.maximum(рама["отклонения"], 1e-12), w
    if mode == "rigid":
        return uv / max(рама["радиус"], 1e-12), w
    raise ValueError("неизвестный способ совмещения: %r" % (mode,))


def build(neurons, side, kind, mode="whiten", assign="capacity",
          align=ALIGN_DEFAULT, shuffle=False, seed=0, data=None,
          fit=FIT_OWN, рама=None):
    """Построить привязку для одной популяции одного глаза.

    ``shuffle`` даёт встречный замер: назначение перемешивается так, что
    **число нейронов на каждый омматидий сохраняется**. Поэтому
    суммарный вход в популяцию совпадает с настоящей привязкой до
    последнего разряда, и различие может дать только пространственная
    структура.
    """
    from . import data as D

    if kind not in PR_TYPES:
        raise ValueError("не тип фоторецептора: %r" % (kind,))
    idx = D.select(neurons if data is None else data,
                   "type=%s+side:%s" % (kind, side))
    if not len(idx):
        raise ValueError("пустая популяция %s на стороне %s"
                         % (kind, side))
    xyz = neurons.iloc[idx][["x_um", "y_um", "z_um"]].to_numpy(float)
    if fit == FIT_JOINT:
        # Общая рама считается ОДИН раз на глаз и передаётся сюда: иначе
        # три вызова `build` посчитали бы её трижды по одному и тому же
        # облаку — те же числа втрое дороже.
        if рама is None:
            рама = _joint_frame(neurons, side, data=data)
        uv, w = _joint_apply(xyz, рама, mode)
        var = рама["доли дисперсии"]
        me = _align(uv, align)
    elif fit == FIT_OWN:
        uv, w, var = plane(xyz)
        me = _align(_fit(uv, mode), align)
    else:
        raise ValueError("неизвестная подгонка: %r; есть %s"
                         % (fit, ", ".join(FITS)))

    om = ommatidia()
    op = _fit(om["pos"] - om["pos"].mean(0), mode)

    spill = 0
    if assign == "nearest":
        who = _assign_nearest(me, op)
    elif assign == "capacity":
        who, spill = _assign_capacity(me, op, PR_TYPES[kind])
    else:
        raise ValueError("неизвестный способ назначения: %r" % (assign,))
    counts = np.bincount(who, minlength=len(op)).astype(np.int64)

    if shuffle:
        # Перестановка, сохраняющая числа: берём тот же список номеров
        # омматидиев и раздаём его нейронам в случайном порядке. Числа
        # на омматидий остаются те же, адресат каждого нейрона меняется.
        g = np.random.default_rng(seed)
        who = who.copy()
        g.shuffle(who)
        assert np.array_equal(
            np.bincount(who, minlength=len(op)).astype(np.int64), counts)

    return Map(side=side, kind=kind, mode=mode, assign=assign,
               fit=fit, align=tuple(align),
               shuffled=bool(shuffle), neurons=idx.astype(np.int64),
               omma=who, counts=counts,
               толщина_листа=float(np.std(w)),
               доля_дисперсии=tuple(float(v) for v in var),
               сверх_места=int(spill))


def build_all(neurons, mode="whiten", assign="capacity",
              align=ALIGN_DEFAULT, shuffle=False, seed=0,
              sides=("left", "right"), kinds=None, fit=FIT_OWN):
    """Привязки для всех типов фоторецепторов обоих глаз.

    При совместной подгонке общая рама считается ОДИН раз на сторону и
    передаётся во все её популяции: это те же числа, но втрое дешевле, и
    главное — одна и та же рама у всех трёх, иначе смысл совместности
    терялся бы на округлении.
    """
    kinds = tuple(kinds or PR_TYPES)
    out = {}
    for s in sides:
        рама = _joint_frame(neurons, s) if fit == FIT_JOINT else None
        for k in kinds:
            out[(s, k)] = build(neurons, s, k, mode=mode, assign=assign,
                                align=align, shuffle=shuffle, seed=seed,
                                fit=fit, рама=рама)
    return out


def fingerprint_all(maps) -> str:
    """Один отпечаток на весь набор привязок — для записи в прогон."""
    h = hashlib.sha256()
    for key in sorted(maps, key=lambda t: (t[0], t[1])):
        h.update(("%s/%s/" % key).encode())
        h.update(maps[key].fingerprint().encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------ яркость в герцы

def intensity(maps, vision, channel="max"):
    """Яркость на каждый нейрон из кадра ``obs["vision"]``.

    Кадр у NeuroMechFly — ``(2, 721, 2)``: два глаза, 721 омматидий, два
    типа фоторецепторов (бледный и жёлтый). Сторона 0 — левый глаз.

    ``channel`` объявляет, что делать с двумя каналами, и это выбор, а
    не данность: ``"max"`` берёт больший (так делает и официальная
    интеграция NeuroMechFly, и так теряется хроматика), ``"pale"`` и
    ``"yellow"`` берут свой канал. Хроматического разделения по типам
    R7 и R8 здесь нет: в кадре его не хватает, а выдумывать нельзя.
    """
    v = np.asarray(vision, dtype=np.float32)
    if v.shape[-2:] != (721, 2):
        raise ValueError("кадр зрения имеет форму %s, ожидалось (…, 721, 2)"
                         % (tuple(v.shape),))
    if channel == "max":
        per = v.max(axis=-1)
    elif channel == "pale":
        per = v[..., 0]
    elif channel == "yellow":
        per = v[..., 1]
    else:
        raise ValueError("неизвестный канал: %r" % (channel,))
    eye = {"left": 0, "right": 1}
    out = {}
    for (side, kind), m in maps.items():
        out[(side, kind)] = per[eye[side]][m.omma]
    return out


def rates(maps, vision, max_hz, ref=1.0, channel="max"):
    """Частота рецепторов в герцах: яркость, насыщение, потолок.

    Преобразование **придумано**, как и у обоняния (`orn_max_hz`): в
    коннектоме нет ни функции яркость-в-частоту, ни усиления. Правило
    объявлено ровно одно и то же для всех фоторецепторов:

        hz = max_hz · min(1, яркость / ref)

    Ни одного нейрона это правило не выделяет, и ни одной реакции не
    задаёт — оно только задаёт масштаб.
    """
    out = {}
    for key, val in intensity(maps, vision, channel=channel).items():
        out[key] = (float(max_hz)
                    * np.minimum(1.0, val / max(float(ref), 1e-12)))
    return out
