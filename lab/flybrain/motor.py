"""Моторный путь: прямые зонды нисходящих и ИНЖЕНЕРНЫЙ декодер в команду телу.

Это единственное место, где лежат нефизиологичные преобразования моторного
пути. Раньше они были рассыпаны по `body.py` и `world.py` комментариями
«придуманная часть»; теперь они собраны здесь, у каждого есть имя, номер
версии и список параметров, и каждое уезжает в `engineered`-блок прогона.
Правило разведения имён, принятое на этом этапе:

* ``readout_*`` — **измерение**. Частота популяции, снятая со счётчика
  спайков движка. Ничего не придумано: это сумма спайков, делённая на
  число нейронов и длительность блока. Такое число можно сравнивать с
  электрофизиологией.
* ``cmd_*`` — **инженерная команда**. Любое число, которое пошло в тело.
  Биологического основания у его величины нет; есть основание у знака и
  у того, какая популяция на него влияет.

Что здесь измерено, а что объявлено
-----------------------------------

**Измерено** (своим запросом к таблице v783 и к матрице связей):

* ``DNp01`` (Giant Fiber) — ровно 2 нейрона, по одному на сторону,
  ``hemibrain_type = Giant Fiber``, оба в GNG;
* ``LC4 → DNp01`` — 104 пары, знаковый вес **+805** (все 104 LC4 точного
  селектора доходят до GF); ``LPLC2 → DNp01`` — 189 пар, **+1080**;
* ``DNp01 → super:motor`` — **ноль** пар из 110 мотонейронов. Через один
  шаг — 82 пары со знаковой суммой **−276**, то есть тормозной, и только
  к головным и шейным. Мотонейронов ног и крыльев в v783 нет физически:
  это мозг без брюшной нервной цепочки, исполнители побега (TTMn, PSI)
  лежат в VNC и в наборе данных отсутствуют.

**Объявлено** (инженерная часть, и она помечена):

* коэффициенты линейной карты «нисходящие → поворот и тяга». Их знаки
  взяты из опубликованных ролей нисходящих нейронов, величины —
  объявление, а не замер;
* сам вид отображения «асимметрия, нормированная на собственный шум
  популяции, → две амплитуды шага». Он перенесён из `body.py` без
  изменения формы: менялось только то, С КАКИХ популяций снимается
  асимметрия;
* мост через обрыв ``GF → мотонейроны``. По умолчанию его нет вовсе —
  см. :class:`GiantFiberBridge`;
* стабилизация головы. Это чужой обученный контроллер, обходящий
  коннектом целиком — см. :class:`HeadStabilizer`.

Чего здесь нет нарочно
----------------------

Нет ни одной заранее запрограммированной последовательности поведения:
ни позы, ни таймера, ни «взлететь через N мс». Всё, что уходит в тело, —
непрерывная функция измеренных в этот момент частот. Если нисходящие
молчат, команда равна опорной, и ничего не происходит.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Версия моторного пути целиком. Меняется, когда меняется ЧИСЛЕННЫЙ
# результат при тех же настройках: по ней видно, сравним ли прогон с
# прежними.
MOTOR_VERSION = 1

__all__ = [
    "MOTOR_VERSION", "DN_PROBES", "SHADOW_PROBES", "PROBE_ROLES",
    "verify_probes", "DnReadout", "MotorCommand", "LegacyAsymDecoder",
    "DnLinearDecoder", "DnTrainedDecoder", "make_decoder", "DECODERS",
    "GiantFiberBridge", "HeadStabilizer", "engineered_block",
]


# ------------------------------------------------------- прямые зонды


@dataclass(frozen=True)
class Probe:
    """Один прямой зонд: точный селектор и то, что по нему ожидается.

    ``n_side`` — сколько нейронов должно оказаться на КАЖДОЙ стороне.
    Проверяется при сборке: если в таблице нейронов стало иначе, это не
    «ну ладно», а повод остановиться — значит сменилась версия набора
    данных или аннотация, и прежние числа несравнимы.
    """

    sel: str
    n_side: int
    note: str = ""


# Шесть прямых зондов вместо усреднения по 1299 нисходящим. Селекторы
# точные и, где имена двух номенклатур столкнулись, адресуют ОДИН
# столбец: `type=` сравнивает с `cell_type`, `hemibrain_type` и `label`
# сразу и берёт объединение, а это для научной выборки молчаливое
# слияние двух популяций (см. докстринг `data.select`).
#
# Почему именно съём в 2 нейрона, а не в 1299. При 209 Гц один Giant
# Fiber поднимает среднее по полушарию нисходящих примерно на 0,32 Гц,
# тогда как обычный уровень этого среднего 2,5…4,1 Гц при 51…88 активных
# нейронах. То есть спайк GF в усреднённом съёме арифметически
# неотличим от шума. Прямой зонд на 2 нейрона его видит ровно.
#
# ВСЕ ШЕСТЬ адресуют один столбец, включая те четыре, где `type=` сегодня
# дал бы то же самое. Проверено понейронно (`probe/motor_audit.py`):
# у `DNp01`, `DNp04`, `DNa02` и `MDN` выборки `type=` и `celltype=`
# совпадают по `root_id`, то есть переход ничего не изменил в числах. Но
# совпадение — свойство сегодняшних ДАННЫХ, а не селектора: ровно на
# `DNa01` и `DNp09` объединение трёх столбцов уже склеило по две разные
# популяции. Полагаться на то, что у остальных столкновения не появится,
# незачем, когда ключ по столбцу стоит столько же.
DN_PROBES = {
    # Giant Fiber: `cell_type` — «DNp01», `hemibrain_type` — «Giant
    # Fiber». Берём столбец `cell_type`.
    "DNp01": Probe("celltype=DNp01", 1,
                   "Giant Fiber, hemibrain_type = Giant Fiber"),
    # Нисходящие, отвечающие на надвигание. DNp04 — пара.
    "DNp04": Probe("celltype=DNp04", 1, "ответ на надвигание"),
    # Рулёжка. `type=DNa01` даёт ЧЕТЫРЕ нейрона: пару с
    # `hemibrain_type = DNa01` (их `cell_type` — `DNae001`, GNG) и
    # отдельную пару с `cell_type = DNa01` (их `hemibrain_type` —
    # `VES006`, область VES). Берём первую: именно она в литературе
    # называется DNa01 и стоит рядом с DNa02 в GNG. Вторая записывается
    # отдельным теневым зондом, чтобы слияние было видно.
    "DNa01": Probe("hemibrain=DNa01", 1, "cell_type = DNae001, GNG"),
    "DNa02": Probe("celltype=DNa02", 1, "рулевой нисходящий"),
    # `type=DNp09` тоже даёт четыре: `hemibrain_type = DNp09` стоит и у
    # `cell_type = DNp09`, и у `cell_type = DNp71`. Берём строго
    # `cell_type = DNp09` — ту пару, которую апстримный опыт зовёт P9.
    "DNp09": Probe("celltype=DNp09", 1, "пара P9 апстримного опыта"),
    # Задний ход. Здесь по два нейрона на сторону, и оба столбца
    # согласны.
    "MDN": Probe("celltype=MDN", 2, "команда заднего хода, 2 на сторону"),
}

# Зонды, которые не управляют ничем и стоят рядом ради честности.
#
# Первые два — те самые популяции, которые `type=` молча подмешивает в
# DNa01 и DNp09. Если они окажутся активны, а DNa01/DNp09 нет, то
# прежние числа по `type=DNa01` были числами другой популяции, и это
# будет видно, а не останется догадкой.
#
# Третий — ЕДИНСТВЕННОЕ, к чему DNp01 в v783 вообще доходит: шейные
# мотонейроны через один шаг, со знаковой суммой −276. Это не выход на
# ноги и не побег; это всё, что в этом наборе данных есть, и оно
# измеряется, а не домысливается.
SHADOW_PROBES = {
    "DNa01_VES": Probe("celltype=DNa01", 1, "подмешивается в type=DNa01, область VES"),
    "DNp71": Probe("hemibrain=DNp09+celltype=DNp71", 1,
                   "подмешивается в type=DNp09"),
    "motor_neck": Probe("super:motor+nerve:CV", 10,
                        "шейные мотонейроны, единственный выход DNp01 в v783"),
    "DN_all": Probe("super:descending", 645,
                    "прежний съём целиком, для сравнения с прямыми зондами"),
}

# Роли, по которым расставлены ЗНАКИ коэффициентов линейной карты. Это
# литература, а не наш замер, и ссылки стоят здесь, а не в отчёте,
# чтобы коэффициент и его основание нельзя было разнести.
PROBE_ROLES = {
    "DNa02": "рулевой; односторонняя активация поворачивает ИПСИлатерально "
             "(Rayshubskiy и др.; Namiki и др., 2018)",
    "DNa01": "рулевой, градуальный и медленнее DNa02, со-активен с ним "
             "(Rayshubskiy и др.)",
    "DNp09": "ипсилатеральный поворот и остановка/замирание "
             "(Cande и др., 2018; Braun и др., 2024)",
    "MDN": "команда ЗАДНЕГО хода, двусторонняя "
           "(Bidaye и др., 2014; Feng и др., 2020)",
    "DNp01": "Giant Fiber: побег, выход в VNC (TTMn, PSI) — в v783 отсутствует",
    "DNp04": "ответ на надвигание; выход в VNC — в v783 отсутствует",
}


def verify_probes(neurons, probes=None, shadow=True, strict=True):
    """Проверить, что каждый зонд существует и что в нём ровно столько.

    Возвращает таблицу-отчёт: имя, селектор, сколько слева, сколько
    справа, ожидалось, совпало ли. ``strict`` — поднять ошибку при
    расхождении, а не записать «не совпало» и поехать дальше.

    Нужно это затем, что молчаливая смена выборки — самая дорогая из
    возможных ошибок: прогон посчитается, числа будут выглядеть
    правдоподобно, и ни одно из них не будет про ту популяцию, про
    которую сказано в отчёте.
    """
    from . import data as D

    # `probes={}` значит «ни одного зонда», а не «по умолчанию»:
    # пустой словарь ложен, и `or` молча подставил бы DN_PROBES.
    items = dict(DN_PROBES if probes is None else probes)
    if shadow:
        items.update(SHADOW_PROBES)
    rows, bad = [], []
    for name, p in items.items():
        try:
            nl = len(D.select(neurons, f"{p.sel}+side:left"))
            nr = len(D.select(neurons, f"{p.sel}+side:right"))
            nt = len(D.select(neurons, p.sel))
        except ValueError as e:
            rows.append({"зонд": name, "селектор": p.sel, "ошибка": str(e),
                         "совпало": False})
            bad.append(f"{name}: селектор {p.sel!r} не разобрался ({e})")
            continue
        ok = (nl == p.n_side and nr == p.n_side) if p.n_side else (nt > 0)
        # У `DN_all` и `motor_neck` сторон не по одному, и равенство
        # сторон у них не обещано: там проверяется только «не меньше».
        if name in ("DN_all", "motor_neck"):
            ok = nl >= p.n_side and nr >= p.n_side
        rows.append({"зонд": name, "селектор": p.sel, "слева": nl,
                     "справа": nr, "всего": nt, "ожидалось на сторону":
                     p.n_side, "совпало": bool(ok), "примечание": p.note})
        if not ok:
            bad.append("%s: селектор %r дал %d/%d на сторону, ожидалось "
                       "%d/%d" % (name, p.sel, nl, nr, p.n_side, p.n_side))
    if bad and strict:
        raise ValueError("прямые зонды не сошлись с таблицей нейронов:\n  "
                         + "\n  ".join(bad))
    return rows


class DnReadout:
    """ИЗМЕРЕНИЕ: частота каждого прямого зонда по сторонам.

    Ничего не преобразует и ничего не помнит. На входе — накопленный за
    блок управления вектор спайков движка (``acc``, по нейрону) и
    длительность блока в секундах; на выходе — словарь
    ``{имя: (Гц слева, Гц справа)}``, считанный как

        Гц = сумма спайков / (число нейронов × длительность блока),

    ровно так же, как считался прежний съём по полушарию. Отличается
    только то, по каким нейронам идёт сумма.

    **Квантование, о котором надо знать.** У зонда в ОДИН нейрон на
    сторону при блоке 20 мс частота принимает значения 0, 50, 100 … Гц:
    один спайк — это 50 Гц. Асимметрия такого зонда на одном шаге равна
    −1, 0 или +1 и ничего промежуточного принимать не может. Градуальной
    её делает только сглаживание по времени (τ в декодере), и это не
    приём, а единственный выход: популяция из двух нейронов иначе не
    измеряется.
    """

    def __init__(self, neurons, device, probes=None, shadow=True,
                 strict=True, pairs=None):
        """``pairs`` — дополнительные пары селекторов ``{имя: (слева, справа)}``.

        Через них подключается ПРЕЖНИЙ съём по полушарию (`__readout__`):
        он не зонд по типу, а выбранная в настройке популяция, и
        проверять его на «1 нейрон на сторону» бессмысленно. Благодаря
        этому старый декодер читает свой вход из того же измерителя, а
        не из отдельной ветки кода.
        """
        import torch

        from . import data as D

        self.rows = verify_probes(neurons, probes=probes, shadow=shadow,
                                  strict=strict)
        items = dict(DN_PROBES if probes is None else probes)
        if shadow:
            items.update(SHADOW_PROBES)
        self.probes = items
        self.motor_names = list(DN_PROBES if probes is None
                                else probes)
        self._idx = {}
        self._n = {}
        sels = {name: (f"{p.sel}+side:left", f"{p.sel}+side:right")
                for name, p in items.items()}
        sels.update(dict(pairs or {}))
        self.selectors = sels
        self.names = list(sels)
        flat, self._slices = [], []
        at = 0
        for name, (ql, qr) in sels.items():
            il = D.select(neurons, ql)
            ir = D.select(neurons, qr)
            self._idx[name] = (torch.as_tensor(il, device=device),
                               torch.as_tensor(ir, device=device))
            self._n[name] = (max(len(il), 1), max(len(ir), 1))
            flat.append(il)
            flat.append(ir)
            self._slices.append((name, at, at + len(il),
                                 at + len(il) + len(ir)))
            at += len(il) + len(ir)
        # Один вектор индексов на все зонды, и один съём с устройства на
        # шаг вместо двадцати. Это не косметика: каждое `float(...)` с
        # CUDA — синхронизация, и десять зондов по две стороны стоили
        # около 6 % шага управления (51,6 с против 48,7 с на эпизоде в
        # 2 с). Арифметика при этом та же до бита: в `acc` лежат целые
        # числа спайков, они точны и во float32, и во float64, поэтому
        # сумма на устройстве и сумма в numpy совпадают, а не «почти».
        self._flat = (torch.as_tensor(np.concatenate(flat).astype(np.int64),
                                      device=device) if flat else
                      torch.zeros(0, dtype=torch.int64, device=device))

    def indices(self, names=None):
        """Индексы нейронов зондов одним массивом — для маски вмешательства."""
        out = []
        for name in (names or self.motor_names):
            il, ir = self._idx[name]
            out.append(il.detach().cpu().numpy())
            out.append(ir.detach().cpu().numpy())
        if not out:
            return np.array([], np.int64)
        return np.unique(np.concatenate(out)).astype(np.int64)

    def read(self, acc, block_s):
        vals = acc[self._flat].detach().cpu().numpy().astype(np.float64)
        out = {}
        for name, a, b, c in self._slices:
            nl, nr = self._n[name]
            out[name] = (float(vals[a:b].sum()) / (nl * block_s),
                         float(vals[b:c].sum()) / (nr * block_s))
        return out

    def counts(self):
        return {k: (len(v[0]), len(v[1])) for k, v in self._idx.items()}


# ------------------------------------------------------- команда телу


@dataclass
class MotorCommand:
    """Ровно два числа для ``HybridTurningController`` и всё, чем они вышли.

    ``amp_l``/``amp_r`` — амплитуды шага левых и правых трёх ног, то, что
    контроллер FlyGym ожидает как «нисходящий сигнал поворота».
    Остальное — для записи: по нему видно, какая популяция сдвинула
    команду и на сколько.
    """

    amp_l: float
    amp_r: float
    u_turn: float = 0.0
    u_drive: float = 0.0
    escape: float = 0.0
    z: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def as_array(self):
        return np.array([self.amp_l, self.amp_r], dtype=float)


@dataclass
class DecoderShape:
    """Общие числа формы команды. Перенесены из `BodyConfig` без правок."""

    amp_base: float = 0.9
    amp_span: float = 0.8
    amp_min: float = 0.1
    amp_max: float = 1.4
    cross: bool = True
    turn_sign: float = 1.0
    dn_tau_s: float = 0.5
    z_clip: float = 3.0


class Decoder:
    """Общий контракт всех декодеров. ИНЖЕНЕРНАЯ часть целиком.

    Жизненный цикл ровно как у прежнего кода `body.py`, чтобы прогоны
    остались сравнимыми:

    1. ``reset()``;
    2. на разогреве, при симметричном входе, по шагу —
       ``observe(readout, dt_s)``. Разогрев нужен затем, что коннектом
       не симметричен лево-право, и у любой популяции есть ненулевая
       покоящаяся асимметрия; её надо вычесть, иначе большое усиление
       превращает её в постоянный поворот и муха просто кружит;
    3. ``calibrate()`` — посчитать перекос и шум по второй половине
       разогрева и вернуть их словарём (он уезжает в снимок прогона);
    4. в эпизоде по шагу — ``command(readout, dt_s)``.

    ``spec()`` отдаёт блок для ``engineered``: имя, версия, все
    параметры, и, если декодер обучен, файл весов с sha256.
    """

    name = "?"
    version = 0
    kind = "motor_decoder"

    def __init__(self, shape: DecoderShape | None = None):
        self.shape = shape or DecoderShape()
        self.reset()

    # ---- служебное ----

    def reset(self):
        self._sm = {}
        self._warm_asym = {}
        self._warm_mean = {}
        self._warm_diff = {}
        self.bias = {}
        self.noise = {}
        self.base = {}
        self.base_noise = {}
        self.bias_hz = {}
        self.noise_hz = {}
        self._calibrated = False

    def reset_state(self):
        """Сбросить сглаживание, сохранив калибровку.

        Зовётся между разогревом и эпизодом: прежний код `body.py`
        ставил ``el = er = None`` ровно здесь, и фильтр в эпизоде
        начинался с первого измерения, а не с конца разогрева. Если это
        не воспроизвести, первые шаги эпизода пойдут иначе и эталонные
        прогоны разойдутся.
        """
        self._sm = {}

    def _smooth(self, readout, dt_s):
        k = math.exp(-dt_s / max(self.shape.dn_tau_s, 1e-6))
        for name, (l, r) in readout.items():
            prev = self._sm.get(name)
            self._sm[name] = ((l, r) if prev is None else
                              (k * prev[0] + (1 - k) * l,
                               k * prev[1] + (1 - k) * r))

    @staticmethod
    def _asym(l, r):
        return (l - r) / (l + r + 1e-9)

    def observe(self, readout, dt_s):
        """Шаг разогрева: сгладить и запомнить, как гуляет симметричный вход."""
        self._smooth(readout, dt_s)
        for name, (l, r) in self._sm.items():
            self._warm_asym.setdefault(name, []).append(self._asym(l, r))
            self._warm_mean.setdefault(name, []).append((l + r) / 2.0)
            self._warm_diff.setdefault(name, []).append(l - r)

    def calibrate(self):
        """Перекос и шум по ВТОРОЙ части разогрева, как и раньше.

        Первая треть отбрасывается: сглаживание с τ = 0,5 с за это время
        ещё не вышло из начального значения, и её дисперсия — не шум
        сети, а переходный процесс фильтра.
        """
        out = {}
        for name, vals in self._warm_asym.items():
            tail = vals[len(vals) // 3:]
            self.bias[name] = float(np.mean(tail)) if tail else 0.0
            self.noise[name] = (float(np.std(tail)) if len(tail) > 1
                                else 1e-4)
            m = self._warm_mean[name][len(vals) // 3:]
            self.base[name] = float(np.mean(m)) if m else 0.0
            self.base_noise[name] = (float(np.std(m)) if len(m) > 1
                                     else 1e-4)
            d = self._warm_diff[name][len(vals) // 3:]
            self.bias_hz[name] = float(np.mean(d)) if d else 0.0
            self.noise_hz[name] = float(np.std(d)) if len(d) > 1 else 0.0
            out[name] = {
                "перекос": round(self.bias[name], 6),
                "шум": round(self.noise[name], 6),
                "опора, Гц": round(self.base[name], 4),
                "шум опоры, Гц": round(self.base_noise[name], 4),
                "перекос, Гц": round(self.bias_hz[name], 4),
                "шум, Гц": round(self.noise_hz[name], 4),
            }
        self._calibrated = True
        return out

    def z_asym(self, name):
        """Асимметрия зонда, приведённая к [-1, 1]. Две нормировки.

        ``norm = "noise"`` — прежняя: нормировка на собственный шум
        популяции, измеренный на разогреве. Это та часть прежнего пути,
        которая была измерена, а не придумана, и для популяции в 341
        нейрон она работает.

        ``norm = "rate"`` — для прямых зондов, и вот почему она
        понадобилась. **Замер:** у зонда ``DNa02`` (1 нейрон на сторону)
        покоящаяся частота 0,00 Гц, значит и измеренный на разогреве шум
        его асимметрии равен нулю. Деление на ``max(шум, 1e-4)``
        превращает нормировку в компаратор: дозовая кривая выходила
        ступенькой — 0 Гц даёт ``cmd_turn`` 0, а уже 5,26 Гц (один спайк
        за блок после сглаживания) даёт сразу +0,556, и дальше до 200 Гц
        не меняется ни на единицу последнего знака. То есть «зависимость
        от активности» была бы двоичной, а это не декодер, а реле — ровно
        то, про что `world.py` предупреждает в комментарии к
        ``turn_gain_rad_s``.

        Поэтому у прямых зондов масштаб ОБЪЯВЛЕН в герцах:

            z = clip(((Гц слева − Гц справа) − перекос_Гц) / probe_ref_hz, ±1)

        ``probe_ref_hz = 100`` — это два спайка за блок 20 мс на одном
        нейроне (один спайк даёт 50 Гц). Число объявлено, а не измерено,
        и оно помечено как инженерное наравне с коэффициентами. Перекос
        при этом остаётся замером: он снимается на разогреве, в тех же
        герцах.
        """
        sm = self._sm.get(name)
        if sm is None:
            return 0.0
        if getattr(self, "norm", "noise") == "rate":
            d = (sm[0] - sm[1]) - self.bias_hz.get(name, 0.0)
            return float(np.clip(d / max(self.probe_ref_hz, 1e-9),
                                 -1.0, 1.0))
        a = self._asym(*sm) - self.bias.get(name, 0.0)
        z = a / max(self.noise.get(name, 1e-4), 1e-9)
        c = self.shape.z_clip
        return float(np.clip(z, -c, c)) / c

    def z_mean(self, name):
        """Общая (двусторонняя) часть зонда, приведённая к [-1, 1].

        Нормировка та же, что у :meth:`z_asym`, и по той же причине: у
        молчащего зонда шум общей части тоже равен нулю.
        """
        sm = self._sm.get(name)
        if sm is None:
            return 0.0
        m = (sm[0] + sm[1]) / 2.0 - self.base.get(name, 0.0)
        if getattr(self, "norm", "noise") == "rate":
            return float(np.clip(m / max(self.probe_ref_hz, 1e-9),
                                 -1.0, 1.0))
        z = m / max(self.base_noise.get(name, 1e-4), 1e-9)
        c = self.shape.z_clip
        return float(np.clip(z, -c, c)) / c

    def _shape_command(self, u_turn, u_drive, escape=0.0, z=None, detail=None):
        s = self.shape
        mid = s.amp_base * (1.0 + u_drive)
        mid *= (1.0 + escape)
        amp_l = mid - s.amp_span * s.turn_sign * u_turn
        amp_r = mid + s.amp_span * s.turn_sign * u_turn
        if not s.cross:
            amp_l, amp_r = amp_r, amp_l
        return MotorCommand(
            amp_l=float(np.clip(amp_l, s.amp_min, s.amp_max)),
            amp_r=float(np.clip(amp_r, s.amp_min, s.amp_max)),
            u_turn=float(u_turn), u_drive=float(u_drive),
            escape=float(escape), z=z or {}, detail=detail or {})

    # ---- переопределяется ----

    def command(self, readout, dt_s, escape=0.0) -> MotorCommand:
        """``escape`` — вклад инженерного моста через обрыв GF, 0 если его нет.

        Он приходит сюда готовым числом из :class:`GiantFiberBridge`, а
        не считается декодером: так видно, что это ОТДЕЛЬНАЯ инженерная
        подмена, а не часть декодера, и выключается она отдельно.
        """
        raise NotImplementedError

    def spec(self):
        return {"имя": self.name, "версия": self.version,
                "вид": self.kind,
                "параметры": {k: v for k, v in vars(self.shape).items()}}


class LegacyAsymDecoder(Decoder):
    """v0: то, что было. Среднее по ОДНОЙ популяции на полушарие.

    Сохранён дословно, а не «примерно»: эталонные прогоны обязаны
    считаться как раньше, а значит у прежней арифметики должен быть
    работающий код, а не описание. Читает один зонд (по умолчанию съём,
    выбранный в `BodyConfig.readout`) и делает в точности прежнее:

        сглаживание τ → asym − перекос → z = clip(asym/шум, ±3)
        → amp_l = base − (span/3)·z, amp_r = base + (span/3)·z.

    Почему этого было недостаточно, сказано числом, и число хуже, чем
    принято думать. При съёме по `super:descending` в среднем 1299
    нейронов, Giant Fiber — 2 из них, и его спайк от шума не отличается.
    Но съём по умолчанию в теле — не этот, а `BodyConfig.readout =
    "ALPN"`, и там ещё резче: `class:ALPN` — 685 нейронов, пересечение
    с `super:descending` **ноль**, с `DNp01` **ноль**, `super_class` у
    всех 685 равен `central`. То есть по умолчанию прежний путь ведёт
    ноги СЕНСОРНОЙ популяцией (проекционные нейроны антеннальной доли),
    в которой нет ни одного премоторного нейрона, и Giant Fiber не
    «растворён в среднем», а отсутствует в съёме полностью. Замер —
    `probe/motor_audit.py`.

    **Внимание:** этот декодер НЕ читает ``shape.turn_sign``. Амплитуды
    считаются дословно прежним выражением, иначе эпизод перестал бы
    совпадать с эталонным побитово. То есть `BodyConfig.turn_sign` при
    `motor = "legacy"` (значение по умолчанию) не делает ничего, и это
    оставлено нарочно: применить его здесь значило бы сломать шлюз.
    """

    name = "legacy_asym"
    version = 0

    def __init__(self, shape=None, readout_name="__readout__"):
        self.readout_name = readout_name
        super().__init__(shape)

    def command(self, readout, dt_s, escape=0.0):
        self._smooth(readout, dt_s)
        s = self.shape
        sm = self._sm.get(self.readout_name)
        a = self._asym(*sm) - self.bias.get(self.readout_name, 0.0)
        # ВНИМАНИЕ: порядок умножений здесь не косметика. Прежний код
        # считал `gain = amp_span / 3.0`, а потом `amp_base - gain * z`.
        # Если вместо этого посчитать `amp_base - amp_span * (z / 3.0)`,
        # результат разойдётся в последнем бите мантиссы — и этого
        # хватает: замер `probe/motor_equal.py` показал расхождение
        # траектории уже к сотому шагу (конечная поза 2,481/1,433 против
        # 2,112/2,134, выравнивание 0,301 против 0,231). Ходьба сама
        # раскачивает один бит до миллиметров, поэтому выражение
        # перенесено дословно, а не «эквивалентно».
        z = float(np.clip(a / max(self.noise.get(self.readout_name, 1e-4),
                                  1e-9), -3.0, 3.0))
        gain = s.amp_span / 3.0
        amp_l = s.amp_base - gain * z
        amp_r = s.amp_base + gain * z
        if escape:
            amp_l *= (1.0 + escape)
            amp_r *= (1.0 + escape)
        if not s.cross:
            amp_l, amp_r = amp_r, amp_l
        return MotorCommand(
            amp_l=float(np.clip(amp_l, s.amp_min, s.amp_max)),
            amp_r=float(np.clip(amp_r, s.amp_min, s.amp_max)),
            u_turn=z / 3.0, u_drive=0.0, escape=float(escape),
            z={self.readout_name: z / 3.0}, detail={"z_raw": z, "asym": a})

    def spec(self):
        d = super().spec()
        d["съём"] = self.readout_name
        d["происхождение"] = ("перенесено из `body.py` без изменения "
                              "арифметики; числа объявлены вручную")
        return d


# Коэффициенты линейной карты, версия 1. ЗНАКИ взяты из опубликованных
# ролей (`PROBE_ROLES`), ВЕЛИЧИНЫ объявлены здесь и ничем не измерены.
# Это главная инженерная вольность декодера, и она стоит отдельным
# словарём именно затем, чтобы её нельзя было не заметить.
#
# Почему DNa02 весит больше всех: это единственный нисходящий, у
# которого опубликована количественная связь активности с угловой
# скоростью поворота. DNa01 вдвое меньше: та же роль, но медленнее и
# градуальнее. DNp09 ещё меньше: поворот у него не единственный эффект.
# MDN, DNp01 и DNp04 в повороте не участвуют вовсе — ноль здесь не
# «ещё не настроено», а утверждение.
TURN_WEIGHTS_V1 = {
    "DNa02": 1.0,
    "DNa01": 0.5,
    "DNp09": 0.3,
    "MDN": 0.0,
    "DNp01": 0.0,
    "DNp04": 0.0,
}

# Тяга (двусторонняя часть). MDN — задний ход, поэтому знак минус:
# двусторонний рост MDN уменьшает амплитуду шага вперёд. DNp09 —
# остановка, минус поменьше. Рулевым здесь делать нечего.
DRIVE_WEIGHTS_V1 = {
    "MDN": -0.5,
    "DNp09": -0.2,
    "DNa02": 0.0,
    "DNa01": 0.0,
    "DNp01": 0.0,
    "DNp04": 0.0,
}


class DnLinearDecoder(Decoder):
    """v1, ВАРИАНТ А: линейная карта с ОБЪЯВЛЕННЫМИ коэффициентами.

    Контракт входа
    --------------
    ``{имя зонда: (Гц слева, Гц справа)}`` за один блок управления
    (20 мс) по шести прямым зондам: DNp01, DNp04, DNa01, DNa02, DNp09,
    MDN. Каждый — 1 или 2 нейрона на сторону, селекторы точные.

    Контракт выхода
    ---------------
    :class:`MotorCommand` с двумя амплитудами в ``[amp_min, amp_max]``.
    Ровно два числа: на «ровно двух» держатся поля записи, ряды,
    отпечатки, графики и отчёт, и ломать это ради декодера нельзя.

    Арифметика
    ----------
    **По умолчанию ``norm = "rate"``, и считается вот это:**

        z_i   = clip(((Гц слева_i − Гц справа_i) − перекос_Гц_i)
                     / probe_ref_hz, ±1),     probe_ref_hz = 100
        u_turn  = Σ w_i·z_i / Σ|w_i|
        u_drive = Σ g_j·m_j / Σ|g_j| · drive_span
        amp = amp_base·(1 + u_drive) ∓ amp_span·turn_sign·u_turn

    Масштаб здесь ОБЪЯВЛЕН (``probe_ref_hz`` — два спайка за блок 20 мс
    на одном нейроне), а перекос остаётся замером на разогреве.

    При ``norm = "noise"`` вместо первой строки берётся прежняя форма v0,
    ``z_i = clip((асимметрия_i − перекос_i)/шум_i, ±3)/3``, то есть
    нормировка на ИЗМЕРЕННЫЙ шум популяции. Для прямых зондов она не
    годится, и это замер, а не вкус: у зонда в один нейрон на сторону
    покой 0,00 Гц, измеренный шум асимметрии равен нулю, и деление на
    ``max(шум, 1e-4)`` превращает нормировку в компаратор — дозовая
    кривая выходит ступенькой. Подробности и числа — в
    :meth:`Decoder.z_asym`.

    Проверено, что при ``rate`` кривая градуальна: 11 различных значений
    команды на 12 точках стимула, 92,7 % размаха даёт середина диапазона
    активности (`probe/motor_grade.py`). Потолок при разнице сторон
    100 Гц обрезает верх диапазона, и это следствие объявленного
    ``probe_ref_hz``.

    Откуда числа
    ------------
    ``w`` — :data:`TURN_WEIGHTS_V1`, ``g`` — :data:`DRIVE_WEIGHTS_V1`.
    Знаки из литературы (:data:`PROBE_ROLES`), величины объявлены.
    Перекос и шум — замер на разогреве этого же прогона.

    Чего этот декодер НЕ утверждает
    -------------------------------
    Что ``turn_sign = +1`` правильный. Соответствие «ипси/контра» между
    опубликованной разметкой и аннотацией v783 не установлено:
    известно, что между апстримной разметкой и нашей лево и право
    систематически перевёрнуты (``P9_left`` у них — ``DNp09`` со
    ``side='right'`` у нас, и так же ``DNa01``, ``DNa02``). Поэтому
    ``turn_sign`` остаётся объявленным параметром со значением по
    умолчанию как у v0, а не выводом.
    """

    name = "dn_linear"
    version = 1

    def __init__(self, shape=None, turn_weights=None, drive_weights=None,
                 drive_span=0.3, norm="rate", probe_ref_hz=100.0):
        self.turn_weights = dict(turn_weights or TURN_WEIGHTS_V1)
        self.drive_weights = dict(drive_weights or DRIVE_WEIGHTS_V1)
        self.drive_span = float(drive_span)
        # По умолчанию `rate`, и это решение по замеру, а не по вкусу:
        # у зонда в один нейрон покой 0,00 Гц, измеренный шум нулевой, и
        # нормировка на него вырождается в компаратор. Подробности и
        # числа — в :meth:`Decoder.z_asym`.
        self.norm = norm
        self.probe_ref_hz = float(probe_ref_hz)
        super().__init__(shape)

    def command(self, readout, dt_s, escape=0.0):
        self._smooth(readout, dt_s)
        zs = {k: self.z_asym(k) for k in self.turn_weights}
        ms = {k: self.z_mean(k) for k in self.drive_weights}
        wsum = sum(abs(v) for v in self.turn_weights.values()) or 1.0
        gsum = sum(abs(v) for v in self.drive_weights.values()) or 1.0
        u_turn = sum(self.turn_weights[k] * zs[k] for k in zs) / wsum
        u_drive = (sum(self.drive_weights[k] * ms[k] for k in ms) / gsum
                   * self.drive_span)
        return self._shape_command(
            u_turn=float(np.clip(u_turn, -1.0, 1.0)),
            u_drive=float(np.clip(u_drive, -self.drive_span,
                                  self.drive_span)),
            escape=escape, z=zs, detail={"m": ms})

    def spec(self):
        d = super().spec()
        d["вес поворота"] = dict(self.turn_weights)
        d["вес тяги"] = dict(self.drive_weights)
        d["параметры"]["drive_span"] = self.drive_span
        d["параметры"]["norm"] = self.norm
        d["параметры"]["probe_ref_hz"] = self.probe_ref_hz
        d["происхождение"] = (
            "знаки коэффициентов — опубликованные роли нисходящих "
            "нейронов; величины объявлены вручную и ничем не измерены; "
            "перекос и шум каждого зонда — замер на разогреве этого "
            "прогона")
        d["роли"] = {k: PROBE_ROLES.get(k, "") for k in self.turn_weights}
        d["не утверждается"] = (
            "соответствие ипси/контра между опубликованной разметкой и "
            "аннотацией v783: turn_sign объявлен, а не выведен")
        return d


class DnTrainedDecoder(Decoder):
    """v1, ВАРИАНТ Б: обученный линейный декодер по тем же шести зондам.

    Контракт входа и выхода тот же, что у варианта А, и это условие
    задачи: два варианта обязаны быть взаимозаменяемы, иначе выбор между
    ними — не выбор, а переделка.

    Чем отличается
    --------------
    ``u_turn`` считается не объявленными коэффициентами, а весами из
    файла:

        u_turn = clip(w·z + b, ±1),

    где ``z`` — тот же вектор нормированных асимметрий, что у варианта А
    (по умолчанию ``norm = "rate"``, то есть масштаб в герцах с
    объявленным ``probe_ref_hz``, а НЕ нормировка на измеренный шум), в
    ЗАФИКСИРОВАННОМ порядке зондов из файла. Нормировка — часть
    обучения: веса подогнаны под тот масштаб ``z``, в котором собиралась
    выборка, поэтому ``norm`` и ``probe_ref_hz`` читаются из файла
    весов, если они там записаны.

    На чём обучен
    -------------
    На собственных прогонах LIF БЕЗ тела: сеть гоняется заданным
    расписанием асимметрии обонятельного входа, и гребневая регрессия
    ищет веса, по которым вектор ``z`` предсказывает ЗНАЕМУЮ асимметрию
    раздражителя. То есть это декодер стимула из популяции в обычном
    нейробиологическом смысле: «что нисходящие знают о том, какая
    антенна в шлейфе».

    Чем это опасно, и это надо сказать прямо
    ----------------------------------------
    Цель обучения — раздражитель. Значит декодер подогнан под связь
    «вход сети → активность зондов → вход сети», и его хорошая точность
    сама по себе НЕ доказывает, что команда идёт из сети: ровно так же
    он выучил бы любой случайный коррелят входа. Доказывает это только
    встречный замер с маской на съёмную популяцию (см. `MOTOR.md` §3):
    при заглушённых зондах вектор ``z`` постоянен, и выход декодера
    постоянен при ЛЮБЫХ весах. Поэтому встречный замер обязателен для
    обоих вариантов, а не только для обученного.

    Кроме того, файл весов — это НОВЫЙ вход опыта. Он версионируется как
    семя: имя, версия, sha256 и список зондов едут в ``engineered``, и
    прогон с другим файлом весов сравнивать с прежними нельзя.
    """

    name = "dn_trained"
    version = 1

    def __init__(self, shape=None, weights_path=None, norm="rate",
                 probe_ref_hz=100.0):
        # Нормировка входа — часть ОБУЧЕНИЯ: веса подогнаны под тот
        # масштаб `z`, в котором собиралась выборка, и менять его после
        # обучения нельзя. Поэтому значения берутся из файла весов, если
        # они там есть, а аргументы остаются только для первого обучения.
        self.norm = norm
        self.probe_ref_hz = float(probe_ref_hz)
        if weights_path is None:
            raise ValueError(
                "обученному декодеру нужен файл весов: он и есть его "
                "содержимое. Собрать — `probe/motor_train.py`")
        self.path = Path(weights_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path}: файла весов нет. Собрать — "
                "`probe/motor_train.py`")
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        z = np.load(self.path, allow_pickle=False)
        self.probe_order = [str(s) for s in z["probe_order"]]
        self.w = np.asarray(z["w"], dtype=float)
        self.b = float(np.asarray(z["b"]).reshape(-1)[0])
        self.train_meta = {k: z[k].tolist() for k in z.files
                           if k.startswith("meta_")}
        if "meta_norm" in z.files:
            self.norm = str(np.asarray(z["meta_norm"]).reshape(-1)[0])
        if "meta_probe_ref_hz" in z.files:
            self.probe_ref_hz = float(
                np.asarray(z["meta_probe_ref_hz"]).reshape(-1)[0])
        if len(self.w) != len(self.probe_order):
            raise ValueError(
                "%s: весов %d, а зондов в файле %d — файл испорчен"
                % (self.path, len(self.w), len(self.probe_order)))
        super().__init__(shape)

    def command(self, readout, dt_s, escape=0.0):
        self._smooth(readout, dt_s)
        zs = {k: self.z_asym(k) for k in self.probe_order}
        vec = np.array([zs[k] for k in self.probe_order], dtype=float)
        u_turn = float(np.clip(float(self.w @ vec) + self.b, -1.0, 1.0))
        return self._shape_command(u_turn=u_turn, u_drive=0.0,
                                   escape=escape, z=zs)

    def spec(self):
        d = super().spec()
        d["файл весов"] = str(self.path)
        d["sha256"] = self.sha256
        d["параметры"]["norm"] = self.norm
        d["параметры"]["probe_ref_hz"] = self.probe_ref_hz
        d["порядок зондов"] = list(self.probe_order)
        d["веса"] = [round(float(v), 6) for v in self.w]
        d["смещение"] = round(self.b, 6)
        d["обучение"] = self.train_meta
        d["происхождение"] = (
            "гребневая регрессия на собственных прогонах LIF без тела; "
            "цель обучения — известная асимметрия обонятельного входа, "
            "а не чьи-то измеренные суставы. Файл весов — новый вход "
            "опыта и версионируется как семя")
        d["предупреждение"] = (
            "цель обучения — раздражитель, поэтому точность декодера НЕ "
            "доказывает, что команда идёт из сети. Это доказывает только "
            "встречный замер с маской на съёмную популяцию")
        return d


DECODERS = {
    "legacy": LegacyAsymDecoder,
    "dn_linear": DnLinearDecoder,
    "dn_trained": DnTrainedDecoder,
}


def make_decoder(name, shape=None, **kw):
    if name not in DECODERS:
        raise ValueError("нет декодера %r; есть: %s"
                         % (name, ", ".join(DECODERS)))
    return DECODERS[name](shape=shape, **kw)


# ------------------------------------------- обрыв GF -> мотонейроны


class GiantFiberBridge:
    """Мост через обрыв «Giant Fiber → мотонейроны». ИНЖЕНЕРНАЯ ПОДМЕНА.

    Положение дел, измеренное, а не предполагаемое
    ----------------------------------------------
    ``DNp01 → super:motor`` — **ноль** пар из 110 мотонейронов. Через
    один шаг — 82 пары со знаковой суммой **−276**, тормозной, и только
    к головным и шейным мотонейронам. Причина физическая: ``super:motor``
    в v783 — это 110 нейронов, все в ``NERVE`` (глотание 40, хоботок 26,
    шея 20, антенны 14, прочее 10). Мотонейронов ног и крыльев в наборе
    НЕТ: FlyWire v783 — мозг без брюшной нервной цепочки, а исполнители
    побега (TTMn, PSI) лежат в VNC.

    Решение: по умолчанию моста НЕТ
    -------------------------------
    ``kind = "none"``. Активность DNp01 измеряется прямым зондом и
    записывается (``readout_DNp01_*``), к телу не идёт ничего. Основание:
    недостающее звено — не пробел в аннотации, который можно восполнить
    точнее, а отсутствующая в наборе данных часть нервной системы. Любая
    подстановка — объявление, а объявление, молча меняющее поведение,
    хуже отсутствующего поведения: прогон посчитается, муха дёрнется, и
    в отчёте это будет выглядеть реакцией коннектома.

    Что при этом НЕ теряется: единственное, к чему DNp01 в v783
    действительно доходит, — шейные мотонейроны, и притом тормозно. Это
    измеряется теневым зондом ``motor_neck`` и попадает в запись. То
    есть биологическое следствие спайка GF в этом наборе данных не
    выдумывается и не замалчивается, а измеряется.

    Если мост всё же нужен
    ----------------------
    ``kind = "gain_v1"``: непрерывный множитель амплитуды шага,

        escape = clip(Гц(DNp01) / gf_ref_hz, 0, 1)
        amp   *= (1 + escape_gain · escape)

    и ничего больше. Это НЕ анимация побега: ни позы, ни таймера, ни
    последовательности — только множитель, равный нулю, когда GF молчит,
    и считанный из частоты ЭТОГО шага. Числа ``gf_ref_hz`` и
    ``escape_gain`` объявлены; биологического основания у их величин нет,
    у знака — тоже нет, потому что соответствующего пути в данных нет.
    Включённый мост пишет оговорку в ``timing["оговорки"]``, и прогон с
    ним с эталонными не сравнивается.
    """

    KINDS = ("none", "gain_v1")
    kind_version = {"none": 0, "gain_v1": 1}

    def __init__(self, kind="none", gf_ref_hz=100.0, escape_gain=0.5,
                 probe="DNp01"):
        if kind not in self.KINDS:
            raise ValueError("нет моста %r; есть: %s"
                             % (kind, ", ".join(self.KINDS)))
        self.kind = kind
        self.gf_ref_hz = float(gf_ref_hz)
        self.escape_gain = float(escape_gain)
        self.probe = probe

    @property
    def active(self):
        return self.kind != "none"

    def escape(self, readout):
        """Во сколько инженерная команда умножит амплитуду. 0 — ни во сколько."""
        if not self.active:
            return 0.0
        l, r = readout.get(self.probe, (0.0, 0.0))
        gf = max(l, r)
        e = float(np.clip(gf / max(self.gf_ref_hz, 1e-9), 0.0, 1.0))
        return self.escape_gain * e

    def spec(self):
        d = {
            "имя": "gf_bridge",
            "версия": self.kind_version[self.kind],
            "вид": self.kind,
            "зонд": self.probe,
            "параметры": {"gf_ref_hz": self.gf_ref_hz,
                          "escape_gain": self.escape_gain},
        }
        if not self.active:
            d["происхождение"] = (
                "моста нет: DNp01 → super:motor в v783 это ноль пар из "
                "110 мотонейронов, а мотонейронов ног и крыльев в наборе "
                "нет физически (мозг без брюшной нервной цепочки). "
                "Активность DNp01 измеряется и записывается, к телу не "
                "идёт ничего")
        else:
            d["происхождение"] = (
                "ИНЖЕНЕРНАЯ ПОДМЕНА отсутствующего в наборе данных пути "
                "GF → мотонейроны ног и крыльев. Непрерывный множитель "
                "амплитуды от измеренной частоты DNp01; величины "
                "gf_ref_hz и escape_gain объявлены и ничем не измерены. "
                "Заранее запрограммированной последовательности побега "
                "здесь нет")
        return d

    def caveat(self):
        if not self.active:
            return None
        return (
            "включён инженерный мост `gf_bridge/%s` через обрыв "
            "«Giant Fiber → мотонейроны». Биологического пути нет: "
            "DNp01 → super:motor в v783 — ноль пар из 110 мотонейронов, "
            "мотонейронов ног и крыльев в наборе нет (мозг без брюшной "
            "нервной цепочки, TTMn и PSI лежат в VNC). Множитель "
            "амплитуды escape_gain=%g при опоре %g Гц объявлен вручную. "
            "С эталонными прогонами такой эпизод не сравнивается."
            % (self.kind, self.escape_gain, self.gf_ref_hz))


# ------------------------------------------------- стабилизация головы


HEAD_STAB_DIR = ("flygym_gymnasium", "data", "trained_models",
                 "head_stabilization")


def head_stab_paths():
    """Где лежат опубликованные веса. Ищутся в установленном пакете."""
    import flygym_gymnasium

    base = Path(flygym_gymnasium.__file__).parent / "data" \
        / "trained_models" / "head_stabilization"
    return base / "all_dofs_model.ckpt", base / "joint_angle_scaler_params.pkl"


class HeadStabilizer:
    """Опубликованная обученная стабилизация головы. ИНЖЕНЕРНАЯ ДЕТАЛЬ ТЕЛА.

    Что это
    -------
    MLP ``Linear(48→32) → ReLU → Linear(32→32) → ReLU → Linear(32→2)``,
    около 2600 параметров: вход — 42 угла суставов ног и 6 двоичных
    признаков контакта лап с полом, выход — целевые крен и тангаж шеи.
    Веса опубликованы авторами NeuroMechFly v2 и **уже лежат в нашем
    venv**, в ``flygym_gymnasium/data/trained_models/head_stabilization``.
    Работает на процессоре.

    Почему это инженерная деталь, а не часть мозга
    ----------------------------------------------
    Это обученный контроллер, обходящий коннектом целиком: от
    проприоцепции до приводов шеи здесь обученные веса, ни одного
    синапса FlyWire. В отчётах он обязан стоять рядом с ``fast_physics``,
    а не рядом с LIF. Основание включать его всё-таки: по опубликованному
    замеру без стабилизации головы слежение за объектом на неровном
    рельефе ломается, а жёсткость шейных суставов 10 — это просто
    значение FlyGym по умолчанию, то есть «голова живёт как пружина» —
    незаданная настройка, а не свойство модели.

    Переходник без новых зависимостей
    ---------------------------------
    Штатный загрузчик (``HeadStabilizationInferenceWrapper``) тянет
    ``lightning`` и ``torchmetrics``, которых в нашем venv нет и ставить
    их запрещено. Поэтому ``state_dict`` читается прямо ``torch.load``
    и заливается в свою трёхслойную сеть. Предобработка перенесена из
    штатной обёртки ДОСЛОВНО, и это важнее, чем кажется: углы
    нормируются сохранёнными ``mean``/``std``, силы контакта берутся по
    норме, суммируются по шести сегментам на ногу и сравниваются с
    порогом ``(0.5, 1, 3)``, продублированным на шесть ног.

    Ловушка имён, на которой легко сломаться
    ----------------------------------------
    Шейная степень свободы **крена** в модели называется
    ``joint_Head_yaw``. Имя оставлено авторами для совместимости, и
    отображение ``[крен, тангаж] → [joint_Head_yaw, joint_Head]`` в
    FlyGym — не описка. Мы сюда не вмешиваемся вовсе: порядок двух
    выходных чисел задаёт ``Fly.pre_step``, наш переходник только
    отдаёт их в том же порядке, в каком отдавала штатная обёртка.
    """

    name = "head_stabilization"
    version = 1
    # Опубликованные значения: 500 в работе про слежение на коннектомной
    # модели зрительной доли, 1000 в зрительном таксисе. Берём 500 —
    # меньшее из двух опубликованных, и это объявлено, а не подобрано.
    default_neck_kp = 500.0

    def __init__(self, neck_kp=None, contact_force_thr=(0.5, 1.0, 3.0)):
        import pickle

        import torch

        self.neck_kp = float(self.default_neck_kp if neck_kp is None
                             else neck_kp)
        ckpt, scaler = head_stab_paths()
        if not ckpt.exists() or not scaler.exists():
            raise FileNotFoundError(
                "опубликованных весов стабилизации головы нет рядом с "
                "flygym_gymnasium: %s" % ckpt)
        self.ckpt_path, self.scaler_path = ckpt, scaler
        raw_c, raw_s = ckpt.read_bytes(), scaler.read_bytes()
        self.ckpt_sha256 = hashlib.sha256(raw_c).hexdigest()
        self.scaler_sha256 = hashlib.sha256(raw_s).hexdigest()
        self.ckpt_bytes = len(raw_c)

        # Переходник. `lightning` не нужен: в файле обычный
        # `state_dict`, и имена слоёв совпадают с нашими.
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = obj["state_dict"] if "state_dict" in obj else obj
        sd = {k: v for k, v in sd.items() if k.startswith("layer")}
        self.epoch = int(obj.get("epoch", -1)) if isinstance(obj, dict) else -1

        net = torch.nn.Sequential(
            torch.nn.Linear(48, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 2))
        mapping = {"layer1.weight": "0.weight", "layer1.bias": "0.bias",
                   "layer2.weight": "2.weight", "layer2.bias": "2.bias",
                   "layer3.weight": "4.weight", "layer3.bias": "4.bias"}
        missing = sorted(set(mapping) - set(sd))
        if missing:
            raise ValueError("в файле весов нет слоёв: %s" % ", ".join(missing))
        net.load_state_dict({mapping[k]: sd[k] for k in mapping})
        net.eval()
        self.net = net
        self.n_params = int(sum(p.numel() for p in net.parameters()))

        p = pickle.loads(raw_s)
        self.mean = np.asarray(p["mean"], dtype=np.float64)
        self.std = np.asarray(p["std"], dtype=np.float64)
        self.thr = np.asarray([*contact_force_thr, *contact_force_thr],
                              dtype=np.float64)
        self.calls = 0
        self._last = np.zeros(2)

    def __call__(self, joint_angles, contact_forces):
        """Контракт FlyGym: (42 угла, (36, 3) силы) → (крен, тангаж), рад."""
        import torch

        a = (np.asarray(joint_angles, dtype=np.float64) - self.mean) / self.std
        f = np.linalg.norm(np.asarray(contact_forces, dtype=np.float64),
                           axis=1)
        f = f.reshape(6, 6).sum(axis=1)
        x = np.concatenate([a, f >= self.thr]).astype(np.float32)
        with torch.no_grad():
            out = self.net(torch.from_numpy(x[None, :])).numpy().squeeze()
        self.calls += 1
        self._last = np.asarray(out, dtype=float)
        return self._last

    def spec(self):
        return {
            "имя": self.name,
            "версия": self.version,
            "вид": "trained_mlp",
            "файл весов": str(self.ckpt_path),
            "sha256": self.ckpt_sha256,
            "байт": self.ckpt_bytes,
            "файл нормировки": str(self.scaler_path),
            "sha256 нормировки": self.scaler_sha256,
            "параметров": self.n_params,
            "сеть": "Linear(48->32) ReLU Linear(32->32) ReLU Linear(32->2)",
            "параметры": {"neck_kp": self.neck_kp,
                          "contact_force_thr": self.thr[:3].tolist()},
            "происхождение": (
                "опубликованные веса NeuroMechFly v2, уже лежат в venv; "
                "загружены переходником через torch.load без lightning и "
                "torchmetrics. Предобработка перенесена из штатной "
                "обёртки дословно"),
            "это не мозг": (
                "обученный контроллер, обходящий коннектом: от "
                "проприоцепции до приводов шеи ни одного синапса "
                "FlyWire"),
        }

    def caveat(self):
        return (
            "включена опубликованная ОБУЧЕННАЯ стабилизация головы "
            "(`head_stabilization` v%d, neck_kp=%g, sha256 весов %s…). "
            "Это инженерная деталь тела, обходящая коннектом: от "
            "проприоцепции ног до приводов шеи работают обученные веса "
            "NeuroMechFly v2, ни одного синапса FlyWire. Приводы на шею "
            "добавляются только при включённой стабилизации, поэтому "
            "модель тела у такого эпизода ДРУГАЯ, и с эталонными "
            "прогонами он не сравнивается."
            % (self.version, self.neck_kp, self.ckpt_sha256[:12]))


# --------------------------------------------------- блок `engineered`


def engineered_block(decoder=None, bridge=None, head_stab=None,
                     readout_rows=None, extra=None):
    """Собрать блок ``engineered`` прогона: что нефизиологично и какой версии.

    Кладётся в снимок прогона рядом с ``timing`` и ``calibration``.
    Читается глазами и машинно: по ``имя`` + ``версия`` + ``sha256``
    видно, сравним ли прогон с другим, не разбирая кода.
    """
    out = {"версия моторного пути": MOTOR_VERSION, "детали": []}
    for obj in (decoder, bridge, head_stab):
        if obj is None:
            continue
        out["детали"].append(obj.spec())
    if readout_rows is not None:
        out["прямые зонды"] = readout_rows
    if extra:
        out.update(extra)
    return out
