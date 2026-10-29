"""Ускоренный контроллер походки: вендоренная копия шага FlyGym.

ПРОИСХОЖДЕНИЕ. Код шага взят из `flygym_gymnasium` 1.3.2,
`flygym_gymnasium/examples/locomotion/turning_controller.py`,
класс ``HybridTurningController`` (FlyGym / NeuroMechFly v2). Лицензия
апстрима — **Apache License 2.0**; копия сделана по её условиям, с
указанием происхождения и перечнем изменений (ниже). **Установленный
пакет не изменён**: здесь подкласс, который переопределяет только
``step``, а всё остальное — поля, правила, пространства действий, сброс —
достаётся по наследству из пакета.

ЗАЧЕМ. В разложении шага управления контроллера не видно: он зовётся из
того же цикла, что и физика, и прячется внутри строки «физика». Замер
(`probe/ctrl_prof.py`) выделил его отдельной строкой — и это не мелочь:
на шаг управления 20 мс (100 шагов физики) питоновская часть контроллера
стоит столько же, сколько сам `mj_step`.

ЧТО ИЗМЕНЕНО, три правки, и все три — об одном и том же: вместо
шестикратного повторения одного и того же питоновского кода делается
один вызов на все шесть ног.

1. **Повторное наблюдение не читается заново.** Стоковый шаг начинается
   с ``super().get_observation()`` — полного наблюдения (сенсоры суставов,
   поза, кватернион через `scipy.Rotation`, силы контактов с поправкой на
   прилипание, положения по именам, запах). Но предыдущий шаг уже
   закончился ровно этим же вычислением в ``Fly.post_step``, и между ними
   состояние MuJoCo не меняется: `mj_forward` внутри того наблюдения снял
   флаг «грязного» состояния (`dm_control` зовёт его на первом же чтении
   через биндинг), а дальше до следующего `mj_step` никто в `mjData` не
   пишет. Поэтому здесь берётся ТОТ ЖЕ объект наблюдения — не копия и не
   пересчёт, а он сам. Если состояние всё-таки помечено грязным (живая
   правка через биндинг), наблюдение читается заново, как в стоке.
   Оговорка, за которой стоит замер, а не рассуждение: пассивное окно
   MuJoCo умеет двигать состояние мышью мимо `dm_control`, и при
   открытом окне повторное чтение оставлено (`reuse_obs=False`, это
   ставит `body.py`).

2. **Кусочный полином один на шесть ног.** ``PreprogrammedSteps`` держит
   по кубическому сплайну на ногу, и шаг зовёт их по одному: 6 вызовов на
   шаг физики, 600 на шаг управления, и почти всё время уходит на
   питоновскую обвязку `PPoly.__call__`, а не на сам полином. Здесь
   коэффициенты шести сплайнов сложены в один ``PPoly`` (узлы у них
   общие — `np.linspace(0, 2π, L)`), и вызов один. Числа те же по
   построению: это те же коэффициенты, те же узлы и тот же код scipy,
   просто столбцов в нём 42 вместо 7. Проверено побитово на 210 000
   значениях.

3. **Правила и фазовый множитель — массивами.** Правило спотыкания,
   правило втягивания, поправки, фазовый множитель (`interp1d`,
   линейная интерполяция) и прилипание считаются на все шесть ног сразу.
   Ноги в стоковом цикле между собой не связаны — каждая читает и пишет
   только свой элемент, — поэтому порядок действий сохраняется. Линейная
   интерполяция выписана теми же выражениями, что в
   `scipy.interpolate.interp1d._call_linear`, и сверена с ней побитово на
   120 000 значениях; проекция сил контактов считается одним `np.dot` на
   18 строк вместо шести по три — тоже сверено побитово (20 000 наборов).

Чего здесь НЕТ. Ни одна частота обновления не понижена, ни один шаг не
пропущен, ни одна величина не заменена приближением: по шагам и по
числам это тот же контроллер. Сверка — `probe/gait_equal.py` (та же
арифметика на случайных входах) и `probe/body_equal.py` (отпечатки
эпизода целиком).

Состояние для продолжения из снимка (`flybrain.replay`) здесь то же, что
у стока: фазы и амплитуды живут в том же ``CPGNetwork`` и пишутся в него
на месте, счётчики поправок — те же три массива на самом контроллере.
Ничего нового в состояние не добавлено, и ничего из него не ушло.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build", "controller_class"]

TWO_PI = 2 * np.pi


def _base():
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    return HybridTurningController


def build(**kwargs):
    """Собрать контроллер: ``gait="fast"`` — здешний, иначе стоковый."""
    gait = kwargs.pop("gait", "fast")
    reuse_obs = kwargs.pop("reuse_obs", True)
    if gait == "legacy":
        return _base()(**kwargs)
    if gait != "fast":
        raise ValueError("gait должен быть 'fast' или 'legacy', а не %r"
                         % (gait,))
    return controller_class()(reuse_obs=reuse_obs, **kwargs)


def _make_class():
    from flygym_gymnasium.simulation import SingleFlySimulation

    Base = _base()

    class FastHybridTurningController(Base):
        """``HybridTurningController``, у которого шаг считается массивами."""

        def __init__(self, *args, reuse_obs=True, **kwargs):
            super().__init__(*args, **kwargs)
            # Раскраска ног в видео зовёт `change_segment_color` из той
            # самой ветки, которую здесь нет: при ней работает стоковый
            # шаг, иначе картинка молча перестала бы помечать поправки.
            self._fast_ok = not self.draw_corrections
            self.reuse_obs = reuse_obs
            self._obs_cache = None
            self._tables()

        # --- таблицы: всё, что в стоковом шаге пересчитывается на ногу ---

        def _tables(self):
            from scipy.interpolate import PPoly

            ps = self.preprogrammed_steps
            legs = ps.legs
            self._legs = list(legs)
            n = len(legs)
            self._rows = np.arange(n)

            # Один кусочный полином на все шесть ног. Узлы обязаны быть
            # общими — иначе сложить коэффициенты нельзя, и это
            # проверяется, а не предполагается.
            knots = ps._psi_funcs[legs[0]].x
            for leg in legs:
                if not np.array_equal(ps._psi_funcs[leg].x, knots):
                    raise ValueError(
                        "у ноги %s свои узлы сплайна: сложить шесть "
                        "сплайнов в один нельзя" % leg)
                if ps._psi_funcs[leg].extrapolate != "periodic":
                    raise ValueError(
                        "сплайн ноги %s не периодический: продолжение за "
                        "период считалось бы иначе" % leg)
            c = np.concatenate([ps._psi_funcs[leg].c for leg in legs],
                               axis=2)
            self._ndof = ps._psi_funcs[legs[0]].c.shape[2]
            self._psi = PPoly.construct_fast(c, knots,
                                             extrapolate="periodic")

            self._neutral = np.array(
                [np.asarray(ps.neutral_pos[leg]).ravel() for leg in legs],
                dtype=np.float64)
            self._corr_vec = np.array(
                [self.correction_vectors[leg[1]] for leg in legs],
                dtype=np.float64)
            # `swing_period` читается ПОСЛЕ `_init_phasic_gain`: тот
            # удлиняет окно переноса на π/4 прямо в этом словаре, и
            # `PreprogrammedSteps._swing_end_arr` помнит ещё прежнее.
            self._swing = np.array([ps.swing_period[leg] for leg in legs],
                                   dtype=np.float64)
            # Инверсия правых ног. Индексируется НОМЕРОМ НОГИ, а не
            # номером степени свободы — так в апстриме, и так же здесь:
            # значение берётся из того же списка тем же индексом.
            inv = np.ones(n, dtype=np.float64)
            for i, leg in enumerate(legs):
                if leg[0] == "R":
                    inv[i] = self.right_leg_inversion[i]
            self._inv = inv

            # Фазовый множитель: узлы и значения тех же `interp1d`.
            fx = [np.asarray(self.phasic_multiplier[leg].x, dtype=np.float64)
                  for leg in legs]
            if len({len(v) for v in fx}) != 1:
                raise ValueError("у фазовых множителей разное число узлов")
            self._px = np.array(fx)
            self._py = np.array(
                [np.asarray(self.phasic_multiplier[leg]._y,
                            dtype=np.float64).ravel() for leg in legs])
            self._pmax = self._px.shape[1] - 1

            self._stumb = np.concatenate(
                [self.stumbling_sensors[leg] for leg in legs])
            self._n_stumb = len(self.stumble_segments)

            r = self.correction_rates["retraction"]
            s = self.correction_rates["stumbling"]
            self._rate_r = (r[0] * self.timestep, r[1] * self.timestep)
            self._rate_s = (s[0] * self.timestep, s[1] * self.timestep)

        # --- наблюдение ---

        def invalidate_observation(self):
            """Забыть наблюдение прошлого шага.

            Зовётся тем, кто поставил состояние модели мимо шага
            (`flybrain.replay.restore_body`): после этого прошлое
            наблюдение относится к другому состоянию.
            """
            self._obs_cache = None

        def _rules_observation(self):
            if self.reuse_obs and self._obs_cache is not None \
                    and not self.physics.is_dirty:
                return self._obs_cache
            return SingleFlySimulation.get_observation(self)

        # --- части стокового шага, посчитанные массивами ---

        def _phasic(self, phase):
            """То же, что шесть ``interp1d`` линейной интерполяции.

            Выражения — из ``scipy.interpolate.interp1d._call_linear``:
            поиск вставки слева, обрезка в [1, len-1], затем
            ``t*y_hi + (1-t)*y_lo`` именно в два деления, как там.
            """
            px, py = self._px, self._py
            hi = np.clip((px < phase[:, None]).sum(1), 1, self._pmax)
            lo = hi - 1
            r = self._rows
            x_lo, x_hi = px[r, lo], px[r, hi]
            y_lo, y_hi = py[r, lo], py[r, hi]
            return (((phase - x_lo) / (x_hi - x_lo)) * y_hi
                    + ((x_hi - phase) / (x_hi - x_lo)) * y_lo)

        def _joint_angles(self, phases, magnitudes):
            """Углы всех шести ног: один вызов кусочного полинома."""
            n, d = len(self._legs), self._ndof
            psi = self._psi(phases).reshape(n, n, d)[self._rows, self._rows]
            return self._neutral + magnitudes[:, None] * (psi - self._neutral)

        # --- сам шаг ---

        def step(self, action):
            """Шаг. Арифметика та же, что в апстриме; см. заголовок модуля."""
            if not self._fast_ok:
                return super().step(action)

            cpg = self.cpg_network
            amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
            freqs = self.intrinsic_freqs.copy()
            freqs[:3] *= 1 if action[0] > 0 else -1
            freqs[3:] *= 1 if action[1] > 0 else -1
            cpg.intrinsic_amps = amps
            cpg.intrinsic_freqs = freqs

            obs = self._rules_observation()

            leg_retract = self._retraction_rule_find_leg(obs)
            self._update_persistence_counter()
            persistent = self.retraction_persistence_counter > 0

            cpg.step()

            # Правило втягивания. `max(0, …)` стока — это `np.maximum`:
            # значения совпадают, включая обе нули.
            cond_r = persistent
            if leg_retract is not None:
                cond_r = cond_r | (self._rows == leg_retract)
            rc = np.where(cond_r,
                          self.retraction_correction + self._rate_r[0],
                          np.maximum(0.0, self.retraction_correction
                                     - self._rate_r[1]))
            self.retraction_correction[:] = rc

            # Правило спотыкания: проекция сил контактов на курс. Одним
            # `np.dot` на 18 строк вместо шести по три — сверено побитово.
            proj = np.dot(obs["contact_forces"][self._stumb],
                          obs["fly_orientation"])
            cond_s = (proj.reshape(-1, self._n_stumb)
                      < self.stumbling_force_threshold).any(axis=1)
            sc = np.where(cond_s,
                          self.stumbling_correction + self._rate_s[0],
                          np.maximum(0.0, self.stumbling_correction
                                     - self._rate_s[1]))
            self.stumbling_correction[:] = sc

            # Поправка втягивания имеет приоритет, и когда она есть,
            # поправка спотыкания сбрасывается — как в стоке.
            use_r = rc > 0
            net = np.where(use_r, rc, sc)
            self.stumbling_correction[use_r] = 0.0

            net = np.clip(net, 0, self.max_increment) * self._inv
            phases = cpg.curr_phases
            mod = phases % TWO_PI
            net = net * self._phasic(mod)

            angles = self._joint_angles(phases, cpg.curr_magnitudes)
            angles += net[:, None] * self._corr_vec
            adhesion = ~((self._swing[:, 0] < mod)
                         & (mod < self._swing[:, 1]))

            act = {"joints": np.array(angles.ravel()),
                   "adhesion": adhesion.astype(int)}
            obs, reward, terminated, truncated, info = \
                SingleFlySimulation.step(self, act)
            self._obs_cache = obs
            info["net_corrections"] = np.array(net)
            info.update(act)
            return obs, reward, terminated, truncated, info

        def reset(self, seed=None, **kwargs):
            obs, info = super().reset(seed=seed, **kwargs)
            # Наблюдение сброса — текущее: `after_reset` зовёт
            # `mj_forward`, и флаг «грязного» состояния снят.
            self._obs_cache = obs
            return obs, info

    return FastHybridTurningController


_CLS = None


def controller_class():
    """Класс быстрого контроллера; собирается при первом обращении.

    База лежит в venv и тянет за собой MuJoCo и dm_control; импорт
    `flybrain.gait` не должен этого требовать, иначе любой разбор
    параметров командной строки платил бы сборкой физики.
    """
    global _CLS
    if _CLS is None:
        _CLS = _make_class()
    return _CLS
