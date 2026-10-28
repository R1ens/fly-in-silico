"""A 2D arena, and the wiring that connects it to the connectome.

Be clear about what is what here.  The brain is the real thing: 138 639
neurons from FlyWire, the published LIF dynamics, verified against upstream.
Everything in this file is invented -- how a smell becomes a firing rate in the
antennal receptors, and how the firing of descending neurons becomes a walking
speed and a turn.  The fly has no body and no ventral nerve cord in this model,
so that transformation does not exist anywhere to be copied; it is a guess with
plausible numbers.

Treat an episode as "what happens when you drive the real network through an
invented body", not as fly behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- the arena


@dataclass
class Odour:
    """A Gaussian plume."""

    x: float
    y: float
    strength: float = 1.0
    sigma: float = 12.0  # mm

    def at(self, x, y):
        d2 = (x - self.x) ** 2 + (y - self.y) ** 2
        return self.strength * math.exp(-d2 / (2.0 * self.sigma ** 2))


@dataclass
class Food:
    x: float
    y: float
    radius: float = 2.5  # mm


@dataclass
class Arena:
    width: float = 60.0  # mm
    height: float = 60.0
    foods: list = field(default_factory=list)
    odours: list = field(default_factory=list)

    def concentration(self, x, y):
        return sum(o.at(x, y) for o in self.odours)

    def gradient(self, x, y):
        """Analytic gradient of the odour field, pointing up-gradient."""
        gx = gy = 0.0
        for o in self.odours:
            c = o.at(x, y)
            gx += -(x - o.x) / (o.sigma ** 2) * c
            gy += -(y - o.y) / (o.sigma ** 2) * c
        return gx, gy

    def on_food(self, x, y):
        for f in self.foods:
            if (x - f.x) ** 2 + (y - f.y) ** 2 <= f.radius ** 2:
                return f
        return None

    def nearest_food_distance(self, x, y):
        if not self.foods:
            return float("inf")
        return min(math.hypot(x - f.x, y - f.y) for f in self.foods)

    def wall_distance_ahead(self, x, y, theta):
        """Distance to the first wall along the heading, in mm."""
        dx, dy = math.cos(theta), math.sin(theta)
        best = float("inf")
        if dx > 1e-6:
            best = min(best, (self.width - x) / dx)
        elif dx < -1e-6:
            best = min(best, -x / dx)
        if dy > 1e-6:
            best = min(best, (self.height - y) / dy)
        elif dy < -1e-6:
            best = min(best, -y / dy)
        return max(best, 0.0)

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "foods": [{"x": f.x, "y": f.y, "r": f.radius} for f in self.foods],
            "odours": [{"x": o.x, "y": o.y, "s": o.strength, "sigma": o.sigma}
                       for o in self.odours],
        }


def default_arena(seed=0):
    """Two food patches, each with its own plume, placed off-centre."""
    rng = np.random.default_rng(seed)
    w = h = 60.0
    foods, odours = [], []
    for fx, fy in ((44.0, 46.0), (14.0, 18.0)):
        jx, jy = rng.uniform(-3, 3, 2)
        foods.append(Food(fx + jx, fy + jy, 3.0))
        odours.append(Odour(fx + jx, fy + jy, 1.0, 13.0))
    return Arena(w, h, foods, odours)


# ------------------------------------------------------------------ the fly


@dataclass
class FlyState:
    x: float
    y: float
    theta: float  # radians, 0 = +x
    v: float = 0.0  # mm/s
    omega: float = 0.0  # rad/s
    eaten: int = 0
    feeding: bool = False


# Which neurons each sensory channel drives.  Selector syntax is data.select.
SENSORS = {
    "odour_left": "class:olfactory+side:left",
    "odour_right": "class:olfactory+side:right",
    "taste": "class:gustatory",
    "loom_left": "type:LPLC2+side:left",
    "loom_right": "type:LPLC2+side:right",
}

# Descending output, read out per hemisphere.
#
# ВНИМАНИЕ: это съём по ВСЕМУ полушарию — 645 и 646 нейронов. Giant Fiber
# в нём 2 из 1299, и его спайк арифметически неотличим от шума: при
# 209 Гц один GF поднимает это среднее примерно на 0,32 Гц, а обычный
# уровень 2,5…4,1 Гц при 51…88 активных нейронах. Прямые зонды по
# названным типам живут в :mod:`flybrain.motor` (``DN_PROBES``), и для
# научного утверждения о конкретном нисходящем нейроне нужны они, а не
# это среднее. Здесь селекторы оставлены как есть: по ним посчитаны все
# прежние прогоны.
MOTOR = {
    "dn_left": "super:descending+side:left",
    "dn_right": "super:descending+side:right",
}

# Extra bilateral populations recorded alongside the motor readout, to find
# out where a left/right difference in the stimulus stops being visible.
# Each entry is (left selector, right selector); ORN is the driven population
# itself and serves as the sanity check.
PROBES = {
    "ORN": ("class:olfactory+side:left", "class:olfactory+side:right"),
    "ALPN": ("class:ALPN+side:left", "class:ALPN+side:right"),
    "LH": ("region:LH+side:left", "region:LH+side:right"),
    "MBON": ("type:MBON+side:left", "type:MBON+side:right"),
    "DN": ("super:descending+side:left", "super:descending+side:right"),
    # ВНИМАНИЕ, ловушка имён. `type=DNa01` даёт ЧЕТЫРЕ нейрона, а не два:
    # пару с `hemibrain_type = DNa01` (их `cell_type` — `DNae001`, GNG) и
    # отдельную пару с `cell_type = DNa01` (их `hemibrain_type` —
    # `VES006`, область VES). Так же `type=DNp09`: в нём ещё и пара
    # `cell_type = DNp71`. Эти три строки оставлены как были — по ним
    # посчитаны прежние прогоны, и менять их значило бы молча изменить
    # чужие числа. Для нового разбора есть `motor.DN_PROBES`, где
    # селекторы адресуют ОДИН столбец, а размер проверяется.
    "DNa02": ("type=DNa02+side:left", "type=DNa02+side:right"),
    "DNa01": ("type=DNa01+side:left", "type=DNa01+side:right"),
    "DNp09": ("type=DNp09+side:left", "type=DNp09+side:right"),
}

def dn_probe_candidates():
    """Прямые зонды моторного пути как пары селекторов, под своими именами.

    Отдельно от :func:`probe_candidates` нарочно, и причина
    арифметическая: тот список отбрасывает всё, где меньше восьми
    нейронов на сторону, а у прямых зондов их один-два — то есть они бы
    в него просто не попали. Подкрутить порог нельзя: он защищает
    общий перебор примерно 250 популяций от групп, где асимметрия
    двух нейронов — это не измерение, а монетка. Для прямых зондов это
    препятствием не является, потому что их асимметрию делает
    измеримой сглаживание по времени, а не число нейронов, — но знать
    об этом должен тот, кто их просит, и поэтому они живут здесь.

    Собираются из `motor.DN_PROBES`, а не выписаны второй раз: два
    списка одних и тех же селекторов неизбежно разъехались бы.
    """
    from .motor import DN_PROBES, SHADOW_PROBES

    out = {}
    for src in (DN_PROBES, SHADOW_PROBES):
        for name, p in src.items():
            out["зонд:" + name] = (f"{p.sel}+side:left",
                                   f"{p.sel}+side:right")
    return out

# Named cell families worth screening as a steering readout, on top of every
# neuropil taken as a whole.  The central-complex ones are here because PFL2
# and PFL3 are the steering output of the compass in the real animal, and the
# descending families because that is where a motor command ought to live.
PROBE_FAMILIES = {
    "ALPN": "class:ALPN",
    "ALLN": "class:ALLN",
    "uniglomerular_PN": "type:_[lva]PN",
    "LHAV": "type:LHAV",
    "LHPV": "type:LHPV",
    "LHCENT": "type:LHCENT",
    "KC": "type:KC",
    "MBON": "type:MBON",
    "DAN": "class:DAN",
    "APL": "type:APL",
    "ER": "type:ER",
    "EPG": "type:EPG",
    "PEN": "type:PEN",
    "PFL": "type:PFL",
    "PFL2": "type:PFL2",
    "PFL3": "type:PFL3",
    "PFN": "type:PFN",
    "FC": "type:FC",
    "hDelta": "type:hDelta",
    "vDelta": "type:vDelta",
    "TuBu": "type:TuBu",
    "MeTu": "type:MeTu",
    "DN_all": "super:descending",
    "DNa": "type:DNa",
    "DNb": "type:DNb",
    "DNc": "type:DNc",
    "DNd": "type:DNd",
    "DNg": "type:DNg",
    "DNp": "type:DNp",
    "DNge": "type:DNge",
    "DNpe": "type:DNpe",
    "DNa01": "type=DNa01",
    "DNa02": "type=DNa02",
    "DNa03": "type=DNa03",
    "DNa04": "type=DNa04",
    "DNa05": "type=DNa05",
    "DNa06": "type=DNa06",
    "DNp09": "type=DNp09",
    "ascending": "super:ascending",
    "motor": "super:motor",
}


_CANDIDATE_CACHE = {}


def probe_candidates(neurons, min_per_side=8):
    """Every bilateral population worth screening: neuropils and families.

    Returns ``{name: (left_selector, right_selector)}``, keeping only the ones
    that actually have neurons on both sides.

    Building the list means running about 150 selectors over 138 000 rows, so
    the result is cached: the live interface asks for it on every page load.
    """
    from . import data as D

    key = (len(neurons), min_per_side)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]

    out = {}
    for region in sorted(neurons["region"].unique()):
        out[f"region:{region}"] = (f"region:{region}+side:left",
                                   f"region:{region}+side:right")
    for name, sel in PROBE_FAMILIES.items():
        out[name] = (f"{sel}+side:left", f"{sel}+side:right")

    keep = {}
    for name, (ql, qr) in out.items():
        try:
            nl, nr = len(D.select(neurons, ql)), len(D.select(neurons, qr))
        except ValueError:
            continue
        if nl >= min_per_side and nr >= min_per_side:
            keep[name] = (ql, qr)
    _CANDIDATE_CACHE[key] = keep
    return keep


@dataclass
class BodyParams:
    """Every number in the invented half, in one place."""

    # A real fly's antennae are ~0.3 mm apart and it resolves gradients mostly
    # by casting its head, which this loop has no way to do.  The baseline is
    # exaggerated to 2 mm so that a bilateral difference exists at all.
    antenna_lateral_mm: float = 2.0
    antenna_forward_mm: float = 1.0

    orn_max_hz: float = 150.0
    # Kept low enough that the receptors do not saturate in the middle of the
    # arena -- once both sides sit at the ceiling the gradient is invisible.
    orn_gain: float = 1.0  # concentration -> fraction of orn_max_hz
    grn_hz: float = 200.0
    loom_max_hz: float = 150.0
    loom_range_mm: float = 14.0  # wall closer than this starts to loom

    # Which bilateral population the steering is read from.  Defaults to the
    # whole descending output; `cli lateral` finds candidates that actually
    # carry the odour side, and `arena --motor NAME` swaps one in.
    motor_left: str = MOTOR["dn_left"]
    motor_right: str = MOTOR["dn_right"]

    # readout rate (Hz, mean over the hemisphere) -> movement.
    # None means: measure it at the start of the episode, since a different
    # readout population sits at a completely different baseline.
    dn_ref_hz: float | None = 6.0  # rate that counts as "full drive"
    # 645 descending neurons counted over 20 ms carry enough Poisson noise to
    # swamp the odour-driven asymmetry, so the readout is integrated first.
    dn_tau_s: float = 0.12
    speed_base_mm_s: float = 3.0
    speed_max_mm_s: float = 22.0
    # None means: measure the spread of the *smoothed* asymmetry during the
    # warm-up, when only noise is varying, and set the gain so that one sigma
    # of it turns the fly by ``turn_target_rad_s``.  Calibrating to full turn
    # instead makes the fly spin: in the arena the real bilateral difference
    # is only about 12 %, so the signal sits near the noise and a relay-like
    # gain just follows the noise sign.
    turn_gain_rad_s: float | None = 5.0
    turn_target_rad_s: float = 0.6
    # The connectome is not left/right symmetric (341 ALPN on one side, 344 on
    # the other, and different wiring), so a population's asymmetry has a
    # non-zero resting value.  Multiplied by a large gain that becomes a
    # constant turn and the fly just circles.  Measured during the warm-up
    # with both antennae driven equally, then subtracted.
    asym_bias: float = 0.0
    turn_max_rad_s: float = 4.5
    turn_sign: float = 1.0  # +1: more left-DN activity turns left
    feed_pause_s: float = 0.4  # standing still on food

    smooth: float = 0.45  # low-pass on speed and turn, per control step

    # Control condition: feed the left antenna's concentration to the right
    # receptors and vice versa.  If the turning bias really comes from the
    # bilateral pathway through the brain, this should push it below chance.
    swap_antennae: bool = False


class Body:
    """Sensing and acting.  Keeps no brain state, only motor smoothing."""

    def __init__(self, params: BodyParams | None = None):
        self.p = params or BodyParams()
        self._v = 0.0
        self._w = 0.0
        self._feed_timer = 0.0
        self._dn_l = None
        self._dn_r = None

    # ---- environment -> firing rates -------------------------------------

    def sense(self, arena: Arena, fly: FlyState):
        p = self.p
        c, s = math.cos(fly.theta), math.sin(fly.theta)
        fx = fly.x + p.antenna_forward_mm * c
        fy = fly.y + p.antenna_forward_mm * s
        # left antenna is 90 deg counter-clockwise from the heading
        lx = fx - p.antenna_lateral_mm * s
        ly = fy + p.antenna_lateral_mm * c
        rx = fx + p.antenna_lateral_mm * s
        ry = fy - p.antenna_lateral_mm * c

        c_left = arena.concentration(lx, ly)
        c_right = arena.concentration(rx, ry)
        in_l, in_r = (c_right, c_left) if p.swap_antennae else (c_left, c_right)
        orn_l = p.orn_max_hz * min(1.0, p.orn_gain * in_l)
        orn_r = p.orn_max_hz * min(1.0, p.orn_gain * in_r)

        taste = p.grn_hz if arena.on_food(fly.x, fly.y) else 0.0

        # Walls loom: the closer the wall in front, the stronger, and the side
        # it is on decides which lobula gets it.
        d_ahead = arena.wall_distance_ahead(fly.x, fly.y, fly.theta)
        loom = 0.0
        if d_ahead < p.loom_range_mm:
            loom = p.loom_max_hz * (1.0 - d_ahead / p.loom_range_mm)
        d_l = arena.wall_distance_ahead(fly.x, fly.y, fly.theta + 0.6)
        d_r = arena.wall_distance_ahead(fly.x, fly.y, fly.theta - 0.6)
        if d_l + d_r > 0:
            share_l = d_r / (d_l + d_r)  # wall nearer on the left -> more left
        else:
            share_l = 0.5

        return {
            "odour_left": orn_l,
            "odour_right": orn_r,
            "taste": taste,
            "loom_left": loom * share_l * 2.0,
            "loom_right": loom * (1.0 - share_l) * 2.0,
            "_c_left": c_left,
            "_c_right": c_right,
            "_wall_ahead": d_ahead,
        }

    # ---- descending firing -> movement -----------------------------------

    def act(self, dn_left_hz, dn_right_hz, on_food, dt_s):
        p = self.p
        if self._dn_l is None:
            self._dn_l, self._dn_r = dn_left_hz, dn_right_hz
        else:
            k = math.exp(-dt_s / max(p.dn_tau_s, 1e-6))
            self._dn_l = k * self._dn_l + (1 - k) * dn_left_hz
            self._dn_r = k * self._dn_r + (1 - k) * dn_right_hz
        dn_left_hz, dn_right_hz = self._dn_l, self._dn_r

        total = (dn_left_hz + dn_right_hz) / 2.0
        drive = min(1.0, total / p.dn_ref_hz)
        asym = ((dn_left_hz - dn_right_hz) / (dn_left_hz + dn_right_hz + 1e-6)
                - p.asym_bias)

        v_target = p.speed_base_mm_s + drive * (p.speed_max_mm_s - p.speed_base_mm_s)
        w_target = p.turn_sign * p.turn_gain_rad_s * asym
        w_target = float(np.clip(w_target, -p.turn_max_rad_s, p.turn_max_rad_s))

        if on_food:
            self._feed_timer = p.feed_pause_s
        if self._feed_timer > 0.0:
            self._feed_timer -= dt_s
            v_target = 0.0

        a = p.smooth
        self._v = a * self._v + (1 - a) * v_target
        self._w = a * self._w + (1 - a) * w_target
        return self._v, self._w, drive, asym

    @property
    def feeding(self):
        return self._feed_timer > 0.0


def advance(arena: Arena, fly: FlyState, v, omega, dt_s):
    """Integrate one control step and bounce off the walls."""
    fly.theta = (fly.theta + omega * dt_s) % (2 * math.pi)
    fly.v, fly.omega = v, omega
    nx = fly.x + v * math.cos(fly.theta) * dt_s
    ny = fly.y + v * math.sin(fly.theta) * dt_s

    margin = 1.0
    bounced = False
    if nx < margin or nx > arena.width - margin:
        nx = float(np.clip(nx, margin, arena.width - margin))
        fly.theta = math.pi - fly.theta
        bounced = True
    if ny < margin or ny > arena.height - margin:
        ny = float(np.clip(ny, margin, arena.height - margin))
        fly.theta = -fly.theta
        bounced = True
    fly.theta %= 2 * math.pi
    fly.x, fly.y = nx, ny
    return bounced
