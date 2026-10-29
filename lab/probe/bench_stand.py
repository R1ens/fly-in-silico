"""Оснастка стенда: чем мерить, чтобы замеру можно было верить.

Модуль ничего не меряет сам. В нём лежит то, без чего числа на этой
машине не значат ничего, и каждая часть отвечает на свою жалобу к
прежним замерам.

**Фон гуляет.** На машине одновременно считают другие ветки, и
видеокарта бывает занята на 96 % ещё до того, как стенд запустился.
Поэтому :class:`Gpu` снимает загрузку и память видеокарты в отдельной
нити и умеет сказать, каким фон был **в окне** конкретного замера, а
:class:`Cpu` отдельно считает процессорное время своего процесса и всей
системы. Абсолютное число без этих двух рядом — не измерение, а слух.

**Два прогона подряд сравнивать нельзя.** :func:`pairs` гоняет варианты
перемежающимися кругами (A, B, A, B, …) и считает отношение **внутри
пары**, а не между средними: так дрейф фона между началом и концом
замера вычитается. Возвращается медиана отношений, разброс и число пар.

**Среднее скрывает хвост.** :func:`stat` печатает медиану, p95 и
максимум рядом: в базовой линии шаг управления гулял от 282 до 450 мс
при среднем 337, и по одному среднему этого не видно.

**Разложение обязано сходиться.** :class:`Split` складывает составляющие
и печатает невязку со знаком. Если невязка больше пары процентов шага,
разложение неверно, и «узкое место» по нему называть нельзя.

**Счётчики на точное равенство.** :class:`Counters` считает вызовы
движка LIF, шаги физики, кадры сетчатки и взятия генератора и снимает
отпечаток состояния генератора. Отпечатки выходов не ловят «ускорились,
потому что перестали считать»; счётчики ловят.
"""

import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from ctypes import wintypes

import numpy as np

# Шаг управления и его деление. Ожидаемые значения, против которых
# счётчики сверяются на точное равенство; сам стенд берёт их из кода
# (`sim.DT`, `BodyConfig.physics_dt`) и сверяет с этими.
CONTROL_MS = 20.0
PHYS_DT = 2e-4
PHYS_PER_STEP = int(round(CONTROL_MS / 1000.0 / PHYS_DT))     # 100
LIF_PER_STEP = 200                                            # 20 мс / 0,1 мс


# --------------------------------------------------------------- статистика

def stat(vals, nd=2):
    """Медиана, хвост и разброс. Среднее тоже, но не вместо хвоста."""
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return {}
    n = len(v)

    def q(p):
        if n == 1:
            return v[0]
        i = (n - 1) * p
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return v[lo] * (1 - (i - lo)) + v[hi] * (i - lo)

    med = q(0.5)
    return {
        "n": n,
        "медиана": round(med, nd),
        "среднее": round(sum(v) / n, nd),
        "мин": round(v[0], nd),
        "p90": round(q(0.9), nd),
        "p95": round(q(0.95), nd),
        "макс": round(v[-1], nd),
        "хвост p95/медиана": round(q(0.95) / med, 3) if med else None,
        "размах макс/мин": round(v[-1] / v[0], 3) if v[0] else None,
    }


def fp(a):
    """Отпечаток массива от сырых байтов: ловит расхождение в последнем бите."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ------------------------------------------------------- загрузка процессора

class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    @property
    def ns100(self):
        return (self.high << 32) | self.low


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("pf", wintypes.DWORD),
                ("peak_ws", ctypes.c_size_t), ("ws", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                ("c", ctypes.c_size_t), ("d", ctypes.c_size_t),
                ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]


def _mem():
    """Рабочий набор процесса: (сейчас, пик) в МБ, или (None, None).

    Типы объявляются, а не оставляются на усмотрение ctypes. Без
    `restype` псевдодескриптор (HANDLE)−1 портится при передаче в
    64-битный вызов, вызов тихо не срабатывает, и функция отдаёт ноль
    вместо отказа — на этом в проекте уже получили «память не
    записывалась никогда» (`probe/bench_v2.py`, `rss_mb`).
    """
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_PMC),
                                               wintypes.DWORD]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                          ctypes.byref(pmc), pmc.cb):
            return None, None
        if not pmc.peak_ws:
            return None, None
        return round(pmc.ws / 1e6, 1), round(pmc.peak_ws / 1e6, 1)
    except Exception:                                         # noqa: BLE001
        return None, None


def rss_now_mb():
    return _mem()[0]


def rss_peak_mb():
    return _mem()[1]


class Cpu:
    """Процессорное время: своё и системное. Без psutil — его тут нет."""

    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.k32.GetCurrentProcess.argtypes = []
        self.k32.GetProcessTimes.restype = wintypes.BOOL
        self.k32.GetProcessTimes.argtypes = [
            wintypes.HANDLE] + [ctypes.POINTER(_FILETIME)] * 4
        self.k32.GetSystemTimes.restype = wintypes.BOOL
        self.k32.GetSystemTimes.argtypes = [ctypes.POINTER(_FILETIME)] * 3
        self.cores = os.cpu_count() or 1

    def _proc(self):
        a, b, c, d = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
        if not self.k32.GetProcessTimes(self.k32.GetCurrentProcess(),
                                        ctypes.byref(a), ctypes.byref(b),
                                        ctypes.byref(c), ctypes.byref(d)):
            return None
        return (c.ns100 + d.ns100) * 1e-7

    def _sys(self):
        idle, kern, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not self.k32.GetSystemTimes(ctypes.byref(idle),
                                       ctypes.byref(kern),
                                       ctypes.byref(user)):
            return None
        return (idle.ns100 * 1e-7, (kern.ns100 + user.ns100) * 1e-7)

    def mark(self):
        return (time.perf_counter(), self._proc(), self._sys())

    def since(self, m0):
        """Загрузка за промежуток: своя и машинная, в ядрах и в процентах."""
        t1, p1, s1 = self.mark()
        t0, p0, s0 = m0
        wall = max(t1 - t0, 1e-9)
        out = {"стена, с": round(wall, 3), "ядер в машине": self.cores}
        if p0 is not None and p1 is not None:
            out["свой процесс, ядер"] = round((p1 - p0) / wall, 3)
            out["свой процесс, % машины"] = round(
                100.0 * (p1 - p0) / wall / self.cores, 1)
        if s0 is not None and s1 is not None:
            busy = (s1[1] - s0[1]) - (s1[0] - s0[0])
            out["вся машина, ядер"] = round(busy / wall, 2)
            out["вся машина, %"] = round(100.0 * busy / wall / self.cores, 1)
        return out


# ------------------------------------------------------------- видеокарта

class Gpu(threading.Thread):
    """Загрузка и память видеокарты, снятая в нити рядом с замером.

    Видеокарта на этой машине общая: на ней считают другие ветки, и
    `memory.used` — машинная занятость, а не наша. Поэтому рядом с
    каждым уровнем печатается и машинная (отсюда), и своя
    (`torch.cuda.max_memory_allocated`); путать их нельзя.
    """

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self.rows = []
        self._stop = threading.Event()
        self.ok = self._probe() is not None

    @staticmethod
    def _probe():
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if out.returncode:
                return None
            row = out.stdout.strip().split("\n")[0].split(",")
            return [float(x.strip()) for x in row]
        except Exception:                                     # noqa: BLE001
            return None

    def run(self):
        while not self._stop.is_set():
            v = self._probe()
            if v is not None:
                self.rows.append((time.perf_counter(), v))
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()

    def window(self, t0, t1):
        """Каким был фон именно в окне замера, а не «вообще»."""
        sel = [v for t, v in self.rows if t0 <= t <= t1]
        if not sel:
            return {"проб": 0, "почему": "в окне ни одной пробы nvidia-smi"}
        util = [v[0] for v in sel]
        mem = [v[1] for v in sel]
        return {
            "проб": len(sel),
            "загрузка видеокарты, % медиана": round(statistics.median(util), 1),
            "загрузка видеокарты, % мин": round(min(util), 1),
            "загрузка видеокарты, % макс": round(max(util), 1),
            "память видеокарты занято машиной, МиБ медиана":
                round(statistics.median(mem)),
            "память видеокарты всего, МиБ": sel[0][2],
        }


def gpu_mine():
    """Своя память на видеокарте: то, что держит наш torch, а не машина."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "своя, выделено пик, МБ":
                round(torch.cuda.max_memory_allocated() / 1e6, 1),
            "своя, зарезервировано пик, МБ":
                round(torch.cuda.max_memory_reserved() / 1e6, 1),
            "своя, выделено сейчас, МБ":
                round(torch.cuda.memory_allocated() / 1e6, 1),
        }
    except Exception:                                         # noqa: BLE001
        return {}


def gpu_reset():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                         # noqa: BLE001
        pass


# --------------------------------------------------------------- разложение

class Split:
    """Составляющие шага и невязка со знаком.

    Правило у разложения одно: составляющие замеряются каждая своим
    таймером вокруг настоящей работы, а окно шага — по стене, и дальше
    печатается разница. Невязка здесь не украшение: в базовой линии она
    −2,44 мс (0,7 % шага) ровно потому, что кадр живого вида и обмен
    лежат ВНЕ окна, по которому считалось «всего». Пока невязка не
    названа числом со знаком, разложение проверить нельзя.
    """

    def __init__(self, names):
        self.names = list(names)
        self.rows = []

    def add(self, total_ms, parts):
        row = {"всего, мс": total_ms}
        row.update({k: float(parts.get(k, 0.0)) for k in self.names})
        row["невязка, мс"] = total_ms - sum(row[k] for k in self.names)
        self.rows.append(row)
        return row

    def report(self, nd=3):
        if not self.rows:
            return {}
        tot = stat([r["всего, мс"] for r in self.rows], nd)
        out = {"шагов в разложении": len(self.rows), "всего, мс": tot}
        med_tot = tot["медиана"]
        for k in self.names:
            s = stat([r[k] for r in self.rows], nd)
            s["доля от медианы шага"] = (round(s["медиана"] / med_tot, 4)
                                        if med_tot else None)
            out[k] = s
        res = [r["невязка, мс"] for r in self.rows]
        med_res = statistics.median(res)
        out["невязка"] = {
            "медиана, мс": round(med_res, nd),
            "медиана со знаком, % шага": (round(100.0 * med_res / med_tot, 3)
                                          if med_tot else None),
            "мин, мс": round(min(res), nd), "макс, мс": round(max(res), nd),
            "сошлось в пределах 2 % шага":
                bool(med_tot and abs(med_res) / med_tot < 0.02),
        }
        return out


# ------------------------------------------------------------------ счётчики

class Counters:
    """Счётчики на точное равенство плюс отпечаток генератора.

    Что они ловят, чего не ловят отпечатки выходов: прогон, который
    «ускорился», потому что перестал считать. Отпечаток сошёлся бы
    побитово, если выход взят из записи, — а счётчик вызовов движка при
    этом ноль, и подделать его, не позвав движок, нельзя: он растёт
    внутри самого шага.
    """

    def __init__(self):
        self.lif = 0          # вызовов sim.Engine.step через обёртку
        self.poisson = 0      # взятий пуассоновского входа (генератор)
        self.phys = 0         # вызовов body.step
        self.phys_inner = 0   # вызовов physics.step (чистый MuJoCo)
        self.retina = 0       # кадров сетчатки, поданных в сеть
        self.eye = 0          # пересъёмов глаз (Fly._update_vision)
        self.gen_fp = []
        self.per_step = []
        self._prev = None

    def snap(self, engine=None):
        """Срез на границе шага управления: дальше считаются приросты."""
        cur = {"вызовов движка LIF": self.lif,
               "взятий генератора": self.poisson,
               "шагов body.step": self.phys,
               "шагов physics.step": self.phys_inner,
               "кадров сетчатки в сеть": self.retina,
               "пересъёмов глаз": self.eye}
        if engine is not None:
            cur["вызовов по счётчику движка"] = int(getattr(engine, "calls", 0))
            try:
                self.gen_fp.append(fp(engine.gen.get_state().cpu().numpy()))
            except Exception:                                 # noqa: BLE001
                pass
        if self._prev is not None:
            self.per_step.append({k: cur[k] - self._prev.get(k, 0)
                                  for k in cur})
        self._prev = cur

    def verdict(self, want):
        """Точное равенство по каждому счётчику. Не «около», а равенство."""
        out = {"шагов со срезом": len(self.per_step), "проверки": {}}
        for key, need in want.items():
            got = sorted({r.get(key, 0) for r in self.per_step})
            out["проверки"][key] = {
                "обязано на шаг управления": need,
                "встретилось": got if len(got) < 8 else got[:8] + ["…"],
                "точное равенство": bool(got == [need]),
            }
        if self.gen_fp:
            uniq = len(set(self.gen_fp))
            out["генератор"] = {
                "отпечатков": len(self.gen_fp), "различных": uniq,
                "состояние двигалось на каждом шаге":
                    bool(uniq == len(self.gen_fp)),
                "первый": self.gen_fp[0], "последний": self.gen_fp[-1],
            }
        out["всё сошлось"] = (
            all(v["точное равенство"] for v in out["проверки"].values())
            and (not self.gen_fp
                 or out["генератор"]["состояние двигалось на каждом шаге"]))
        return out


# ------------------------------------------------------- перемежающиеся пары

def pairs(run_a, run_b, n=5, name_a="A", name_b="B", warm=True, log=print):
    """Перемежающиеся круги A, B, A, B, … и отношение ВНУТРИ пары.

    Почему не два прогона подряд. Фон на этой машине гуляет: загрузка
    видеокарты доходила до 96 % при простаивающем стенде. Два прогона
    подряд отличаются и фоном, и порядком, и разделить это нельзя.
    Перемежение оставляет фону возможность испортить обе половины пары
    примерно одинаково, и тогда отношение внутри пары от дрейфа почти не
    зависит.

    Одна пара не даёт ничего, и это записано числом — размахом по
    парам, — а не оговоркой.
    """
    if warm:
        log("  круг разогрева (в зачёт не идёт)")
        run_a()
        run_b()
    rows = []
    for i in range(n):
        a = run_a()
        b = run_b()
        ratio = (a / b) if b else None
        rows.append({"пара": i + 1, name_a: round(a, 4), name_b: round(b, 4),
                     "отношение": None if ratio is None else round(ratio, 4)})
        log("  пара %d: %s %.4f, %s %.4f, отношение %s"
            % (i + 1, name_a, a, name_b, b,
               "—" if ratio is None else ("%.4f" % ratio)))
    rr = [r["отношение"] for r in rows if r["отношение"]]
    out = {"пар": len(rr), "пары": rows,
           "что сравнивалось": "%s / %s" % (name_a, name_b)}
    if rr:
        out["отношение"] = {
            "медиана": round(statistics.median(rr), 4),
            "мин": round(min(rr), 4), "макс": round(max(rr), 4),
            "размах, % медианы": round(
                100.0 * (max(rr) - min(rr)) / statistics.median(rr), 1),
        }
        if len(rr) > 1:
            out["отношение"]["СО по парам"] = round(statistics.stdev(rr), 4)
    return out


def save(path, rep):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("отчёт: " + str(path))
