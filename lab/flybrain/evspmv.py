"""Событийное умножение связей: сборка ядра и обвязка для движка LIF.

Зачем это вообще. Шаг движка упирается не в обращения к драйверу, а в
видеокарту, и на видеокарте — в одно ядро: `csrmv` по матрице связей,
86 % шага (`LIF_PERF.md`). Оно читает все 15 091 983 связи на каждом
тике 0,1 мс, а спайкуют при этом 28…103 нейрона из 138 639, то есть
нужны 0,04 % связей — 99,96 % прочитанного умножается на ноль.

Здесь то же произведение считается событийно, своим ядром (`ev_spmv.cu`):
складываются только строки спайкующих нейронов, атомарными добавлениями
в произвольном порядке. Почему это остаётся ПОБИТОВО тем же, а не
приближением, написано в заголовке `.cu` и проверяется здесь
:func:`exactness` — тремя числами, а не рассуждением.

**Отказ сборки обязан быть громким.** В этом проекте уже есть механизм,
который отказывает молча: `sim.Engine.use_graphs` возвращает False и
оставляет шаг втрое медленнее, ничего не сообщая. Второй такой отравил
бы все будущие замеры: прогон шёл бы на `csrmv` с потерей втрое, а в
отчёте стояло бы «событийное». Поэтому здесь нет ни одного тихого
отката: всякая нехватка — :class:`EventKernelUnavailable` с указанием,
ЧЕГО не хватает и где это искали. Откат на `csrmv` есть, но выбирается
он явно — `spmv="эталонный"` или `FLYBRAIN_SPMV=эталонный`.

Что требуется на машине (замерено, а не предположено; см. `LIF_EVENT.md`):

* `nvcc` из CUDA Toolkit — ищется в ``FLYBRAIN_NVCC``, затем в ``PATH``,
  затем в ``CUDA_PATH``/``CUDA_PATH_V*``, затем в стандартном каталоге
  установки (самая свежая версия);
* `vcvars64.bat` из Visual Studio — ищется в ``FLYBRAIN_VCVARS``, затем
  через ``vswhere.exe``, затем в стандартных каталогах 2022/2019 всех
  изданий. На Windows `nvcc` без хостового компилятора не работает;
* видеокарта с вычислительной способностью, которую знает этот `nvcc`
  (архитектура берётся у устройства, а не вписана числом).

В `.venv` не ставится ничего: ядро собирается в отдельный каталог
(``FLYBRAIN_EXT_DIR``, по умолчанию ``%TEMP%/flybrain-ext``) и зовётся
через `ctypes`. Собранная DLL запоминается по отпечатку источника и
архитектуры, поэтому сборка (около 5 с) идёт один раз, а не на каждый
прогон.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Предел точности суммирования целых во float32. Выше него сумма целых
# уже не обязана быть точной, и побитовое равенство с `csrmv` перестаёт
# следовать из порядка слагаемых.
F32_EXACT_INT = 1 << 24

# Нитей в блоке. 128 — это чуть больше среднего числа исходящих связей
# (108,9), то есть обычная строка проходится одной итерацией. Меняется
# через ``FLYBRAIN_EV_THREADS`` — для замеров, а не для настройки на
# глаз.
THREADS = int(os.environ.get("FLYBRAIN_EV_THREADS", "128"))

SRC = Path(__file__).with_name("ev_spmv.cu")
ABI = 2            # обязано совпасть с `ev_spmv_abi()` в .cu

_LIB = None        # загруженная DLL, одна на процесс
_BUILD = None      # чем и как собрано: уходит в отчёт прогона


class EventKernelUnavailable(RuntimeError):
    """Событийное ядро недоступно, и это сказано громко.

    Поднимается вместо любого тихого отката: на `csrmv` движок
    отступает только по явной просьбе.
    """


def _say_missing(what, looked):
    raise EventKernelUnavailable(
        "событийное умножение связей не собрано: не найден %s.\n"
        "Искал: %s\n"
        "Либо укажите путь переменной окружения, либо выберите эталонное "
        "умножение явно: spmv=\"эталонный\" (или FLYBRAIN_SPMV=эталонный). "
        "Тихого отката на csrmv здесь нет нарочно: он стоит втрое, и "
        "молчащий откат отравил бы все замеры."
        % (what, "\n       ".join(str(p) for p in looked)))


def find_nvcc():
    """Путь к `nvcc` или громкий отказ со списком, где искали."""
    looked = []
    env = os.environ.get("FLYBRAIN_NVCC")
    if env:
        # Указанный путь — указание, а не подсказка. Если его нет, дальше
        # не ищется: поиск в обход явной просьбы и есть тихий откат, а
        # собрали бы тогда не тем, чем просили.
        looked.append("FLYBRAIN_NVCC=%s" % env)
        if Path(env).exists():
            return Path(env)
        _say_missing("nvcc по указанному FLYBRAIN_NVCC", looked)
    got = shutil.which("nvcc")
    looked.append("PATH (which nvcc) -> %s" % (got or "нет"))
    if got:
        return Path(got)
    for key in sorted(os.environ):
        if key == "CUDA_PATH" or key.startswith("CUDA_PATH_V"):
            cand = Path(os.environ[key]) / "bin" / "nvcc.exe"
            looked.append("%s -> %s" % (key, cand))
            if cand.exists():
                return cand
    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    looked.append("%s\\v*\\bin\\nvcc.exe" % root)
    if root.is_dir():
        cands = sorted(root.glob("v*/bin/nvcc.exe"))
        if cands:
            return cands[-1]
    _say_missing("nvcc (CUDA Toolkit)", looked)


def find_vcvars():
    """Путь к `vcvars64.bat` или громкий отказ со списком, где искали.

    На Windows `nvcc` без хостового компилятора не соберёт ничего, и
    отказ от него приходит не как «нет MSVC», а как невнятная ошибка
    вида «Cannot find compiler 'cl.exe' in PATH». Поэтому хостовой
    компилятор ищется отдельно и называется отдельно.
    """
    looked = []
    env = os.environ.get("FLYBRAIN_VCVARS")
    if env:
        # Как и с `nvcc`: указанный путь — указание. Нет его — отказ, а не
        # поиск мимо просьбы.
        looked.append("FLYBRAIN_VCVARS=%s" % env)
        if Path(env).exists():
            return Path(env)
        _say_missing("vcvars64.bat по указанному FLYBRAIN_VCVARS", looked)
    vswhere = Path(os.environ.get("ProgramFiles(x86)",
                                  r"C:\Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    looked.append("vswhere.exe (%s)" % vswhere)
    if vswhere.exists():
        try:
            got = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=60)
            root = (got.stdout or "").strip().splitlines()
            if root:
                cand = (Path(root[0]) / "VC" / "Auxiliary" / "Build"
                        / "vcvars64.bat")
                looked.append("vswhere -> %s" % cand)
                if cand.exists():
                    return cand
        except (OSError, subprocess.SubprocessError) as exc:
            looked.append("vswhere не ответил: %s" % exc)
    for pf in (r"C:\Program Files\Microsoft Visual Studio",
               r"C:\Program Files (x86)\Microsoft Visual Studio"):
        base = Path(pf)
        if not base.is_dir():
            continue
        pat = "*/*/VC/Auxiliary/Build/vcvars64.bat"
        looked.append("%s\\%s" % (base, pat.replace("/", "\\")))
        cands = sorted(base.glob(pat))
        if cands:
            return cands[-1]
    _say_missing("vcvars64.bat (хостовой компилятор Visual Studio)", looked)


def arch_flag():
    """`-gencode` под ТО устройство, на котором считаем.

    Архитектура спрашивается у видеокарты, а не вписана числом: ядро,
    собранное под `sm_89`, на другой карте просто не запустится, и узнать
    об этом из невнятной ошибки запуска хуже, чем собрать под своё.
    """
    import torch

    if not torch.cuda.is_available():
        raise EventKernelUnavailable(
            "событийное умножение связей требует CUDA, а torch её не "
            "видит (torch.cuda.is_available() == False). Выберите "
            "эталонное умножение явно: spmv=\"эталонный\".")
    major, minor = torch.cuda.get_device_capability()
    sm = "%d%d" % (major, minor)
    return "-gencode", "arch=compute_%s,code=sm_%s" % (sm, sm), sm


def ext_dir():
    base = os.environ.get("FLYBRAIN_EXT_DIR")
    if not base:
        base = Path(os.environ.get("TEMP") or os.environ.get("TMP")
                    or ".") / "flybrain-ext"
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    return out


def build(force=False, verbose=False):
    """Собрать (или взять из кэша) DLL с событийным ядром.

    Возвращает ``(lib, info)``: загруженную библиотеку и словарь
    зависимостей сборки, который уходит в отчёт прогона. Любая нехватка —
    :class:`EventKernelUnavailable`, никогда не None и никогда не тихий
    откат.
    """
    global _LIB, _BUILD
    if _LIB is not None and not force:
        return _LIB, _BUILD

    if sys.platform != "win32":
        raise EventKernelUnavailable(
            "сборка событийного ядра написана под Windows (vcvars64.bat, "
            "__declspec(dllexport)). На %s выберите эталонное умножение "
            "явно: spmv=\"эталонный\"." % sys.platform)
    if not SRC.exists():
        raise EventKernelUnavailable(
            "нет источника ядра %s — он часть пакета, а не генерируется "
            "на месте" % SRC)

    nvcc = find_nvcc()
    vcvars = find_vcvars()
    gen_a, gen_b, sm = arch_flag()
    src_text = SRC.read_text(encoding="utf-8")
    key = hashlib.sha256(
        ("%s|%s|%s|%s" % (src_text, gen_b, ABI, nvcc)).encode("utf-8")
    ).hexdigest()[:16]

    where = ext_dir()
    cu = where / ("ev_spmv_%s.cu" % key)
    dll = where / ("ev_spmv_%s.dll" % key)
    log = where / ("ev_spmv_%s.log" % key)
    info = {"ядро": "событийное", "архитектура": "sm_%s" % sm,
            "nvcc": str(nvcc), "vcvars64": str(vcvars),
            "источник": str(SRC), "dll": str(dll),
            "нитей в блоке": THREADS, "abi": ABI}

    if force or not dll.exists():
        cu.write_text(src_text, encoding="utf-8")
        # Через .bat, а не строкой в `cmd /c`: кавычки в пути к
        # vcvars64 subprocess на Windows переэкранирует, и вызов
        # ломается.
        bat = where / ("build_%s.bat" % key)
        bat.write_text(
            "@echo off\r\n"
            'call "%s" >nul\r\n'
            'if errorlevel 1 exit /b 90\r\n'
            '"%s" -O3 -shared %s %s -o "%s" "%s"\r\n'
            % (vcvars, nvcc, gen_a, gen_b, dll, cu),
            encoding="utf-8")
        got = subprocess.run(["cmd", "/c", str(bat)], capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=600)
        log.write_text("=== stdout ===\n%s\n=== stderr ===\n%s"
                       % (got.stdout, got.stderr), encoding="utf-8")
        if verbose:
            print(got.stdout[-4000:])
            print(got.stderr[-4000:])
        if got.returncode != 0 or not dll.exists():
            raise EventKernelUnavailable(
                "nvcc вернул %d и DLL не появилась.\n"
                "  nvcc:    %s\n  vcvars:  %s\n  арх:     sm_%s\n"
                "  журнал:  %s\n"
                "последние строки:\n%s\n%s"
                % (got.returncode, nvcc, vcvars, sm, log,
                   (got.stdout or "")[-1500:], (got.stderr or "")[-1500:]))
        info["собрано сейчас"] = True
    else:
        info["собрано сейчас"] = False

    lib = ctypes.CDLL(str(dll))
    lib.ev_spmv_abi.restype = ctypes.c_int
    lib.ev_spmv_abi.argtypes = []
    got_abi = int(lib.ev_spmv_abi())
    if got_abi != ABI:
        raise EventKernelUnavailable(
            "загруженная DLL %s отвечает abi %d, а ожидается %d: кэш "
            "разошёлся с источником. Уберите каталог %s и соберите снова."
            % (dll, got_abi, ABI, where))
    for name in ("ev_spmv_i32", "ev_spmv_i64"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = ([ctypes.c_void_p] * 6
                       + [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_void_p])
    _LIB, _BUILD = lib, info
    return lib, info


# --- условия побитового равенства -------------------------------------


def exactness(w_csr):
    """Три числа, из которых следует побитовое равенство с `csrmv`.

    Возвращает словарь с замерами и признаком ``годится``. Ничего не
    поднимает: решение принимает вызывающий, потому что у него есть
    право на явное «считай приближённо».

    * ``значения целые`` — значения матрицы целые. Тогда произведение на
      0.0/1.0 точно и сумма — это сумма целых.
    * ``макс сумма модулей по строке`` — предел, до которого сумма целых
      во float32 точна при ЛЮБОМ порядке сложения: 2^24. Считается по
      строкам ``W[post, :]``, то есть по тому самому результату, который
      собирается атомарными добавлениями.
    * ``запас`` — во сколько раз предел дальше замеренного максимума.
    """
    import torch

    val = w_csr.values()
    crow = w_csr.crow_indices()
    frac = (val - torch.round(val)).abs().max()
    integral = bool(float(frac) == 0.0)
    # Сумма модулей по строкам CSR: разность кумулятивной суммы на
    # границах строк. Так это один проход по значениям, а не 138 639
    # срезов.
    cum = torch.cumsum(val.abs().to(torch.float64), 0)
    cum = torch.cat([torch.zeros(1, dtype=cum.dtype, device=cum.device), cum])
    rows = cum[crow.to(torch.int64)]
    worst = float((rows[1:] - rows[:-1]).max())
    return {
        "значения целые": integral,
        "макс нецелая часть": float(frac),
        "мин значение": float(val.min()),
        "макс значение": float(val.max()),
        "макс сумма модулей по строке": worst,
        "предел точности float32": F32_EXACT_INT,
        "запас": round(F32_EXACT_INT / worst, 1) if worst else None,
        "годится": bool(integral and 0.0 < worst < F32_EXACT_INT),
    }


class Events:
    """Матрица связей по пресинаптическому нейрону плюс вызов ядра.

    Держит транспонированный CSR (строка — исходящие связи нейрона) и
    постоянный буфер результата. Буфер постоянный не для экономии: шаг
    пишется в CUDA-граф, и граф на повторе обязан писать в ту же память.
    """

    def __init__(self, w_csr, batch, device, threads=None, verbose=False):
        import torch

        from . import data as D

        self.lib, self.build_info = build(verbose=verbose)
        self.threads = int(threads or THREADS)
        self.batch = int(batch)
        if w_csr.shape[0] != w_csr.shape[1]:
            # Ядро считает решётку по `n` и адресует выход по тому же
            # `n`. На неквадратной матрице это дало бы не отказ, а молча
            # другие числа, поэтому проверка здесь, а не в комментарии.
            raise EventKernelUnavailable(
                "матрица связей не квадратная: %s. Событийное ядро "
                "считает W[post, pre] при post и pre из одного набора "
                "нейронов." % (tuple(w_csr.shape),))
        self.n = int(w_csr.shape[0])
        with D._quiet_sparse():
            wt = w_csr.t().to_sparse_csr()
        self.crow = wt.crow_indices().contiguous()
        self.col = wt.col_indices().contiguous()
        self.val = wt.values().contiguous()
        if self.crow.dtype != self.col.dtype:
            raise EventKernelUnavailable(
                "индексы транспонированной матрицы разных типов: crow %s, "
                "col %s" % (self.crow.dtype, self.col.dtype))
        if self.col.dtype == torch.int32:
            self._call = self.lib.ev_spmv_i32
        elif self.col.dtype == torch.int64:
            self._call = self.lib.ev_spmv_i64
        else:
            raise EventKernelUnavailable(
                "индексы матрицы типа %s: ядро знает только int32 и int64"
                % self.col.dtype)
        self.out = torch.zeros(self.batch, self.n, device=device)
        self._p = (self.crow.data_ptr(), self.col.data_ptr(),
                   self.val.data_ptr())
        self._pout = self.out.data_ptr()
        deg = (self.crow[1:] - self.crow[:-1]).to(torch.int64)
        self.build_info = dict(self.build_info)
        self.build_info.update({
            "тип индексов": str(self.col.dtype),
            "исходящих: максимум": int(deg.max()),
            "исходящих: среднее": round(float(deg.to(torch.float64).mean()), 1),
            "связей": int(self.val.numel()),
            "матрица по пресинаптическому, МБ": round(
                (self.crow.numel() * self.crow.element_size()
                 + self.col.numel() * self.col.element_size()
                 + self.val.numel() * self.val.element_size())
                / 1048576.0, 1),
        })

    def mm(self, sp, mo):
        """``out = W @ (sp * mo)`` по партиям. Возвращает свой буфер.

        ``sp`` и ``mo`` — [batch, n] и ОБЯЗАНЫ быть непрерывными: ядро
        читает их линейно. Проверка стоит здесь, а не в комментарии,
        потому что ошибка вида «пришёл срез» дала бы не отказ, а молча
        другие числа.
        """
        if sp.shape != (self.batch, self.n) or mo.shape != sp.shape:
            raise ValueError("ждал [%d, %d], пришло %s и %s"
                             % (self.batch, self.n, tuple(sp.shape),
                                tuple(mo.shape)))
        if not (sp.is_contiguous() and mo.is_contiguous()):
            raise ValueError("спайки и маска обязаны быть непрерывными")
        import torch

        self.out.zero_()
        rc = self._call(self._p[0], self._p[1], self._p[2],
                        sp.data_ptr(), mo.data_ptr(), self._pout,
                        self.n, self.batch, self.threads,
                        torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                "запуск событийного ядра вернул код CUDA %d" % rc)
        return self.out
