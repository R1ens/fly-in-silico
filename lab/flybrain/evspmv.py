"""Событийное умножение связей: доставка ядра и обвязка для движка LIF.

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

ДВЕ ДОРОГИ ДО ВИДЕОКАРТЫ, И ОДИН СЧЁТ
--------------------------------------

Счёт один, а доставить его на карту можно двумя способами. Они требуют
на машине пользователя разного, и это главное различие:

* :data:`LOAD_PTX` (**по умолчанию**) — `ev_spmv.ptx`, собранный здесь
  один раз и положенный в дерево рядом с источником. На машине
  пользователя его догружает JIT драйвера: `cuModuleLoadDataEx` плюс
  `cuLaunchKernel` из `nvcuda.dll`, который идёт с драйвером и есть на
  любой машине, где вообще работает torch. Хостовой обёртки на C больше
  нет вовсе, поэтому **ни `nvcc`, ни MSVC не нужны**. Замерено, а не
  предположено (`probe/kernel_ship.py`): с `FLYBRAIN_NVCC` и
  `FLYBRAIN_VCVARS`, уведёнными на несуществующие пути, движок считает
  и даёт тот же отпечаток;
* :data:`LOAD_NVCC` — прежняя дорога: `nvcc` собирает `ev_spmv.cu` в
  обычную DLL на месте, её зовут через `ctypes`. Требует `nvcc` и MSVC.
  Оставлена не для совместимости, а как ЭТАЛОН ДОСТАВКИ: по ней мерилась
  скорость до появления PTX, и она же проверяет, что JIT-дорога даёт те
  же байты. Выбирается явно: ``FLYBRAIN_EV_LOAD=nvcc``.

Обе дороги запускают ОДНИ И ТЕ ЖЕ два ядра из одного `ev_spmv.cu`
(`ev_spmv_i32_k`, `ev_spmv_i64_k`) — у них имена на простом C именно
для того, чтобы PTX-дорога искала их по имени, а не по правилам
искажения имён компилятора.

**Отказ обязан быть громким.** В этом проекте уже есть механизм,
который отказывает молча: `sim.Engine.use_graphs` возвращает False и
оставляет шаг втрое медленнее, ничего не сообщая. Второй такой отравил
бы все будущие замеры: прогон шёл бы на `csrmv` с потерей втрое, а в
отчёте стояло бы «событийное». Поэтому здесь нет ни одного тихого
отката, и в частности **дороги не подменяют друг друга**: если выбрана
PTX-дорога, а артефакта нет или он разошёлся с источником — это
:class:`EventKernelUnavailable`, а не молчаливая сборка через `nvcc`.
Откат на `csrmv` есть, но выбирается он явно — `spmv="эталонный"` или
`FLYBRAIN_SPMV=эталонный`.

ПРОИСХОЖДЕНИЕ АРТЕФАКТА
-----------------------

`ev_spmv.ptx` — не бинарник неизвестного происхождения: это текст, он
читается и сравнивается построчно, а рядом лежит `ev_spmv.ptx.json` —
чем собран, из какого источника, с какими флагами и с какими
отпечатками. Проверяется это в двух местах:

* при загрузке — отпечаток самого PTX и **отпечаток источника**: если
  `ev_spmv.cu` правили, а PTX не пересобрали, движок откажется громко.
  Иначе правка источника молча не доезжала бы до карты, а это ровно тот
  класс расхождений, которого здесь быть не должно;
* пробой `probe/kernel_ship.py --происхождение` — она пересобирает PTX
  из источника тем же `nvcc` и сверяет с лежащим в дереве побайтово.

В `.venv` не ставится ничего: драйверный API зовётся через `ctypes` на
`nvcuda.dll`, а дорога через `nvcc` собирает в отдельный каталог
(``FLYBRAIN_EXT_DIR``, по умолчанию ``%TEMP%/flybrain-ext``).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
PTX = Path(__file__).with_name("ev_spmv.ptx")
PTX_MANIFEST = Path(__file__).with_name("ev_spmv.ptx.json")
ABI = 3            # обязано совпасть с `ev_spmv_abi()` в .cu и с манифестом

# Имена ядер в модуле. Простые, на C: PTX-дорога ищет их по имени.
KERNELS = {"int32": b"ev_spmv_i32_k", "int64": b"ev_spmv_i64_k"}

# Виртуальная архитектура артефакта. `compute_52` — не «пониже на
# всякий случай», а решение: ядру нужен только `atomicAdd` по float32
# (есть с compute_20), и чем ниже виртуальная цель, тем больше поколений
# карт покрывает JIT драйвера вперёд. Замерено, что на `sm_89` PTX с
# `compute_52` даёт ТЕ ЖЕ БАЙТЫ и ту же скорость, что родной `sm_89`
# cubin (`probe/kernel_ship.py`, §скорость): выбор виртуальной цели на
# арифметику не влияет, потому что в счёте одно умножение и атомарное
# добавление, и сливать их в FMA не с чем.
PTX_TARGET = "compute_52"

# Версия ISA, с которой артефакт КЛАДЁТСЯ В ДЕРЕВО. `nvcc` 12.8 пишет в
# PTX `.version 8.7`, и это не свойство счёта, а свойство компилятора:
# драйвер обязан понимать объявленную версию, поэтому артефакт от
# нынешнего `nvcc` требовал бы драйвера R570 и новее. Колёса torch под
# cu128 при этом работают и на драйверах от R525 — то есть оставался бы
# промежуток, где torch считает, а наше ядро не загружается. Ровно та
# зависимость, которую вся эта дорога убирает.
#
# Поэтому ОДНА директива понижается (:func:`lower_ptx_version`), и
# значение выбрано замером, а не вкусом (`probe/kernel_ship.py`, §7):
# принимаются 8.7…4.1, а 4.0 отвергается, и отвергается с называнием
# причины — «PTX .version 4.0 does not support .target sm_52». То есть
# границу задаёт не арифметика, а выбранная виртуальная цель: `sm_52`
# появился в CUDA 6.5, а это и есть ISA 4.1. Ниже спускаться было бы
# можно только вместе с целью, а это сузило бы набор карт снизу и
# меняло бы ВТОРУЮ директиву — чего преобразование делать не вправе.
#
# ISA 4.1 отвечает CUDA 6.5, то есть драйверам R340 и новее: промежуток
# R525…R569 закрыт с большим запасом.
PTX_VERSION = "4.1"

# Директивы PTX, которые читаются и правятся по одной. Регулярные
# выражения привязаны к началу строки и к концу: `.version` в тексте
# ровно одна, и преобразование обязано это проверить, а не надеяться.
_VERSION_RE = re.compile(r"^\.version[ \t]+(\S+)[ \t]*$", re.M)
_TARGET_RE = re.compile(r"^\.target[ \t]+(\S+)[ \t]*$", re.M)

# Какой дорогой доставлять ядро. Одна переменная, одно значение:
# ``ptx`` (по умолчанию) или ``nvcc``. Подмены одной дороги другой нет
# ни в какую сторону — выбранная либо работает, либо отказывает громко.
LOAD_PTX = "ptx"
LOAD_NVCC = "nvcc"
LOAD_CHOICES = (LOAD_PTX, LOAD_NVCC)
LOAD_DEFAULT = os.environ.get("FLYBRAIN_EV_LOAD", LOAD_PTX)

_LIB = None        # загруженное ядро, одно на процесс
_BUILD = None      # чем и как доставлено: уходит в отчёт прогона


class EventKernelUnavailable(RuntimeError):
    """Событийное ядро недоступно, и это сказано громко.

    Поднимается вместо любого тихого отката: ни на другую дорогу
    доставки, ни на `csrmv` движок сам не отступает.
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


def _sha(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


# --- поиск средств сборки (нужны ТОЛЬКО дороге через nvcc) ------------


def find_nvcc():
    """Путь к `nvcc` или громкий отказ со списком, где искали.

    Нужен дороге `LOAD_NVCC` и генератору артефакта. На машине
    пользователя, где дорога `LOAD_PTX`, не зовётся вовсе.
    """
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

    На Windows `nvcc` без хостового компилятора не соберёт ничего — даже
    PTX: отказ приходит не как «нет MSVC», а как невнятная ошибка вида
    «Cannot find compiler 'cl.exe' in PATH». Поэтому хостовой компилятор
    ищется отдельно и называется отдельно. Нужен только здесь, при
    сборке артефакта и на дороге `LOAD_NVCC`.
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


def device_sm():
    """Вычислительная способность ТОГО устройства, на котором считаем."""
    import torch

    if not torch.cuda.is_available():
        raise EventKernelUnavailable(
            "событийное умножение связей требует CUDA, а torch её не "
            "видит (torch.cuda.is_available() == False). Выберите "
            "эталонное умножение явно: spmv=\"эталонный\".")
    major, minor = torch.cuda.get_device_capability()
    return "%d%d" % (major, minor)


def arch_flag():
    """`-gencode` под ТО устройство, на котором считаем.

    Архитектура спрашивается у видеокарты, а не вписана числом: DLL,
    собранная под `sm_89`, на другой карте просто не запустится, и узнать
    об этом из невнятной ошибки запуска хуже, чем собрать под своё.
    """
    sm = device_sm()
    return "-gencode", "arch=compute_%s,code=sm_%s" % (sm, sm), sm


def ext_dir():
    base = os.environ.get("FLYBRAIN_EXT_DIR")
    if not base:
        base = Path(os.environ.get("TEMP") or os.environ.get("TMP")
                    or ".") / "flybrain-ext"
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_nvcc(args, where, tag, timeout=600):
    """Позвать `nvcc` под `vcvars64`. Возвращает (CompletedProcess, журнал).

    Через .bat, а не строкой в `cmd /c`: кавычки в пути к vcvars64
    subprocess на Windows переэкранирует, и вызов ломается.
    """
    nvcc = find_nvcc()
    vcvars = find_vcvars()
    bat = where / ("build_%s.bat" % tag)
    bat.write_text(
        "@echo off\r\n"
        'call "%s" >nul\r\n'
        'if errorlevel 1 exit /b 90\r\n'
        '"%s" %s\r\n' % (vcvars, nvcc, args),
        encoding="utf-8")
    got = subprocess.run(["cmd", "/c", str(bat)], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         timeout=timeout)
    log = where / ("build_%s.log" % tag)
    log.write_text("=== команда ===\n%s %s\n=== stdout ===\n%s\n"
                   "=== stderr ===\n%s"
                   % (nvcc, args, got.stdout, got.stderr), encoding="utf-8")
    return got, log, nvcc, vcvars


# --- понижение версии ISA: одна директива и ничего больше -------------


def ptx_version(text):
    """Объявленная версия ISA. Директива обязана быть ровно одна."""
    found = _VERSION_RE.findall(text)
    if len(found) != 1:
        raise EventKernelUnavailable(
            "в PTX директив `.version` %d, а ожидается ровно одна: текст "
            "не похож на вывод nvcc для одного модуля" % len(found))
    return found[0]


def ptx_target(text):
    """Объявленная цель. Нужна для проверки, что её НЕ тронули."""
    found = _TARGET_RE.findall(text)
    if len(found) != 1:
        raise EventKernelUnavailable(
            "в PTX директив `.target` %d, а ожидается ровно одна"
            % len(found))
    return found[0]


def _ver_tuple(v):
    return tuple(int(x) for x in str(v).split("."))


def lower_ptx_version(text, to=None):
    """Понизить ОДНУ директиву `.version` и больше ничего не тронуть.

    Возвращает ``(текст, было, стало)``.

    Зачем это вообще, написано у :data:`PTX_VERSION`: версия ISA — это
    свойство компилятора, а не счёта, и она задаёт нижнюю границу
    драйвера. Понижение расширяет набор машин, на которых артефакт
    загрузится, и **не может** изменить результат: в счёте одно
    умножение и атомарное добавление, FMA сливать не с чем, а если
    пониженная версия не поддержит какую-то конструкцию, JIT драйвера
    откажется ГРОМКО (замерено: `.version 4.0` отвергается с «does not
    support .target sm_52»), а не посчитает иначе.

    Почему это функция, а не правка файла руками. Артефакт в научном
    репозитории обязан выводиться из источника воспроизводимо. Здесь
    преобразование описано кодом, применяется к тому, что выдал `nvcc`,
    и **проверяет себя на месте**: сравнение построчно обязано дать
    ровно одно расхождение, и оно обязано быть в `.version`. Если
    преобразование когда-нибудь зацепит вторую строку — оно откажет, а
    не отдаст молча другой артефакт.
    """
    to = str(to or PTX_VERSION)
    have = ptx_version(text)
    target_before = ptx_target(text)
    if _ver_tuple(to) > _ver_tuple(have):
        raise EventKernelUnavailable(
            "просили поставить версию ISA %s вместо %s — это ПОВЫШЕНИЕ. "
            "Поднимать версию нельзя: она объявляет то, чего в тексте "
            "может не быть, и сузила бы набор драйверов вместо того чтобы "
            "расширить." % (to, have))
    out = _VERSION_RE.sub(".version " + to, text, count=1)

    # Проверка на месте, а не в комментарии: ровно одно расхождение, и
    # оно в `.version`. Без неё правка регулярным выражением однажды
    # зацепила бы лишнее и отдала бы другой артефакт молча.
    a, b = text.splitlines(), out.splitlines()
    if len(a) != len(b):
        raise EventKernelUnavailable(
            "понижение версии ISA изменило число строк: %d -> %d"
            % (len(a), len(b)))
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b), 1) if x != y]
    if len(diff) != 1:
        raise EventKernelUnavailable(
            "понижение версии ISA тронуло %d строк, а должно одну: %s"
            % (len(diff), [i for i, _, _ in diff]))
    line, was, now = diff[0]
    if not (_VERSION_RE.fullmatch(was) and _VERSION_RE.fullmatch(now)):
        raise EventKernelUnavailable(
            "понижение версии ISA тронуло не ту строку (%d): было %r, "
            "стало %r" % (line, was, now))
    if ptx_target(out) != target_before:
        raise EventKernelUnavailable(
            "понижение версии ISA изменило цель: %s -> %s"
            % (target_before, ptx_target(out)))
    return out, have, to


def ptx_transform_note(was, now):
    """Строка в манифест: каким преобразованием одно получено из другого."""
    return ("понижена одна директива: `.version %s` -> `.version %s` "
            "(`evspmv.lower_ptx_version`); в тексте не изменено больше "
            "ничего, цель `.target` та же" % (was, now))


# --- сборка артефакта PTX (делается ЗДЕСЬ, не у пользователя) ---------


def nvcc_version(nvcc=None):
    """Строка версии `nvcc` — в манифест, чтобы было чем собрано."""
    nvcc = Path(nvcc or find_nvcc())
    got = subprocess.run([str(nvcc), "--version"], capture_output=True,
                         text=True, timeout=120)
    for line in (got.stdout or "").splitlines():
        if "release" in line:
            return line.strip()
    return (got.stdout or "").strip().splitlines()[-1:] or "неизвестно"


def compile_ptx(target=None, where=None, verbose=False, version=None):
    """Собрать PTX из `ev_spmv.cu`: ``(артефакт, манифест, как_выдал_nvcc)``.

    Это делается на машине разработчика один раз, результат кладётся в
    дерево. На машине пользователя не зовётся: там PTX уже есть.

    Возвращаются ДВА текста: то, что выдал `nvcc`, и то, что ложится в
    дерево — второе получается из первого понижением одной директивы
    `.version` (:func:`lower_ptx_version`, зачем — у :data:`PTX_VERSION`).
    Оба нужны вызывающему, потому что сверка происхождения идёт в две
    ступени: пересобрать и сверить с «как выдал `nvcc`», потом применить
    понижение и сверить с положенным в дерево.

    Манифест возвращается рядом с текстами, чтобы происхождение считалось
    из того же вызова, которым артефакт и собран — иначе они разъехались
    бы при первой же правке.
    """
    target = str(target or PTX_TARGET)
    where = Path(where or ext_dir())
    where.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        raise EventKernelUnavailable(
            "нет источника ядра %s — он часть пакета, а не генерируется "
            "на месте" % SRC)
    src_text = SRC.read_text(encoding="utf-8")
    tag = hashlib.sha256(("ptx|%s|%s" % (src_text, target))
                         .encode("utf-8")).hexdigest()[:16]
    cu = where / ("ev_spmv_%s.cu" % tag)
    out = where / ("ev_spmv_%s.ptx" % tag)
    cu.write_text(src_text, encoding="utf-8")
    # `-Wno-deprecated-gpu-targets` — не замалчивание ошибки: nvcc 12.8
    # предупреждает, что офлайн-сборка под цели ниже compute_75 когда-то
    # будет убрана. Предупреждение записано в манифест, а не потеряно.
    flags = ["-ptx", "-arch=%s" % target, "-Wno-deprecated-gpu-targets",
             "-o", str(out), str(cu)]
    args = " ".join(('"%s"' % a) if (" " in a or "\\" in a) else a
                    for a in flags)
    got, log, nvcc, vcvars = _run_nvcc(args, where, tag)
    if verbose:
        print(got.stdout[-4000:])
        print(got.stderr[-4000:])
    if got.returncode != 0 or not out.exists():
        raise EventKernelUnavailable(
            "nvcc вернул %d и PTX не появился.\n"
            "  nvcc:    %s\n  vcvars:  %s\n  цель:    %s\n  журнал:  %s\n"
            "последние строки:\n%s\n%s"
            % (got.returncode, nvcc, vcvars, target, log,
               (got.stdout or "")[-1500:], (got.stderr or "")[-1500:]))
    raw = out.read_text(encoding="utf-8")
    missing = [k.decode() for k in KERNELS.values()
               if (".entry %s" % k.decode()) not in raw]
    if missing:
        raise EventKernelUnavailable(
            "в собранном PTX нет точек входа %s. Имена ядер в "
            "`ev_spmv.cu` разошлись с `KERNELS` в evspmv.py — PTX-дорога "
            "искала бы их по имени и не нашла." % missing)
    # Понижение одной директивы. Оно и есть то, что ложится в дерево;
    # `raw` остаётся рядом как «как выдал nvcc», чтобы сверка
    # происхождения шла в две ступени и обе были проверяемы.
    text, was, now = lower_ptx_version(raw, version)
    manifest = {
        "артефакт": PTX.name,
        "источник": SRC.name,
        "отпечаток источника": _sha(src_text),
        "отпечаток артефакта": _sha(text),
        "отпечаток как выдал nvcc": _sha(raw),
        "ядра": [k.decode() for k in KERNELS.values()],
        "abi": ABI,
        "цель": target,
        "цель в артефакте": ptx_target(text),
        "версия ptx": now,
        "версия ptx как выдал nvcc": was,
        "преобразование": ptx_transform_note(was, now),
        "собран чем": nvcc_version(nvcc),
        "nvcc при сборке": str(nvcc),
        "хостовой компилятор при сборке": str(vcvars),
        "флаги": flags[:3],
        "предупреждения nvcc": [
            ln.strip() for ln in ((got.stdout or "") + (got.stderr or ""))
            .splitlines() if "warning" in ln.lower()],
        "как пересобрать и сверить":
            "lab/probe/kernel_ship.py --происхождение",
    }
    return text, manifest, raw


# --- драйверный API через ctypes --------------------------------------


# Коды CUresult, которые здесь надо называть по-человечески: по ним
# видно, ЧТО именно не сошлось, а не просто «ошибка 222».
_CU_ERR = {
    100: "нет устройства CUDA (CUDA_ERROR_NO_DEVICE)",
    101: "неверное устройство (CUDA_ERROR_INVALID_DEVICE)",
    200: "образ модуля не разобран (CUDA_ERROR_INVALID_IMAGE): "
         "файл PTX испорчен",
    209: "в образе нет кода для этой карты "
         "(CUDA_ERROR_NO_BINARY_FOR_GPU)",
    218: "PTX не принят (CUDA_ERROR_INVALID_PTX)",
    219: "неверный контекст графики (CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)",
    221: "в драйвере нет JIT-компилятора "
         "(CUDA_ERROR_JIT_COMPILER_NOT_FOUND)",
    222: "версия PTX новее, чем понимает драйвер "
         "(CUDA_ERROR_UNSUPPORTED_PTX_VERSION) — нужен драйвер посвежее",
}

_CU_JIT_ERROR_LOG_BUFFER = 5
_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6


class _Driver:
    """`nvcuda.dll` — драйверный API CUDA через `ctypes`.

    Почему через `ctypes`, а не `cuda-python`: в `.venv` ставить ничего
    нельзя, а `nvcuda.dll` ставится вместе с драйвером и есть на любой
    машине, где torch вообще видит карту. Именно это и снимает
    требование `nvcc`/MSVC: хостового кода на C больше нет.
    """

    _inst = None

    def __init__(self):
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        try:
            self.cu = (ctypes.WinDLL(name) if sys.platform == "win32"
                       else ctypes.CDLL(name))
        except OSError as exc:
            raise EventKernelUnavailable(
                "не загрузился %s — драйверный API CUDA. Он ставится "
                "вместе с драйвером NVIDIA; если torch видит карту, он "
                "обязан быть. Ошибка загрузчика: %s\n"
                "Выберите эталонное умножение явно: spmv=\"эталонный\"."
                % (name, exc))
        cu = self.cu
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cu.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        cu.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
            ctypes.c_char_p]
        cu.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 6
            + [ctypes.c_uint, ctypes.c_void_p,
               ctypes.POINTER(ctypes.c_void_p),
               ctypes.POINTER(ctypes.c_void_p)])
        self.check(cu.cuInit(0), "cuInit")
        ver = ctypes.c_int()
        self.check(cu.cuDriverGetVersion(ctypes.byref(ver)),
                   "cuDriverGetVersion")
        self.version = int(ver.value)

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def why(self, rc):
        msg = ctypes.c_char_p()
        try:
            self.cu.cuGetErrorString(ctypes.c_int(rc), ctypes.byref(msg))
            text = (msg.value or b"").decode("utf-8", "replace")
        except Exception:
            text = ""
        known = _CU_ERR.get(int(rc))
        parts = [p for p in (known, text) if p]
        return "CUresult %d (%s)" % (rc, "; ".join(parts) or "без описания")

    def check(self, rc, what):
        if rc != 0:
            raise EventKernelUnavailable(
                "драйверный API CUDA: %s -> %s" % (what, self.why(rc)))

    def context(self):
        """Текущий контекст, а если его нет — первичный контекст карты.

        torch ставит первичный контекст текущим в своём потоке, когда
        впервые что-то считает; модуль обязан грузиться в ТОТ ЖЕ
        контекст, иначе ручки функций в нём недействительны. Если
        контекста нет (загрузка идёт раньше первой работы torch),
        первичный берётся явно — тот же самый, что возьмёт потом torch.
        """
        import torch

        ctx = ctypes.c_void_p()
        self.check(self.cu.cuCtxGetCurrent(ctypes.byref(ctx)),
                   "cuCtxGetCurrent")
        if ctx.value:
            return ctx, "текущий контекст потока"
        dev = int(torch.cuda.current_device())
        self.check(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                   "cuDevicePrimaryCtxRetain")
        self.check(self.cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        return ctx, "первичный контекст устройства %d" % dev


class PtxKernel:
    """Ядро из PTX, положенного в дерево, через JIT драйвера.

    Ни `nvcc`, ни MSVC на машине не нужны. Происхождение артефакта
    проверяется ДО загрузки, и расхождение — громкий отказ.
    """

    road = LOAD_PTX

    def __init__(self, verbose=False):
        if not PTX.exists():
            raise EventKernelUnavailable(
                "нет артефакта ядра %s. Это PTX, собранный один раз и "
                "положенный в дерево; на машине пользователя он не "
                "собирается, а читается. Если дерево неполное — верните "
                "файл из истории; если источник правили — пересоберите:\n"
                "    lab/probe/kernel_ship.py --пересобрать\n"
                "Либо выберите эталонное умножение явно: "
                "spmv=\"эталонный\"." % PTX)
        if not PTX_MANIFEST.exists():
            raise EventKernelUnavailable(
                "нет происхождения артефакта %s. Без него %s — бинарник "
                "неизвестного происхождения, и грузить его нельзя. "
                "Пересоберите: lab/probe/kernel_ship.py --пересобрать"
                % (PTX_MANIFEST, PTX.name))
        man = json.loads(PTX_MANIFEST.read_text(encoding="utf-8"))
        text = PTX.read_text(encoding="utf-8")
        got = _sha(text)
        if got != man.get("отпечаток артефакта"):
            raise EventKernelUnavailable(
                "артефакт %s разошёлся со своим происхождением:\n"
                "  в файле:   %s\n  в манифесте: %s\n"
                "Это значит, что PTX правили руками или он испорчен. "
                "Пересоберите из источника: "
                "lab/probe/kernel_ship.py --пересобрать"
                % (PTX.name, got, man.get("отпечаток артефакта")))
        # Версия и цель читаются ИЗ САМОГО артефакта и сверяются с
        # манифестом. Отпечаток это уже покрывает, но при расхождении
        # сказать «версия в файле 8.7, а в манифесте 4.1» полезнее, чем
        # «отпечатки разные»: сразу видно, что кто-то правил `.version`
        # руками вместо пересборки.
        got_ver, got_target = ptx_version(text), ptx_target(text)
        want_ver = man.get("версия ptx")
        want_target = man.get("цель в артефакте")
        if want_ver and got_ver != want_ver:
            raise EventKernelUnavailable(
                "в артефакте %s объявлена версия ISA %s, а в манифесте "
                "%s. Пересоберите: lab/probe/kernel_ship.py --пересобрать"
                % (PTX.name, got_ver, want_ver))
        if want_target and got_target != want_target:
            raise EventKernelUnavailable(
                "в артефакте %s объявлена цель %s, а в манифесте %s. "
                "Пересоберите: lab/probe/kernel_ship.py --пересобрать"
                % (PTX.name, got_target, want_target))
        if int(man.get("abi", -1)) != ABI:
            raise EventKernelUnavailable(
                "артефакт %s собран под abi %s, а движок ждёт %d: "
                "артефакт и обвязка разошлись. Пересоберите: "
                "lab/probe/kernel_ship.py --пересобрать"
                % (PTX.name, man.get("abi"), ABI))
        # Главная проверка против МОЛЧАЛИВОГО расхождения: правка
        # `ev_spmv.cu` без пересборки PTX означала бы, что на карте
        # считает прежнее ядро, а в дереве лежит новый счёт. Это не
        # поймал бы ни один шлюз равенства, потому что обе дороги
        # грузили бы один и тот же устаревший артефакт.
        if SRC.exists():
            src_fp = _sha(SRC.read_text(encoding="utf-8"))
            if src_fp != man.get("отпечаток источника"):
                raise EventKernelUnavailable(
                    "источник %s правили, а артефакт %s не пересобирали:\n"
                    "  источник сейчас: %s\n  в манифесте:     %s\n"
                    "На карте считало бы ПРЕЖНЕЕ ядро, а в дереве лежал "
                    "бы новый счёт — молча. Пересоберите:\n"
                    "    lab/probe/kernel_ship.py --пересобрать"
                    % (SRC.name, PTX.name, src_fp,
                       man.get("отпечаток источника")))
        self.manifest = man
        drv = _Driver.get()
        self.drv = drv
        ctx, how = drv.context()
        sm = device_sm()

        image = text.encode("utf-8") + b"\0"
        err = ctypes.create_string_buffer(8192)
        opts = (ctypes.c_int * 2)(_CU_JIT_ERROR_LOG_BUFFER,
                                  _CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        vals = (ctypes.c_void_p * 2)(
            ctypes.cast(err, ctypes.c_void_p).value,
            ctypes.c_void_p(len(err)).value)
        mod = ctypes.c_void_p()
        t0 = time.perf_counter()
        rc = drv.cu.cuModuleLoadDataEx(ctypes.byref(mod), image, 2, opts,
                                       vals)
        jit_s = time.perf_counter() - t0
        if rc != 0:
            raise EventKernelUnavailable(
                "JIT драйвера не принял %s (цель %s, карта sm_%s):\n"
                "  %s\n  журнал JIT: %s\n"
                "Если сказано, что версия PTX новее драйвера — нужен "
                "драйвер посвежее либо пересборка артефакта под более "
                "старый nvcc. Выбрать эталонное умножение явно: "
                "spmv=\"эталонный\"."
                % (PTX.name, man.get("цель"), sm, drv.why(rc),
                   err.value.decode("utf-8", "replace").strip() or "пуст"))
        self.mod = mod
        self.fn = {}
        for dtype, name in KERNELS.items():
            fn = ctypes.c_void_p()
            drv.check(drv.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name),
                      "cuModuleGetFunction(%s)" % name.decode())
            self.fn[dtype] = fn
        if verbose:
            print("PTX загружен за %.3f с, цель %s, карта sm_%s"
                  % (jit_s, man.get("цель"), sm))
        self.info = {
            "ядро": "событийное",
            "дорога доставки": "PTX + JIT драйвера (nvcc и MSVC не нужны)",
            "архитектура": "sm_%s" % sm,
            "источник": str(SRC),
            "артефакт": str(PTX),
            "цель артефакта": man.get("цель"),
            "версия ptx": man.get("версия ptx"),
            "версия ptx как выдал nvcc": man.get("версия ptx как выдал nvcc"),
            "преобразование артефакта": man.get("преобразование"),
            "отпечаток артефакта": man.get("отпечаток артефакта"),
            "отпечаток как выдал nvcc": man.get("отпечаток как выдал nvcc"),
            "отпечаток источника": man.get("отпечаток источника"),
            "собран чем": man.get("собран чем"),
            "версия драйвера": drv.version,
            "контекст": how,
            "загрузка (JIT), с": round(jit_s, 3),
            "нитей в блоке": THREADS,
            "abi": ABI,
            "собрано сейчас": False,
        }

    def launcher(self, dtype, n, batch, threads):
        """Запускатель под конкретные размеры. Возвращает ``call(ptrs…)``.

        Параметры запуска собираются один раз: на повторе меняются только
        указатели, а решётка и блок постоянны — иначе шаг не записался бы
        в CUDA-граф.
        """
        key = str(dtype).replace("torch.", "")
        if key not in self.fn:
            raise EventKernelUnavailable(
                "индексы матрицы типа %s: ядро знает только %s"
                % (dtype, ", ".join(self.fn)))
        fn = self.fn[key]
        drv = self.drv
        slots = [ctypes.c_void_p(0) for _ in range(6)]
        c_n = ctypes.c_int(int(n))
        params = (ctypes.c_void_p * 7)(
            *[ctypes.addressof(s) for s in slots],
            ctypes.addressof(c_n))
        grid = (int(n), int(batch), 1)
        block = (int(threads), 1, 1)

        def call(crow, col, val, sp, mo, out, stream):
            slots[0].value = crow
            slots[1].value = col
            slots[2].value = val
            slots[3].value = sp
            slots[4].value = mo
            slots[5].value = out
            return drv.cu.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                0, ctypes.c_void_p(stream), params, None)

        # Ссылки на ctypes-объекты держатся в замыкании `call`; без них
        # `params` указывал бы в освобождённую память.
        call._keep = (slots, c_n, params)
        return call

    def why(self, rc):
        return self.drv.why(rc)


class DllKernel:
    """Ядро из DLL, собранной `nvcc` на месте. Требует `nvcc` и MSVC.

    Эталон доставки: по этой дороге мерилась скорость до появления PTX, и
    по ней же проверяется, что JIT-дорога даёт те же байты.
    """

    road = LOAD_NVCC

    def __init__(self, force=False, verbose=False):
        if sys.platform != "win32":
            raise EventKernelUnavailable(
                "сборка событийного ядра через nvcc написана под Windows "
                "(vcvars64.bat, __declspec(dllexport)). На %s выберите "
                "дорогу PTX (FLYBRAIN_EV_LOAD=ptx) или эталонное "
                "умножение явно: spmv=\"эталонный\"." % sys.platform)
        if not SRC.exists():
            raise EventKernelUnavailable(
                "нет источника ядра %s — он часть пакета, а не "
                "генерируется на месте" % SRC)
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
        built_now = False
        t0 = time.perf_counter()
        if force or not dll.exists():
            cu.write_text(src_text, encoding="utf-8")
            args = ('-O3 -shared %s %s -o "%s" "%s"'
                    % (gen_a, gen_b, dll, cu))
            got, log, nvcc, vcvars = _run_nvcc(args, where, key)
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
                       (got.stdout or "")[-1500:],
                       (got.stderr or "")[-1500:]))
            built_now = True
        took = time.perf_counter() - t0

        lib = ctypes.CDLL(str(dll))
        lib.ev_spmv_abi.restype = ctypes.c_int
        lib.ev_spmv_abi.argtypes = []
        got_abi = int(lib.ev_spmv_abi())
        if got_abi != ABI:
            raise EventKernelUnavailable(
                "загруженная DLL %s отвечает abi %d, а ожидается %d: кэш "
                "разошёлся с источником. Уберите каталог %s и соберите "
                "снова." % (dll, got_abi, ABI, where))
        self.lib = lib
        self._call = {}
        for key_t, name in (("int32", "ev_spmv_i32"),
                            ("int64", "ev_spmv_i64")):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = ([ctypes.c_void_p] * 6
                           + [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p])
            self._call[key_t] = fn
        self.info = {
            "ядро": "событийное",
            "дорога доставки": "nvcc -> DLL (требует nvcc и MSVC)",
            "архитектура": "sm_%s" % sm,
            "nvcc": str(nvcc), "vcvars64": str(vcvars),
            "источник": str(SRC), "dll": str(dll),
            "сборка, с": round(took, 3),
            "нитей в блоке": THREADS, "abi": ABI,
            "собрано сейчас": built_now,
        }

    def launcher(self, dtype, n, batch, threads):
        key = str(dtype).replace("torch.", "")
        if key not in self._call:
            raise EventKernelUnavailable(
                "индексы матрицы типа %s: ядро знает только %s"
                % (dtype, ", ".join(self._call)))
        fn = self._call[key]
        n, batch, threads = int(n), int(batch), int(threads)

        def call(crow, col, val, sp, mo, out, stream):
            return fn(crow, col, val, sp, mo, out, n, batch, threads, stream)

        return call

    def why(self, rc):
        return "код CUDA %d" % rc


def build(force=False, verbose=False, load=None):
    """Доставить (или взять из кэша) событийное ядро.

    Возвращает ``(kern, info)``: объект с ``launcher()`` и словарь
    происхождения, который уходит в отчёт прогона. Любая нехватка —
    :class:`EventKernelUnavailable`, никогда не None и никогда не тихий
    откат и не подмена дороги.
    """
    global _LIB, _BUILD
    name = str(LOAD_DEFAULT if load is None else load)
    if name not in LOAD_CHOICES:
        raise ValueError("дорога доставки ядра %r не из списка %s"
                         % (name, LOAD_CHOICES))
    if _LIB is not None and not force and _LIB.road == name:
        return _LIB, _BUILD
    kern = (PtxKernel(verbose=verbose) if name == LOAD_PTX
            else DllKernel(force=force, verbose=verbose))
    _LIB, _BUILD = kern, kern.info
    return _LIB, _BUILD


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

    def __init__(self, w_csr, batch, device, threads=None, verbose=False,
                 load=None):
        import torch

        from . import data as D

        self.kern, self.build_info = build(verbose=verbose, load=load)
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
        if self.col.dtype not in (torch.int32, torch.int64):
            raise EventKernelUnavailable(
                "индексы матрицы типа %s: ядро знает только int32 и int64"
                % self.col.dtype)
        self._call = self.kern.launcher(self.col.dtype, self.n, self.batch,
                                        self.threads)
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
                        torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError("запуск событийного ядра вернул %s"
                               % self.kern.why(rc))
        return self.out
