"""Пускатель FlyBrain: находит проект, проверяет его и открывает окно.

Внутри собранного `FlyBrain.exe` нет ни torch, ни CUDA, ни Taichi, ни
MuJoCo, ни коннектома. Он написан на одной стандартной библиотеке и делает
четыре вещи:

1. **находит проект** — от себя и вверх, пока не встретится каталог с
   `lab\\flybrain`; текущая рабочая папка при этом не важна вовсе;
2. **проверяет, что запускать есть чем** — окружение, модули проекта,
   файлы коннектома, — и при нехватке показывает окно с тем, чего именно
   нет, а не исчезает молча;
3. **не даёт запустить себя дважды** — именованный мьютекс на время
   работы окна, иначе рядом поднялся бы второй работник с теми же
   четырьмя гигабайтами данных;
4. **не оставляет хвостов** — окно и его работник живут в job-объекте
   Windows с `KILL_ON_JOB_CLOSE`, так что обрыв пускателя уносит всё
   дерево, а не оставляет работника висеть.

Почему не упаковываем весь стек. torch с CUDA, Taichi с Vulkan, MuJoCo и
четыре гигабайта коннектома в одном файле — это многоминутная сборка
гигабайтного артефакта после каждого обновления драйвера или колеса.
Пускатель собирается за секунды и обновляется правкой одного файла.

Почему именно PyInstaller, а не заглушка от setuptools (она весит 17 КБ
против восьми мегабайт). Заглушка запускает питон, путь к которому вшит в
неё строкой, и **своего интерпретатора не несёт**. Проверено: со сломанным
окружением она печатает «Cannot open …-script.py» и выходит с кодом 2, а
оконный её вариант не показывает вообще ничего. Пункт про понятную
диагностику так не выполнить: чтобы объяснить, что окружения нет, нужен
интерпретатор, который от этого окружения не зависит.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

APP_NAME = "FlyBrain"

# Подставляется при сборке: корень проекта на этой машине. Используется
# ТОЛЬКО если относительный поиск не нашёл ничего — например, когда exe
# унесли на рабочий стол. Это первая местная сборка, и знать свой дом ей
# не зазорно; переносимость этим не обещается.
BUILD_ROOT = r""

# Консольный близнец отличается только именем файла.
CONSOLE = Path(sys.argv[0]).stem.lower().endswith("console")


def _console_utf8():
    """Сказать консоли, что текст будет в utf-8, и писать в utf-8.

    `fly.cmd` не зовёт `chcp` и правильно делает: cmd.exe следит за своим
    местом в батнике по байтам, и смена кодовой страницы на ходу режет
    оставшиеся строки не там. Здесь не батник, а exe — переключать можно,
    и нужно: без этого русские подписи выходят кракозябрами.
    """
    if not CONSOLE:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:                                         # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                     # noqa: BLE001
            pass


_console_utf8()

ERROR_ALREADY_EXISTS = 183
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_EXTENDED_LIMIT_INFORMATION = 9


# ------------------------------------------------------------------ где мы

def project_root():
    """Корень проекта: каталог, в котором лежит `lab\\flybrain`.

    Ищем от самого exe вверх, а не от текущей рабочей папки: по двойному
    щелчку из проводника рабочей папкой бывает что угодно, вплоть до
    `C:\\Windows\\System32`.
    """
    start = Path(sys.executable if getattr(sys, "frozen", False)
                 else __file__).resolve().parent
    for base in (start, *start.parents):
        if (base / "lab" / "flybrain" / "cli.py").is_file():
            return base
    env = os.environ.get("FLYBRAIN_ROOT")
    if env and (Path(env) / "lab" / "flybrain" / "cli.py").is_file():
        return Path(env)
    if BUILD_ROOT and (Path(BUILD_ROOT) / "lab" / "flybrain"
                       / "cli.py").is_file():
        return Path(BUILD_ROOT)
    return None


def preflight(root):
    """Быстрая проверка по файлам. Возвращает список бед, пустой — если всё.

    Только то, что видно по диску за миллисекунды: тяжёлую проверку
    (torch, CUDA, драйверы) делает `fly doctor`, и зовут его лишь когда
    окно не поднялось.
    """
    bad = []
    py = root / ".venv" / "Scripts" / "python.exe"
    pyw = root / ".venv" / "Scripts" / "pythonw.exe"
    if not py.exists():
        bad.append(("окружение проекта", str(py),
                    "почините его: " + str(root / "fix-venv.cmd")))
    elif not pyw.exists() and not CONSOLE:
        bad.append(("pythonw.exe в окружении", str(pyw),
                    "окно без консоли запускается именно им; "
                    "почините окружение"))
    if not (root / "lab" / "flybrain" / "cli.py").is_file():
        bad.append(("модули проекта", str(root / "lab" / "flybrain"),
                    "каталог проекта неполон"))
    data = root / "upstream" / "data"
    for name in ("2025_Completeness_783.csv", "2025_Connectivity_783.parquet"):
        if not (data / name).exists():
            bad.append(("данные коннектома", str(data / name),
                        "клонируйте upstream заново"))
    return bad


# ------------------------------------------------------------ разговор с человеком

def say(title, lines):
    """Сказать человеку. Без консоли — окном, с консолью — в консоль."""
    text = "\n".join(lines)
    if CONSOLE:
        print(title)
        print(text)
        try:
            input("\nEnter — закрыть окно. ")
        except (EOFError, OSError):
            pass
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x40000)
    except Exception:                                         # noqa: BLE001
        print(title)
        print(text)


def diagnose(root, tail):
    """Тяжёлая диагностика — только когда окно не поднялось.

    Зовём `fly doctor` тем же окружением: он проверяет torch и CUDA,
    видеокарту, файлы модели, таблицу нейронов и место на диске. Своей
    проверки на то же самое у пускателя нет и не нужно.
    """
    py = root / ".venv" / "Scripts" / "python.exe"
    out = []
    if tail:
        out.append("Последнее, что сказало приложение:")
        out.extend("  " + t for t in tail[-12:])
        out.append("")
    if not py.exists():
        out.append("Окружения нет, диагностику запускать нечем.")
        return out
    out.append("Проверка окружения (fly doctor):")
    try:
        p = subprocess.run(
            [str(py), "-m", "flybrain.cli", "doctor"],
            cwd=str(root / "lab"), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        lines = (p.stdout or "").strip().splitlines()
        out.extend("  " + ln for ln in lines[-24:])
        if p.returncode != 0:
            out.append("  (doctor вернул код %d)" % p.returncode)
    except subprocess.TimeoutExpired:
        out.append("  doctor не ответил за пять минут")
    except OSError as exc:
        out.append("  doctor не запустился: %s" % exc)
    return out


# ---------------------------------------------------- один экземпляр и job

def single_instance(root):
    """Мьютекс на время работы. Вернуть (можно ли, ручка).

    Имя привязано к корню проекта: две копии проекта в разных каталогах
    друг другу не мешают, а вот второй запуск одной и той же копии — это
    второй работник, второй коннектом в памяти и драка за одну папку
    результатов.
    """
    try:
        import ctypes
        import hashlib

        tag = hashlib.sha1(str(root).lower().encode("utf-8")).hexdigest()[:16]
        h = ctypes.windll.kernel32.CreateMutexW(None, True,
                                                "Local\\FlyBrain-" + tag)
        if not h:
            return True, None            # не вышло — не мешаем запуску
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            ctypes.windll.kernel32.CloseHandle(h)
            return False, None
        return True, h
    except Exception:                                         # noqa: BLE001
        return True, None


def make_job():
    """Job-объект, который уносит всё дерево при закрытии ручки.

    Работник — внук пускателя (окно порождает его само), а дети членов
    job по умолчанию попадают в тот же job. Поэтому обрыв пускателя
    гасит и окно, и работника: висящих процессов с коннектомом в памяти
    не остаётся.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.windll.kernel32
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None, None
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = \
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
                job, JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return None, None
        return job, k32
    except Exception:                                         # noqa: BLE001
        return None, None


# ------------------------------------------------------------------- пуск

def main():                                                   # noqa: C901
    root = project_root()
    if root is None:
        say(APP_NAME + ": проект не найден", [
            "Не нашёл каталог проекта FlyBrain.",
            "",
            "Пускатель ищет папку, в которой лежит lab\\flybrain\\cli.py, "
            "начиная от себя и выше.",
            "",
            "Положите FlyBrain.exe внутрь проекта (например в папку "
            "release рядом с lab, out и .venv) или задайте переменную "
            "окружения FLYBRAIN_ROOT.",
        ])
        return 2

    bad = preflight(root)
    if bad:
        lines = ["Не хватает того, без чего приложение не поднимется:", ""]
        for what, where, hint in bad:
            lines.append("• %s" % what)
            lines.append("    %s" % where)
            if hint:
                lines.append("    %s" % hint)
        lines.append("")
        lines.append("Корень проекта: " + str(root))
        say(APP_NAME + ": не хватает файлов", lines)
        return 3

    py_name = "python.exe" if CONSOLE else "pythonw.exe"
    py = root / ".venv" / "Scripts" / py_name
    probe = root / ".venv" / "Scripts" / "python.exe"

    # Интерпретатор может стоять на месте и при этом не запускаться —
    # так бывает после переноса окружения. Проверяем до окна, иначе
    # человек увидит только мгновенное исчезновение.
    try:
        rc = subprocess.run([str(probe), "-c", "import sys"],
                            capture_output=True, timeout=120).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        say(APP_NAME + ": окружение не запускается",
            ["Интерпретатор окружения не стартует.", "", str(exc), "",
             "Почините: " + str(root / "fix-venv.cmd")])
        return 4
    if rc != 0:
        say(APP_NAME + ": окружение не запускается",
            ["Интерпретатор окружения отвечает кодом %d." % rc, "",
             "Почините: " + str(root / "fix-venv.cmd")])
        return 5

    args = list(sys.argv[1:])
    if "--check" in args:
        say(APP_NAME + ": проверка окружения",
            ["Корень проекта: " + str(root), "",
             *diagnose(root, [])])
        return 0

    ok, mutex = single_instance(root)
    if not ok:
        say(APP_NAME + " уже запущен", [
            "Окно FlyBrain уже открыто.",
            "",
            "Второй экземпляр поднял бы второго работника: ещё один "
            "коннектом в памяти видеокарты и две задачи, пишущие в одну "
            "папку результатов. Поэтому запуск остановлен.",
            "",
            "Если окна не видно, закройте его из диспетчера задач "
            "(python.exe или pythonw.exe) и запустите снова.",
        ])
        return 0

    job, k32 = make_job()
    cmd = [str(py), "-m", "flybrain.cli", "app", *args]
    cwd = str(root / "lab")
    creation = 0 if CONSOLE else getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, creationflags=creation,
            stdout=None if CONSOLE else subprocess.PIPE,
            stderr=None if CONSOLE else subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except OSError as exc:
        say(APP_NAME + ": не запустилось",
            ["Не удалось запустить приложение.", "", str(exc), "",
             "Команда: " + " ".join(cmd)])
        return 6

    if job and k32:
        try:
            k32.AssignProcessToJobObject(job, int(proc._handle))
        except Exception:                                     # noqa: BLE001
            pass

    tail = []
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-200]
    rc = proc.wait()
    lived = time.time() - started

    # Окно закрылось сразу и неудачно — это и есть тот случай, ради
    # которого пускатель остаётся жить: показать, что именно случилось.
    if rc != 0:
        lines = ["Приложение завершилось с кодом %d через %.1f с."
                 % (rc, lived), ""]
        lines += diagnose(root, tail)
        say(APP_NAME + ": приложение не запустилось", lines)
    if mutex:
        try:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(mutex)
        except Exception:                                     # noqa: BLE001
            pass
    if job and k32:
        try:
            k32.CloseHandle(job)
        except Exception:                                     # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
