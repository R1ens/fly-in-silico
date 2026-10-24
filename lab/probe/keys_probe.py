"""Разведка ввода в живом окне GGUI.

Показывает всё, что доходит до окна, двумя путями сразу и рядом:

* `flybrain.desk.keys.Keys` — подмена оконной процедуры Win32;
* `ti.ui.Window.get_events` — то, что GGUI отдаёт сам.

Из-за этого видно не только «пришло ли», но и «что при этом теряет GGUI».
Отчёт пишется в JSON при любом выходе, таблица охвата печатается в консоль.

Два режима:

``--selftest``
    Сообщения подставляются сами, через `SendMessage` прямо в оконную
    процедуру, а модификаторы — через `SetKeyboardState`. Нужен потому, что
    окно, запущенное из автоматики, не композитится: нажать клавишу некому,
    а проверить перехват надо. Код выхода — число непокрытых пунктов.

без ключа
    Окно ждёт живых нажатий `--seconds` секунд (по умолчанию 60) и
    отчитывается тем, что успели нажать. Выход — Esc или крестик.

Подсказка на экране печатается латиницей сознательно: во встроенном шрифте
imgui нет кириллицы, и русский текст там выйдет вопросительными знаками.
Правду показывает консоль и JSON — там кодировка своя.

Запуск:
    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\keys_probe.py --selftest
    ..\\.venv\\Scripts\\python.exe probe\\keys_probe.py --seconds 60
"""

import argparse
import ctypes as C
import ctypes.wintypes as W
import json
import sys
import time
from pathlib import Path

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import keys as K                             # noqa: E402
from flybrain.desk.textinput import TextInput                   # noqa: E402

TITLE = "keys probe"
OUT = Path(__file__).resolve().parent / "keys_report.json"

# Что обязательно должно дойти. Группы — те же, что в задании, чтобы по
# таблице сразу было видно, какой именно класс клавиш потерялся.
WANT = {
    "латиница": list("abcdefghijklmnopqrstuvwxyz"),
    "кириллица": list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "цифры верхнего ряда": list("0123456789"),
    "знаки": list(":+=,.-/\\;'[]()_*"),
    "пробел": [" "],
    "регистр": list("ABЮЯ"),
    "управление": ["backspace", "delete", "enter", "tab", "escape"],
    "стрелки": ["left", "right", "up", "down"],
    "навигация": ["home", "end", "pageup", "pagedown"],
    "цифры numpad": ["numpad0", "numpad5", "numpad9"],
    "сочетания": ["ctrl+c", "ctrl+v", "ctrl+a", "ctrl+x", "shift+left",
                  "ctrl+left"],
    "функциональные": ["f1", "f2", "f3", "f4", "f5"],
    "колесо": ["вверх", "вниз"],
}

# ---------------------------------------------------------------------------
# подстановка сообщений для --selftest

u32 = C.WinDLL("user32", use_last_error=True)
u32.SendMessageW.argtypes = [W.HWND, C.c_uint, C.c_ulonglong, C.c_longlong]
u32.SendMessageW.restype = C.c_longlong
u32.MapVirtualKeyW.argtypes = [C.c_uint, C.c_uint]
u32.SetKeyboardState.argtypes = [C.POINTER(C.c_ubyte)]
u32.GetKeyboardState.argtypes = [C.POINTER(C.c_ubyte)]

MAPVK_VK_TO_VSC = 0


def _lparam(vk, repeat=False):
    """Собрать lParam как это делает Windows: скан-код и бит прежнего нажатия.

    Скан-код настоящий, а не единица: GLFW по нему опознаёт клавишу в своей
    старой процедуре, и подсунутая единица сбивала бы её состояние клавиш.
    """
    sc = u32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF
    lp = 1 | (sc << 16)
    if repeat:
        lp |= 0x40000000
    return lp


class Inject:
    """Подстановщик сообщений прямо в оконную процедуру.

    `SendMessage`, а не `PostMessage`: сообщение из очереди сбрасывает
    таблицу состояния клавиш потока, и поддельный Ctrl до перехвата не
    доживает — проверено, модификаторы приходили пустыми.
    """

    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.state = (C.c_ubyte * 256)()
        u32.GetKeyboardState(self.state)

    def _mods(self, shift=False, ctrl=False, alt=False):
        self.state[K.VK_SHIFT] = 0x80 if shift else 0
        self.state[K.VK_CONTROL] = 0x80 if ctrl else 0
        self.state[K.VK_MENU] = 0x80 if alt else 0
        u32.SetKeyboardState(self.state)

    def char(self, ch, **mods):
        self._mods(**mods)
        # WM_CHAR несёт блок UTF-16, поэтому символ вне BMP уходит парой —
        # так его и шлёт Windows, и так его должен склеить перехват.
        buf = ch.encode("utf-16-le")
        for i in range(0, len(buf), 2):
            u32.SendMessageW(self.hwnd, K.WM_CHAR,
                             int.from_bytes(buf[i:i + 2], "little"), 0)
        self._mods()

    def key(self, vk, repeat=False, **mods):
        self._mods(**mods)
        u32.SendMessageW(self.hwnd, K.WM_KEYDOWN, vk, _lparam(vk, repeat))
        self._mods()

    def wheel(self, clicks):
        delta = int(round(clicks * 120)) & 0xFFFF
        u32.SendMessageW(self.hwnd, K.WM_MOUSEWHEEL, delta << 16, 0)


VK = {"backspace": 0x08, "tab": 0x09, "enter": 0x0D, "escape": 0x1B,
      "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
      "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
      "insert": 0x2D, "delete": 0x2E,
      "a": 0x41, "c": 0x43, "v": 0x56, "x": 0x58,
      "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74}


def inject_all(inj):
    """Подставить по одному представителю каждой проверяемой группы."""
    for group in ("латиница", "кириллица", "цифры верхнего ряда", "знаки",
                  "пробел", "регистр"):
        for ch in WANT[group]:
            inj.char(ch, shift=ch.isupper())
    for name in WANT["управление"] + WANT["стрелки"] + WANT["навигация"]:
        inj.key(VK[name])
    # Клавиши numpad дают тот же WM_CHAR, что верхний ряд, и отличить их
    # можно только по коду клавиши — поэтому проверяются сочетанием.
    for i in (0, 5, 9):
        inj.key(0x60 + i, ctrl=True)
    for name in ("c", "v", "a", "x"):
        inj.key(VK[name], ctrl=True)
    inj.key(VK["left"], shift=True)
    inj.key(VK["left"], ctrl=True)
    for name in WANT["функциональные"]:
        inj.key(VK[name])
    inj.key(VK["backspace"], repeat=True)
    inj.wheel(1)
    inj.wheel(-1)


# ---------------------------------------------------------------------------
# учёт охвата


def mark(cov, ev):
    """Отметить, какой пункт задания закрыло это событие."""
    if ev.kind == "wheel":
        cov["колесо"]["вверх" if ev.wheel > 0 else "вниз"] = True
        return
    if ev.kind == "text":
        ch = ev.text
        for group in ("латиница", "кириллица", "цифры верхнего ряда",
                      "знаки", "пробел", "регистр"):
            if ch in cov[group]:
                cov[group][ch] = True
        return
    n = ev.name
    for group in ("управление", "стрелки", "навигация", "функциональные"):
        if n in cov[group]:
            cov[group][n] = True
    if ev.ctrl and n in "0123456789" and f"numpad{n}" in cov["цифры numpad"]:
        cov["цифры numpad"][f"numpad{n}"] = True
    if ev.ctrl and f"ctrl+{n}" in cov["сочетания"]:
        cov["сочетания"][f"ctrl+{n}"] = True
    if ev.shift and f"shift+{n}" in cov["сочетания"]:
        cov["сочетания"][f"shift+{n}"] = True


def table(cov):
    """Напечатать таблицу охвата и вернуть список непокрытого."""
    missing = []
    print("\n" + "=" * 72)
    print(f"{'группа':<24}{'дошло':>8}  непокрытое")
    print("=" * 72)
    for group, items in cov.items():
        got = sum(1 for v in items.values() if v)
        bad = [k for k, v in items.items() if not v]
        missing += [f"{group}:{k}" for k in bad]
        show = " ".join(repr(k) if len(k) == 1 else k for k in bad[:10])
        if len(bad) > 10:
            show += f" … ещё {len(bad) - 10}"
        print(f"{group:<24}{got:>4}/{len(items):<3}  {show or '—'}")
    print("=" * 72)
    return missing


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="сколько ждать нажатий, потом выход с отчётом")
    ap.add_argument("--selftest", action="store_true",
                    help="подставить сообщения самому, без человека")
    ap.add_argument("--no-native", action="store_true",
                    help="нарочно пойти запасным путём, разбором get_events")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ti.init(arch=ti.vulkan)
    win = ti.ui.Window(TITLE, (820, 420), vsync=True, pos=(80, 80))
    canvas = win.get_canvas()
    gui = win.get_gui()

    src = K.Keys(win, title=TITLE, native=not args.no_native)
    field = TextInput(max_len=120, clipboard=src.clipboard)
    print(f"источник: {src!r}")
    if not src.native:
        print("ВНИМАНИЕ: нативный путь не удался, идёт разбор get_events.")
        for s in K.MISSING_WITHOUT_NATIVE:
            print("  теряется:", s)

    cov = {g: {k: False for k in items} for g, items in WANT.items()}
    log = []
    ggui_raw = []
    ggui_fields = None
    t0 = time.perf_counter()
    injected = False
    frame = 0

    while win.running:
        frame += 1
        now = time.perf_counter() - t0

        # то, что видит сам GGUI — для сравнения, а не для работы
        for e in win.get_events(ti.ui.PRESS):
            if ggui_fields is None:
                ggui_fields = sorted(n for n in dir(e)
                                     if not n.startswith("_"))
            ggui_raw.append({
                "key": repr(e.key),
                "action": repr(getattr(e, "action", None)),
            })

        for ev in src.poll():
            log.append({
                "kind": ev.kind, "text": ev.text, "name": ev.name,
                "wheel": ev.wheel, "shift": ev.shift, "ctrl": ev.ctrl,
                "alt": ev.alt, "repeat": ev.repeat,
                "frame": frame, "t": round(now, 3),
            })
            mark(cov, ev)
            field.apply_event(ev)
            print(f"  {now:7.3f} {ev!r}")
            if ev.kind == "key" and ev.name == "escape":
                win.running = False

        # Подстановка после первых кадров: до первого `show` очередь
        # сообщений ещё не прокручивалась ни разу.
        if args.selftest and not injected and frame > 3:
            if src.hwnd is None:
                print("ПРОВАЛ: подставлять сообщения некуда, hwnd не найден")
                break
            inject_all(Inject(src.hwnd))
            injected = True

        with gui.sub_window("probe", 0.02, 0.02, 0.96, 0.5) as g:
            g.text("press keys: letters, cyrillic, digits, : + = , . - / \\")
            g.text("and Backspace Delete Enter Tab Esc arrows Home End")
            g.text("PgUp PgDn Shift+letter Ctrl+C/V/A F1-F5, mouse wheel")
            g.text("")
            g.text(f"native win32 hook: {src.native}")
            g.text(f"events: {len(log)}   caret: {field.caret}")
            # Кириллицу встроенный шрифт imgui не покажет; правда — в консоли
            g.text("field (ascii only here): "
                   + "".join(c if " " <= c < "\x7f" else "?"
                             for c in field.text))
            g.text(f"quit: Esc    auto-quit in "
                   f"{max(0.0, args.seconds - now):.0f}s")

        canvas.set_background_color((0.07, 0.08, 0.10))
        win.show()

        if args.selftest and injected and frame > 12:
            break
        if now > args.seconds:
            break

    missing = table(cov)
    if ggui_fields:
        print("поля события GGUI:", ggui_fields)
    print(f"перехвачено событий: {len(log)}, "
          f"get_events за то же время: {len(ggui_raw)}")

    report = {
        "taichi": ti.__version__,
        "native": src.native,
        "hwnd": src.hwnd,
        "failure": src.failure,
        "selftest": bool(args.selftest),
        "seconds": args.seconds,
        "frames": frame,
        "ggui_event_fields": ggui_fields,
        "coverage": cov,
        "missing": missing,
        "events": log,
        "ggui_raw": ggui_raw,
        "field_text": field.text,
        "missing_without_native": list(K.MISSING_WITHOUT_NATIVE),
    }
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт:", args.out)

    src.close()
    win.destroy()
    return len(missing) if args.selftest else 0


if __name__ == "__main__":
    sys.exit(main())
