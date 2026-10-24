"""Источник событий клавиатуры и колеса для окна GGUI.

Зачем вообще отдельный источник. `ti.ui.Window.get_events` отдаёт события с
единственным полем `key`, и в нём либо односимвольное имя клавиши в нижнем
регистре, либо одно из полутора десятков имён (`BackSpace`, `Return`, …).
Этого хватает на горячие клавиши и ни на что больше: ни кириллицы, ни
запятой с двоеточием, ни регистра, ни вставки, ни колеса мыши — события
колеса в GGUI нет вовсе.

Как это обойдено. Окно GGUI — это окно GLFW: класс `GLFW30`, и на нём
выставлено оконное свойство `GLFW`, по которому его и опознать среди трёх
окон, которые GLFW создаёт в потоке. Сам C-интерфейс GLFW внутри
`taichi_python.pyd` слинкован статически и наружу не выведен (в таблице
экспорта из `glfw*` нет ничего, кроме методов самого Taichi, принимающих
`GLFWwindow*`), поэтому `glfwSetCharCallback` вызвать нельзя — ни из pyd, ни
из отдельной `glfw3.dll`: у неё свой экземпляр библиотеки и своя раскладка
структур, и передавать ей чужой указатель значит портить память.

Поэтому источник берётся на один слой ниже GLFW — там, откуда GLFW сам всё
и берёт: оконная процедура Win32 подменяется своей, а старая вызывается из
неё же, так что GGUI и его imgui продолжают получать всё как раньше. Оттуда
приходят:

* `WM_CHAR` — готовый кодовый блок UTF-16 с учётом раскладки, регистра,
  мёртвых клавиш и ввода через IME. Это и есть источник **текста**;
* `WM_KEYDOWN` — коды клавиш управления, уже с автоповтором от Windows;
* `WM_MOUSEWHEEL` — колесо, которого в GGUI нет.

Буфер обмена тоже берётся напрямую из Win32, без GLFW.

Если подмена не удалась (не Windows, окна не нашлось), источник молча
переходит на разбор `get_events`, и наружу разница видна только по флагу
`native`. Что при этом теряется, перечислено в `MISSING_WITHOUT_NATIVE`.
"""

import time

__all__ = ["Event", "Keys", "WinClipboard", "MISSING_WITHOUT_NATIVE"]

MISSING_WITHOUT_NATIVE = (
    "кириллица и любая нелатинская раскладка",
    "знаки препинания, кроме пробела",
    "различение регистра букв",
    "колесо мыши",
    "автоповтор при удержании приходит не от системы, а подделывается",
)

# ---------------------------------------------------------------------------
# событие

_MODS = ("shift", "ctrl", "alt")


class Event:
    """Одно событие ввода.

    `kind` — `"text"`, `"key"` или `"wheel"`:

    * `"text"` — в `text` один символ, уже с учётом раскладки и регистра;
    * `"key"` — в `name` имя управляющей клавиши в нижнем регистре
      (`left`, `right`, `home`, `end`, `backspace`, `delete`, `enter`,
      `tab`, `escape`, `up`, `down`, `pageup`, `pagedown`, `insert`,
      `f1`…`f12`), а для сочетаний с Ctrl или Alt — сама буква или цифра
      (`a`, `c`, `v`, `x`, `1`…);
    * `"wheel"` — в `wheel` дельта в «щелчках», вверх положительна.

    Буква при обычном наборе приходит только как `"text"`: имя клавиши для
    неё не выдаётся, иначе каждый нажатый символ приходил бы дважды.
    """

    __slots__ = ("kind", "text", "name", "wheel", "shift", "ctrl", "alt",
                 "repeat")

    def __init__(self, kind, text="", name="", wheel=0.0,
                 shift=False, ctrl=False, alt=False, repeat=False):
        self.kind = kind
        self.text = text
        self.name = name
        self.wheel = wheel
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.repeat = repeat

    @classmethod
    def char(cls, ch, **mods):
        return cls("text", text=ch, **mods)

    @classmethod
    def key(cls, name, **mods):
        return cls("key", name=name, **mods)

    @classmethod
    def scroll(cls, delta, **mods):
        return cls("wheel", wheel=float(delta), **mods)

    def __repr__(self):
        on = "+".join(m for m in _MODS if getattr(self, m))
        on = ("[" + on + "]") if on else ""
        rep = "*" if self.repeat else ""
        if self.kind == "text":
            return f"<text {self.text!r}{on}{rep}>"
        if self.kind == "key":
            return f"<key {self.name}{on}{rep}>"
        return f"<wheel {self.wheel:+g}{on}>"


# ---------------------------------------------------------------------------
# Win32

try:                                    # ctypes есть всегда, user32 — нет
    import ctypes as _C
    import ctypes.wintypes as _W

    _u32 = _C.WinDLL("user32", use_last_error=True)
    _k32 = _C.WinDLL("kernel32", use_last_error=True)
except Exception:                       # noqa: BLE001
    _C = _W = _u32 = _k32 = None

GWLP_WNDPROC = -4
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_CHAR, WM_SYSCHAR, WM_UNICHAR = 0x0102, 0x0106, 0x0109
WM_MOUSEWHEEL = 0x020A
UNICODE_NOCHAR = 0xFFFF

VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12

# Клавиши, у которых есть смысл имени. Буквы и цифры сюда не входят: их
# имена выдаются только при зажатом Ctrl или Alt, и считаются из кода.
VK_NAMES = {
    0x08: "backspace", 0x09: "tab", 0x0D: "enter", 0x1B: "escape",
    0x21: "pageup", 0x22: "pagedown", 0x23: "end", 0x24: "home",
    0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down",
    0x2D: "insert", 0x2E: "delete", 0x14: "capslock",
}
for _i in range(1, 13):                 # F1…F12
    VK_NAMES[0x6F + _i] = f"f{_i}"

# Имена, которыми те же клавиши зовутся в `get_events` GGUI.
GGUI_NAMES = {
    "BackSpace": "backspace", "Return": "enter", "Tab": "tab",
    "Escape": "escape", "Up": "up", "Down": "down",
    "Left": "left", "Right": "right", "CapsLock": "capslock",
}
# Эти GGUI отдаёт как события, но полю ввода они не нужны.
GGUI_SKIP = {"Shift", "Alt", "Control", "LMB", "MMB", "RMB"}

# Что повторять при удержании, когда повтор приходится подделывать.
REPEATABLE = ("left", "right", "up", "down", "home", "end",
              "backspace", "delete")
REPEAT_DELAY = 0.35
REPEAT_RATE = 28.0


def _mods_now():
    """Модификаторы на момент сообщения.

    `GetKeyState`, а не `GetAsyncKeyState`: первый отдаёт состояние на
    момент разбираемого сообщения, и именно его читает сам GLFW. Со вторым
    быстрое отпускание Shift приводило бы к потере регистра.
    """
    return (
        _u32.GetKeyState(VK_SHIFT) < 0,
        _u32.GetKeyState(VK_CONTROL) < 0,
        _u32.GetKeyState(VK_MENU) < 0,
    )


class WinClipboard:
    """Буфер обмена Windows напрямую, без GLFW.

    Формат один — `CF_UNICODETEXT`: другие (`CF_TEXT`, `CF_OEMTEXT`) Windows
    сама сводит к нему, а обратного пересчёта через кодовую страницу
    кириллица не переживает.
    """

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    def __init__(self, tries=6, pause=0.01):
        # Буфером владеет одно окно на всю систему, и `OpenClipboard` честно
        # отказывает, пока он занят чужим процессом. Поэтому не один заход.
        self.tries = tries
        self.pause = pause

    def _open(self):
        for _ in range(self.tries):
            if _u32.OpenClipboard(None):
                return True
            time.sleep(self.pause)
        return False

    def get(self):
        if _u32 is None or not self._open():
            return ""
        try:
            h = _u32.GetClipboardData(self.CF_UNICODETEXT)
            if not h:
                return ""
            p = _k32.GlobalLock(h)
            if not p:
                return ""
            try:
                return _C.wstring_at(p)
            finally:
                _k32.GlobalUnlock(h)
        except Exception:               # noqa: BLE001
            return ""
        finally:
            _u32.CloseClipboard()

    def set(self, text):
        if _u32 is None or not self._open():
            return False
        try:
            buf = str(text).encode("utf-16-le") + b"\x00\x00"
            h = _k32.GlobalAlloc(self.GMEM_MOVEABLE, len(buf))
            if not h:
                return False
            p = _k32.GlobalLock(h)
            if not p:
                return False
            _C.memmove(p, buf, len(buf))
            _k32.GlobalUnlock(h)
            _u32.EmptyClipboard()
            # После удачного SetClipboardData память принадлежит системе и
            # освобождать её нельзя.
            return bool(_u32.SetClipboardData(self.CF_UNICODETEXT, h))
        except Exception:               # noqa: BLE001
            return False
        finally:
            _u32.CloseClipboard()


def _bind():
    """Объявить типы вызовов Win32 — один раз на процесс."""
    if _u32 is None or getattr(_bind, "done", False):
        return
    LRESULT = _C.c_longlong if _C.sizeof(_C.c_void_p) == 8 else _C.c_long
    _u32.GetClassNameW.argtypes = [_W.HWND, _W.LPWSTR, _C.c_int]
    _u32.GetWindowTextW.argtypes = [_W.HWND, _W.LPWSTR, _C.c_int]
    _u32.IsWindowVisible.argtypes = [_W.HWND]
    _u32.GetPropW.argtypes = [_W.HWND, _W.LPCWSTR]
    _u32.GetPropW.restype = _C.c_void_p
    _u32.SetWindowLongPtrW.argtypes = [_W.HWND, _C.c_int, _C.c_void_p]
    _u32.SetWindowLongPtrW.restype = _C.c_void_p
    _u32.GetWindowLongPtrW.argtypes = [_W.HWND, _C.c_int]
    _u32.GetWindowLongPtrW.restype = _C.c_void_p
    _u32.CallWindowProcW.argtypes = [_C.c_void_p, _W.HWND, _C.c_uint,
                                     _C.c_ulonglong, _C.c_longlong]
    _u32.CallWindowProcW.restype = LRESULT
    _u32.GetKeyState.argtypes = [_C.c_int]
    _u32.GetKeyState.restype = _C.c_short
    _u32.OpenClipboard.argtypes = [_W.HWND]
    _u32.GetClipboardData.argtypes = [_C.c_uint]
    _u32.GetClipboardData.restype = _C.c_void_p
    _u32.SetClipboardData.argtypes = [_C.c_uint, _C.c_void_p]
    _u32.SetClipboardData.restype = _C.c_void_p
    _k32.GlobalAlloc.argtypes = [_C.c_uint, _C.c_size_t]
    _k32.GlobalAlloc.restype = _C.c_void_p
    _k32.GlobalLock.argtypes = [_C.c_void_p]
    _k32.GlobalLock.restype = _C.c_void_p
    _k32.GlobalUnlock.argtypes = [_C.c_void_p]
    _bind.LRESULT = LRESULT
    _bind.ENUMPROC = _C.WINFUNCTYPE(_C.c_bool, _W.HWND, _W.LPARAM)
    _bind.WNDPROC = _C.WINFUNCTYPE(LRESULT, _W.HWND, _C.c_uint,
                                   _C.c_ulonglong, _C.c_longlong)
    _u32.EnumThreadWindows.argtypes = [_W.DWORD, _bind.ENUMPROC, _W.LPARAM]
    _bind.done = True


def find_window(title=None):
    """Найти HWND окна GGUI в текущем потоке.

    В потоке живут три окна класса `GLFW30`: само окно, скрытое `Dummy
    Window` (его GLFW держит для выбора формата пикселей) и `GLFW message
    window` без свойства `GLFW`. Поэтому отбор идёт по свойству, заголовку
    и видимости, а не по классу.
    """
    if _u32 is None:
        return None
    _bind()
    seen = []

    @_bind.ENUMPROC
    def _collect(hwnd, _):
        cls = _C.create_unicode_buffer(64)
        _u32.GetClassNameW(hwnd, cls, 64)
        if cls.value != "GLFW30" or not _u32.GetPropW(hwnd, "GLFW"):
            return True
        txt = _C.create_unicode_buffer(512)
        _u32.GetWindowTextW(hwnd, txt, 512)
        seen.append((hwnd, txt.value, bool(_u32.IsWindowVisible(hwnd))))
        return True

    _u32.EnumThreadWindows(_k32.GetCurrentThreadId(), _collect, 0)
    if title:
        for hwnd, txt, _vis in seen:
            if txt == title:
                return hwnd
    for hwnd, txt, vis in seen:
        if vis and txt and txt != "Dummy Window":
            return hwnd
    for hwnd, txt, _vis in seen:
        if txt and txt != "Dummy Window":
            return hwnd
    return None


# ---------------------------------------------------------------------------
# источник


class Keys:
    """Ввод за кадр. Создаётся один раз на окно, опрашивается раз в кадр.

    ::

        keys = Keys(window, title="Мозг мухи")
        ...
        for ev in keys.poll():
            field.apply_event(ev)

    `native` говорит, удалась ли подмена оконной процедуры. Всё остальное
    от этого не зависит.
    """

    def __init__(self, window=None, title=None, native=True, queue_max=512):
        self.window = window
        self.title = title
        self.hwnd = None
        self.native = False
        self.clipboard = WinClipboard() if _u32 is not None else None
        if self.clipboard is None:
            from .textinput import MemoryClipboard
            self.clipboard = MemoryClipboard()

        self._queue = []
        self._queue_max = queue_max
        self._proc = None               # ссылку держать обязательно
        self._old = None
        self._pending_high = 0          # половина суррогатной пары
        self._held = {}                 # подделка повтора в запасном пути
        self.failure = None

        if native:
            try:
                self._hook()
            except Exception as exc:    # noqa: BLE001
                self.failure = repr(exc)
                self.native = False
        else:
            self.failure = "нативный путь выключен вызывающим"

    # ------------------------------------------------------------------

    def _hook(self):
        hwnd = find_window(self.title)
        if hwnd is None:
            self.failure = "окна GLFW в этом потоке не нашлось"
            return
        _bind()
        old = _u32.GetWindowLongPtrW(hwnd, GWLP_WNDPROC)
        if not old:
            self.failure = "не удалось прочитать старую оконную процедуру"
            return

        queue = self._queue

        @_bind.WNDPROC
        def proc(h, msg, wp, lp):
            # Внутри — только разбор и запись в список: эта процедура
            # вызывается из чужого цикла разбора сообщений, и всё тяжёлое
            # или падающее здесь уронит окно, а не кадр.
            try:
                self._on_message(msg, wp, lp)
                if msg == WM_UNICHAR and wp == UNICODE_NOCHAR:
                    return 1            # да, мы понимаем WM_UNICHAR
            except Exception:           # noqa: BLE001
                pass
            # Старую процедуру звать обязательно: на ней держится и GGUI, и
            # его imgui, и закрытие окна по крестику.
            return _u32.CallWindowProcW(old, h, msg, wp, lp)

        _u32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, _C.cast(proc, _C.c_void_p))
        self.hwnd, self._old, self._proc = hwnd, old, proc
        self.native = True
        del queue

    def _push(self, ev):
        if len(self._queue) < self._queue_max:
            self._queue.append(ev)

    def _on_message(self, msg, wp, lp):
        if msg in (WM_CHAR, WM_SYSCHAR, WM_UNICHAR):
            self._on_char(msg, wp)
            return
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            self._on_key(wp, lp)
            return
        if msg == WM_MOUSEWHEEL:
            sh, ct, al = _mods_now()
            delta = _C.c_short((wp >> 16) & 0xFFFF).value / 120.0
            self._push(Event.scroll(delta, shift=sh, ctrl=ct, alt=al))

    def _on_char(self, msg, wp):
        code = int(wp)
        if msg == WM_UNICHAR:
            if code == UNICODE_NOCHAR:
                return
        else:
            # WM_CHAR несёт блок UTF-16, и символы вне BMP приходят парой.
            if 0xD800 <= code <= 0xDBFF:
                self._pending_high = code
                return
            if 0xDC00 <= code <= 0xDFFF:
                if not self._pending_high:
                    return
                code = (0x10000 + ((self._pending_high - 0xD800) << 10)
                        + (code - 0xDC00))
                self._pending_high = 0
            else:
                self._pending_high = 0
        if code < 0x20 or code == 0x7F:
            # Backspace, Enter, Tab и Ctrl+буква тоже приходят как WM_CHAR,
            # но управляющими кодами. Текстом они не являются — их место в
            # событиях WM_KEYDOWN, где видно, что именно нажато.
            return
        sh, ct, al = _mods_now()
        self._push(Event.char(chr(code), shift=sh, ctrl=ct, alt=al))

    def _on_key(self, wp, lp):
        vk = int(wp)
        if vk in (VK_SHIFT, VK_CONTROL, VK_MENU):
            return
        sh, ct, al = _mods_now()
        rep = bool(int(lp) & 0x40000000)        # бит 30 — клавиша уже была нажата
        name = VK_NAMES.get(vk)
        if name is None:
            # Буквы и цифры имеют смысл как имя только в сочетании: сам
            # символ придёт следующим сообщением WM_CHAR.
            if not (ct or al):
                return
            if 0x41 <= vk <= 0x5A:
                name = chr(vk + 32)
            elif 0x30 <= vk <= 0x39:
                name = chr(vk)
            elif 0x60 <= vk <= 0x69:
                name = chr(vk - 0x60 + 0x30)
            else:
                return
        self._push(Event.key(name, shift=sh, ctrl=ct, alt=al, repeat=rep))

    # ------------------------------------------------------------------

    def poll(self):
        """События, накопленные с прошлого кадра, в порядке прихода."""
        if self.native:
            out, self._queue[:] = list(self._queue), []
            return out
        return self._poll_ggui()

    def _poll_ggui(self):
        """Запасной путь: разбор `window.get_events`.

        Приходит сюда то же, что видит обычный код на GGUI, и теряется всё
        из `MISSING_WITHOUT_NATIVE`. Держится ради не-Windows и ради случая,
        когда окно почему-то не нашлось.
        """
        if self.window is None:
            return []
        sh = bool(self.window.is_pressed("Shift"))
        ct = bool(self.window.is_pressed("Control"))
        al = bool(self.window.is_pressed("Alt"))
        out = []
        for e in self.window.get_events("Press"):
            k = str(e.key)
            if k in GGUI_SKIP:
                continue
            name = GGUI_NAMES.get(k)
            if name is not None:
                out.append(Event.key(name, shift=sh, ctrl=ct, alt=al))
            elif len(k) == 1:
                if ct or al:
                    out.append(Event.key(k.lower(), shift=sh, ctrl=ct, alt=al))
                else:
                    out.append(Event.char(k, shift=sh, ctrl=ct, alt=al))
        out.extend(self._fake_repeat(sh, ct, al))
        return out

    def _fake_repeat(self, sh, ct, al):
        """Повтор при удержании там, где его не даёт система.

        В нативном пути повтор приходит от Windows вместе с сообщением, и
        подделывать нечего; здесь же остаётся только опрашивать состояние
        клавиш и считать время самому.
        """
        now = time.monotonic()
        out = []
        ggui = {v: k for k, v in GGUI_NAMES.items()}
        for name in REPEATABLE:
            key = ggui.get(name)
            if key is None or not self.window.is_pressed(key):
                self._held.pop(name, None)
                continue
            t0 = self._held.get(name)
            if t0 is None:
                self._held[name] = now + REPEAT_DELAY
                continue
            while now >= t0:
                out.append(Event.key(name, shift=sh, ctrl=ct, alt=al,
                                     repeat=True))
                t0 += 1.0 / REPEAT_RATE
            self._held[name] = t0
        return out

    # ------------------------------------------------------------------

    def close(self):
        """Вернуть окну его оконную процедуру.

        Без этого ctypes освободит нашу процедуру раньше окна, и первое же
        сообщение после сборки мусора уйдёт в освобождённую память.
        """
        if self.native and self.hwnd is not None and self._old is not None:
            try:
                _u32.SetWindowLongPtrW(self.hwnd, GWLP_WNDPROC,
                                       _C.c_void_p(self._old))
            except Exception:           # noqa: BLE001
                pass
        self.native = False
        self._proc = None
        self._old = None
        self._queue[:] = []

    def __del__(self):
        try:
            self.close()
        except Exception:               # noqa: BLE001
            pass

    def __repr__(self):
        way = "win32" if self.native else f"get_events ({self.failure})"
        return f"<Keys {way} hwnd={self.hwnd}>"
