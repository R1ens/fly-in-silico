"""Модель однострочного поля ввода.

Здесь нет ни рисования, ни обращений к окну: поле знает только строку,
каретку и выделение. Из-за этого всё поведение — замена выделения, ход по
словам, предел длины, фильтр для числовых полей — проверяется без окна,
синтетическими событиями (`probe/textinput_test.py`).

Поле однострочное намеренно: в приложении им правят запрос-селектор
(`class:gustatory+side:left`), имя прогона и числа. Поэтому перевод строки не
вставляется никогда, а многострочная вставка схлопывается — иначе в поле
окажется текст, который оно не сможет показать.

События приходят извне и опознаются по полям (`kind`, `text`, `name`,
`shift`, `ctrl`, `alt`, `repeat`) — ровно то, что отдаёт `desk.keys.Event`.
Импорта оттуда здесь нет сознательно: модель не должна тянуть за собой
ctypes и Win32, чтобы её можно было прогнать где угодно.
"""

import re

__all__ = ["TextInput", "MemoryClipboard", "DIGITS", "FLOAT", "INT", "QUERY"]

# Переводы строки и табуляции, которые вставка схлопывает в один пробел.
_BREAKS = re.compile(r"[\r\n\t\v\f]+")

# Готовые фильтры. Пробел в числовые не входит: его сюда нечего ронять, а
# «1 2» потом не разберётся как число.
DIGITS = "0123456789"
INT = DIGITS + "-"
FLOAT = DIGITS + "-+.eE"
# Запрос-селектор: имена полей, значения и разделители `:` и `+`.
QUERY = ("abcdefghijklmnopqrstuvwxyz"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         + DIGITS + ":+_-*.")


class MemoryClipboard:
    """Буфер обмена в памяти процесса.

    Нужен затычкой: модель проверяется без окна, а настоящий буфер Windows
    даёт `desk.keys.WinClipboard`. Поле ввода про разницу не знает.
    """

    def __init__(self, text=""):
        self._text = text

    def get(self):
        return self._text

    def set(self, text):
        self._text = text


def _cls(ch):
    """Класс символа для хода по словам: пробел, слово, знак.

    Три класса, а не два, потому что в запросах `class:gustatory+side:left`
    разделители — это знаки, и Ctrl+стрелка должна останавливаться на них,
    а не перепрыгивать весь запрос как одно слово.
    """
    if ch.isspace():
        return 0
    if ch.isalnum() or ch == "_":
        return 1
    return 2


class TextInput:
    """Однострочное поле: строка, каретка, выделение.

    `caret` — номер позиции между символами, от 0 до `len(text)`.
    `sel_anchor` — вторая граница выделения или `None`, если выделения нет.
    """

    def __init__(self, text="", max_len=None, allow=None, clipboard=None,
                 on_change=None):
        self.max_len = max_len
        # `allow` — либо строка допустимых символов, либо предикат. Строка
        # удобнее для числовых полей, предикат — для чего-то вроде «только
        # то, что переварит разборщик запроса».
        self.allow = allow
        self.clipboard = clipboard if clipboard is not None else MemoryClipboard()
        self.on_change = on_change

        self.text = ""
        self.caret = 0
        self.sel_anchor = None
        self.set_text(text)

    # ------------------------------------------------------------------
    # строка и фильтр

    def _keep(self, ch):
        if self.allow is None:
            return True
        if callable(self.allow):
            return bool(self.allow(ch))
        return ch in self.allow

    def _clean(self, s):
        """Привести вставляемое к тому, что поле вообще может хранить."""
        if not s:
            return ""
        # Схлопываем в пробел, а не выбрасываем: иначе склеятся слова из
        # разных строк. Фильтр поля потом уберёт и этот пробел, если он в
        # нём не разрешён.
        s = _BREAKS.sub(" ", s)
        s = "".join(ch for ch in s if ch >= " " and ch != "\x7f")
        return "".join(ch for ch in s if self._keep(ch))

    def _touch(self):
        if self.on_change is not None:
            self.on_change(self)

    def set_text(self, s):
        """Заменить всё содержимое; каретка — в конец, выделение снято."""
        s = self._clean(str(s))
        if self.max_len is not None:
            s = s[: self.max_len]
        self.text = s
        self.caret = len(s)
        self.sel_anchor = None
        self._touch()
        return self.text

    # ------------------------------------------------------------------
    # выделение

    def selection(self):
        """Границы выделения `(a, b)`, a <= b. Без выделения — `(caret, caret)`."""
        if self.sel_anchor is None:
            return (self.caret, self.caret)
        a, b = self.sel_anchor, self.caret
        return (a, b) if a <= b else (b, a)

    def has_selection(self):
        a, b = self.selection()
        return a != b

    def selected_text(self):
        a, b = self.selection()
        return self.text[a:b]

    def select_all(self):
        self.sel_anchor = 0
        self.caret = len(self.text)
        return self.selection()

    def deselect(self):
        self.sel_anchor = None

    def _anchor(self, select):
        """Поставить или снять якорь перед движением каретки."""
        if select:
            if self.sel_anchor is None:
                self.sel_anchor = self.caret
        else:
            self.sel_anchor = None

    def _drop_selection(self):
        """Вырезать выделенное из строки; вернуть, было ли что вырезать."""
        a, b = self.selection()
        if a == b:
            return False
        self.text = self.text[:a] + self.text[b:]
        self.caret = a
        self.sel_anchor = None
        return True

    # ------------------------------------------------------------------
    # правка

    def insert(self, s):
        """Вставить текст вместо выделения, не выходя за предел длины."""
        s = self._clean(str(s))
        had = self._drop_selection()
        if not s:
            if had:
                self._touch()
            return self.text
        if self.max_len is not None:
            room = self.max_len - len(self.text)
            if room <= 0:
                if had:
                    self._touch()
                return self.text
            s = s[:room]
        i = self.caret
        self.text = self.text[:i] + s + self.text[i:]
        self.caret = i + len(s)
        self._touch()
        return self.text

    def backspace(self):
        """Удалить выделение, иначе символ перед кареткой."""
        if self._drop_selection():
            self._touch()
            return self.text
        if self.caret > 0:
            self.text = self.text[: self.caret - 1] + self.text[self.caret:]
            self.caret -= 1
            self._touch()
        return self.text

    def delete(self):
        """Удалить выделение, иначе символ под кареткой."""
        if self._drop_selection():
            self._touch()
            return self.text
        if self.caret < len(self.text):
            self.text = self.text[: self.caret] + self.text[self.caret + 1:]
            self._touch()
        return self.text

    def delete_word_left(self):
        """Ctrl+Backspace: снести слово перед кареткой."""
        if self._drop_selection():
            self._touch()
            return self.text
        j = self._word_pos(-1)
        if j != self.caret:
            self.text = self.text[:j] + self.text[self.caret:]
            self.caret = j
            self._touch()
        return self.text

    def delete_word_right(self):
        """Ctrl+Delete: снести слово за кареткой."""
        if self._drop_selection():
            self._touch()
            return self.text
        j = self._word_pos(1)
        if j != self.caret:
            self.text = self.text[: self.caret] + self.text[j:]
            self._touch()
        return self.text

    # ------------------------------------------------------------------
    # движение каретки

    def move(self, delta, select=False):
        """Сдвинуть каретку на `delta` символов.

        Без `select` при выделении каретка не едет, а садится на его край —
        так ведут себя все поля ввода, и это даёт привычный способ снять
        выделение, ничего не потеряв.
        """
        if not select and self.sel_anchor is not None and self.has_selection():
            a, b = self.selection()
            self.caret = a if delta < 0 else b
            self.sel_anchor = None
            return self.caret
        self._anchor(select)
        self.caret = max(0, min(len(self.text), self.caret + delta))
        if select and self.sel_anchor == self.caret:
            self.sel_anchor = None
        return self.caret

    def _word_pos(self, delta):
        """Куда встанет каретка при Ctrl+стрелке, как в обычных полях.

        Вправо: добрать текущий класс символов, затем пробелы, — каретка
        встаёт на начало следующего слова. Влево: сначала пробелы, затем
        класс символа перед кареткой.
        """
        n = len(self.text)
        i = self.caret
        if delta > 0:
            if i >= n:
                return n
            k = _cls(self.text[i])
            if k != 0:
                while i < n and _cls(self.text[i]) == k:
                    i += 1
            while i < n and _cls(self.text[i]) == 0:
                i += 1
            return i
        if i <= 0:
            return 0
        while i > 0 and _cls(self.text[i - 1]) == 0:
            i -= 1
        if i > 0:
            k = _cls(self.text[i - 1])
            while i > 0 and _cls(self.text[i - 1]) == k:
                i -= 1
        return i

    def move_word(self, delta, select=False):
        """Ctrl+стрелка: шаг по словам."""
        if not select:
            self.sel_anchor = None
        else:
            self._anchor(True)
        self.caret = self._word_pos(1 if delta > 0 else -1)
        if select and self.sel_anchor == self.caret:
            self.sel_anchor = None
        return self.caret

    def home(self, select=False):
        if not select:
            self.sel_anchor = None
        else:
            self._anchor(True)
        self.caret = 0
        if select and self.sel_anchor == 0:
            self.sel_anchor = None
        return self.caret

    def end(self, select=False):
        if not select:
            self.sel_anchor = None
        else:
            self._anchor(True)
        self.caret = len(self.text)
        if select and self.sel_anchor == self.caret:
            self.sel_anchor = None
        return self.caret

    # ------------------------------------------------------------------
    # буфер обмена

    def copy(self):
        """Скопировать выделение. Без выделения буфер не трогаем.

        Именно не трогаем, а не кладём пустую строку: иначе Ctrl+C мимо
        выделения затёр бы то, что человек копировал в другом окне.
        """
        s = self.selected_text()
        if s:
            self.clipboard.set(s)
        return s

    def cut(self):
        s = self.copy()
        if s:
            self._drop_selection()
            self._touch()
        return s

    def paste(self):
        return self.insert(self.clipboard.get() or "")

    # ------------------------------------------------------------------
    # события

    def apply_event(self, ev):
        """Отработать событие из `desk.keys`.

        Возвращает `True`, если событие поле забрало себе. Чужое (Enter,
        Tab, Esc, колесо, стрелки вверх-вниз) не забирается: ими окно
        переключает поля и закрывает панели, и поле не должно это глотать.
        """
        kind = getattr(ev, "kind", "")

        if kind == "text":
            s = getattr(ev, "text", "")
            if not s:
                return False
            # Ctrl+буква приходит и как символ тоже, если окно пустило
            # сообщение дальше; текст при зажатом Ctrl — не текст.
            if getattr(ev, "ctrl", False) and not getattr(ev, "alt", False):
                return False
            self.insert(s)
            return True

        if kind != "key":
            return False

        name = getattr(ev, "name", "")
        shift = bool(getattr(ev, "shift", False))
        ctrl = bool(getattr(ev, "ctrl", False))

        if name == "left":
            (self.move_word(-1, shift) if ctrl else self.move(-1, shift))
            return True
        if name == "right":
            (self.move_word(1, shift) if ctrl else self.move(1, shift))
            return True
        if name == "home":
            self.home(shift)
            return True
        if name == "end":
            self.end(shift)
            return True
        if name == "backspace":
            (self.delete_word_left() if ctrl else self.backspace())
            return True
        if name == "delete":
            # Shift+Delete и Shift/Ctrl+Insert — старые сочетания для буфера;
            # на части клавиатур ими пользуются вместо Ctrl+X и Ctrl+V.
            if shift:
                self.cut()
            elif ctrl:
                self.delete_word_right()
            else:
                self.delete()
            return True
        if name == "insert":
            if ctrl:
                self.copy()
            elif shift:
                self.paste()
            else:
                return False
            return True

        if ctrl and name == "a":
            self.select_all()
            return True
        if ctrl and name == "c":
            self.copy()
            return True
        if ctrl and name == "x":
            self.cut()
            return True
        if ctrl and name == "v":
            self.paste()
            return True

        return False

    # ------------------------------------------------------------------

    def __repr__(self):
        a, b = self.selection()
        return (f"TextInput({self.text!r}, caret={self.caret}, "
                f"sel=({a},{b}))")
