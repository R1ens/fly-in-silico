"""Проверка модели поля ввода — без окна.

Поле прогоняется синтетическими событиями `desk.keys.Event`, теми же, что
придут из подмены оконной процедуры. Окно здесь не нужно и не создаётся:
модель про рисование ничего не знает, и проверять её живым окном значило бы
проверять заодно растеризатор и раскладку.

Запуск:
    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\textinput_test.py
Код выхода — число провалов.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk.keys import Event                            # noqa: E402
from flybrain.desk.textinput import (                           # noqa: E402
    FLOAT, QUERY, MemoryClipboard, TextInput,
)

QUERY_TEXT = "class:gustatory+side:left"

fails = []
done = 0


def ok(name, got, want):
    """Одно утверждение. Печатает строку таблицы, копит провалы."""
    global done
    done += 1
    good = got == want
    mark = "  ok " if good else "ПРОВАЛ"
    print(f"{mark} {name}")
    if not good:
        print(f"       получено {got!r}")
        print(f"       ожидалось {want!r}")
        fails.append(name)
    return good


def state(f):
    """Чем поле отвечает: строка, каретка, границы выделения."""
    a, b = f.selection()
    return (f.text, f.caret, a, b)


def type_text(f, s, **mods):
    """Набрать строку посимвольно, как это делает WM_CHAR."""
    for ch in s:
        f.apply_event(Event.char(ch, **mods))


def press(f, name, n=1, **mods):
    for i in range(n):
        f.apply_event(Event.key(name, repeat=i > 0, **mods))


def section(t):
    print("\n" + "-" * 68)
    print(t)
    print("-" * 68)


# ---------------------------------------------------------------------------
section("1. набор")

f = TextInput()
type_text(f, "sugar-run")
ok("1 латиница посимвольно", state(f), ("sugar-run", 9, 9, 9))

f = TextInput()
type_text(f, "прогон №7")
ok("2 кириллица посимвольно", state(f), ("прогон №7", 9, 9, 9))

f = TextInput()
type_text(f, QUERY_TEXT)
ok("3 запрос-селектор целиком", f.text, QUERY_TEXT)
ok("4 каретка в конце набранного", f.caret, len(QUERY_TEXT))

f = TextInput()
type_text(f, "Ab")                      # регистр приходит готовым из WM_CHAR
ok("5 регистр букв сохраняется", f.text, "Ab")


# ---------------------------------------------------------------------------
section("2. выделение и замена")

f = TextInput(QUERY_TEXT)
press(f, "left", 4, shift=True)
ok("6 Shift+Left выделяет справа налево", state(f),
   (QUERY_TEXT, 21, 21, 25))
ok("7 выделенное — это 'left'", f.selected_text(), "left")

type_text(f, "right")
ok("8 набор заменяет выделение", state(f),
   ("class:gustatory+side:right", 26, 26, 26))

f = TextInput(QUERY_TEXT)
f.select_all()
ok("9 select_all выделяет всё", f.selection(), (0, 25))
type_text(f, "x")
ok("10 набор поверх select_all", state(f), ("x", 1, 1, 1))

f = TextInput("abcdef")
f.caret = 2
press(f, "right", 3, shift=True)
press(f, "left")
ok("11 Left при выделении садится на левый край", state(f),
   ("abcdef", 2, 2, 2))

f = TextInput("abcdef")
f.caret = 2
press(f, "right", 3, shift=True)
press(f, "right")
ok("12 Right при выделении садится на правый край", state(f),
   ("abcdef", 5, 5, 5))


# ---------------------------------------------------------------------------
section("3. Home, End, пределы")

f = TextInput(QUERY_TEXT)
press(f, "home")
ok("13 Home в начало", f.caret, 0)
press(f, "end")
ok("14 End в конец", f.caret, 25)
press(f, "home", shift=True)
ok("15 Shift+Home выделяет до начала", f.selection(), (0, 25))

f = TextInput("ab")
press(f, "left", 5)
ok("16 каретка не уходит левее нуля", f.caret, 0)
press(f, "right", 5)
ok("17 каретка не уходит правее строки", f.caret, 2)


# ---------------------------------------------------------------------------
section("4. Ctrl+стрелки по словам")

f = TextInput(QUERY_TEXT)
press(f, "home")
steps = []
for _ in range(7):
    press(f, "right", ctrl=True)
    steps.append(f.caret)
ok("18 Ctrl+Right по словам и разделителям", steps,
   [5, 6, 15, 16, 20, 21, 25])

press(f, "end")
steps = []
for _ in range(7):
    press(f, "left", ctrl=True)
    steps.append(f.caret)
ok("19 Ctrl+Left по словам и разделителям", steps,
   [21, 20, 16, 15, 6, 5, 0])

f = TextInput("имя прогона тут")
press(f, "end")
press(f, "left", ctrl=True)
ok("20 Ctrl+Left через пробел на кириллице", f.caret, 12)
press(f, "left", ctrl=True, shift=True)
ok("21 Ctrl+Shift+Left выделяет слово", f.selected_text(), "прогона ")


# ---------------------------------------------------------------------------
section("5. удаление")

f = TextInput(QUERY_TEXT)
press(f, "left", 4, shift=True)
press(f, "backspace")
ok("22 Backspace на выделении сносит выделение", state(f),
   ("class:gustatory+side:", 21, 21, 21))

f = TextInput("абвгд")
press(f, "end")
press(f, "backspace")
ok("23 Backspace без выделения — один символ", state(f), ("абвг", 4, 4, 4))

f = TextInput("абвгд")
press(f, "home")
press(f, "delete")
ok("24 Delete без выделения — символ под кареткой", state(f),
   ("бвгд", 0, 0, 0))

f = TextInput(QUERY_TEXT)
press(f, "end")
press(f, "backspace", ctrl=True)
ok("25 Ctrl+Backspace сносит слово", f.text, "class:gustatory+side:")

f = TextInput(QUERY_TEXT)
press(f, "home")
press(f, "delete", ctrl=True)
ok("26 Ctrl+Delete сносит слово справа", f.text, ":gustatory+side:left")

f = TextInput("ab")
press(f, "home")
press(f, "backspace", 3)
ok("27 Backspace в начале ничего не портит", state(f), ("ab", 0, 0, 0))


# ---------------------------------------------------------------------------
section("6. буфер обмена")

clip = MemoryClipboard()
f = TextInput(QUERY_TEXT, clipboard=clip)
f.select_all()
f.apply_event(Event.key("c", ctrl=True))
ok("28 Ctrl+C кладёт выделение в буфер", clip.get(), QUERY_TEXT)

f.apply_event(Event.key("end"))
f.apply_event(Event.char("|"))
f.apply_event(Event.key("v", ctrl=True))
ok("29 Ctrl+V вставляет из буфера", f.text, QUERY_TEXT + "|" + QUERY_TEXT)

f = TextInput("вырезать это", clipboard=clip)
press(f, "home")
press(f, "right", 8, shift=True)
f.apply_event(Event.key("x", ctrl=True))
ok("30 Ctrl+X вырезает", (f.text, clip.get()), (" это", "вырезать"))

clip.set("без выделения буфер не трогаем")
f = TextInput("abc", clipboard=clip)
f.apply_event(Event.key("c", ctrl=True))
ok("31 Ctrl+C мимо выделения не затирает буфер", clip.get(),
   "без выделения буфер не трогаем")

clip.set("первая\r\nвторая\nтретья\tчетвёртая")
f = TextInput(clipboard=clip)
f.apply_event(Event.key("v", ctrl=True))
ok("32 многострочная вставка схлопывается в одну строку", f.text,
   "первая вторая третья четвёртая")

clip.set("class:gustatory\n+side:left")
f = TextInput(clipboard=clip, allow=QUERY)
f.apply_event(Event.key("v", ctrl=True))
ok("33 фильтр запроса убирает и пробел от схлопывания", f.text, QUERY_TEXT)


# ---------------------------------------------------------------------------
section("7. фильтр и предел длины")

f = TextInput(allow=FLOAT)
type_text(f, "рейт 12.5 Гц abc")
ok("34 числовое поле берёт только числовое", f.text, "12.5")

f = TextInput(allow=FLOAT)
type_text(f, "-1.5e-3")
ok("35 числовое поле пропускает знак и экспоненту", f.text, "-1.5e-3")

f = TextInput("", max_len=5)
type_text(f, "abcdefgh")
ok("36 предел длины при наборе", f.text, "abcde")

clip = MemoryClipboard("0123456789")
f = TextInput("ab", max_len=5, clipboard=clip)
press(f, "end")
f.apply_event(Event.key("v", ctrl=True))
ok("37 предел длины при вставке обрезает вставляемое", f.text, "ab012")

f = TextInput("abcde", max_len=5)
press(f, "home")
press(f, "right", 2, shift=True)
type_text(f, "ZZ")
ok("38 на пределе выделение всё равно заменяется", f.text, "ZZcde")

f = TextInput("слишком длинное начальное значение", max_len=8)
ok("39 предел длины применяется и к начальному тексту", f.text, "слишком ")


# ---------------------------------------------------------------------------
section("8. повтор при удержании и чужие события")

f = TextInput("1234567890")
press(f, "end")
press(f, "backspace", 5)                # первое нажатие плюс четыре повтора
ok("40 повтор удержания стирает по символу", f.text, "12345")

f = TextInput("абвгдежз")
press(f, "home")
press(f, "right", 4)
ok("41 повтор удержания двигает каретку", f.caret, 4)

f = TextInput("abc")
took = [f.apply_event(Event.key(n)) for n in
        ("enter", "tab", "escape", "up", "down", "pageup", "f1")]
ok("42 Enter, Tab, Esc, вверх-вниз и F1 поле не забирает", took,
   [False] * 7)
ok("43 чужие события не меняют текст", state(f), ("abc", 3, 3, 3))

ok("44 колесо поле не забирает", f.apply_event(Event.scroll(1.0)), False)

f = TextInput("abc")
# Ctrl+C приходит ещё и управляющим символом, если окно пустило его дальше
f.apply_event(Event.char("c", ctrl=True))
ok("45 символ при зажатом Ctrl не вставляется", f.text, "abc")

f = TextInput("abc")
f.apply_event(Event.char("@", alt=True))
ok("46 AltGr-символ вставляется", f.text, "abc@")

f = TextInput()
f.set_text("имя\nс переводом")
ok("47 set_text схлопывает перевод строки", f.text, "имя с переводом")

f = TextInput("abc")
f.apply_event(Event.key("a", ctrl=True))
type_text(f, "новое")
ok("48 Ctrl+A и набор заменяют всё", state(f), ("новое", 5, 5, 5))


# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print(f"утверждений {done}, провалов {len(fails)}")
if fails:
    for n in fails:
        print("  провал:", n)
print("=" * 68)
sys.exit(len(fails))
