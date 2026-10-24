"""Сборщик запроса: состояние и сборка строки селектора.

Виджетов здесь нет — только модель. Окно показывает списки, а собранная
строка обязана быть ровно тем, что принимает `data.select`, иначе интерфейс
разойдётся с командной строкой.

Почему сборщик, а не поле ввода. Набор констант клавиш в Taichi — это
модификаторы, стрелки и Esc; буквы приходят строками (проверено: `o s k e w
r i j g p h t`), а про знаки `:` `+` `=` `,` неизвестно ничего. Собирать
`class:olfactory+side:left` вслепую из того, что может не прийти, — плохая
ставка. К тому же окно и так знает все 36 нейропилей, все классы и 40
семейств: выбрать из списка быстрее, чем набрать, и опечататься нельзя.
Поле ввода остаётся поверх этого как второй путь, а не как единственный.
"""

__all__ = ["KEYS", "Term", "QueryBuilder", "describe_key"]

# Ключи DSL, которые имеет смысл предлагать списком. `id:` и `idx:` сюда не
# попадают намеренно: это ручной ввод номеров, им место в поле.
KEYS = (
    ("type", "тип", "regex по cell_type, hemibrain_type и подписи"),
    ("type=", "тип точно", "совпадение целиком, без regex"),
    ("class", "класс", "cell_class: olfactory, gustatory, descending…"),
    ("super", "надкласс", "super_class: central, optic, sensory…"),
    ("region", "нейропиль", "область мозга целиком"),
    ("superregion", "группа", "крупная группа нейропилей"),
    ("nt", "медиатор", "преобладающий нейромедиатор"),
    ("side", "сторона", "left, right, center"),
)

KEY_LABEL = {k: label for k, label, _ in KEYS}


def describe_key(key):
    for k, label, hint in KEYS:
        if k == key:
            return label, hint
    return key, ""


class Term:
    """Одно слагаемое запроса: ключ и значение."""

    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def text(self):
        # `type=` уже несёт знак равенства, второй двоеточия не нужно
        return (self.key + self.value) if self.key.endswith("=") \
            else (self.key + ":" + self.value)

    def label(self):
        return KEY_LABEL.get(self.key.rstrip("="), self.key) + " " + self.value

    def __eq__(self, other):
        return (isinstance(other, Term) and other.key == self.key
                and other.value == self.value)

    def __repr__(self):
        return "Term(" + self.text() + ")"


class QueryBuilder:
    """Запрос как список групп.

    В DSL `+` это пересечение, `,` — объединение, и приоритет такой, что
    `a+b,c+d` читается как «(a и b) или (c и d)». Поэтому модель — список
    групп, а каждая группа — список слагаемых, соединённых пересечением.
    Так в окне не нужно объяснять приоритет: группа рисуется строкой.
    """

    def __init__(self, groups=None):
        self.groups = [list(g) for g in groups] if groups else [[]]
        self.active = 0

    # ------------------------------------------------------------------

    def text(self):
        parts = ["+".join(t.text() for t in g) for g in self.groups if g]
        return ",".join(parts)

    def is_empty(self):
        return not any(self.groups)

    def summary(self):
        """Человеческое описание: «запах слева, или вкус»."""
        parts = []
        for g in self.groups:
            if g:
                parts.append(" и ".join(t.label() for t in g))
        return ", или ".join(parts) if parts else "пусто"

    # ------------------------------------------------------------------

    def add(self, key, value):
        """Добавить слагаемое в текущую группу.

        Повторный ключ в одной группе заменяет прежнее значение: пересечение
        `side:left+side:right` всегда пусто, и предлагать такое бессмысленно.
        """
        if not value:
            return False
        term = Term(key, value)
        g = self.groups[self.active]
        if term in g:
            return False
        base = key.rstrip("=")
        for i, t in enumerate(g):
            if t.key.rstrip("=") == base:
                g[i] = term
                return True
        g.append(term)
        return True

    def remove(self, group_i, term_i):
        if 0 <= group_i < len(self.groups):
            g = self.groups[group_i]
            if 0 <= term_i < len(g):
                del g[term_i]
                if not g and len(self.groups) > 1:
                    del self.groups[group_i]
                    self.active = min(self.active, len(self.groups) - 1)
                return True
        return False

    def new_group(self):
        """Начать объединение: «или ещё вот это»."""
        if self.groups and not self.groups[-1]:
            self.active = len(self.groups) - 1
            return False
        self.groups.append([])
        self.active = len(self.groups) - 1
        return True

    def clear(self):
        self.groups = [[]]
        self.active = 0

    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, text):
        """Разобрать строку DSL обратно в группы.

        Нужно, чтобы пресет или сохранённый прогон открывались в сборщике, а
        не только в поле. Незнакомое слагаемое попадает в группу как есть:
        потерять часть запроса хуже, чем показать её неразобранной.
        """
        groups = []
        for chunk in (text or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            terms = []
            for part in chunk.split("+"):
                part = part.strip()
                if not part:
                    continue
                if "=" in part and part.index("=") < part.find(":") % (len(part) + 1):
                    k, _, v = part.partition("=")
                    terms.append(Term(k + "=", v))
                elif ":" in part:
                    k, _, v = part.partition(":")
                    terms.append(Term(k, v))
                else:
                    # голое слово в DSL значит `type:`
                    terms.append(Term("type", part))
            if terms:
                groups.append(terms)
        return cls(groups or [[]])


def facets(neurons, limit=400):
    """Списки значений на каждый ключ, с числом нейронов.

    Считается в работнике, а не в окне: это проход по таблице на 138 639
    строк. Окно получает готовое и только показывает.
    """
    import pandas as pd

    out = {}
    cols = {"class": "cell_class", "super": "super_class",
            "region": "region", "superregion": "super_region",
            "nt": "top_nt", "side": "side"}
    for key, col in cols.items():
        if col not in neurons.columns:
            continue
        vc = neurons[col].value_counts()
        out[key] = [(str(k), int(v)) for k, v in vc.items()
                    if str(k) not in ("", "nan", "None")][:limit]

    # Типы: их тысячи, поэтому отдаём самые населённые, а остальное ищется
    # фильтром по буквам.
    labels = neurons["label"] if "label" in neurons.columns else None
    if labels is not None:
        vc = labels.value_counts()
        out["type"] = [(str(k), int(v)) for k, v in vc.items()
                       if str(k) not in ("", "nan", "unnamed")][:limit * 4]
    return out
