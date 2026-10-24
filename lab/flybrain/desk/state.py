"""Состояние приложения, разложенное по принадлежности.

Прежний интерфейс держал всё в одном объекте окна, и из-за этого
переключение вкладки трогало то, что к вкладке отношения не имеет.
Здесь шесть независимых держателей, и граница между ними проведена по
времени жизни и по владельцу:

``WorkerState``     что окно знает о процессе-работнике; пишет только приём
                    сообщений
``RunState``        одна задача: фаза, время, этап, прогресс; пишет приём
                    сообщений, читают все
``SelectionState``  что выбрано: область, нейрон, селектор; живёт дольше
                    задачи и переживает её
``CameraState``     вид одного вьюпорта; у каждого вьюпорта свой, и
                    переключение рабочего пространства его не трогает
``WorkspaceState``  какое пространство открыто и как разложены его зоны
``AppState``        всё вместе плюс параметры сборки прогона

Правило, из-за которого это и разделено: **переключение пространства или
вкладки меняет только `WorkspaceState`**. Ни одна задача от этого не
начинается и не прекращается.

Параметры прогона здесь не дублируются: их схема — `spec.RunSpec`, одна
на окно и на CLI. Здесь лежит лишь то, что окно показывает.
"""

import time
from dataclasses import dataclass, field

__all__ = ["PHASES", "WorkerState", "RunState", "SelectionState",
           "CameraState", "WorkspaceState", "AppState", "VISION",
           "WORKSPACES", "BODY_VIEWS", "BrainState", "BRAIN_MODES",
           "SEL_VIEWS", "SCIENCE"]

# Фазы задачи названы явно. Бриф требует именно этого: «не маскировать
# 27 секунд инициализации отсутствием обратной связи».
PHASES = ("idle", "prep", "run", "cancel", "done", "fail")

WORKSPACES = (("experiment", "Эксперимент"), ("brain", "Мозг"),
              ("body", "Тело"), ("sandbox", "Песочница"),
              ("analysis", "Анализ"))

# Виды камеры тела: ключ, подпись, имя камеры FlyGym (не используется —
# без объекта камеры их в модели нет, см. probe/cams.py) и настройка:
# сдвиг азимута к курсу мухи, возвышение, множитель охвата к габариту
# мухи.
#
# Сдвиги взяты с картинок, а не из рассуждения: `probe/views.py` даёт
# четыре снимка, и на них видно прямо — при сдвиге 0° видно брюшко и
# крылья (это «сзади»), при 180° — лицо с красными глазами («спереди»),
# при 90° чистый профиль головой вправо. Первая попытка вывести это
# через скалярные произведения дала обратный знак, и подписи видов не
# совпадали с кадром.
#
# Множитель охвата: габарит мухи около 5 мм, при 1,3 она занимает
# примерно 5 % кадра против 1,8 % при прежних 2,2 (probe/look_body.py).
BODY_VIEWS = (
    ("back", "сзади", "camera_back_track_game", (0.0, -18.0, 1.30)),
    ("side", "сбоку", "camera_right", (90.0, -14.0, 1.45)),
    ("top", "сверху", "camera_top", (0.0, -89.0, 1.35)),
    ("front", "спереди", "camera_front_zoomin", (180.0, -16.0, 1.20)),
)


# Честность про зрение. Четыре разных вещи, которые легко подменить одна
# другой; интерфейс обязан их различать и не рисовать несуществующий
# поток сигнала.
VISION = {
    "camera": ("Кадр камеры", "есть",
               "Картинка глазами мухи, как её рендерит MuJoCo."),
    "retina": ("Сигнал омматидиев", "есть",
               "721 омматидий на глаз, два типа фоторецепторов. "
               "Числа настоящие, из FlyGym."),
    "wired": ("Связь с сетью", "нет",
              "Сигнал омматидиев никуда не подаётся: входа в оптические "
              "доли не существует."),
    "lobes": ("Активность оптических долей", "нет",
              "96 495 нейронов оптических долей молчат весь прогон. "
              "Любая их подсветка была бы выдумкой."),
}


@dataclass
class WorkerState:
    """Что окно знает о процессе со счётом. Пишется только приёмом."""

    alive: bool = False
    booting: bool = True
    stage: str = ""
    progress: float = 0.0
    device: str = ""
    n_neurons: int = 0
    n_edges: int = 0
    regions: tuple = ()
    readouts: tuple = ()
    last_error: str = ""
    hello_at: float = 0.0

    @property
    def ready(self):
        return self.alive and not self.booting


@dataclass
class RunState:
    """Одна задача. Время модельное и по стене — врозь, как требует бриф."""

    job_id: str = ""
    kind: str = ""
    title: str = ""              # название вида по-русски
    phase: str = "idle"
    stage: str = ""              # что именно делается сейчас
    progress: float = 0.0        # 0..1; -1 означает «неизвестно сколько»
    t_sim: float = 0.0           # модельное время, с
    t_target: float = 0.0        # сколько заказано, с
    started: float = 0.0         # perf_counter на момент пуска
    finished: float = 0.0
    message: str = ""
    out_dir: str = ""
    cancelled: bool = False      # остановлен по кнопке или по should_stop
    saved: bool = False          # файлы прогона легли на диск

    @property
    def outcome(self):
        """Чем кончилось: три разных исхода, а не «готово / не готово»."""
        if self.phase == "fail":
            return "fail"
        if self.phase == "done":
            return "cancelled" if self.cancelled else "done"
        return self.phase

    def reset(self, job_id="", kind=""):
        self.__init__(job_id=job_id, kind=kind)

    def begin(self, job_id, kind, t_target=0.0):
        self.reset(job_id, kind)
        self.phase = "prep"
        self.stage = "подготовка"
        self.progress = -1.0
        self.t_target = float(t_target)
        self.started = time.perf_counter()

    @property
    def busy(self):
        return self.phase in ("prep", "run", "cancel")

    @property
    def t_wall(self):
        if not self.started:
            return 0.0
        end = self.finished or time.perf_counter()
        return end - self.started

    @property
    def realtime(self):
        """Доля реального времени. Пока ничего не посчитано — None."""
        w = self.t_wall
        return (self.t_sim / w) if (w > 0.05 and self.t_sim > 0) else None

    @property
    def unknown_length(self):
        return self.progress is not None and self.progress < 0.0


@dataclass
class SelectionState:
    """Что выбрано. Переживает задачу: после прогона выбор не сбрасывается."""

    query: str = "class:gustatory+side:left"
    region: str = ""
    neuron: int = -1
    readout: str = ""
    n_matched: int = 0
    matched_note: str = ""


# Потолок отдаления по вьюпортам. У песочницы он больше: источники
# запаха стоят в 14 мм от мухи, а охват камеры слежения считается по её
# габариту (около 5 мм), и при множителе четыре источник оказывается у
# самого края кадра — взять его мышью нельзя.
CAMERA_DIST_MAX = {"sandbox": 12.0}


@dataclass
class CameraState:
    """Вид одного вьюпорта.

    `view` — какой из `BODY_VIEWS` выбран. `yaw`/`pitch`/`dist` — поправка
    поверх него для свободного вращения. `fit` просят, когда вид надо
    пересчитать под габарит тела.
    """

    view: str = "back"
    yaw: float = 0.0
    pitch: float = 0.0
    dist: float = 1.0            # множитель к расстоянию вида
    free: bool = False           # ушли от именованной камеры вручную
    fit: bool = True             # пересчитать охват на ближайшем кадре
    layers: tuple = ()           # включённые отладочные слои
    # Потолок отдаления — у каждого вьюпорта свой. Общий был ошибкой:
    # подняв его ради песочницы (источники стоят в 14 мм, а охват
    # считается по габариту мухи в 5 мм), я поднял его и в «Теле», где
    # он достигается кнопкой «отдалить» и колесом, а не только
    # ползунком с его собственными пределами.
    dist_max: float = 4.0

    def reset(self):
        self.yaw = self.pitch = 0.0
        self.dist = 1.0
        self.free = False
        self.fit = True

    def nudge(self, dyaw, dpitch):
        self.yaw += float(dyaw)
        self.pitch = max(-89.0, min(89.0, self.pitch + float(dpitch)))
        self.free = True

    def zoom(self, k):
        self.dist = max(0.3, min(self.dist_max, self.dist * float(k)))
        self.free = True


# Режимы карты мозга. «Выделение» — не третий цвет, а состояние поверх
# цвета: оно живёт отдельно и переключением режима не сбрасывается.
BRAIN_MODES = (("anatomy", "Анатомия"), ("activity", "Активность"))

# Как показывать выделение. Настройка независимая: переключение цвета
# её не трогает, и наоборот.
SEL_VIEWS = (("context", "Контекст", "выбранное ярко, остальной мозг "
                                     "приглушён, но различим"),
             ("only", "Только", "показаны только нейроны набора, "
                                "остального мозга нет"),
             ("all", "Весь мозг", "карта без акцента на выделении"))


# Оговорки о научной стороне. Показываются во вкладке «Сведения» и
# нигде не прячутся: это условия, при которых картинку вообще можно
# читать.
SCIENCE = (
    ("Источник структуры", "FlyWire v783",
     "Связи и положения нейронов взяты из опубликованного коннектома. "
     "Это структура одной мухи, а не усреднённый вид."),
    ("Модель динамики", "LIF, Shiu и др., Nature 2024",
     "Интегратор с утечкой и порогом. Упрощение: нет дендритной "
     "геометрии, нет активных токов, нет задержек проведения."),
    ("Вес связи", "число синапсов × wScale",
     "Это счёт синапсов из реконструкции, умноженный на общий "
     "коэффициент, а не измеренная физиологическая сила."),
    ("Знак медиатора", "предсказан",
     "Возбуждающий или тормозный определён предсказанием медиатора по "
     "снимкам, а не измерен для каждого нейрона."),
    ("Нейромодуляция и пластичность", "нет",
     "Веса не меняются по ходу прогона, модуляторных путей нет."),
    ("Зрительный вход", "не подключён",
     "Омматидии считаются, но в оптические доли ничего не подаётся: "
     "96 495 нейронов оптических долей молчат."),
    ("Как читать", "не запись живой мухи",
     "Это поведение модели при заданном стимуле. Причинных выводов из "
     "одной картинки активности делать нельзя."),
)


@dataclass
class BrainState:
    """Карта коннектома: что выбрано, что показано, что спрошено.

    Выделение и режим цвета намеренно разведены: по брифу переключение
    цвета не должно уничтожать выбор.
    """

    mode: str = "anatomy"
    query: str = "region:AL"
    query_note: str = ""          # сколько нашлось или в чём ошибка
    query_bad: bool = False

    sel_idx: object = None        # numpy-массив индексов или None
    sel_label: str = ""
    sel_n: int = 0
    sel_src: str = ""             # search | region | link
    sel_facts: dict = field(default_factory=dict)

    region: int = -1              # выбранный нейропиль
    hover: int = -1               # нейропиль под курсором
    hover_neuron: int = -1

    labels: bool = True
    only_active: bool = False
    point: float = 1.0            # множитель размера точки
    dim: float = 0.07             # во сколько гаснет невыделенное

    sel_view: str = "context"
    links_sort: str = "synapses"
    links_dir: str = "out"
    links_by: str = "region"
    links_rows: list = field(default_factory=list)
    links_state: str = "idle"     # idle | wait | ready | error | none
    links_note: str = ""
    links_for: str = ""
    links_ms: float = 0.0

    pending: dict = field(default_factory=dict)   # req_id -> вид запроса
    ready: bool = False

    def clear_selection(self):
        self.sel_idx = None
        self.sel_label = ""
        self.sel_n = 0
        self.sel_src = ""
        self.sel_facts = {}
        self.region = -1
        self.links_rows = []
        self.links_state = "idle"
        self.links_for = ""


@dataclass
class WorkspaceState:
    """Какое пространство открыто и как разложены зоны у каждого.

    Раскладка хранится **на пространство**: вернувшись в «Тело», владелец
    должен увидеть ту ширину рейки, которую сам и поставил.
    """

    current: str = "body"
    docks: dict = field(default_factory=dict)
    tabs: dict = field(default_factory=dict)

    def dock(self, name=None):
        from .layout import DockState
        name = name or self.current
        st = self.docks.get(name)
        if st is None:
            st = self.docks[name] = DockState()
        return st

    def tab(self, name=None, default=0):
        return self.tabs.get(name or self.current, default)

    def set_tab(self, index, name=None):
        self.tabs[name or self.current] = int(index)

    def switch(self, name):
        """Сменить пространство. Больше не делает ничего — и не должно."""
        if any(name == k for k, _ in WORKSPACES):
            self.current = name
            return True
        return False


# ------------------------------------------------------------- песочница

# Слоты источника запаха по умолчанию. Ближе, чем в научных эпизодах
# (там 25 и 38 мм): камера слежения считает охват по габариту мухи, пять
# миллиметров, и источник в тридцати миллиметрах в кадр не попадает
# вовсе. Здесь он должен быть виден вместе с мухой, иначе его нельзя
# взять в сцене.
SANDBOX_SLOTS = ((12.0, 8.0), (12.0, -8.0), (-12.0, 8.0), (-12.0, -8.0))

# Яркость насыщения рецепторов для сеанса. Задаётся числом, а не мерится
# на разогреве: сеанс начинается с выключенными источниками, и
# измеренная по началу яркость была бы нулевой — первый же включённый
# источник насытил бы рецепторы до потолка.
#
# Число взято из замера: источник силой 1 на расстоянии 14 мм даёт
# яркость около 0,005 (обратный квадрат расстояния, `OdorArena`).
SANDBOX_ODOUR_REF = 0.005

# Высота слоя, по которому ведётся перетаскивание: маркеры источников
# стоят на 1,5 мм над полом.
SANDBOX_Z = 1.5


@dataclass
class SandboxSlot:
    """Слот источника запаха — так, как его видит окно.

    Это намерение пользователя, а не состояние модели. Состояние модели
    приходит подтверждением от работника, и до него слот показывается
    как «ожидает применения».
    """

    x: float = 0.0
    y: float = 0.0
    strength: float = 1.0
    on: bool = False

    def as_payload(self):
        return {"x": round(float(self.x), 4), "y": round(float(self.y), 4),
                "strength": round(float(self.strength), 4),
                "on": bool(self.on)}


@dataclass
class SandboxState:
    """Состояние рабочего пространства «Песочница».

    Три вещи, которые здесь держатся отдельно и путать их нельзя:

    * `slots` — чего хочет пользователь;
    * `applied` — что работник подтвердил как состояние модели;
    * `pending` — команды, ушедшие и ещё не подтверждённые.

    Пока они различаются, окно показывает именно это различие: иначе
    подпись «источник перемещён» означала бы «кнопка нажата».
    """

    slots: list = field(default_factory=lambda: [
        SandboxSlot(x=x, y=y) for x, y in SANDBOX_SLOTS])
    picked: int = -1              # выбранный слот, -1 — ни один
    drag: int = -1                # слот, который тянут мышью
    drag_from: tuple = ()         # откуда начали тянуть, для отмены

    # --- сеанс ---
    t_run: float = 0.0            # 0 — без заранее заданного конца
    keep_s: float = 20.0          # окно записи активности, с
    odour_ref: float = SANDBOX_ODOUR_REF
    # Частота рецепторов при насыщении — та же, что у `BodyConfig`.
    # Держится здесь, чтобы показания могли сказать, когда оба канала
    # упёрлись в потолок и двусторонней разницы не стало.
    orn_max_hz: float = 150.0
    seed: int = 0
    readout: str = "ALPN"

    # --- прямой стимул ---
    query: str = "region:AL+side:left"
    stim_hz: float = 100.0
    stim_on: bool = False
    stim_note: str = ""
    stim_bad: bool = False

    # --- что подтвердил работник ---
    applied_slots: list = field(default_factory=list)
    applied_stim: list = field(default_factory=list)
    pending: dict = field(default_factory=dict)   # cmd_id -> запись
    events: list = field(default_factory=list)
    events_max: int = 400

    # --- измеренная скорость ---
    speed_ms: float = 0.0         # модельного времени на последней выборке
    speed_wall: float = 0.0
    ratio: float = 0.0            # модельное к настенному, измеренное
    paused: bool = False
    cam: dict = None              # поза камеры того кадра, что на экране
    cam_seq: int = -1             # номер кадра, которому поза принадлежит
    # Держать правку до следующего шага вместо применения на ближайшей
    # границе. Нужно это паузе: так правку можно подготовить, а
    # применится она ровно тем шагом, который сделает «Один шаг».
    on_next: bool = False
    control_ms: float = 20.0      # шаг управления сеанса

    def payload(self):
        return [s.as_payload() for s in self.slots]

    def new_session(self):
        """Забыть всё, что относилось к прошлому сеансу.

        Без этого второй сеанс в том же окне начинался с чужими
        числами: модельное время идёт заново, значит разность для
        скорости отрицательна, а «применённые раздражители» и «ждут
        применения» первые секунды описывают прошлый прогон.
        """
        self.applied_slots = []
        self.applied_stim = []
        self.pending = {}
        self.cam = None
        self.cam_seq = -1
        self.speed_ms = 0.0
        self.speed_wall = 0.0
        self.ratio = 0.0
        self.paused = False
        self.drag = -1

    def note(self, what, t_ms=None, detail="", level="info"):
        """Запись в журнал событий — с модельным временем, если оно есть."""
        self.events.append({"что": what, "модельное, мс": t_ms,
                            "по стене": time.time(), "подробно": detail,
                            "уровень": level})
        if len(self.events) > self.events_max:
            del self.events[:len(self.events) - self.events_max]


@dataclass
class AppState:
    """Всё состояние окна в одном месте, но не в одной куче."""

    worker: WorkerState = field(default_factory=WorkerState)
    run: RunState = field(default_factory=RunState)
    sel: SelectionState = field(default_factory=SelectionState)
    brain: BrainState = field(default_factory=BrainState)
    ws: WorkspaceState = field(default_factory=WorkspaceState)
    sb: SandboxState = field(default_factory=SandboxState)
    cams: dict = field(default_factory=dict)

    # параметры сборки прогона: ровно те, что уходят в RunSpec
    t_run: float = 2.0
    seed: int = 0
    bin_ms: float = 10.0
    rate_hz: float = 180.0
    readout: str = ""
    out_name: str = ""

    # Путь по арене копим сами из записей шага: в рядах, которые шлёт
    # задача, координат нет, а менять её схему ради картинки в окне
    # незачем.
    track: list = field(default_factory=list)
    track_max: int = 20000

    # Последний законченный прогон живёт отдельно от текущего: «готов к
    # запуску» и «прошлый прогон закончился вот так» — разные вещи, и
    # смешивать их в одном состоянии значит выдавать прошлые числа за
    # состояние окна.
    last: object = None
    # Что запустится по «Пуску» в текущем пространстве: оболочка пишет
    # сюда фразу, чтобы подсказка внизу не угадывала вид опыта сама.
    ready_line: str = "Готов к запуску."

    log: list = field(default_factory=list)
    log_max: int = 400

    def camera(self, key):
        """Вид одного вьюпорта. У каждого ключа — своё состояние.

        Песочница берёт свой ключ, а не чужой: камера «Тела» — тот же
        объект, который читает `send_camera`, и широкое отдаление
        песочницы уезжало прямо в «Тело». Своё состояние и свой потолок.
        """
        st = self.cams.get(key)
        if st is None:
            st = self.cams[key] = CameraState(
                dist_max=CAMERA_DIST_MAX.get(key, 4.0))
        return st

    def note(self, text, level="info"):
        self.log.append((time.time(), level, str(text)))
        if len(self.log) > self.log_max:
            del self.log[:len(self.log) - self.log_max]
