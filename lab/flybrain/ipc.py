"""Сообщения между окном и работником: типы, кодеки, очереди, кольцо кадров.

Здесь нет ни процессов, ни torch, ни mujoco — только данные. Это сделано
нарочно: окно не должно тянуть за собой движок, а всё, что описано в этом
модуле, проверяется в одном процессе, без разведения очередей по памяти.

Разделение труда такое. Окно живёт на 60 Гц и ни на чём не ждёт. Работник
считает мозг и шлёт кадры по мере готовности, а это 3–12 Гц: бин 10 мс
закрывается раз в 86 мс, шаг управления 20 мс — раз в 300 мс. Поэтому
кадры активности — расходный материал: очередь под них ограничена, и при
переполнении кадр молча выбрасывается. Всё остальное (состояние, ряды,
конец, ошибка, артефакт) терять нельзя, и для этого есть вторая очередь,
без потолка.

Картинка тела через очередь не ходит: кадр 640x480x3 это 0,9 МБ, и копия
на каждый шаг управления съела бы больше, чем сам рендер. Вместо этого
:class:`FrameRing` — кольцо слотов в разделяемой памяти, где обмен стоит
ноль, а блит в окне — 0,47 мс.
"""

from __future__ import annotations

import queue
from multiprocessing import shared_memory
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "Hello", "Status", "Frame", "Series", "Log", "Artifact", "Done", "Error",
    "Answer", "Submit", "Cancel", "Query", "Shutdown",
    "STATES", "MAX_FRAME_POINTS", "FRAME_QUEUE",
    "pack_frame", "Channel", "FrameRing",
]

# Больше 60 тысяч точек в кадре не рисует ни вьюер в отчёте, ни окно, а
# платить за них пересылкой приходится на каждом бине.
MAX_FRAME_POINTS = 60000

# Глубина очереди кадров: при 12 Гц это пять секунд отставания, после
# которых догонять уже бессмысленно — лучше показать свежий кадр.
FRAME_QUEUE = 64

STATES = ("queued", "setup", "warmup", "running", "saving", "done", "error")


# ------------------------------------------------------- работник -> окно


class Hello(NamedTuple):
    """Один раз после разогрева: всё, что окну нужно знать о коннектоме.

    Сюда же уходит список популяций: его сборка — это полторы сотни
    разборов селекторов по 138 тысячам строк, 14,4 с. Пока он не готов,
    окно всё равно не может предложить, с чего читать выход, так что
    ждать его один раз на старте честнее, чем на первом же щелчке.
    """

    n_neurons: int
    device: str
    regions: list                   # имена областей, в порядке кодов
    region_super: dict              # область -> крупная группа
    populations: dict               # имя -> (селектор слева, селектор справа)
    sensors: dict                   # канал -> селектор
    readouts: list                  # имена популяций, пригодных как выход
    bbox: dict                      # {"lo": [x, y, z], "hi": [...]}, мкм
    positions_quantised: Any        # uint16[N, 3], нормированные в bbox
    region_code: Any                # uint8[N], индекс в regions
    frame_ring: dict | None = None  # имя и форма кольца кадров тела


class Status(NamedTuple):
    job_id: str
    state: str                      # см. STATES
    phase: str = ""                 # что именно делается, человеку
    progress: float = 0.0           # 0..1, -1 если неизвестно
    text: str = ""


class Frame(NamedTuple):
    """Разреженная активность за один бин: только сработавшие нейроны."""

    job_id: str
    t_ms: float
    idx: Any                        # int32[k]
    val: Any                        # uint8[k], спайков за бин
    extra: dict | None = None       # запись шага тела: поза, dn_l, dn_r, ...


class Series(NamedTuple):
    """Скалярные ряды для графиков: одна точка на шаг."""

    job_id: str
    t_ms: float
    values: dict


class Log(NamedTuple):
    job_id: str
    level: str                      # info | warn | error
    text: str


class Artifact(NamedTuple):
    job_id: str
    kind: str                       # report | video | run | ring
    path: str


class Done(NamedTuple):
    job_id: str
    summary: dict


class Error(NamedTuple):
    job_id: str
    message: str
    tb: str = ""


class Answer(NamedTuple):
    """Ответ на :class:`Query`. Идёт в неограниченную очередь: окно ждёт его."""

    req_id: str
    what: str
    payload: Any


# ------------------------------------------------------- окно -> работник


class Submit(NamedTuple):
    job_id: str
    spec: dict


class Cancel(NamedTuple):
    job_id: str


class Query(NamedTuple):
    req_id: str
    what: str
    arg: Any = None


class Shutdown(NamedTuple):
    pass


class Control(NamedTuple):
    """Команда, меняющая идущий расчёт на ходу.

    Приходит в ту же очередь, что `Submit` и `Cancel`, но разбирается не
    между задачами, а **внутри** задачи, на границе шага управления:
    прерывать шаг физики посреди вычисления нельзя, а между шагами
    состояние целостно.

    `cmd_id` — сквозной номер, по которому окно потом узнаёт свою
    команду в подтверждении. `at_t_ms` — пожелание применить не раньше
    заданного модельного времени; `None` значит «на ближайшей границе».
    """

    cmd_id: str
    action: str                 # pause | resume | step | stop | set
    payload: Any = None
    at_t_ms: float | None = None


class Applied(NamedTuple):
    """Подтверждение ФАКТИЧЕСКОГО применения, а не получения.

    Окно обязано показывать именно это: между нажатием и применением
    лежит шаг управления, который на теле занимает четверть секунды, и
    выдавать отправку за применение значит врать о состоянии модели.

    `outcome`: applied | rejected | cancelled | error.
    """

    cmd_id: str
    action: str
    t_model_ms: float
    wall_s: float
    outcome: str
    values: Any = None
    why: str = ""


# ------------------------------------------------------------- кодек кадра


def pack_frame(counts, limit=MAX_FRAME_POINTS):
    """``counts`` int16[N] -> (idx int32[k], val uint8[k]).

    Перенесено из веб-пульта как есть: при 138 639 нейронах ненулевых в
    бине обычно тысячи, так что плотный массив ушёл бы почти пустым. При
    перегрузе оставляем самые громкие нейроны — обрезать хвост в единицу
    спайка незаметно, а потолок держит кадр в предсказуемом размере.
    """
    counts = np.asarray(counts)
    idx = np.nonzero(counts)[0].astype(np.int32)
    if len(idx) > limit:
        idx = idx[np.argsort(-counts[idx])][:limit]
    val = np.clip(counts[idx], 0, 255).astype(np.uint8)
    return idx, val


# ---------------------------------------------------------------- очереди


class Channel:
    """Две очереди в одном: расходная для кадров и надёжная для остального.

    Бросать можно только :class:`Frame`. Если окно не успевает разбирать
    кадры, лишний кадр теряется молча (счётчик в :attr:`dropped`), потому
    что показать устаревший кадр всё равно некому. Состояние, ряды, конец
    и ошибка идут во вторую очередь, где потолка нет: пропавший
    :class:`Done` оставил бы окно с вечным «идёт счёт».
    """

    def __init__(self, frames=None, events=None, frame_slots=FRAME_QUEUE):
        if frames is None or events is None:
            import multiprocessing as mp

            frames = frames or mp.Queue(maxsize=frame_slots)
            events = events or mp.Queue()
        self.frames = frames
        self.events = events
        self.dropped = 0

    def send(self, msg):
        """Отправить сообщение. Возвращает False, если кадр выброшен."""
        if isinstance(msg, Frame):
            try:
                self.frames.put_nowait(msg)
            except queue.Full:
                self.dropped += 1
                return False
            return True
        self.events.put(msg)
        return True

    def drain(self, max_frames=None):
        """Всё накопившееся, не блокируя: сначала события, потом кадры.

        Порядок важен: ``Done`` должен дойти до окна в том же разборе, что
        и последние кадры прогона, иначе кнопка «пуск» оживёт на кадр позже.
        """
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        n = 0
        while max_frames is None or n < max_frames:
            try:
                out.append(self.frames.get_nowait())
            except queue.Empty:
                break
            n += 1
        return out

    def wait(self, timeout=None):
        """Блокирующее чтение из надёжной очереди; None по таймауту."""
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """Закрыть обе очереди, не повиснув на недочитанных кадрах.

        Почему это не просто `q.close()` на обе. У `mp.Queue` есть
        питающий поток, и при выходе из интерпретатора
        `multiprocessing.util._exit_function` его ДОЖИДАЕТСЯ. Если в
        очереди осталось то, что никто не прочитал, ожидание не
        кончается никогда: процесс печатает весь свой вывод и повисает
        насмерть. Поймано замером — `probe/bodyfinal_timing.py` досчитала
        эпизод, записала отчёт и провисела семнадцать часов на выходе,
        потому что складывала кадры в очередь, у которой в этом прогоне
        не было читателя вовсе.

        Поэтому очереди закрываются РАЗНО, и это не осторожность, а
        устройство обмена:

        * `frames` — кадры картинки, которые по замыслу МОЖНО бросать
          при переполнении (у очереди на них есть `maxsize`, и окно
          отстающие кадры не ждёт). Значит недочитанный кадр терять
          разрешено, и ожидание питающего потока отменяется;
        * `events` — `Status`, `Series`, `Done`, `Error`, `Artifact`.
          Их терять нельзя: по `Done` окно оживляет кнопку «пуск», а по
          `Error` показывает причину. Здесь ожидание сохраняется, как
          было.
        """
        frames_close = getattr(self.frames, "close", None)
        if frames_close is not None:
            cancel = getattr(self.frames, "cancel_join_thread", None)
            if cancel is not None:
                cancel()
            frames_close()
        events_close = getattr(self.events, "close", None)
        if events_close is not None:
            events_close()


# ------------------------------------------------------- кольцо кадров тела


# Счётчик держим в начале того же блока памяти, с запасом на кэш-линию,
# чтобы читатель не трогал страницу, в которую пишется кадр.
_HEADER = 64


class FrameRing:
    """Кольцо кадров картинки в разделяемой памяти.

    Писатель занимает слоты по кругу и увеличивает счётчик только после
    того, как кадр записан целиком. Читатель смотрит счётчик и берёт
    предыдущий слот — то есть заведомо дописанный. При четырёх слотах
    писатель догонит читателя не раньше, чем через три кадра, а кадры
    приходят раз в 300 мс: ни блокировок, ни копий.
    """

    def __init__(self, shape=(480, 640, 3), slots=4, name=None, create=True):
        self.shape = tuple(int(v) for v in shape)
        self.slots = int(slots)
        self.frame_bytes = int(np.prod(self.shape))
        size = _HEADER + self.frame_bytes * self.slots
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.name = self.shm.name
        self._seq = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:8])
        self._slots = np.ndarray(
            (self.slots,) + self.shape, dtype=np.uint8,
            buffer=self.shm.buf[_HEADER:_HEADER + self.frame_bytes * self.slots])
        if create:
            self._seq[0] = 0

    @classmethod
    def open(cls, name, shape=(480, 640, 3), slots=4):
        return cls(shape=shape, slots=slots, name=name, create=False)

    @classmethod
    def from_descriptor(cls, desc):
        return cls.open(desc["name"], tuple(desc["shape"]), int(desc["slots"]))

    def descriptor(self):
        """То, что нужно передать читателю: сама память не пикуется."""
        return {"name": self.name, "shape": list(self.shape), "slots": self.slots}

    def write(self, frame):
        seq = int(self._seq[0])
        self._slots[seq % self.slots] = frame
        self._seq[0] = seq + 1
        return seq + 1

    def latest(self):
        """(номер кадра, вид на самый свежий целый кадр); (0, None) если пусто."""
        seq = int(self._seq[0])
        if seq <= 0:
            return 0, None
        return seq, self._slots[(seq - 1) % self.slots]

    @property
    def seq(self):
        return int(self._seq[0])

    def close(self):
        # Виды на буфер держат ссылку на mmap, и close() на нём падает,
        # пока они живы.
        self._seq = None
        self._slots = None
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
