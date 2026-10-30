"""Сеанс: работник за процессом, кадры, накопление активности.

Окно живёт на 60 Гц, а кадры приходят на 3–12 Гц при стимуляции и на 2–2,6 Гц
в теле — мозг считается 1167 шагов в секунду, и бин 10 мс закрывается раз в
86 мс. Если рисовать то, что пришло, картинка будет дёргаться ступенями.
Поэтому активность здесь не показывается как есть, а копится с затуханием, и
окно каждый кадр рисует сглаженное состояние, а не последний кадр работника.

Второе, ради чего этот модуль есть: окно не должно знать ни про torch, ни про
mujoco, ни про multiprocessing. Оно спрашивает `sess.activity` и `sess.frame`.
"""

import math
import time
import uuid

import numpy as np

from .. import ipc

__all__ = ["Session"]

# Постоянная затухания активности. Та же, что в веб-пульте и в отчёте: за
# 50 мс след гаснет вдвое, и при бине 10 мс картинка течёт, а не мигает.
TAU_MS = 50.0

# Шкала подстраивается под пик, но отпускает медленно — иначе один громкий
# нейрон надолго гасит всё остальное.
SCALE_MIN, SCALE_MAX = 10.0, 400.0
SCALE_DECAY = 0.99


class Session:
    """Работник, его кадры и всё, что из них накоплено."""

    def __init__(self, device=None, start=True):
        self.device = device
        self.proc = None
        self.chan = None
        self.ring = None

        self.hello = None
        self.n = 0
        self.ready = False
        self.error = None

        self.job_id = None
        self.boot_phase = ""
        self.boot_progress = 0.0
        self.state = "idle"
        self.phase = ""
        self.progress = 0.0
        self.t_ms = 0.0

        self.act = None            # Гц на нейрон, сглаженные
        self.disp = None           # 0..1 для карты
        self.scale = SCALE_MIN
        self.bin_ms = 10.0

        self.series = {}           # имя -> список (t_ms, значение)
        self.log = []              # строки журнала
        self.extra = {}            # последняя запись шага тела
        self.summary = None
        self.artifacts = []

        # Управление идущим расчётом. `sent` — что окно послало,
        # `applied` — что работник подтвердил применённым. Два списка, а
        # не один, именно потому, что это разные события: между ними
        # лежит шаг управления.
        self.sent = []
        self.applied = []
        self._born = time.perf_counter()

        self.body_seq = 0
        self.body_frame = None
        self._body_from = 0
        self.last_n = 0
        self.last_spikes = 0
        self.frames_seen = 0
        self.frames_dropped = 0
        self._last_frame_at = 0.0

        self._labels = None        # подписи нейронов, читаются по требованию
        self._facets = None        # списки значений для сборщика запроса
        self._pending = {}         # req_id -> что спрашивали
        self._answers = {}

        if start:
            self.start()

    # ------------------------------------------------------------------ жизнь

    def start(self):
        import multiprocessing as mp

        from .. import worker

        ctx = mp.get_context("spawn")
        self.cmd = ctx.Queue()
        frames = ctx.Queue(maxsize=ipc.FRAME_QUEUE)
        events = ctx.Queue()
        self.chan = ipc.Channel(frames=frames, events=events)
        self.proc = ctx.Process(
            target=worker.run, args=(self.cmd, self.chan, self.device),
            daemon=True, name="flybrain-worker")
        self.proc.start()

    def close(self):
        """Отпустить работника и ВСЁ, что держит этот процесс.

        Три вещи, и две из них здесь прежде не делались.

        **Нить-кормилица очереди команд.** `mp.Queue` заводит её на
        первом `put`, и `multiprocessing` при выходе процесса ждёт её
        без срока. Та же ловушка, что разобрана в `ipc.Channel.close`
        (там процесс стенда провисел шесть минут, а
        `probe/bodyfinal_timing.py` — семнадцать часов): остаток в
        трубе не даёт нити кончиться. Команды — последнее, что эта
        очередь несёт, и после `Shutdown` они никому не нужны, поэтому
        кормилицу отпускаем явно.

        **Очереди кадров и событий.** `Channel.close()` отпускает
        кормилицу КАДРОВ всегда, а событий — по просьбе. Здесь просьба
        уместна: читатель этих событий — само окно, и оно закрывается.

        **Разделяемая память кольца.** `ring.close()` отпускает
        отображение, но имя остаётся в системе до `unlink()`. Без него
        каждый сеанс оставляет блок на 15,8 МБ (восемь слотов
        960×720×3 плюс заголовок), и за долгую смену окна их
        накапливаются десятки.

        Кольцо создаёт РАБОТНИК (`worker.Worker.boot`), а окно
        присоединяется к нему по описателю из `Hello`, то есть
        владельцем здесь оказывается не этот процесс. Отвязывать всё
        равно надо, и именно здесь: работник к этому моменту уже
        дожат (`join`, при упорстве `terminate`) и отвязать за собой
        ничего не успеет, а имя останется. На Windows `unlink` у
        `shared_memory` и так ничего не делает — сегмент уходит, когда
        закрыт последний описатель, — поэтому вызов безвреден; на
        POSIX он снимает имя, которое больше некому держать.

        Каждый шаг в своём `try`: закрытие не должно падать на том, что
        уже закрыто, иначе следующие шаги не исполнятся вовсе.
        """
        if self.proc is not None and self.proc.is_alive():
            try:
                self.cmd.put(ipc.Shutdown())
                self.proc.join(timeout=3.0)
            except Exception:                                  # noqa: BLE001
                pass
            if self.proc.is_alive():
                self.proc.terminate()
        cmd = getattr(self, "cmd", None)
        if cmd is not None:
            for name in ("cancel_join_thread", "close"):
                fn = getattr(cmd, name, None)
                if fn is None:
                    continue
                try:
                    fn()
                except Exception:                              # noqa: BLE001
                    pass
        if self.chan is not None:
            try:
                self.chan.close(drop_events=True)
            except Exception:                                  # noqa: BLE001
                pass
        if self.ring is not None:
            try:
                self.ring.close()
            except Exception:                                  # noqa: BLE001
                pass
            try:
                self.ring.unlink()
            except Exception:                                  # noqa: BLE001
                pass

    @property
    def alive(self):
        return self.proc is not None and self.proc.is_alive()

    @property
    def busy(self):
        return self.state in ("queued", "setup", "warmup", "running", "saving")

    # ------------------------------------------------------------------ команды

    def submit(self, spec):
        """Поставить задачу. Возвращает её номер или None, если занят."""
        if self.busy or not self.ready:
            return None
        self.job_id = uuid.uuid4().hex[:8]
        # Шаг записи нужен для перевода спайков в герцы: у стимуляции это бин,
        # у арены и тела — шаг управления.
        self.bin_ms = float(spec.get("bin_ms") or spec.get("bin")
                            or spec.get("control_ms") or 10.0)
        self.state = "queued"
        self.phase = ""
        self.progress = 0.0
        self.t_ms = 0.0
        self.series = {}
        self.summary = None
        self.extra = {}
        self.frames_seen = 0
        # Кадр тела и его счётчик тоже принадлежат прошлому прогону.
        # Без сброса окно показывало прошлую муху всю подготовку нового
        # эпизода — то есть выдавало старые данные за новые. Обнулить
        # поля мало: в общей памяти лежит прошлый кадр, и первый же
        # опрос кольца возвращал его обратно. Поэтому запоминаем, с
        # какого номера кадры считаются новыми.
        self.body_frame = None
        self.body_seq = 0
        self.last_n = 0
        self.last_spikes = 0
        self._body_from = int(self.ring.seq) if self.ring is not None else 0
        self.reset_activity()
        self.cmd.put(ipc.Submit(self.job_id, dict(spec)))
        return self.job_id

    def cancel(self):
        if self.job_id is not None:
            self.cmd.put(ipc.Cancel(self.job_id))

    def control(self, action, payload=None, at_t_ms=None):
        """Изменить идущий расчёт. Возвращает номер команды.

        Команда уходит в ту же очередь, что и задачи, но разбирается
        задачей на границе шага управления: прерывать шаг физики посреди
        вычисления нельзя. Поэтому отправка — ещё не применение, и
        подтверждение придёт отдельным сообщением с модельным временем,
        в которое команду действительно применили.

        `at_t_ms` просит применить не раньше заданного модельного
        времени; `None` значит «на ближайшей границе».
        """
        cmd_id = uuid.uuid4().hex[:8]
        self.cmd.put(ipc.Control(cmd_id, str(action), payload, at_t_ms))
        self.sent.append({"команда": cmd_id, "действие": action,
                          "подана, с": time.perf_counter() - self._born,
                          "целевое модельное, мс": at_t_ms})
        del self.sent[:-200]
        return cmd_id

    def take_applied(self):
        """Забрать подтверждения, пришедшие с прошлого раза."""
        got, self.applied = self.applied, []
        return got

    def ask(self, what, arg=None):
        """Спросить работника. Ответ придёт в `answers`, не блокируя окно."""
        req = uuid.uuid4().hex[:8]
        self._pending[req] = what
        self.cmd.put(ipc.Query(req, what, arg))
        return req

    def answer(self, req):
        return self._answers.pop(req, None)

    # ------------------------------------------------------------------ приём

    def reset_activity(self):
        if self.n:
            self.act = np.zeros(self.n, dtype=np.float32)
            self.disp = np.zeros(self.n, dtype=np.float32)
        self.scale = SCALE_MIN

    def pump(self, dt=None):
        """Разобрать почту работника. Зовётся раз в кадр окна."""
        if self.chan is None:
            return []
        got = []
        for msg in self.chan.drain():
            got.append(msg)
            self._take(msg)
        self._decay(dt)
        self._poll_body()
        return got

    def _take(self, msg):
        if isinstance(msg, ipc.Hello):
            self._on_hello(msg)
        elif isinstance(msg, ipc.Frame):
            self._on_frame(msg)
        elif isinstance(msg, ipc.Series):
            for k, v in msg.values.items():
                self.series.setdefault(k, []).append((msg.t_ms, float(v)))
        elif isinstance(msg, ipc.Status):
            # Работник шлёт состояние разогрева под именем задачи "boot".
            # Принимать его за состояние прогона нельзя: окно считало, что
            # всё уже готово, и кнопка «Пуск» не срабатывала ни разу.
            if msg.job_id == "boot":
                self.boot_phase = msg.phase
                self.boot_progress = msg.progress
            elif msg.job_id == self.job_id:
                self.state = msg.state
                self.phase = msg.phase
                self.progress = msg.progress
                # Причина отказа приходит текстом в том же сообщении, и
                # без неё окно показывало «Отказ» без единого слова.
                if msg.state == "error" and msg.text:
                    self.error = msg.text
                    self.log.append("ошибка: " + str(msg.text))
            if msg.text:
                self.log.append(msg.text)
        elif isinstance(msg, ipc.Log):
            self.log.append(msg.text)
            if len(self.log) > 2000:
                del self.log[:1000]
        elif isinstance(msg, ipc.Artifact):
            self.artifacts.append((msg.kind, msg.path))
        elif isinstance(msg, ipc.Done):
            if msg.job_id == self.job_id:
                self.state = "done"
                self.summary = msg.summary
                self.frames_dropped = int(
                    msg.summary.get("frames_dropped", 0) or 0)
        elif isinstance(msg, ipc.Error):
            if msg.job_id in (self.job_id, "boot"):
                self.state = "error"
                self.error = msg.message
            self.log.append("ошибка: " + msg.message)
        elif isinstance(msg, ipc.Applied):
            # Подтверждение ФАКТИЧЕСКОГО применения. Окно показывает
            # именно его: между нажатием и применением лежит шаг
            # управления, и выдавать отправку за применение нельзя.
            self.applied.append(msg)
            del self.applied[:-200]
        elif isinstance(msg, ipc.Answer):
            self._answers[msg.req_id] = msg.payload
            self._pending.pop(msg.req_id, None)

    def _on_hello(self, h):
        self.hello = h
        self.n = int(h.n_neurons)
        self.device = h.device
        self.act = np.zeros(self.n, dtype=np.float32)
        self.disp = np.zeros(self.n, dtype=np.float32)
        if h.frame_ring:
            try:
                self.ring = ipc.FrameRing.from_descriptor(h.frame_ring)
            except Exception as exc:                           # noqa: BLE001
                self.log.append("кольцо кадров недоступно: " + repr(exc)[:120])
        self.ready = True

    def _on_frame(self, f):
        self.t_ms = f.t_ms
        self.frames_seen += 1
        # Последний кадр — отдельная величина: сглаженный след это уже
        # не он, и путать их нельзя.
        self.last_n = int(len(f.idx))
        self.last_spikes = int(f.val.sum()) if len(f.val) else 0
        self._last_frame_at = time.perf_counter()
        if f.extra:
            self.extra = f.extra
        if self.act is None or not len(f.idx):
            return
        # Спайки за бин переводятся в герцы: так шкала не зависит от бина.
        # Индексы от `np.nonzero` уникальны, поэтому обычная выборка здесь
        # и правильна, и в разы быстрее `np.add.at`.
        hz = f.val.astype(np.float32) * (1000.0 / max(self.bin_ms, 1e-3))
        self.act[f.idx] += hz * (1.0 - self._decay_k())

    def _decay_k(self):
        return math.exp(-self.bin_ms / TAU_MS)

    def _rescale(self):
        """Шкала по тому, что показано, а не по мгновенному пику кадра.

        Мгновенный пик бина — это 255 спайков за 10 мс, то есть 25 500 Гц;
        сравнивать с ним накопленный след бессмысленно, шкала сразу упиралась
        в потолок и карта выходила тусклой. Вверх шкала идёт сразу, чтобы
        вспышку не срезало, вниз — медленно, иначе картинка дышит.
        """
        peak = float(self.act.max()) if self.act is not None else 0.0
        target = min(SCALE_MAX, max(SCALE_MIN, peak))
        if target > self.scale:
            self.scale = target
        else:
            self.scale += (target - self.scale) * (1.0 - SCALE_DECAY) * 5.0

    def _decay(self, dt):
        """Затухание по времени окна, а не по приходу кадров.

        Если считать затухание только на кадрах работника, картинка будет
        стоять между ними и прыгать на каждом: кадры идут в десять раз реже,
        чем окно рисует.
        """
        if self.act is None:
            return
        if dt is None:
            dt = 1.0 / 60.0
        # Затухание работает, только пока идёт счёт, и ни мгновением
        # дольше: при постоянной 50 мс полсекунды запаса уже гасят след
        # в ноль, а смотрят на результат как раз после прогона.
        if self.busy:
            self.act *= math.exp(-(dt * 1000.0) / TAU_MS)
        self._rescale()
        np.divide(self.act, max(self.scale, 1e-6), out=self.disp)
        np.clip(self.disp, 0.0, 1.0, out=self.disp)

    def _poll_body(self):
        if self.ring is None:
            return
        seq, view = self.ring.latest()
        if view is not None and seq > self._body_from                 and seq != self.body_seq:
            self.body_seq = seq
            self.body_frame = view

    # ------------------------------------------------------------------ подписи

    def labels(self):
        """Подписи нейронов для подсказки при наведении.

        Читаются окном напрямую из кэша: это 4,3 МБ parquet и 0,11 с, тогда
        как запрос в работника стоит около 150 мс и приходит асинхронно — на
        наведение мышью такое не годится. Торч и mujoco окно при этом
        по-прежнему не тянет, нужен только pandas.
        """
        if self._labels is None:
            import pandas as pd

            from .. import data as D

            cols = ["label", "cell_class", "side", "top_nt"]
            try:
                df = pd.read_parquet(D.NEURONS_CACHE, columns=cols)
                self._labels = {c: df[c].to_numpy().astype(str) for c in cols}
            except Exception as exc:                           # noqa: BLE001
                self.log.append("подписи недоступны: " + repr(exc)[:110])
                self._labels = {}
        return self._labels

    def facets(self):
        """Списки значений для сборщика запроса: ключ -> [(значение, сколько)].

        Читается из того же кэша, что и подписи. В работник за этим ходить
        незачем: он занят счётом, а таблица одна и та же.
        """
        if self._facets is None:
            import pandas as pd

            from . import query as Q
            from .. import data as D

            try:
                cols = ["label", "cell_class", "super_class", "region",
                        "super_region", "top_nt", "side"]
                df = pd.read_parquet(D.NEURONS_CACHE, columns=cols)
                self._facets = Q.facets(df)
            except Exception as exc:                           # noqa: BLE001
                self.log.append("списки для запроса недоступны: "
                                + repr(exc)[:110])
                self._facets = {}
        return self._facets

    def describe(self, i):
        """Строка подсказки про один нейрон."""
        lab = self.labels()
        if not lab or i is None or i >= len(lab.get("label", ())):
            return ""
        parts = [lab["label"][i]]
        cls = lab["cell_class"][i]
        if cls and cls not in ("nan", "None", ""):
            parts.append(cls)
        side = lab["side"][i]
        if side in ("left", "right"):
            parts.append("слева" if side == "left" else "справа")
        nt = lab["top_nt"][i]
        if nt and nt not in ("nan", "None", ""):
            parts.append(nt)
        return " · ".join(parts)

    # ------------------------------------------------------------------ карта

    def geometry(self):
        """Готовые данные для карты: позиции, коды областей, цвета.

        Работник шлёт позиции квантованными в uint16 — 0,8 МБ вместо 1,7 и
        ровно та же точность, что видна на экране. Здесь они разворачиваются
        обратно и приводятся к тому виду, в котором их ждёт `BrainView`:
        центр в нуле, наибольший размах в единицу, y и z перевёрнуты (во
        FlyWire y растёт вентрально, и без переворота мозг смотрит затылком).
        """
        if self.hello is None:
            return None
        from .. import regions as R
        from . import theme

        h = self.hello
        q = np.asarray(h.positions_quantised, dtype=np.float32) / 65535.0
        lo = np.asarray(h.bbox["lo"], dtype=np.float32)
        hi = np.asarray(h.bbox["hi"], dtype=np.float32)
        p = lo + q * (hi - lo)
        p -= 0.5 * (p.min(0) + p.max(0))
        p /= max(float(np.abs(p).max()), 1e-6)
        p[:, 1] *= -1.0
        p[:, 2] *= -1.0

        names = list(h.regions)
        sup = dict(h.region_super)
        cols = theme.region_colors({n: sup.get(n, R.super_region(n)) for n in names})
        rgb = np.asarray([cols[n] for n in names], dtype=np.float32)
        code = np.asarray(h.region_code, dtype=np.int32)
        return (np.ascontiguousarray(p), code, names,
                np.ascontiguousarray(rgb), cols)

    # ------------------------------------------------------------------ сводка

    def region_levels(self, region_code, n_regions, counts=None):
        """Средняя активность по областям — для полос в колонке показаний."""
        if self.act is None:
            return np.zeros(n_regions, dtype=np.float32)
        sums = np.bincount(region_code, weights=self.act, minlength=n_regions)
        if counts is None:
            counts = np.bincount(region_code, minlength=n_regions)
        return (sums / np.maximum(counts, 1)).astype(np.float32)

    def series_array(self, name, last=None):
        s = self.series.get(name)
        if not s:
            return np.zeros(0, dtype=np.float32)
        vals = np.asarray([v for _, v in s], dtype=np.float32)
        return vals[-last:] if last else vals

    @property
    def n_active(self):
        return int((self.act > 0.5).sum()) if self.act is not None else 0

    @property
    def peak_hz(self):
        return float(self.act.max()) if self.act is not None else 0.0

    @property
    def stale_s(self):
        """Сколько секунд назад пришёл последний кадр — для индикатора связи."""
        if not self._last_frame_at:
            return 0.0
        return time.perf_counter() - self._last_frame_at
