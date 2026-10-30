"""Процесс-работник: коннектом, веса и тело живут здесь, а не в окне.

Зачем отдельный процесс, а не поток. Загрузка матрицы связей — 4,4 с и
302 МБ видеопамяти, сборка тела NeuroMechFly — ещё десяток секунд, а счёт
идёт 1167 шагов движка в секунду и не отпускает GIL предсказуемо. Окно при
этом обязано отвечать на 60 Гц. Поэтому всё тяжёлое — здесь, а окно не
импортирует ни torch, ни mujoco и держит только numpy.

Что уходит наружу: :class:`~flybrain.ipc.Hello` один раз после разогрева,
дальше ``Status``/``Frame``/``Series``/``Log`` по ходу задачи и
``Done``/``Error`` в конце. Картинка тела идёт в обход очередей, через
кольцо в разделяемой памяти.

Кадры тела снимает собственный ``mujoco.Renderer``, а не камера FlyGym:
камера отмеряет кадры по модельному времени и на 21 вызов отдала три
кадра, да ещё копит их в ``camera._frames`` без предела. Прямой рендер
стоит 0,53 мс на кадр и работает в одном процессе с окном на Vulkan.
"""

from __future__ import annotations

import contextlib
import json
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import data as D, ipc, regions as R, world

# Кадр тела: тот же размер, что проверен замером (0,53 мс на рендер,
# 0,47 мс на блит), и четыре слота — запас на три кадра вперёд.
# Кадр тела: 960x720 вместо 640x480. Во вьюпорте шириной под девятьсот
# точек прежний кадр приходилось растягивать, и сцена выглядела мыльной.
# Цена — 2,1 МБ на кадр вместо 0,9, кольцо на четыре слота это 8 МБ.
# Больше нельзя: оффскрин-буфер модели ограничен 1280x720.
RING_SHAPE = (720, 960, 3)
RING_SLOTS = 4

POPULATIONS_CACHE = D.CACHE / "populations.json"

# Как часто обновлять Status внутри задачи. Кадры идут 3–12 Гц, и слать
# состояние на каждый из них смысла нет — полоса прогресса и так ровная.
STATUS_EVERY_S = 0.25


# --------------------------------------------------------- перехват печати


class _LogStream:
    """Подменяет stdout задачи: ``print`` и ``tqdm`` внутри движка иначе
    уходят в никуда — у процесса нет консоли. Строки шлём как ``Log``.
    """

    encoding = "utf-8"

    def __init__(self, send, job_id, level="info"):
        self._send = send
        self._job = job_id
        self._level = level
        self._buf = ""
        self._live = True

    def detach(self):
        """После задачи поток глохнет: absl запоминает stderr навсегда и
        пишет в него из своих обработчиков, а задача уже кончилась.
        """
        self._live = False

    def write(self, text):
        if not self._live:
            return len(text)
        self._buf += text
        # tqdm перерисовывает строку через \r, поэтому режем и по нему.
        while True:
            cut = max(self._buf.find("\n"), self._buf.find("\r"))
            if cut < 0:
                break
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            line = line.strip()
            if line:
                self._send(ipc.Log(self._job, self._level, line[:300]))
        return len(text)

    def flush(self):
        pass

    def close(self):
        self.detach()

    def writable(self):
        return True

    def isatty(self):
        return False


# ------------------------------------------------------------------ задачи


class Worker:
    def __init__(self, cmd_queue, channel, device=None):
        import torch

        self.cmd = cmd_queue
        self.ch = channel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Живой вид тела. Своя камера MuJoCo, а не камера FlyGym: та
        # отмеряет кадры по модельному времени. И не именованная камера
        # из config.yaml: без объекта камеры FlyGym их в модели нет вовсе
        # (проверено: `probe/cams.py` насчитал ноль), а вписывать их туда
        # значило бы менять модель, по которой считает командная строка.
        # Своя камера следит за грудью, поворачивается вслед за курсом и
        # слушается окна.
        self.cam_pref = {"az": 0.0, "el": -18.0, "k": 1.3,
                         "dist": 0.0, "follow": True}

        # Индекс связей строится при первом запросе и живёт до конца
        # процесса: он стоит секунд, а нужен каждой вкладке «Связи».
        self._links_index = None
        self._links_ms = 0.0

        self._pending = []          # команды, пришедшие посреди задачи
        self._control = []          # управление идущим расчётом
        self._cancelled = set()
        self._stop = False
        self._job = None

        self._status("boot", "setup", "коннектом", 0.1)
        self.neurons = D.load_neurons()
        self._status("boot", "setup", "матрица связей", 0.3)
        self.weights = D.load_weights(device=self.device)
        self._status("boot", "setup", "популяции", 0.6)
        self.populations = load_populations(self.neurons)
        # Тот же список, но в памяти процесса: `body.run_episode` зовёт
        # `world.probe_candidates` сам, и без этого первый эпизод платил
        # 14 секунд за пересчёт того, что уже лежит на диске.
        world._CANDIDATE_CACHE[(len(self.neurons), 20)] = dict(
            self.populations)
        self._status("boot", "warmup", "геометрия", 0.9)
        self.ring = ipc.FrameRing(shape=RING_SHAPE, slots=RING_SLOTS)
        self.ch.send(self._hello())
        self._status("boot", "done", "готов", 1.0)

    # ------------------------------------------------------------- живой вид

    def _renderer(self, body):
        """Рендерер MuJoCo и камера слежения для живого вида тела.

        Первый вызов стоит около полутора секунд — создание контекста, —
        дальше кадр обходится в пару миллисекунд.
        """
        import mujoco

        model = body.physics.model.ptr
        ren = mujoco.Renderer(model, RING_SHAPE[0], RING_SHAPE[1])
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = _thorax_id(model)
        # Небо выключено. Со скайбоксом верх кадра — чистый белый, и в
        # тёмном окне это било пятном; без него фон чёрный, горизонт на
        # месте, сцена садится в оболочку, а не спорит с ней
        # (probe/look_body.py: медиана яркости 127 против 103).
        self._rnd_skybox = int(mujoco.mjtRndFlag.mjRND_SKYBOX)

        # Пол приглушён вдвое. Он занимает большую часть кадра, и в
        # светлом виде это была плита, светящаяся рядом с тёмными
        # панелями; на половинной яркости муха выходит вперёд, а клетка
        # по-прежнему читается и даёт чувство движения (медиана
        # яркости 74 против 127). Красится материал: у плоскости он
        # есть, и `geom_rgba` при этом не действует. На физику это не
        # влияет никак, а командная строка этого кода и не исполняет.
        done = set()
        for i in range(model.ngeom):
            if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            m = int(model.geom_matid[i])
            if m >= 0:
                if m not in done:
                    done.add(m)
                    model.mat_rgba[m][:3] = model.mat_rgba[m][:3] * 0.5
            else:
                model.geom_rgba[i][:3] = model.geom_rgba[i][:3] * 0.5
        return ren, cam

    # ------------------------------------------------------------ знакомство

    def _hello(self):
        n = self.neurons
        coords = n[["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        good = np.isfinite(coords).all(axis=1)
        centre = np.nanmean(coords[good], axis=0)
        coords = np.where(good[:, None], coords, centre[None, :]).astype(np.float32)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        quant = np.clip(((coords - lo) / span) * 65535.0, 0, 65535).astype(np.uint16)
        names = sorted(n["region"].unique().tolist())
        codes = n["region"].map({r: i for i, r in enumerate(names)}).to_numpy(np.uint8)
        return ipc.Hello(
            n_neurons=int(len(n)),
            device=self.device,
            regions=names,
            region_super={r: R.super_region(r) for r in names},
            populations=self.populations,
            sensors=dict(world.SENSORS),
            readouts=sorted(self.populations),
            bbox={"lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()},
            positions_quantised=quant,
            region_code=codes,
            frame_ring=self.ring.descriptor(),
        )

    # ------------------------------------------------------------- служебное

    def _status(self, job_id, state, phase="", progress=0.0, text=""):
        self.ch.send(ipc.Status(job_id, state, phase, progress, text))

    def _pump(self):
        """Разобрать очередь команд, не блокируясь.

        Зовётся из колбэков счёта: отмена должна доходить до движка между
        бинами, а не после конца прогона.
        """
        while True:
            try:
                cmd = self.cmd.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, ipc.Cancel):
                self._cancelled.add(cmd.job_id)
            elif isinstance(cmd, ipc.Shutdown):
                self._stop = True
                if self._job is not None:
                    self._cancelled.add(self._job)
            elif isinstance(cmd, ipc.Query):
                # Запросы отвечаем на месте: они только читают таблицу, а
                # заставлять окно ждать конца прогона ради подсчёта нейронов
                # в селекторе — это неотзывчивое поле ввода.
                self._query(cmd)
            elif isinstance(cmd, ipc.Control):
                # Управление идущим расчётом складывается отдельно: его
                # разбирает сама задача, на границе шага управления.
                # Разбирать здесь нельзя — середина шага физики не то
                # место, где можно менять состояние модели.
                self._control.append(cmd)
            else:
                self._pending.append(cmd)

    def _drain_control(self):
        """Отдать задаче то, что пришло с прошлого раза. Не блокирует."""
        self._pump()
        got, self._control = self._control, []
        return got

    def _should_stop(self, job_id):
        def check():
            self._pump()
            return self._stop or job_id in self._cancelled
        return check

    def _next_cmd(self):
        if self._pending:
            return self._pending.pop(0)
        try:
            return self.cmd.get(timeout=0.2)
        except queue.Empty:
            return None

    # ---------------------------------------------------------- главный цикл

    def serve(self):
        while not self._stop:
            cmd = self._next_cmd()
            if cmd is None:
                continue
            if isinstance(cmd, ipc.Shutdown):
                self._stop = True
            elif isinstance(cmd, ipc.Cancel):
                # Отмена догнала уже закончившуюся задачу — помечаем, вдруг
                # она ещё в очереди на выполнение.
                self._cancelled.add(cmd.job_id)
            elif isinstance(cmd, ipc.Query):
                self._query(cmd)
            elif isinstance(cmd, ipc.Submit):
                self._run_job(cmd)
        self._close()

    def _close(self):
        # Кадры брошены ДО выхода из процесса, и это про остановку, а не
        # про скорость. `multiprocessing` на выходе ждёт нить-кормилицу
        # каждой очереди без срока; если окно перестало разбирать кадры
        # (закрылось, подвисло, просто не успело), в трубе остаются
        # непрочитанные байты и работник не выходит никогда — окно его
        # дожмёт `terminate()` через три секунды, но «остановить муху»
        # это уже не остановка, а убийство. Кадр по договору этого
        # обмена расходный, поэтому отпускаем его нить прямо здесь.
        # Очередь событий не трогаем: `Done` и `Error` уже отправлены, и
        # они обязаны дойти — на них держится кнопка «пуск» в окне.
        self.ch.release_frames()
        self.ring.close()
        self.ring.unlink()

    def _query(self, cmd):
        try:
            if cmd.what == "select":
                payload = self._select(str(cmd.arg or ""))
            elif cmd.what == "camera":
                arg = dict(cmd.arg or {})
                for k in ("az", "el", "dist", "k"):
                    if k in arg:
                        self.cam_pref[k] = float(arg[k])
                if "follow" in arg:
                    self.cam_pref["follow"] = bool(arg["follow"])
                payload = dict(self.cam_pref)
            elif cmd.what == "select_idx":
                payload = self._select_idx(str(cmd.arg or ""))
            elif cmd.what == "links":
                payload = self._links(dict(cmd.arg or {}))
            elif cmd.what == "populations":
                payload = self.populations
            elif cmd.what == "describe":
                idx = np.asarray(cmd.arg, dtype=np.int64)
                payload = D.describe(self.neurons, idx).to_dict("index")
            else:
                payload = {"error": f"неизвестный запрос {cmd.what!r}"}
        except Exception as exc:  # noqa: BLE001
            payload = {"error": str(exc)[:300]}
        self.ch.send(ipc.Answer(cmd.req_id, cmd.what, payload))

    def _select(self, query):
        try:
            idx = D.select(self.neurons, query)
        except ValueError as exc:
            return {"error": str(exc)}
        if len(idx) == 0:
            return {"n": 0}
        sub = self.neurons.loc[idx]
        return {
            "n": int(len(idx)),
            "regions": {str(k): int(v) for k, v in
                        sub["region"].value_counts().head(6).items()},
            "types": {str(k): int(v) for k, v in
                      sub["label"].value_counts().head(8).items()},
        }

    def _select_idx(self, query):
        """Селектор -> сами индексы, а не только сводка.

        Карте мозга нужны индексы: по ним строится выделение. Сводка
        остаётся в прежнем запросе `select`, чтобы поле ввода считало
        совпадения, не таская по очереди полмегабайта.
        """
        try:
            idx = D.select(self.neurons, query)
        except ValueError as exc:
            return {"error": str(exc)}
        out = {"n": int(len(idx)),
               "idx": np.asarray(idx, dtype=np.int32)}
        if len(idx):
            sub = self.neurons.loc[idx]
            out["regions"] = {str(k): int(v) for k, v in
                              sub["region"].value_counts().head(8).items()}
            out["types"] = {str(k): int(v) for k, v in
                            sub["label"].value_counts().head(8).items()}
            out["sides"] = {str(k): int(v) for k, v in
                            sub["side"].value_counts().items()}
            # Откуда взялась область: часть нейронов получила её
            # голосованием соседних сом, и на вкладке это должно быть
            # видно, а не выдаваться за аннотацию.
            if "region_src" in sub.columns:
                out["region_src"] = {
                    str(k): int(v) for k, v in
                    sub["region_src"].fillna("аннотация").value_counts(
                    ).items()}
        return out

    def _links(self, arg):
        """Связи выбранного набора. Считает `flybrain.links`.

        Запрос отвечается в `_pump`, то есть доходит и посреди прогона, и
        окно на нём не стоит: оно рисует кадры дальше и заберёт ответ,
        когда тот придёт.
        """
        try:
            from . import links as LK
        except ImportError as exc:
            return {"error": "слой связей недоступен: " + str(exc)[:120]}
        try:
            if self._links_index is None:
                t0 = time.perf_counter()
                self._links_index = LK.build_index(self.weights)
                self._links_ms = (time.perf_counter() - t0) * 1000.0
            idx = arg.get("idx")
            if idx is None and arg.get("query"):
                idx = D.select(self.neurons, str(arg["query"]))
            idx = np.asarray(idx if idx is not None else [], dtype=np.int64)
            if not len(idx):
                return {"error": "пустая выборка"}
            t0 = time.perf_counter()
            res = LK.summarize(self._links_index, self.neurons, idx,
                               direction=str(arg.get("dir", "out")),
                               by=str(arg.get("by", "region")),
                               limit=int(arg.get("limit", 30)))
            return {"rows": res.get("rows", []),
                    "total": res.get("total", {}),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "index_ms": round(self._links_ms, 1),
                    "dir": str(arg.get("dir", "out")),
                    "by": str(arg.get("by", "region")),
                    "n": int(len(idx))}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:300]}

    def _run_job(self, cmd):
        """Исполнить задачу через общий реестр.

        Своего разбора видов задач здесь больше нет: и командная строка, и
        окно собирают `RunSpec`, а `jobs.run` его исполняет. Работнику
        остаётся то, что есть только у него, — рендерер MuJoCo, кольцо кадров
        и сохранение на диск.
        """
        from . import jobs, spec as S

        job_id, raw = cmd.job_id, dict(cmd.spec or {})
        if job_id in self._cancelled:
            self._status(job_id, "done", "отменена до начала", 1.0)
            return
        self._job = job_id
        kind = raw.get("kind", "stim")
        t0 = time.perf_counter()
        dropped0 = self.ch.dropped
        # Папка выбирается ДО счёта: задаче нужно знать, куда класть
        # видео, а двум задачам с одним именем нельзя дать одну папку.
        name = raw.get("save") or raw.get("out")
        out_dir = self._free_dir(str(name)) if name else None
        ctx = _WorkerCtx(self, job_id, out_dir=out_dir)
        ctx._base._drain = self._drain_control
        try:
            res = jobs.run(S.from_dict(raw), ctx)
            if res.summary.get("error"):
                self.ch.send(ipc.Error(job_id, res.summary["error"][:400],
                                       res.summary.get("tb", "")))
                self._status(job_id, "error", kind, 1.0,
                             res.summary["error"][:200])
            else:
                summary = dict(res.summary)
                summary.update(ctx.extras())
                self._save(job_id, raw, summary, res, out_dir)
                summary["wall_s"] = round(time.perf_counter() - t0, 3)
                summary["cancelled"] = (res.cancelled
                                        or job_id in self._cancelled)
                summary["frames_dropped"] = int(self.ch.dropped - dropped0)
                self.ch.send(ipc.Done(job_id, summary))
                self._status(job_id, "done", kind, 1.0)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            # Процесс не падает: окно получит ошибку и сможет послать
            # следующую задачу, веса остаются загруженными. `SystemExit`
            # здесь не исключение, а такая же ошибка данных.
            tb = "".join(traceback.format_exc().splitlines(keepends=True)[-12:])
            self.ch.send(ipc.Error(job_id, str(exc)[:400], tb))
            self._status(job_id, "error", kind, 1.0, str(exc)[:200])
        finally:
            ctx.close()
            self._job = None
            self._cancelled.discard(job_id)

    def _save(self, job_id, raw, summary, res, out_dir=None):
        """Положить прогон на диск, если окно попросило.

        Рядом с привычными файлами пишется `spec.json` — по нему прогон
        повторяется одной кнопкой, и его же читает командная строка, — и
        `run.json` с исходом: вид, отменён ли, сколько шло, какой задачей
        сделан. Без `run.json` отменённый прогон на диске неотличим от
        законченного: файлы те же, а частоты уже подрезаны.
        """
        name = raw.get("save") or raw.get("out")
        if not name or res.obj is None:
            return
        import json

        from . import body as B, loop, sim

        self._status(job_id, "saving", str(name), 1.0)
        # Папка уже выбрана перед счётом и, возможно, уже занята видео
        # этого же прогона: второй раз её подбирать нельзя.
        out = Path(out_dir) if out_dir else self._free_dir(str(name))
        bench = str(summary.get("kind", "")).endswith("-bench")
        if bench:
            # Серия отдаёт не эпизод, а словарь «условие -> числа». Ни
            # `loop.save`, ни `body.save` такого не примут: до этой ветки
            # серия из окна падала при сохранении, уже посчитавшись.
            out.mkdir(parents=True, exist_ok=True)
            (out / "bench.json").write_text(
                json.dumps(res.obj, ensure_ascii=False, indent=2),
                encoding="utf-8")
        elif res.kind == "body":
            B.save(res.obj, out)
        elif res.kind == "arena":
            loop.save(res.obj, out)
        elif res.kind == "lateral":
            out.mkdir(parents=True, exist_ok=True)
            (out / "lateral.json").write_text(
                json.dumps(res.obj, ensure_ascii=False, indent=1),
                encoding="utf-8")
        else:
            sim.save(res.obj, out)
        out.mkdir(parents=True, exist_ok=True)
        if summary.get("verify"):
            # Исход сверки на диск до сих пор не попадал: он жил только в
            # сводке задачи. Без него сохранённая сверка неотличима от
            # обычной стимуляции, а отменённая — от пройденной.
            (out / "verify.json").write_text(
                json.dumps(summary["verify"], ensure_ascii=False, indent=1),
                encoding="utf-8")
        (out / "spec.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "run.json").write_text(json.dumps({
            "version": 1,
            "job": str(job_id),
            "kind": str(res.kind),
            "result_kind": str(summary.get("kind") or res.kind),
            "series": bool(bench),
            "cancelled": bool(res.cancelled or job_id in self._cancelled),
            "t_requested_s": float(raw.get("t") or 0.0),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "desk",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["path"] = str(out)
        self.ch.send(ipc.Artifact(job_id, "run", str(out)))

    @staticmethod
    def _free_dir(name):
        """Папка, в которой ещё нет чужого прогона.

        Имя из окна — отметка времени, и две задачи в одну секунду дали бы
        одну папку: вторая затёрла бы файлы первой и присвоила её `spec.json`.
        Имя может задать и человек, и тогда совпадение тем более вероятно.
        Занятой считается папка, в которой лежит хоть один файл результата:
        `spec.json`, `run.json` или главный файл любого вида. Пустая папка
        и папка, которую этот же прогон только что создал под видео,
        занятыми не считаются.
        """
        busy = ("spec.json", "run.json", "meta.json", "episode.json",
                "embodied.json", "bench.json", "lateral.json")

        def taken(d):
            return any((d / f).exists() for f in busy)

        out = D.OUT / name
        if not taken(out):
            return out
        for i in range(2, 1000):
            alt = D.OUT / ("%s-%d" % (name, i))
            if not taken(alt):
                return alt
        return D.OUT / ("%s-%s" % (name, time.strftime("%H%M%S%f")))


class _WorkerCtx:
    """Контекст реестра в работнике.

    Добавляет к общему контексту ровно то, что есть только здесь: рендерер
    MuJoCo на физической модели контроллера и кольцо кадров картинки.
    """

    def __init__(self, w, job_id, out_dir=None):
        from . import jobs

        self._base = jobs.Ctx(
            emit=w.ch.send, should_stop=w._should_stop(job_id),
            neurons=w.neurons, weights=w.weights, device=w.device,
            job_id=job_id, out_dir=out_dir)
        self.w = w
        self.ren = self.cam = self.data = None
        self.span = 5.0
        self._warm = False
        self.pictures = 0
        self.last_cam = None
        self.render_ms = 0.0
        self._mark = time.perf_counter()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def on_body(self, body):
        self.ren, self.cam = self.w._renderer(body)
        self.data = body.physics.data.ptr
        self.span = _fly_span(self.data, self.cam.trackbodyid)

    def step(self, rec):
        if self.ren is None:
            return
        t = time.perf_counter()
        if self.pictures == 0 and not self._warm:
            self._warm = True
            self.ren.update_scene(self.data, camera=self.cam)
            self.ren.render()
        p = self.w.cam_pref
        # Курс мухи уже посчитан шагом: камера идёт за ним, иначе «сзади»
        # верно только до первого поворота.
        head = float(np.degrees(rec.get("theta", 0.0)))             if p.get("follow", True) else 0.0
        self.cam.azimuth = head + float(p.get("az", 180.0))
        self.cam.elevation = float(p.get("el", -18.0))
        self.cam.distance = (float(p.get("dist") or 0.0)
                             or self.span * float(p.get("k", 1.3)))
        self.ren.update_scene(self.data, camera=self.cam)
        self.ren.scene.flags[self.w._rnd_skybox] = False
        self.w.ring.write(self.ren.render())
        self.pictures += 1
        self.render_ms += (time.perf_counter() - t) * 1000.0
        # Поза камеры, которой снят ИМЕННО этот кадр. Нужна она окну для
        # одного: чтобы щелчок по сцене можно было перевести в
        # координаты модели, а не угадать. Берётся не из настроек, а из
        # разрешённой сцены — там учтены и слежение за мухой, и охват,
        # посчитанный по её габариту, чего окно не знает.
        gc = self.ren.scene.camera[0]
        self.last_cam = {
            "pos": [float(v) for v in gc.pos],
            "forward": [float(v) for v in gc.forward],
            "up": [float(v) for v in gc.up],
            "near": float(gc.frustum_near), "far": float(gc.frustum_far),
            "bottom": float(gc.frustum_bottom), "top": float(gc.frustum_top),
            "centre": float(gc.frustum_center),
            "ortho": bool(gc.orthographic),
            # `RING_SHAPE` — это (высота, ширина): так его принимает
            # `mujoco.Renderer`. Перепутать здесь местами значит дать
            # окну неверную проекцию, и на кадре 960×720 промах выходит
            # под двести пикселей (поймано `probe/scene_pick.py`).
            "w": int(RING_SHAPE[1]), "h": int(RING_SHAPE[0]),
            "pic": int(self.pictures),
        }

    def extras(self):
        if not self.pictures:
            return {}
        return {"picture_frames": self.pictures,
                "render_ms": round(self.render_ms / self.pictures, 3)}

    def close(self):
        if self.ren is not None:
            self.ren.close()
            self.ren = None


def _fly_span(data, tid):
    """Габарит мухи в миллиметрах: по чему подбирается охват камеры.

    Берутся тела в пределах десяти миллиметров от груди — это и есть
    муха; арена сюда не попадает, иначе охват считался бы по всей сцене
    и тело занимало бы десятую часть кадра.
    """
    p = np.asarray(data.xpos)
    d = np.linalg.norm(p - p[tid], axis=1)
    near = d[d < 10.0]
    return float(max(1.5, near.max() * 2.0)) if near.size else 3.0


def _thorax_id(model):
    """Тело, за которым следит камера: грудь мухи, а не мир."""
    import mujoco

    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.endswith("Thorax"):
            return i
    return 0


# ------------------------------------------------------- список популяций


def load_populations(neurons):
    """Популяции для выбора выхода, с кэшем на диске.

    ``world.probe_candidates`` считается 14,4 с — это полторы сотни разборов
    селекторов по 138 тысячам строк. Платить ими за каждое открытие окна
    нельзя, поэтому результат лежит рядом с таблицей нейронов и проверяется
    по её времени изменения: таблица пересобралась — список устарел.
    """
    stamp = D.NEURONS_CACHE.stat().st_mtime if D.NEURONS_CACHE.exists() else 0.0
    if POPULATIONS_CACHE.exists():
        try:
            obj = json.loads(POPULATIONS_CACHE.read_text(encoding="utf-8"))
            if (abs(float(obj.get("neurons_mtime", -1)) - stamp) < 1e-6
                    and int(obj.get("n_neurons", -1)) == len(neurons)):
                return {k: tuple(v) for k, v in obj["populations"].items()}
        except (ValueError, KeyError, OSError):
            pass
    pops = world.probe_candidates(neurons)
    D.CACHE.mkdir(parents=True, exist_ok=True)
    POPULATIONS_CACHE.write_text(json.dumps({
        "neurons_mtime": stamp,
        "n_neurons": int(len(neurons)),
        "populations": {k: list(v) for k, v in pops.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return pops


# ---------------------------------------------------------------- точка входа


def run(cmd_queue, evt_channel, device=None):
    """Точка входа процесса: загрузиться, представиться, крутить команды."""
    # Консоль у дочернего процесса может быть в cp866, и тогда любой символ
    # вне её таблицы валит процесс на первом же выводе.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    worker = None
    try:
        worker = Worker(cmd_queue, evt_channel, device=device)
        worker.serve()
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exc().splitlines(keepends=True)[-12:])
        evt_channel.send(ipc.Error("boot", str(exc)[:400], tb))
        if worker is not None:
            worker._close()
