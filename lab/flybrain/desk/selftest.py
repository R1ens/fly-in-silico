"""Самопроверка настоящего окна: нужна, чтобы принять собранный пускатель.

Это не второй интерфейс и не тестовый режим приложения. Это короткий
сценарий, который гоняет **то же самое окно**, открытое тем же способом,
что и по двойному щелчку: настоящий Shell, настоящий Vulkan, настоящий
работник. Нужен он ровно для одного — проверить выпуск через сам
`FlyBrain.exe`, а не через python или `app.cmd`: иначе «запускается»
осталось бы словом.

Сценарий делает то, что просили проверить: открывает все пять
пространства, считает короткий опыт, убеждается, что результат лёг на
диск, и закрывает окно. Отчёт пишется в JSON, код возврата ненулевой при
любой беде — пускатель на него и смотрит.

Никакого ввода не подделывается: пространства переключаются вызовом, как
в пробах. Попадание мыши по кнопкам проверяют `ana_shot` и `exp_shot` без
окна; здесь проверяется другое — что собранный exe доводит дело до конца.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .state import WORKSPACES

__all__ = ["SelfTest"]


class SelfTest:
    """Шаги сценария. `step` зовётся из цикла окна, по разу на кадр."""

    def __init__(self, out_path, t_run=0.3, limit=420.0):
        self.out = Path(out_path)
        self.t_run = float(t_run)
        self.limit = float(limit)
        self.phase = "boot"
        self.t0 = time.perf_counter()
        self.mark = self.t0
        self.i = 0
        self.frames = 0
        self.worst = 0.0
        self.rep = {"шаги": [], "пространства": {}, "беды": []}
        self._last = time.perf_counter()

    # ------------------------------------------------------------ учёт

    def _note(self, what, **kw):
        kw["что"] = what
        kw["с начала, с"] = round(time.perf_counter() - self.t0, 1)
        self.rep["шаги"].append(kw)

    def _fail(self, why):
        self.rep["беды"].append(why)
        self.phase = "done"

    # ------------------------------------------------------------- шаг

    def step(self, app):
        """Вернуть False, когда окно пора закрывать."""
        now = time.perf_counter()
        self.frames += 1
        self.worst = max(self.worst, (now - self._last) * 1000.0)
        self._last = now
        if now - self.t0 > self.limit:
            self._fail("не уложился в %.0f с" % self.limit)
            return False
        return getattr(self, "_" + self.phase)(app, now)

    def _boot(self, app, now):
        if app.st.worker.ready:
            self._note("работник поднялся",
                       **{"нейронов": app.st.worker.n_neurons,
                          "устройство": app.st.worker.device})
            self.phase = "spaces"
            self.i = 0
            self.mark = now
            return True
        if not app.st.worker.alive and now - self.t0 > 10.0:
            self._fail("работник не живёт")
            return False
        return True

    def _spaces(self, app, now):
        """По очереди все пять: каждому полсекунды настоящих кадров."""
        keys = [k for k, _t in WORKSPACES]
        if self.i >= len(keys):
            self.phase = "start"
            return True
        key = keys[self.i]
        if app.st.ws.current != key:
            app.st.ws.switch(key)
            self.mark = now
            self.worst = 0.0
            return True
        if now - self.mark < 0.5:
            return True
        self.rep["пространства"][key] = {
            "кадров": self.frames, "худший кадр, мс": round(self.worst, 1)}
        self._note("пространство открылось", пространство=key)
        self.i += 1
        self.frames = 0
        return True

    def _start(self, app, now):
        from .. import catalog as K
        from . import ws_experiment as X

        app.st.ws.switch("experiment")
        st = X._state(app)
        X._reset_values(st, "stim")
        st["values"]["t"] = self.t_run
        app.frame()
        spec = X.build_spec(app)
        bad = K.validate(spec)
        if bad:
            self._fail("спека не прошла проверку: " + "; ".join(bad))
            return False
        job = app.start_run(spec, title="Стимуляция")
        if not job:
            self._fail("опыт не запустился")
            return False
        self._note("опыт запущен", задача=job,
                   **{"модельное, с": self.t_run})
        self.phase = "wait"
        self.mark = now
        self.worst = 0.0
        return True

    def _wait(self, app, now):
        if app.st.run.busy or app.st.last is None:
            return True
        last = app.st.last
        folder = Path(last.out_dir) if last.out_dir else None
        files = sorted(x.name for x in folder.iterdir()) if (
            folder and folder.exists()) else []
        self.rep["опыт"] = {
            "исход": last.outcome, "папка": str(folder or ""),
            "файлы": files,
            "модельное, с": round(last.t_sim, 3),
            "по стене, с": round(last.t_wall, 1),
            "худший кадр окна, мс": round(self.worst, 1),
        }
        want = ("bins.npz", "meta.json", "rates.parquet", "run.json",
                "spec.json")
        miss = [w for w in want if w not in files]
        if last.outcome != "done":
            self._fail("опыт кончился как %s: %s" % (last.outcome,
                                                     last.message))
        elif miss:
            self._fail("на диске нет: " + ", ".join(miss))
        else:
            self._note("результат сохранён", папка=folder.name,
                       файлов=len(files))
        self.phase = "done"
        return False

    def _done(self, app, now):
        return False

    # ----------------------------------------------------------- отчёт

    def finish(self, app, sh):
        self.rep["всего, с"] = round(time.perf_counter() - self.t0, 1)
        self.rep["окно"] = {"ширина": sh.w, "высота": sh.h,
                            "dpi": getattr(sh.sc, "dpi", 1.0)}
        self.rep["прошло"] = not self.rep["беды"]
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.rep, ensure_ascii=False,
                                       indent=1), encoding="utf-8")
        return 0 if self.rep["прошло"] else 1
