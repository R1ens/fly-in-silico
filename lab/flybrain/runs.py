"""Каталог сохранённых прогонов: что лежит в `out/` и в каком оно состоянии.

Второго хранилища здесь нет. Файлы те же, что пишут `sim.save`, `loop.save`,
`body.save`, `jobs` и `worker._save`; формат ни одного из них не меняется.
Этот модуль только читает их и отвечает на три вопроса, на которые указатель
`index.py` не отвечает:

* **что это за прогон** — вид, а не «карточка для HTML»;
* **чем он кончился** — завершён, остановлен на полпути, упал, неполон или
  неизвестно. `index.scan` такого различия не делает вовсе, а существование
  каталога успехом не является;
* **сколько это стоит прочитать** — обзор каталога не должен открывать ни
  одного `parquet` и ни одного `npz`.

Про цену отдельно. `index.scan` для каждой папки зовёт `_top_regions`, а он
читает `rates.parquet` целиком: на десятке прогонов это уже секунды, а окно
обязано открыться сразу. Поэтому обзор здесь свой и читает только маленькие
JSON-ы, а тяжёлые таблицы открываются по выбору прогона — функциями ниже.
`index.py` при этом остаётся на месте и используется там, где он к месту:
его `read_json` и его же разбор серии через `_welch`.

Состояния результата названы явно и не пересекаются:

``done``        прогон дошёл до конца (`run.json` говорит `cancelled: false`);
``cancelled``   остановлен на полпути — файлы есть, но длительность в них
                подрезана до посчитанного (`jobs._truncate`);
``failed``      задача отдала ошибку; в папке лежит след, но не результат;
``partial``     главного файла вида нет, а что-то другое есть;
``unknown``     исход не записан. Так выглядят все прогоны командной строки:
                она `run.json` не пишет, и выдумывать за неё «успех» нельзя;
``broken``      файл, по которому прогон опознаётся, лежит в папке, но не
                читается. Запись остаётся видимой, с именем файла и самой
                ошибкой: молча выбросить её или выдать за другой вид — ровно
                то, что делал прежний `index._read_json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import data as D
from .index import read_json

__all__ = ["Run", "STATES", "KIND_TITLES", "scan", "detail", "series_table",
           "rates_table", "bins_npz", "spec_of", "repeatable", "folder_size"]

# Подписи состояний: короткое слово и пояснение, почему оно такое.
STATES = {
    "done": ("завершён", "прогон дошёл до конца"),
    "cancelled": ("остановлен", "остановлен на полпути: длительность в файлах "
                                "подрезана до посчитанного"),
    "failed": ("отказ", "задача отдала ошибку"),
    "partial": ("неполный", "главного файла этого вида в папке нет"),
    "unknown": ("неизвестно", "исход не записан: `run.json` пишет только "
                              "запуск из окна, у командной строки его нет"),
    "broken": ("повреждён", "файл прогона лежит в папке, но не читается"),
}

KIND_TITLES = {
    "stim": "Стимуляция",
    "arena": "Арена",
    "body": "В теле",
    "lateral": "Перебор популяций",
    "verify": "Сверка с апстримом",
    "bench": "Серия",
    "agent": "Агент",
    "unknown": "Неопознанный",
}

# Главный файл вида: по нему решается, есть ли в папке результат вообще.
MAIN_FILE = {
    "stim": "meta.json",
    "verify": "meta.json",
    "arena": "episode.json",
    "body": "embodied.json",
    "lateral": "lateral.json",
    "bench": "bench.json",
    "agent": "journal.md",
}


@dataclass
class Run:
    """Одна папка в `out/`. Ничего тяжёлого внутри не держит."""

    name: str
    path: Path
    mtime: float = 0.0
    kind: str = "unknown"
    state: str = "unknown"
    state_why: str = ""
    title: str = ""
    files: tuple = ()
    spec: dict | None = None
    run_json: dict | None = None
    t_requested: float | None = None
    t_actual: float | None = None
    n_episodes: int = 0          # эпизодов в серии, если это серия
    series_conditions: tuple = ()
    has_report: bool = False
    source: str = ""             # "desk" | "cli" | ""
    errors: dict = field(default_factory=dict)   # имя файла -> что с ним
    # Оговорки прогона: чем он отличается от эталонного. Пишет их тот,
    # кто считал (`timing.оговорки`), а каталог обязан их показать —
    # оговорка, которую никто не видит, оговоркой не является.
    caveats: tuple = ()
    # Происхождение зрения: `timing["зрение"]` прогона или `None`, и
    # `None` значит «зрение было выключено» — так посчитаны все прежние
    # прогоны, и ключа у них нет вовсе. Держится отдельным полем, а не
    # вычитывается из `run_json` по месту: требование «какой привязкой
    # считалось, видно в сохранённом прогоне» выполняется тем, что
    # каталог это ПОКАЗЫВАЕТ, а не тем, что данные где-то лежат.
    vision: dict | None = None
    # Сеанс песочницы: прогон, у которого есть единый таймлайн
    # модельного времени (`timeline.npz`) и не было заранее заданного
    # конца. Это не новый вид прогона, а признак: вид остаётся прежним
    # («В теле», «Арена»), а по признаку «Анализ» показывает таймлайн и
    # называет сеанс сеансом, а не оборванным эпизодом.
    session: bool = False
    timeline: dict | None = None     # схема taймлайна без массивов

    @property
    def kind_title(self):
        return KIND_TITLES.get(self.kind, self.kind)

    @property
    def state_word(self):
        return STATES.get(self.state, (self.state, ""))[0]

    @property
    def when(self):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.mtime))

    def has(self, name):
        return name in self.files

    @property
    def has_timeline(self):
        return "timeline.npz" in self.files and "timeline.json" in self.files

    @property
    def has_state(self):
        """Есть ли снимок состояния: с него сеанс можно продолжить."""
        return "state.npz" in self.files and "state.json" in self.files


def _kind_of(files, spec, run_json):
    """Вид прогона. Запись окна точнее догадки по файлам, поэтому первее.

    Серия узнаётся отдельно: `bench.json` пишут и арена, и тело, и сам вид
    прогона при этом остаётся прежним — в `Run.kind` это «серия», а исходный
    вид виден в спеке.
    """
    if run_json:
        if run_json.get("series"):
            return "bench"
        k = str(run_json.get("kind") or "")
        if k in MAIN_FILE:
            return k
    if "bench.json" in files:
        return "bench"
    if "journal.md" in files:
        return "agent"
    if "lateral.json" in files:
        return "lateral"
    if "embodied.json" in files:
        return "body"
    if "episode.json" in files:
        return "arena"
    if "meta.json" in files:
        # Стимуляция и сверка пишут одно и то же. Теперь сверка оставляет
        # ещё и `verify.json` — по нему она узнаётся и без спеки.
        if "verify.json" in files or (spec and spec.get("kind") == "verify"):
            return "verify"
        return "stim"
    if spec and str(spec.get("kind")) in MAIN_FILE:
        return str(spec["kind"])
    return "unknown"


def _state_of(kind, files, run_json):
    """Чем кончился прогон по тому, что лежит в папке.

    `run.json` отвечает прямо, но он есть только у прогонов из окна. Для
    всех остальных ответ уточняется числами — см. `_state_by_numbers`.
    """
    main = MAIN_FILE.get(kind)
    has_main = bool(main and main in files)
    if run_json is not None:
        if run_json.get("error"):
            return "failed", "в `run.json` записана ошибка задачи"
        if run_json.get("cancelled"):
            return "cancelled", STATES["cancelled"][1]
        if not has_main:
            return "partial", ("`run.json` есть, а главного файла вида "
                               "(%s) нет" % (main or "—"))
        return "done", "`run.json` говорит, что прогон дошёл до конца"
    if kind == "unknown":
        return "unknown", ("в папке нет ни одного файла, по которому вид "
                           "узнаётся")
    if not has_main:
        return "partial", "нет %s — главного файла этого вида" % main
    return "unknown", STATES["unknown"][1]


def _state_by_numbers(r, meta):  # noqa: C901
    """Уточнить исход по заказу против факта, если заказ сохранён.

    Явной отметки об обрыве старые прогоны не несут, но у каждого вида
    есть пара «сколько просили» и «сколько посчитали», и расхождение
    между ними — признак, а не догадка:

    * стимуляция и сверка — `spec.json:t` против `meta.json:config.t_run_s`
      (`jobs._truncate` переписывает длительность на фактическую);
    * арена — `metrics.steps × control_ms` против `config.duration_s`
      (здесь, наоборот, заказ остаётся на месте);
    * тело — `timing.duration_s` против `config.duration_s`: единственное
      место, где обе величины лежат рядом в одном файле;
    * серия — одно условие вместо двух: при остановке на норме контроль
      не считается вовсе.

    Где заказ не сохранён (прогоны командной строки без `spec.json`),
    функция молчит и состояние остаётся «неизвестно». Назвать такой
    прогон успешным было бы выдумкой.
    """
    if r.run_json is not None or r.state not in ("unknown",):
        return
    eps = 0.995
    if r.kind in ("stim", "verify"):
        want = (r.spec or {}).get("t")
        cfg = (meta.get("meta.json") or {}).get("config") or {}
        got = cfg.get("t_run_s")
        if want and got is not None and float(got) < float(want) * eps:
            r.state = "cancelled"
            r.state_why = ("заказано %g с, в файле посчитано %g с: прогон "
                           "оборван (`jobs._truncate`)" % (float(want),
                                                           float(got)))
    elif r.kind == "arena":
        ep = meta.get("episode.json") or {}
        cfg, met = ep.get("config") or {}, ep.get("metrics") or {}
        want = cfg.get("duration_s")
        step = cfg.get("control_ms") or 20.0
        steps = met.get("steps")
        if want and steps:
            got = float(steps) * float(step) / 1000.0
            if got < float(want) * eps:
                r.state = "cancelled"
                r.state_why = ("заказано %g с, посчитано %d шагов по %g мс "
                               "= %g с" % (float(want), int(steps),
                                           float(step), got))
    elif r.kind == "body":
        em = meta.get("embodied.json") or {}
        tim = em.get("timing") or {}
        done, want_n = tim.get("steps_done"), tim.get("steps_requested")
        if done is not None and want_n:
            if int(done) < int(want_n):
                r.state = "cancelled"
                r.state_why = ("пройдено %d шагов управления из "
                               "заказанных %d" % (int(done), int(want_n)))
            return
        want = (em.get("config") or {}).get("duration_s")
        got = tim.get("duration_s")
        if want and got is not None and float(got) < float(want) * eps:
            r.state = "cancelled"
            r.state_why = ("заказано %g с, пройдено %g с (`timing.duration_s`)"
                           % (float(want), float(got)))
    elif r.kind == "bench":
        if len(r.series_conditions) == 1:
            r.state = "cancelled"
            r.state_why = ("в серии одно условие (%s) вместо двух: при "
                           "остановке на норме контроль не считается"
                           % r.series_conditions[0])


def _title_of(run_dir, kind, spec, files):
    """Подпись прогона. Заголовок отчёта, иначе спека, иначе имя папки."""
    if "report.html" in files:
        try:
            head = (run_dir / "report.html").open(
                "r", encoding="utf-8").read(4096)
            i = head.find("<title>")
            j = head.find("</title>", i)
            if i >= 0 and j > i:
                t = head[i + 7:j].strip()
                if t:
                    return t
        except Exception:                                     # noqa: BLE001
            pass
    if spec and spec.get("title"):
        return str(spec["title"])
    return KIND_TITLES.get(kind, kind) + " · " + run_dir.name


def scan(out_dir=None):
    """Перечислить прогоны. Тяжёлых файлов не открывает.

    Возвращает список `Run`, свежие первыми. Папки, в которых непонятно
    что, не выбрасываются: «не узнали» — это тоже ответ, и он честнее
    молчания. Папка `exp-NN` внутри прогона агента в список не попадает:
    она принадлежит ему, и агент показывает её сам.
    """
    out_dir = Path(out_dir or D.OUT)
    if not out_dir.exists():
        return []
    runs = []
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            files = tuple(sorted(x.name for x in p.iterdir() if x.is_file()))
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if not files and not any(p.iterdir()):
            runs.append(Run(name=p.name, path=p, mtime=mtime,
                            state="partial",
                            state_why="папка пуста", title=p.name))
            continue
        errors = {}
        spec, err = read_json(p / "spec.json")
        if err:
            errors["spec.json"] = err
        rj, err = read_json(p / "run.json")
        if err:
            errors["run.json"] = err
        kind = _kind_of(files, spec, rj)
        state, why = _state_of(kind, files, rj)
        r = Run(name=p.name, path=p, mtime=mtime, kind=kind, state=state,
                state_why=why, files=files, spec=spec, run_json=rj,
                has_report="report.html" in files, errors=errors,
                source=str((rj or {}).get("source") or ("desk" if rj
                                                        else "")))
        r.title = _title_of(p, kind, spec, files)
        if spec is not None and spec.get("t") is not None:
            r.t_requested = float(spec["t"])
        elif rj and rj.get("t_requested_s"):
            r.t_requested = float(rj["t_requested_s"])
        _fill_times(r)
        runs.append(r)
    runs.sort(key=lambda x: -x.mtime)
    return runs


def _fill_times(r):
    """Фактическая длительность, эпизоды и уточнение исхода.

    Читаются только маленькие JSON-ы вида: ни `parquet`, ни `npz` обзор
    каталога не открывает. `meta.json` у прогонов с большим стимулом
    весит до 82 КБ — там лежат полные списки индексов, — но разбирается
    всё равно быстрее, чем открылась бы таблица.
    """
    small = {}
    for name in ("meta.json", "episode.json", "embodied.json", "bench.json"):
        if not r.has(name):
            continue
        v, err = read_json(r.path / name)
        if err:
            r.errors[name] = err
        elif v is not None:
            small[name] = v
    # Оговорки собираются со всех видов сразу, одним местом: их пишет
    # тот, кто считал (`timing.оговорки` — быстрая физика в теле, окно
    # записи у длинной сессии, правки условий на ходу), а каталог обязан
    # их показать. Разбирать по видам нечего: ключ один и тот же.
    seen, caveats = set(), []
    for v in small.values():
        for line in ((v.get("timing") or {}).get("оговорки") or []):
            line = str(line)
            if line not in seen:
                seen.add(line)
                caveats.append(line)
    r.caveats = tuple(caveats)

    # Зрение. Берётся из того же `timing`, что и оговорки, и тем же
    # перебором по видам: ключ один. Отсутствие ключа — это «зрение
    # выключено», а не «неизвестно», и подменять его пустым словарём
    # нельзя: пустой словарь читался бы как «зрение было, но про него
    # ничего не записано».
    for v in small.values():
        vis = (v.get("timing") or {}).get("зрение")
        if isinstance(vis, dict):
            r.vision = vis
            break

    # Таймлайн: читается только схема (`timeline.json`), массивы —
    # нет. Обзор каталога не должен открывать ни одного `npz`.
    if r.has_timeline:
        tl, err = read_json(r.path / "timeline.json")
        if err:
            r.errors["timeline.json"] = err
        elif isinstance(tl, dict):
            r.timeline = {k: v for k, v in tl.items() if k != "события"}
            r.timeline["событий"] = len(tl.get("события") or [])
            endless = any("без заранее заданного конца" in str(c)
                          for c in r.caveats)
            r.session = bool(endless or r.run_json is None
                             or (r.run_json or {}).get("cancelled"))

    main = MAIN_FILE.get(r.kind)
    if main and main in r.errors:
        # Запись остаётся видимой: папка найдена, остальные файлы целы.
        # Но успехом она не считается и чужим видом не притворяется.
        r.state = "broken"
        r.state_why = "%s не читается: %s" % (main, r.errors[main])

    if r.kind in ("stim", "verify"):
        m = small.get("meta.json") or {}
        cfg = m.get("config") if isinstance(m.get("config"), dict) else m
        if cfg.get("t_run_s") is not None:
            r.t_actual = float(cfg["t_run_s"])
    elif r.kind == "arena":
        cfg = (small.get("episode.json") or {}).get("config") or {}
        if cfg.get("duration_s") is not None:
            r.t_actual = float(cfg["duration_s"])
            r.t_requested = r.t_requested or float(cfg["duration_s"])
    elif r.kind == "body":
        em = small.get("embodied.json") or {}
        cfg, tim = em.get("config") or {}, em.get("timing") or {}
        if tim.get("duration_s") is not None:
            r.t_actual = float(tim["duration_s"])
        elif cfg.get("duration_s") is not None:
            r.t_actual = float(cfg["duration_s"])
        if cfg.get("duration_s") is not None:
            r.t_requested = r.t_requested or float(cfg["duration_s"])
    elif r.kind == "bench":
        b = small.get("bench.json") or {}
        if isinstance(b, dict):
            r.series_conditions = tuple(b.keys())
            r.n_episodes = sum(int(v.get("n_runs") or 0)
                               for v in b.values() if isinstance(v, dict))
            for v in b.values():
                if isinstance(v, dict) and v.get("duration_s"):
                    r.t_actual = float(v["duration_s"])
                    break
    _state_by_numbers(r, small)


def detail(run):
    """Все маленькие файлы прогона разом. Таблицы не трогает."""
    out = {}
    for name in ("meta.json", "episode.json", "embodied.json", "lateral.json",
                 "bench.json", "probes.json", "verify.json", "spec.json",
                 "run.json", "timeline.json", "state.json"):
        if name not in run.files:
            continue
        v, err = read_json(run.path / name)
        if err:
            run.errors[name] = err
        elif v is not None:
            out[name] = v
    if "narrative.md" in run.files:
        try:
            out["narrative.md"] = (run.path / "narrative.md").read_text(
                encoding="utf-8")[:20000]
        except OSError:
            pass
    return out


def series_table(run, columns=None):
    """`track.parquet` прогона — запись по шагам. Открывается по выбору."""
    path = run.path / "track.parquet"
    if not path.exists():
        return None
    import pandas as pd

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:                                         # noqa: BLE001
        return pd.read_parquet(path)


def rates_table(run, columns=None):
    """`rates.parquet` — активные нейроны прогона и их частоты."""
    path = run.path / "rates.parquet"
    if not path.exists():
        return None
    import pandas as pd

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:                                         # noqa: BLE001
        return pd.read_parquet(path)


def bins_npz(run):
    """`bins.npz` — растр активности по бинам. Файл большой, читать лениво."""
    path = run.path / "bins.npz"
    if not path.exists():
        return None
    import numpy as np

    return np.load(path, allow_pickle=False)


def spec_of(run):
    """Спека прогона как словарь — та, что уйдёт в «Эксперимент» на повтор."""
    if run.spec is not None:
        return dict(run.spec)
    return None


def repeatable(run):
    """Можно ли повторить опыт. Возвращает (можно, почему нет)."""
    if run.spec is None:
        if "spec.json" in run.errors:
            return False, ("`spec.json` в папке есть, но не читается: %s"
                           % run.errors["spec.json"])
        return False, ("в папке нет `spec.json`: прогон сделан командной "
                       "строкой или старым окном, и настройки не сохранены")
    kind = str(run.spec.get("kind") or "")
    if kind not in ("stim", "arena", "body", "lateral", "verify"):
        return False, "вид %r этим окном не запускается" % kind
    try:
        from . import spec as S

        S.from_dict(dict(run.spec))
    except Exception as exc:                                  # noqa: BLE001
        return False, "спека не читается нынешним движком: %s" % str(exc)[:120]
    return True, ""


def folder_size(run):
    """Сколько места занял прогон, байты."""
    total = 0
    try:
        for x in run.path.rglob("*"):
            if x.is_file():
                total += x.stat().st_size
    except OSError:
        pass
    return total
