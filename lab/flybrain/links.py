"""Слой данных о связях коннектома: кто входит в набор нейронов и куда он выходит.

Зачем отдельный слой. Матрица связей — это torch sparse COO ``W[post, pre]``
на 15 091 983 ненулевых значения. Любой вопрос вкладки «Связи» («кто подаёт
на AL?», «куда идёт LC10?») — это выборка подмножества строк или столбцов и
свёртка по группам. Делать это каждый раз обходом всей матрицы нельзя: один
проход по 15 миллионам рёбер стоит около секунды, а поле ввода должно
отвечать на нажатие клавиши.

Как устроено. Матрица уже ``coalesce()``, то есть её индексы отсортированы по
строке (post), затем по столбцу (pre). Значит входящие связи для набора
post-индексов — это непрерывные куски массива, которые берутся срезами. Для
исходящих нужен тот же массив, пересобранный в порядке pre. :func:`build_index`
строит оба порядка один раз и держит их как плоские numpy-массивы:

=========================  ==========================  ==========
массив                     что в нём                   размер
=========================  ==========================  ==========
``post_ptr`` (n+1, int64)  начала строк post            1,1 МБ
``pre_ptr``  (n+1, int64)  начала столбцов pre          1,1 МБ
``in_partner`` (nnz,i32)   pre-индекс, в порядке post   57,6 МБ
``in_value``   (nnz,f32)   значение, в порядке post     57,6 МБ
``out_partner``(nnz,i32)   post-индекс, в порядке pre   57,6 МБ
``out_value``  (nnz,f32)   значение, в порядке pre      57,6 МБ
=========================  ==========================  ==========

Итого 232 МБ сверх самой матрицы (288 МБ: индексы в ней int64). Бюджет —
400 МБ, он не превышен, поэтому скромный вариант не включался. Если бюджет
когда-нибудь станет жёстче, экономить надо так: выбросить
``out_partner``/``out_value`` и хранить вместо них перестановку ``order``
(int32) плюс массив строк ``row`` (int32) — те же 115 МБ, но с двойной
выборкой ``row[order[срез]]`` на каждом запросе исходящих. Ещё скромнее — не
хранить ``row`` вовсе, а восстанавливать его через
``searchsorted(post_ptr, ...)``: 58 МБ на оба порядка, ценой бинарного поиска
по 138 тысячам границ на каждое ребро ответа.

Разделение величин. Четыре разные вещи, которые на вкладке «Связи» легко
перепутать и которые здесь считаются раздельно и никогда не подменяют друг
друга:

* ``neurons``  — сколько РАЗНЫХ нейронов-партнёров участвует в группе;
* ``edges``    — сколько РЁБЕР графа (пар пре-пост) ведёт в группу;
* ``synapses`` — сумма числа синапсов по этим рёбрам (по модулю значения);
* ``exc``/``inh`` — та же сумма, разложенная по знаку значения, то есть по
  тому, тормозный ли пресинаптический нейрон рёбра.

Один нейрон-партнёр может дать несколько рёбер (разные нейроны выборки), а
одно ребро — сотни синапсов, поэтому ``neurons <= edges`` и
``edges <= synapses`` (в норме; равенство бывает только на слабых связях).
``exc + inh == synapses`` тождественно.

Что здесь считается на GPU. Запрос на 50 тысяч нейронов задевает миллионы
рёбер, и на процессоре это уже не отвечает «мгновенно». :func:`summarize` при
``device="auto"`` сама уходит на GPU, когда рёбер много и ``torch.cuda`` есть;
результат одинаков с точностью до сложения float (проверено в
``probe/links_test.py``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D

MATRIX_CACHE = D.CACHE / "region_matrix.json"

# Пустое имя группы. В таблице нейронов `cell_type` и `cell_class` — строки, и
# у неаннотированных нейронов они пустые; `region` для таких заполнен словом
# "unknown" ещё в data.assign_regions.
UNNAMED = "unnamed"

# Нейрон, которого нет в таблице. В норме не бывает (таблица и матрица одной
# длины), но код не должен молча складывать такие рёбра в чужую группу.
MISSING = "not in table"

# Порог ухода на GPU для device="auto". Ниже него перенос выборки на карту и
# обратно дороже самого счёта: 300 тысяч рёбер процессор свёртывает за ~6 мс.
GPU_EDGES = 300_000

DIRECTIONS = ("in", "out")
BY_KEYS = ("region", "type", "class")


# ------------------------------------------------------------------- индекс


@dataclass
class Index:
    """Оба порядка матрицы связей, посчитанные один раз.

    Поля-массивы описаны в докстринге модуля. ``codes`` и ``gpu`` — ленивые
    кэши: коды групп по строке ``by`` и копии массивов в видеопамяти; они
    наполняются при первом запросе, который их требует.
    """

    n: int
    nnz: int
    post_ptr: np.ndarray
    pre_ptr: np.ndarray
    in_partner: np.ndarray
    in_value: np.ndarray
    out_partner: np.ndarray
    out_value: np.ndarray
    build_s: float = 0.0
    codes: dict = field(default_factory=dict)
    gpu: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        """Сколько памяти стоит индекс сверх самой матрицы, в байтах."""
        return int(sum(a.nbytes for a in (
            self.post_ptr, self.pre_ptr, self.in_partner, self.in_value,
            self.out_partner, self.out_value)))

    def arrays(self, direction: str):
        """(указатели, партнёры, значения) для направления."""
        if direction == "in":
            return self.post_ptr, self.in_partner, self.in_value
        if direction == "out":
            return self.pre_ptr, self.out_partner, self.out_value
        raise ValueError(f"direction должно быть 'in' или 'out', а не {direction!r}")


def build_index(weights) -> Index:
    """Построить индекс по разреженной матрице ``W[post, pre]``.

    Стоит один проход по рёбрам плюс одна устойчивая сортировка по столбцу;
    на 15 миллионах рёбер это несколько секунд (замер — в
    ``probe/links_test.py``). Результат переиспользуется всеми запросами.
    """
    t0 = time.perf_counter()
    w = weights if weights.is_coalesced() else weights.coalesce()
    n = int(w.shape[0])

    ind = w.indices()
    if ind.device.type != "cpu":
        ind = ind.cpu()
    ind = ind.numpy()
    row, col = ind[0], ind[1]
    val = w.values()
    if val.device.type != "cpu":
        val = val.cpu()
    # Копии, а не виды на хранилище тензора: индекс живёт дольше, чем ссылка
    # на матрицу у вызывающего, и его размер не должен зависеть от того,
    # держит ли кто-то ещё эту матрицу.
    val = np.array(val.numpy(), dtype=np.float32)
    nnz = int(val.size)

    post_ptr = _pointers(row, n)
    pre_ptr = _pointers(col, n)

    in_partner = col.astype(np.int32)
    # Устойчивая сортировка по столбцу: внутри каждого pre строки остаются
    # возрастающими, как они и шли в post-порядке.
    order = np.argsort(col, kind="stable")
    row32 = row.astype(np.int32)
    out_partner = row32[order]
    out_value = val[order]
    del order, row32

    return Index(
        n=n, nnz=nnz, post_ptr=post_ptr, pre_ptr=pre_ptr,
        in_partner=in_partner, in_value=val,
        out_partner=out_partner, out_value=out_value,
        build_s=time.perf_counter() - t0,
    )


def _pointers(keys: np.ndarray, n: int) -> np.ndarray:
    """Границы групп в отсортированном по ``keys`` массиве: ptr[i]..ptr[i+1]."""
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n), out=ptr[1:])
    return ptr


# --------------------------------------------------------------- группировка


def group_codes(index: Index, neurons: pd.DataFrame, by: str):
    """Код группы для каждого модельного индекса и список имён групп.

    Кэшируется в самом индексе: на 138 тысячах строк факторизация стоит
    десятки миллисекунд, а запросов за время работы окна — сотни.
    """
    if by in index.codes:
        return index.codes[by]
    values = _group_values(neurons, by)
    raw, names = pd.factorize(values, sort=False)
    names = [str(x) for x in names]
    # Последняя корзина — нейроны, которых нет в таблице. Она остаётся пустой
    # при согласованных данных и видна в выдаче, если согласованность нарушена.
    codes = np.full(index.n, len(names), dtype=np.int32)
    where = neurons["idx"].to_numpy(np.int64)
    inside = (where >= 0) & (where < index.n)
    codes[where[inside]] = raw[inside].astype(np.int32)
    names.append(MISSING)
    index.codes[by] = (codes, names)
    return index.codes[by]


def _group_values(neurons: pd.DataFrame, by: str) -> pd.Series:
    """Строка-название группы для каждой строки таблицы нейронов."""
    if by == "region":
        return neurons["region"].astype(str)
    if by == "type":
        ct = neurons["cell_type"].astype(str).str.strip()
        hb = neurons["hemibrain_type"].astype(str).str.strip()
        out = ct.where(ct.str.len() > 0, hb)
        return out.where(out.str.len() > 0, UNNAMED)
    if by == "class":
        cc = neurons["cell_class"].astype(str).str.strip()
        return cc.where(cc.str.len() > 0, UNNAMED)
    raise ValueError(f"by должно быть одним из {BY_KEYS}, а не {by!r}")


# ------------------------------------------------------------------- свёртка


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Склеить полуинтервалы ``[s, s+l)`` в один массив позиций.

    Все ``lens`` должны быть положительными — пустые строки отбрасываются
    вызывающим. Без этого пришлось бы собирать куски питоновским циклом по
    выборке, а она бывает на 50 тысяч нейронов.
    """
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    cuts = np.cumsum(lens)[:-1]
    out[cuts] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


def _agg_cpu(index: Index, codes: np.ndarray, ngroups: int,
             idx: np.ndarray, direction: str) -> dict:
    ptr, partner, value = index.arrays(direction)
    starts = ptr[idx]
    lens = ptr[idx + 1] - starts
    keep = lens > 0
    pos = _ranges(starts[keep], lens[keep])

    part = partner[pos]
    val = value[pos]
    mag = np.abs(val)
    grp = codes[part].astype(np.int64)

    syn = np.bincount(grp, weights=mag, minlength=ngroups)
    exc = np.bincount(grp, weights=np.where(val > 0, mag, 0.0), minlength=ngroups)
    edges = np.bincount(grp, minlength=ngroups)
    # Разных нейронов-партнёров в группе: у нейрона ровно одна группа, поэтому
    # достаточно пересчитать уникальные партнёры по их группам.
    uniq = np.unique(part)
    neu = np.bincount(codes[uniq].astype(np.int64), minlength=ngroups)
    return {
        "syn": syn, "exc": exc, "inh": syn - exc,
        "edges": edges, "neurons": neu,
        "total_edges": int(val.size), "total_partners": int(uniq.size),
    }


def _gpu_arrays(index: Index, torch):
    """Копии массивов индекса в видеопамяти, один раз на процесс."""
    if "in_partner" in index.gpu:
        return index.gpu
    g = index.gpu
    for name in ("post_ptr", "pre_ptr", "in_partner", "in_value",
                 "out_partner", "out_value"):
        g[name] = torch.from_numpy(getattr(index, name)).cuda()
    return g


def _agg_gpu(index: Index, codes: np.ndarray, ngroups: int,
             idx: np.ndarray, direction: str, by: str) -> dict:
    import torch

    g = _gpu_arrays(index, torch)
    ptr = g["post_ptr" if direction == "in" else "pre_ptr"]
    partner = g["in_partner" if direction == "in" else "out_partner"]
    value = g["in_value" if direction == "in" else "out_value"]

    ckey = f"codes:{by}"
    if ckey not in index.gpu:
        index.gpu[ckey] = torch.from_numpy(codes.astype(np.int64)).cuda()
    codes_t = index.gpu[ckey]

    sel = torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64)).cuda()
    starts = ptr[sel]
    lens = ptr[sel + 1] - starts
    keep = lens > 0
    starts, lens = starts[keep], lens[keep]
    total = int(lens.sum().item())
    if total == 0:
        zf = np.zeros(ngroups)
        return {"syn": zf, "exc": zf.copy(), "inh": zf.copy(),
                "edges": np.zeros(ngroups, dtype=np.int64),
                "neurons": np.zeros(ngroups, dtype=np.int64),
                "total_edges": 0, "total_partners": 0}
    offs = torch.cumsum(lens, 0) - lens
    pos = (torch.repeat_interleave(starts - offs, lens)
           + torch.arange(total, device=sel.device, dtype=torch.int64))

    part = partner[pos].to(torch.int64)
    val = value[pos]
    mag = val.abs()
    grp = codes_t[part]

    syn = torch.bincount(grp, weights=mag, minlength=ngroups)
    exc = torch.bincount(grp, weights=torch.where(val > 0, mag, torch.zeros_like(mag)),
                         minlength=ngroups)
    edges = torch.bincount(grp, minlength=ngroups)
    uniq = torch.unique(part)
    neu = torch.bincount(codes_t[uniq], minlength=ngroups)

    syn_np = syn.cpu().numpy().astype(np.float64)
    exc_np = exc.cpu().numpy().astype(np.float64)
    return {
        "syn": syn_np, "exc": exc_np, "inh": syn_np - exc_np,
        "edges": edges.cpu().numpy().astype(np.int64),
        "neurons": neu.cpu().numpy().astype(np.int64),
        "total_edges": total, "total_partners": int(uniq.numel()),
    }


def pick_device(index: Index, idx: np.ndarray, direction: str,
                device: str = "auto") -> tuple[str, int]:
    """Где считать и сколько рёбер задето. Оценка рёбер стоит ``O(|idx|)``."""
    ptr, _, _ = index.arrays(direction)
    edges = int((ptr[idx + 1] - ptr[idx]).sum())
    if device == "cpu":
        return "cpu", edges
    import torch

    have = torch.cuda.is_available()
    if device == "cuda":
        if not have:
            raise RuntimeError("device='cuda', но torch.cuda недоступен")
        return "cuda", edges
    if device != "auto":
        raise ValueError(f"device должно быть auto/cpu/cuda, а не {device!r}")
    return ("cuda" if have and edges >= GPU_EDGES else "cpu"), edges


def summarize(index: Index, neurons: pd.DataFrame, idx, direction: str = "in",
              by: str = "region", limit: int = 30, device: str = "auto") -> dict:
    """Партнёры выбранного набора нейронов, сгруппированные и сосчитанные.

    ``idx`` — модельные индексы выбранных нейронов (повторы отбрасываются).
    ``direction``: ``"in"`` — кто подаёт на выбранных, ``"out"`` — куда подают
    выбранные. ``by``: ``"region"``, ``"type"`` или ``"class"`` — по чему
    группировать ПАРТНЁРОВ. ``limit`` — сколько групп вернуть (``None`` или
    ``0`` — все). ``device``: ``"auto"`` уходит на GPU, когда рёбер много.

    Возвращает словарь с двумя ключами:

    ``rows``
        список групп по убыванию ``synapses``, не длиннее ``limit``; в каждой
        записи ``partner``, ``neurons``, ``edges``, ``synapses``, ``exc``,
        ``inh``, ``direction`` — разные величины, которые нельзя смешивать
        (что есть что — в докстринге модуля);
    ``total``
        итог по ВСЕЙ выборке, независимо от ``limit``: ``partners`` (разных
        нейронов-партнёров), ``edges``, ``synapses``, ``exc``, ``inh``,
        ``groups`` (сколько групп вышло всего), ``hidden_groups`` и
        ``hidden_synapses`` (что не поместилось в ``limit``), плюс
        ``selected``, ``device`` и ``elapsed_ms``.

    Итог считается по всем группам, поэтому сумма ``synapses`` по ``rows`` при
    ``limit=None`` совпадает с ``total["synapses"]``, а при обрезке —
    отличается ровно на ``hidden_synapses``.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction должно быть одним из {DIRECTIONS}, а не {direction!r}")
    t0 = time.perf_counter()
    idx = np.unique(np.ascontiguousarray(idx, dtype=np.int64))
    if idx.size and (idx[0] < 0 or idx[-1] >= index.n):
        raise ValueError("индексы нейронов вне матрицы")

    codes, names = group_codes(index, neurons, by)
    ngroups = len(names)
    empty_total = {
        "direction": direction, "by": by, "selected": int(idx.size),
        "partners": 0, "edges": 0, "synapses": 0.0, "exc": 0.0, "inh": 0.0,
        "groups": 0, "hidden_groups": 0, "hidden_synapses": 0.0,
        "device": "cpu", "elapsed_ms": 0.0,
    }
    if idx.size == 0:
        return {"rows": [], "total": empty_total}

    where, _edges_hint = pick_device(index, idx, direction, device)
    if where == "cuda":
        agg = _agg_gpu(index, codes, ngroups, idx, direction, by)
    else:
        agg = _agg_cpu(index, codes, ngroups, idx, direction)

    live = np.flatnonzero(agg["edges"] > 0)
    order = live[np.argsort(-agg["syn"][live], kind="stable")]
    cut = len(order) if not limit else min(int(limit), len(order))
    rows = [{
        "partner": names[g],
        "neurons": int(agg["neurons"][g]),
        "edges": int(agg["edges"][g]),
        "synapses": float(agg["syn"][g]),
        "exc": float(agg["exc"][g]),
        "inh": float(agg["inh"][g]),
        "direction": direction,
    } for g in order[:cut]]

    syn_all = float(agg["syn"].sum())
    shown = float(agg["syn"][order[:cut]].sum())
    total = {
        "direction": direction, "by": by, "selected": int(idx.size),
        "partners": int(agg["total_partners"]),
        "edges": int(agg["total_edges"]),
        "synapses": syn_all,
        "exc": float(agg["exc"].sum()),
        "inh": float(agg["inh"].sum()),
        "groups": int(len(order)),
        "hidden_groups": int(len(order) - cut),
        "hidden_synapses": syn_all - shown,
        "device": where,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
    return {"rows": rows, "total": total}


# ------------------------------------------------------- нейропиль → нейропиль


def region_matrix(index: Index, neurons: pd.DataFrame,
                  rebuild: bool = False) -> dict:
    """Агрегат связей нейропиль → нейропиль, с кэшем на диске.

    ``M[i][j]`` — из области ``regions[i]`` (пресинаптическая) в область
    ``regions[j]`` (постсинаптическая). Отдаются четыре матрицы: ``synapses``,
    ``edges``, ``exc``, ``inh``.

    Считается одним проходом по всем рёбрам, поэтому стоит около секунды и
    кэшируется рядом с таблицей нейронов — так же, как
    ``worker.load_populations``: кэш проверяется по времени изменения
    ``neurons.parquet``, числу нейронов и числу рёбер, и устаревает сам, когда
    таблица пересобрана.
    """
    stamp = D.NEURONS_CACHE.stat().st_mtime if D.NEURONS_CACHE.exists() else 0.0
    if MATRIX_CACHE.exists() and not rebuild:
        try:
            obj = json.loads(MATRIX_CACHE.read_text(encoding="utf-8"))
            if (abs(float(obj.get("neurons_mtime", -1)) - stamp) < 1e-6
                    and int(obj.get("n_neurons", -1)) == index.n
                    and int(obj.get("nnz", -1)) == index.nnz):
                obj["from_cache"] = True
                return obj
        except (ValueError, KeyError, OSError):
            pass

    t0 = time.perf_counter()
    codes, names = group_codes(index, neurons, "region")
    ng = len(names)
    syn = np.zeros(ng * ng)
    exc = np.zeros(ng * ng)
    edges = np.zeros(ng * ng, dtype=np.int64)

    # Куском по четыре миллиона рёбер: пары областей приходится держать в
    # int64, и на всех 15 миллионах это был бы лишний 120-мегабайтный массив
    # ровно ради одного bincount.
    step = 4_000_000
    counts = np.diff(index.post_ptr)
    # Код области постсинаптического нейрона для каждого ребра. Размножается
    # сразу код, а не индекс нейрона: int32 на ребро вместо int64.
    dst_code = np.repeat(codes, counts)
    for lo in range(0, index.nnz, step):
        hi = min(lo + step, index.nnz)
        src = codes[index.in_partner[lo:hi]].astype(np.int64)
        dst = dst_code[lo:hi].astype(np.int64)
        pair = src * ng + dst
        val = index.in_value[lo:hi]
        mag = np.abs(val)
        syn += np.bincount(pair, weights=mag, minlength=ng * ng)
        exc += np.bincount(pair, weights=np.where(val > 0, mag, 0.0),
                           minlength=ng * ng)
        edges += np.bincount(pair, minlength=ng * ng)
    del dst_code

    out = {
        "neurons_mtime": stamp,
        "n_neurons": int(index.n),
        "nnz": int(index.nnz),
        "regions": names,
        "synapses": syn.reshape(ng, ng).tolist(),
        "exc": exc.reshape(ng, ng).tolist(),
        "inh": (syn - exc).reshape(ng, ng).tolist(),
        "edges": edges.reshape(ng, ng).tolist(),
        "build_s": round(time.perf_counter() - t0, 3),
    }
    D.CACHE.mkdir(parents=True, exist_ok=True)
    MATRIX_CACHE.write_text(json.dumps(out, ensure_ascii=False),
                            encoding="utf-8")
    out["from_cache"] = False
    return out
