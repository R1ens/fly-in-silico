"""Loading the connectome: neuron table, region labels, sparse weight matrix.

Two data sources are joined here:

* ``upstream/data/2025_Completeness_783.csv`` and
  ``2025_Connectivity_783.parquet`` -- the model itself, as shipped by
  eonsystemspbc/fly-brain.  The CSV row order defines the neuron index used by
  every simulator backend, so it is the authority on indices.  It carries no
  names, no classes and no coordinates.
* ``cache/annotations_783.tsv`` -- the FlyWire v783 annotation release of
  Schlegel et al. (2024), which supplies cell types, classes, neurotransmitter,
  side and soma coordinates for the same root ids.

Everything derived is cached under ``cache/`` so the second run is fast.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
import weakref
from pathlib import Path

import numpy as np
import pandas as pd

from . import regions as R

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
CACHE = ROOT / "cache"

# Куда ложатся прогоны. Обычно это `out/` рядом с проектом, но проверкам
# там делать нечего: они за один вечер насыпали сотню папок поверх
# научных результатов, и отличать одно от другого пришлось вручную.
# `FLYBRAIN_OUT` уводит их в отдельный каталог — для проб это временная
# папка, и научные результаты остаются нетронутыми по определению, а не
# по аккуратности.
OUT = Path(os.environ.get("FLYBRAIN_OUT") or (ROOT / "out"))

PATH_COMP = UPSTREAM / "data" / "2025_Completeness_783.csv"
PATH_CONN = UPSTREAM / "data" / "2025_Connectivity_783.parquet"
PATH_ANNOT = CACHE / "annotations_783.tsv"

NEURONS_CACHE = CACHE / "neurons.parquet"
WEIGHT_CACHE = CACHE / "weights_coo.pt"

ANNOT_COLUMNS = [
    "root_id",
    "pos_x",
    "pos_y",
    "pos_z",
    "soma_x",
    "soma_y",
    "soma_z",
    "flow",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemibrain_type",
    "ito_lee_hemilineage",
    "top_nt",
    "side",
    "nerve",
]

# FlyWire voxel size, nanometres
VOXEL_NM = (4.0, 4.0, 40.0)


# ---------------------------------------------------------------- neuron table


def _read_annotations() -> pd.DataFrame:
    if not PATH_ANNOT.exists():
        raise FileNotFoundError(
            f"{PATH_ANNOT} not found. Run: python -m flybrain.cli fetch-annotations"
        )
    df = pd.read_csv(
        PATH_ANNOT, sep="\t", usecols=ANNOT_COLUMNS, low_memory=False
    )
    df = df.drop_duplicates(subset="root_id", keep="first")
    return df.set_index("root_id")


def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``region``, ``region_src`` and ``super_region`` columns in place.

    Confidence drops down the list: a named cell family or neuropil prefix in
    the cell-type name is the strongest signal, then the cell class, then the
    optic-lobe family (for optic neurons only), then the super class, and
    finally a 15-nearest-neighbour vote among somata that did get a label.
    """
    OPTIC_SUPERS = ("optic", "visual_projection", "visual_centrifugal")
    n = len(df)
    region = np.array([None] * n, dtype=object)
    source = np.array(["none"] * n, dtype=object)

    types = df["cell_type"].fillna("").astype(str).to_numpy()
    hb_types = df["hemibrain_type"].fillna("").astype(str).to_numpy()
    classes = df["cell_class"].fillna("").astype(str).to_numpy()
    supers = df["super_class"].fillna("").astype(str).to_numpy()

    for i in range(n):
        reg = R.region_from_type(types[i]) or R.region_from_type(hb_types[i])
        if reg:
            region[i] = reg
            source[i] = "cell_type"
            continue
        reg = R.CLASS_TO_REGION.get(classes[i])
        if reg:
            region[i] = reg
            source[i] = "cell_class"
            continue
        if supers[i] in OPTIC_SUPERS:
            reg = R.region_from_optic_family(types[i]) or R.region_from_optic_family(
                hb_types[i]
            )
            if reg:
                region[i] = reg
                source[i] = "optic_family"
                continue
        reg = R.SUPER_TO_REGION.get(supers[i])
        if reg:
            region[i] = reg
            source[i] = "super_class"

    df["region"] = region
    df["region_src"] = source

    # Nearest-neighbour fill for the rest, using soma position.
    known = df["region"].notna() & df["x_um"].notna()
    unknown = df["region"].isna() & df["x_um"].notna()
    if unknown.any() and known.sum() > 50:
        from scipy.spatial import cKDTree

        coords = df.loc[known, ["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        labels = df.loc[known, "region"].to_numpy(object)
        tree = cKDTree(coords)
        query = df.loc[unknown, ["x_um", "y_um", "z_um"]].to_numpy(np.float32)
        k = min(15, len(coords))
        _, idx = tree.query(query, k=k)
        idx = np.atleast_2d(idx)
        voted = []
        for row in idx:
            vals, counts = np.unique(labels[row], return_counts=True)
            voted.append(vals[int(np.argmax(counts))])
        df.loc[unknown, "region"] = voted
        df.loc[unknown, "region_src"] = "knn_soma"

    df["region"] = df["region"].fillna("unknown")
    df["super_region"] = [R.super_region(a) for a in df["region"]]
    return df


def build_neuron_table() -> pd.DataFrame:
    """Join model indices with FlyWire annotations and derive regions."""
    comp = pd.read_csv(PATH_COMP, index_col=0)
    comp.index.name = "root_id"
    df = pd.DataFrame(
        {
            "root_id": comp.index.to_numpy(np.int64),
            "completed": comp["Completed"].to_numpy(),
        }
    )
    df["idx"] = np.arange(len(df), dtype=np.int32)

    annot = _read_annotations()
    df = df.join(annot, on="root_id")

    # Prefer the soma; fall back to the generic position annotation.
    for axis, voxel in zip("xyz", VOXEL_NM):
        soma = df[f"soma_{axis}"]
        pos = df[f"pos_{axis}"]
        df[f"{axis}_um"] = (soma.where(soma.notna(), pos) * voxel) / 1000.0
    df = df.drop(columns=[f"soma_{a}" for a in "xyz"] + [f"pos_{a}" for a in "xyz"])

    df = assign_regions(df)

    for col in ("cell_type", "hemibrain_type", "cell_class", "cell_sub_class",
                "super_class", "side", "top_nt", "nerve", "flow",
                "ito_lee_hemilineage"):
        df[col] = df[col].fillna("").astype(str)

    df["label"] = np.where(
        df["cell_type"] != "",
        df["cell_type"],
        np.where(df["hemibrain_type"] != "", df["hemibrain_type"], df["cell_class"]),
    )
    df["label"] = df["label"].replace("", "unnamed")
    return df.set_index("idx", drop=False)


def load_neurons(rebuild: bool = False) -> pd.DataFrame:
    if NEURONS_CACHE.exists() and not rebuild:
        df = pd.read_parquet(NEURONS_CACHE)
        return df.set_index("idx", drop=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build_neuron_table()
    df.to_parquet(NEURONS_CACHE, index=False)
    return df


# ------------------------------------------------------------- weight matrix


@contextlib.contextmanager
def _quiet_sparse():
    """Торч ругается на отключённые проверки разреженных тензоров при каждом
    создании матрицы. Проверки нам не нужны — индексы мы строим сами, — а в
    окне управления это предупреждение занимает пол-экрана.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
        yield


def load_weights(device: str = "cpu", rebuild: bool = False):
    """Signed synapse-count matrix W[post, pre] as a torch sparse COO tensor.

    Values are ``Excitatory x Connectivity``: the number of synapses of the
    connection, negative for inhibitory presynaptic neurons.  The simulator
    multiplies them by ``wScale`` millivolts.
    """
    import torch

    if WEIGHT_CACHE.exists() and not rebuild:
        obj = torch.load(WEIGHT_CACHE, weights_only=True)
        with _quiet_sparse():
            w = torch.sparse_coo_tensor(
                obj["indices"], obj["values"], tuple(obj["shape"])
            ).coalesce()
        return w.to(device)

    comp = pd.read_csv(PATH_COMP, index_col=0)
    n = len(comp)
    conn = pd.read_parquet(
        PATH_CONN,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    idx = np.vstack(
        [
            conn["Postsynaptic_Index"].to_numpy(np.int64),
            conn["Presynaptic_Index"].to_numpy(np.int64),
        ]
    )
    val = conn["Excitatory x Connectivity"].to_numpy(np.float32)
    indices = torch.from_numpy(idx)
    values = torch.from_numpy(val)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"indices": indices, "values": values, "shape": (n, n)}, WEIGHT_CACHE
    )
    with _quiet_sparse():
        w = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return w.to(device)


# ----------------------------------------------------------------- selection


def select(df: pd.DataFrame, query: str) -> np.ndarray:
    """Resolve a selection query to model indices.

    Supported forms, comma-separated and combined as a union::

        type:Gr64f           cell_type or hemibrain_type contains (regex, case-insensitive)
        type=MBON01          cell_type or hemibrain_type equals exactly
        class:gustatory      cell_class contains
        super:descending     super_class contains
        region:AL            region equals
        nt:gaba              neurotransmitter contains
        side:left            left / right / center
        id:720575940627652358
        idx:12345

    A bare word is treated as ``type:<word>``.  Terms can be chained with ``+``
    to intersect, e.g. ``type:LC10+side:left``.
    """
    idx = _codes(df)["idx"]
    parts = []
    for part in str(query).split(","):
        # Quotes survive some shells and pasted commands; a selector wrapped in
        # them should still work rather than look like an unknown key.
        part = part.strip().strip("\"'").strip()
        if not part:
            continue
        mask = np.ones(len(df), dtype=bool)
        for term in part.split("+"):
            mask &= _term_mask(df, term.strip())
        parts.append(idx[mask])
    if not parts:
        return np.array([], dtype=np.int64)
    # Было `sorted(set(...))` через список питоновских целых: на 138 тысячах
    # строк это заметно дороже самой выборки. `np.unique` даёт тот же
    # отсортированный уникальный результат.
    return np.unique(np.concatenate(parts)).astype(np.int64)


_SEL_CACHE: dict[int, dict] = {}


def _cache(df: pd.DataFrame) -> dict:
    """Кэш выборок, живущий рядом с таблицей, но не на ней.

    Зачем. Один селектор вида ``region:AL+side:left`` раньше сканировал
    138 639 строк регулярным выражением на каждое слагаемое, а список
    популяций (:func:`world.probe_candidates`) гоняет около 250 таких
    запросов, из которых ``side:left`` встречается в половине. Замер:
    13,7 с на построение списка, 19,0 с на 247 селекторов, и 48 млн
    вызовов ``re.search`` в профиле эпизода в теле — больше, чем вся
    физика MuJoCo.

    Первая попытка держала кэш в ``df.attrs``, и это оказалось ловушкой:
    pandas переносит ``attrs`` на производные таблицы, а
    ``DataFrame.to_parquet`` сериализует их в JSON. Массив numpy в кэше
    ронял сохранение **всех** прогонов — поймано регрессией
    (`probe/cli_check.py`), а не рассуждением.

    Поэтому кэш лежит в словаре модуля по ``id`` таблицы, а
    ``weakref.finalize`` убирает запись, когда таблицу собирает сборщик
    мусора: ни утечки памяти, ни повторного использования ``id`` под
    чужую таблицу. Отпечаток (длина и края столбца ``idx``) стоит
    микросекунды и страхует от подвыборки с тем же ``id``.
    """
    stamp = (len(df), int(df["idx"].iat[0]), int(df["idx"].iat[-1])) \
        if len(df) else (0, -1, -1)
    key = id(df)
    c = _SEL_CACHE.get(key)
    if c is None or c["stamp"] != stamp:
        c = {"stamp": stamp, "столбцы": {}, "слагаемые": {}}
        _SEL_CACHE[key] = c
        weakref.finalize(df, _SEL_CACHE.pop, key, None)
    return c


def _codes(df: pd.DataFrame) -> dict:
    """Числовые столбцы, нужные выборке на каждом запросе."""
    c = _cache(df)
    got = c.get("числа")
    if got is None:
        got = {"idx": df["idx"].to_numpy(),
               "root_id": df["root_id"].to_numpy(np.int64)}
        c["числа"] = got
    return got


def _column(df: pd.DataFrame, col: str):
    """Столбец как ``(коды, различные значения)``.

    Предикат считается по различным значениям, а не по строкам: в
    ``side`` их пять на 138 639 строк, в ``cell_type`` — 8 846. Результат
    тот же до бита, потому что и ``str.contains``, и ``str.fullmatch``
    применяются к каждому элементу независимо, а ``astype(str)``
    выполняется до разбиения — значит пропусков в значениях уже нет и
    ``na=False`` ни на что не влияет.
    """
    c = _cache(df)
    got = c["столбцы"].get(col)
    if got is None:
        codes, uniques = pd.factorize(df[col].astype(str), sort=False)
        got = (codes, pd.Series(uniques, dtype=object))
        c["столбцы"][col] = got
    return got


def _term_mask(df: pd.DataFrame, term: str) -> np.ndarray:
    """Маска одного слагаемого; одинаковые слагаемые считаются один раз."""
    key = term.strip().strip("\"'").strip()
    c = _cache(df)
    got = c["слагаемые"].get(key)
    if got is None:
        got = _term_mask_slow(df, key)
        got.flags.writeable = False  # чтобы `mask &=` не испортил кэш
        c["слагаемые"][key] = got
    return got


def _term_mask_slow(df: pd.DataFrame, term: str) -> np.ndarray:
    term = term.strip().strip("\"'").strip()
    if ":" in term:
        key, value = term.split(":", 1)
        exact = False
    elif "=" in term:
        key, value = term.split("=", 1)
        exact = True
    else:
        key, value, exact = "type", term, False
    key = key.strip().lower()
    value = value.strip()

    def contains(col):
        codes, uniques = _column(df, col)
        hits = uniques.str.contains(value, case=False, regex=True,
                                    na=False).to_numpy()
        return hits[codes]

    def equals(col):
        codes, uniques = _column(df, col)
        hits = uniques.str.fullmatch(value, case=False, na=False).to_numpy()
        return hits[codes]

    hit = equals if exact else contains

    if key in ("type", "cell_type", "t"):
        return hit("cell_type") | hit("hemibrain_type") | hit("label")
    if key in ("class", "cell_class", "c"):
        return hit("cell_class") | hit("cell_sub_class")
    if key in ("super", "super_class", "s"):
        return hit("super_class")
    if key in ("region", "r"):
        # Область сравнивается целиком всегда, даже без `=`.
        codes, uniques = _column(df, "region")
        hits = uniques.str.fullmatch(value, case=False, na=False).to_numpy()
        return hits[codes]
    if key in ("superregion", "super_region"):
        return hit("super_region")
    if key in ("nt", "transmitter"):
        return hit("top_nt")
    if key == "side":
        return hit("side")
    if key == "nerve":
        return hit("nerve")
    if key in ("lineage", "hemilineage"):
        return hit("ito_lee_hemilineage")
    if key in ("id", "root_id"):
        return _codes(df)["root_id"] == np.int64(value)
    if key == "idx":
        return _codes(df)["idx"] == int(value)
    if key == "all":
        return np.ones(len(df), dtype=bool)
    raise ValueError(f"unknown selector key: {key!r} in term {term!r}")


def describe(df: pd.DataFrame, idx) -> pd.DataFrame:
    cols = ["root_id", "label", "cell_class", "super_class", "region", "side", "top_nt"]
    return df.loc[list(idx), cols]


def summary() -> dict:
    df = load_neurons()
    return {
        "neurons": int(len(df)),
        "annotated": int((df["super_class"] != "").sum()),
        "regions": int(df["region"].nunique()),
        "region_source_counts": df["region_src"].value_counts().to_dict(),
        "top_regions": df["region"].value_counts().head(15).to_dict(),
        "super_classes": df["super_class"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
