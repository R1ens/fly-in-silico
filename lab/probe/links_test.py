"""Проверка слоя связей: правильность величин, сверка сумм, замеры.

Главный вопрос здесь не «быстро ли», а «не перепутаны ли величины». Числа
нейронов-партнёров, рёбер и синапсов на вкладке «Связи» различаются в разы, и
подмена одного другим не ломает ничего видимого — она просто врёт. Поэтому
первая и главная проверка — крошечная матрица на шесть нейронов и одиннадцать
связей, посчитанная руками.

Дальше — настоящий коннектом: верхние партнёры для четырёх выборок, сверка
суммы синапсов с прямым подсчётом по матрице, тождество ``exc + inh ==
synapses``, согласование направлений и таблица замеров.

Запуск:
    set PYTHONIOENCODING=utf-8
    set PYTHONUTF8=1
    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\links_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D, links as L  # noqa: E402

bad = 0


def check(name, cond, detail=""):
    global bad
    if not cond:
        bad += 1
    print(("  [ок] " if cond else "  [!!] ") + name
          + ((": " + detail) if detail else ""), flush=True)


def head(title):
    print("\n" + title, flush=True)
    print("-" * len(title), flush=True)


def row_of(res, partner):
    for r in res["rows"]:
        if r["partner"] == partner:
            return r
    return None


def same(res, partner, neurons, edges, synapses, exc, inh):
    """Сверить все пять величин группы с посчитанными руками."""
    r = row_of(res, partner)
    if r is None:
        check(f"группа {partner}", False, "её нет в выдаче")
        return
    got = (r["neurons"], r["edges"], r["synapses"], r["exc"], r["inh"])
    want = (neurons, edges, float(synapses), float(exc), float(inh))
    check(f"{partner}: нейронов/рёбер/синапсов/возб/торм"
          f" = {got[0]}/{got[1]}/{got[2]:.0f}/{got[3]:.0f}/{got[4]:.0f}",
          got == want, "" if got == want else f"ждали {want}")


# ------------------------------------------------- крошечная матрица руками


def tiny():
    """Шесть нейронов, одиннадцать связей, оба знака.

    Знак — свойство пресинаптического нейрона: 0, 1, 4 возбуждающие,
    2, 3, 5 тормозные, и все их рёбра идут со своим знаком.

    Области: 0, 1, 5 — AL; 2, 3 — LH; 4 — MB.
    """
    neurons = pd.DataFrame({
        "idx": np.arange(6, dtype=np.int32),
        "region": ["AL", "AL", "LH", "LH", "MB", "AL"],
        "cell_type": ["ORN1", "ORN2", "LH1", "", "KC1", ""],
        "hemibrain_type": ["", "", "", "LH2x", "", ""],
        "cell_class": ["olfactory", "olfactory", "LHON", "LHON",
                       "Kenyon_Cell", ""],
    }).set_index("idx", drop=False)

    # (pre, post, синапсы со знаком пресинаптического нейрона)
    edges = [
        (0, 2, +10), (0, 3, +5), (1, 2, +7),
        (2, 4, -3), (3, 4, -8), (5, 4, -1),
        (0, 4, +2), (1, 4, +4),
        (4, 0, +6), (4, 1, +9), (2, 0, -12),
    ]
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int64)
    val = np.array([e[2] for e in edges], dtype=np.float32)
    with D._quiet_sparse():
        w = torch.sparse_coo_tensor(
            np.vstack([post, pre]), val, (6, 6)).coalesce()
    return neurons, w, len(edges)


def test_tiny():
    head("1. Крошечная матрица: все величины сверены руками")
    neurons, w, n_edges = tiny()
    index = L.build_index(w)
    check(f"индекс собран: нейронов {index.n}, рёбер {index.nnz}",
          index.n == 6 and index.nnz == n_edges)

    # --- входящие для нейрона 4: 0(+2), 1(+4), 2(-3), 3(-8), 5(-1)
    res = L.summarize(index, neurons, np.array([4]), "in", "region", limit=30,
                      device="cpu")
    print("  входящие для нейрона 4 по областям:")
    same(res, "LH", 2, 2, 11, 0, 11)
    same(res, "AL", 3, 3, 7, 6, 1)
    check("порядок по убыванию синапсов",
          [r["partner"] for r in res["rows"]] == ["LH", "AL"])
    t = res["total"]
    check(f"итог: партнёров {t['partners']}, рёбер {t['edges']},"
          f" синапсов {t['synapses']:.0f}, групп {t['groups']}",
          (t["partners"], t["edges"], t["synapses"], t["groups"])
          == (5, 5, 18.0, 2))
    check("поле direction отдаётся как пришло",
          all(r["direction"] == "in" for r in res["rows"]))

    # --- исходящие для нейрона 0: 2(+10), 3(+5), 4(+2)
    res = L.summarize(index, neurons, np.array([0]), "out", "region",
                      device="cpu")
    print("  исходящие от нейрона 0 по областям:")
    same(res, "LH", 2, 2, 15, 15, 0)
    same(res, "MB", 1, 1, 2, 2, 0)
    t = res["total"]
    check(f"итог: партнёров {t['partners']}, рёбер {t['edges']},"
          f" синапсов {t['synapses']:.0f}",
          (t["partners"], t["edges"], t["synapses"]) == (3, 3, 17.0))

    # --- главный случай: нейронов 1, рёбер 2, синапсов 15 — три разных числа
    res = L.summarize(index, neurons, np.array([0, 1]), "in", "region",
                      device="cpu")
    print("  входящие для {0,1}: один партнёр даёт два ребра")
    same(res, "MB", 1, 2, 15, 15, 0)
    same(res, "LH", 1, 1, 12, 0, 12)
    r = row_of(res, "MB")
    check("нейроны != рёбра != синапсы (1 / 2 / 15)",
          r["neurons"] == 1 and r["edges"] == 2 and r["synapses"] == 15.0)

    # --- группировка по типу: 3 берётся из hemibrain_type, 5 безымянный
    res = L.summarize(index, neurons, np.array([4]), "in", "type",
                      device="cpu")
    print("  входящие для нейрона 4 по типам:")
    for p, syn in (("LH2x", 8), ("ORN2", 4), ("LH1", 3), ("ORN1", 2),
                   (L.UNNAMED, 1)):
        same(res, p, 1, 1, syn, syn if p.startswith("ORN") else 0,
             0 if p.startswith("ORN") else syn)
    check("hemibrain_type подхвачен, когда cell_type пуст",
          row_of(res, "LH2x") is not None)

    # --- группировка по классу
    res = L.summarize(index, neurons, np.array([4]), "in", "class",
                      device="cpu")
    print("  входящие для нейрона 4 по классам:")
    same(res, "LHON", 2, 2, 11, 0, 11)
    same(res, "olfactory", 2, 2, 6, 6, 0)
    same(res, L.UNNAMED, 1, 1, 1, 0, 1)

    # --- обрезка по limit и честный итог
    res = L.summarize(index, neurons, np.array([4]), "in", "type", limit=2,
                      device="cpu")
    t = res["total"]
    check(f"limit=2: выдано {len(res['rows'])}, скрыто групп"
          f" {t['hidden_groups']}, скрыто синапсов {t['hidden_synapses']:.0f}",
          len(res["rows"]) == 2 and t["hidden_groups"] == 3
          and t["hidden_synapses"] == 6.0 and t["groups"] == 5)
    check("итог считается по всем группам, а не по выданным",
          t["synapses"] == 18.0 and t["partners"] == 5)

    # --- пустая выборка
    res = L.summarize(index, neurons, np.array([], dtype=np.int64), "in")
    check("пустая выборка: пустые строки и нулевой итог",
          res["rows"] == [] and res["total"]["synapses"] == 0.0)

    # --- GPU даёт то же самое
    if torch.cuda.is_available():
        a = L.summarize(index, neurons, np.array([0, 1, 4]), "in", "region",
                        device="cpu")
        b = L.summarize(index, neurons, np.array([0, 1, 4]), "in", "region",
                        device="cuda")
        check("GPU и процессор согласны на крошечной матрице",
              a["rows"] == b["rows"]
              and a["total"]["synapses"] == b["total"]["synapses"])

    # --- матрица областей на крошечном примере
    #     AL->LH: 10+5+7 = 22 синапса, 3 ребра; LH->MB: 3+8 = 11; AL->MB: 2+4+1=7
    print("  матрица областей на крошечном примере:")
    codes, names = L.group_codes(index, neurons, "region")
    syn = np.zeros((len(names), len(names)))
    cnt = np.zeros((len(names), len(names)), dtype=np.int64)
    for p in range(index.n):
        lo, hi = index.pre_ptr[p], index.pre_ptr[p + 1]
        for k in range(lo, hi):
            syn[codes[p], codes[index.out_partner[k]]] += abs(
                index.out_value[k])
            cnt[codes[p], codes[index.out_partner[k]]] += 1
    i_al, i_lh, i_mb = (names.index("AL"), names.index("LH"),
                        names.index("MB"))
    check(f"AL->LH = {syn[i_al, i_lh]:.0f} синапсов / {cnt[i_al, i_lh]} рёбер",
          syn[i_al, i_lh] == 22.0 and cnt[i_al, i_lh] == 3)
    check(f"LH->MB = {syn[i_lh, i_mb]:.0f} синапсов / {cnt[i_lh, i_mb]} рёбер",
          syn[i_lh, i_mb] == 11.0 and cnt[i_lh, i_mb] == 2)
    check(f"AL->MB = {syn[i_al, i_mb]:.0f} синапсов / {cnt[i_al, i_mb]} рёбер",
          syn[i_al, i_mb] == 7.0 and cnt[i_al, i_mb] == 3)


# ------------------------------------------------------ настоящий коннектом


def show(res, title, k=10):
    t = res["total"]
    print(f"\n  {title}")
    print(f"    выбрано {t['selected']}, партнёров {t['partners']},"
          f" рёбер {t['edges']}, синапсов {t['synapses']:.0f}"
          f" (возб {t['exc']:.0f} / торм {t['inh']:.0f}),"
          f" групп {t['groups']}, на {t['device']} за {t['elapsed_ms']:.1f} мс")
    print("    {:<22} {:>8} {:>9} {:>11} {:>11} {:>11}".format(
        "партнёр", "нейронов", "рёбер", "синапсов", "возб.", "торм."))
    for r in res["rows"][:k]:
        print("    {:<22} {:>8} {:>9} {:>11.0f} {:>11.0f} {:>11.0f}".format(
            r["partner"][:22], r["neurons"], r["edges"], r["synapses"],
            r["exc"], r["inh"]))


def direct_sum(index, idx, direction):
    """Прямой подсчёт суммы синапсов по массивам индекса, без группировки.

    Считается в лоб маской по всем 15 миллионам рёбер — это и есть
    независимая от :func:`summarize` ветка.
    """
    mark = np.zeros(index.n, dtype=bool)
    mark[idx] = True
    if direction == "in":
        counts, value = np.diff(index.post_ptr), index.in_value
    else:
        counts, value = np.diff(index.pre_ptr), index.out_value
    owner = np.repeat(mark, counts)
    # Складывать float32 в float32 на миллионах слагаемых нельзя: на всём
    # коннектоме накапливается расхождение в двадцать синапсов.
    return float(np.abs(value[owner]).sum(dtype=np.float64)), int(owner.sum())


def test_real(index, df):
    head("2. Настоящий коннектом: верхние партнёры")
    sets = {
        "region:AL": D.select(df, "region:AL"),
        "type:LC10": D.select(df, "type:LC10"),
        "class:gustatory+side:left": D.select(df, "class:gustatory+side:left"),
    }
    one = int(sets["type:LC10"][0])
    sets[f"один нейрон idx:{one}"] = np.array([one])

    for name, idx in sets.items():
        for direction in ("in", "out"):
            res = L.summarize(index, df, idx, direction, "region", limit=10)
            show(res, f"{name} — {'входящие' if direction == 'in' else 'исходящие'}"
                      f" (n={len(idx)}), по областям")

    print("\n  для разнообразия — по типам и классам:")
    show(L.summarize(index, df, sets["type:LC10"], "out", "type", limit=10),
         "type:LC10 — исходящие, по типам партнёров")
    show(L.summarize(index, df, sets["region:AL"], "in", "class", limit=10),
         "region:AL — входящие, по классам партнёров")

    head("3. Сверка сумм")
    for name, idx in sets.items():
        for direction in ("in", "out"):
            res = L.summarize(index, df, idx, direction, "region", limit=None,
                              device="cpu")
            t = res["total"]
            by_group = sum(r["synapses"] for r in res["rows"])
            d_syn, d_edges = direct_sum(index, idx, direction)
            ok = (abs(by_group - d_syn) < 1.0 and abs(t["synapses"] - d_syn) < 1.0
                  and t["edges"] == d_edges)
            check(f"{name} / {direction}: по группам {by_group:.0f},"
                  f" итог {t['synapses']:.0f}, прямо по матрице {d_syn:.0f};"
                  f" рёбер {t['edges']} против {d_edges}", ok)
            check(f"{name} / {direction}: exc + inh == synapses",
                  abs(t["exc"] + t["inh"] - t["synapses"]) < 1.0,
                  f"{t['exc']:.0f} + {t['inh']:.0f}")
            per_row = all(abs(r["exc"] + r["inh"] - r["synapses"]) < 1e-6
                          for r in res["rows"])
            check(f"{name} / {direction}: exc + inh == synapses в каждой группе",
                  per_row)
            mono = all(r["neurons"] <= r["edges"] for r in res["rows"])
            check(f"{name} / {direction}: нейронов <= рёбер в каждой группе",
                  mono)

    head("4. Согласование направлений")
    al = sets["region:AL"]
    out = L.summarize(index, df, al, "out", "region", limit=None, device="cpu")
    target = None
    for r in out["rows"]:
        if r["partner"] != "AL":
            target = r
            break
    region = target["partner"]
    back = D.select(df, f"region:{region}")
    inc = L.summarize(index, df, back, "in", "region", limit=None, device="cpu")
    mirror = row_of(inc, "AL")
    check(f"AL -> {region}: исходящие AL дают {target['synapses']:.0f} синапсов"
          f" и {target['edges']} рёбер; входящие {region} от AL дают"
          f" {mirror['synapses']:.0f} и {mirror['edges']}",
          abs(target["synapses"] - mirror["synapses"]) < 1.0
          and target["edges"] == mirror["edges"])
    # А `neurons` в этих двух записях совпадать не обязано, и это не ошибка:
    # слева считаются разные нейроны области-партнёра, справа — разные
    # нейроны AL. Печатаем, чтобы разница была видна и не пугала.
    print(f"    (нейронов-партнёров: {target['neurons']} со стороны {region},"
          f" {mirror['neurons']} со стороны AL — это разные множества)")

    if torch.cuda.is_available():
        head("5. GPU против процессора на настоящей выборке")
        idx = sets["region:AL"]
        a = L.summarize(index, df, idx, "in", "region", limit=None, device="cpu")
        b = L.summarize(index, df, idx, "in", "region", limit=None, device="cuda")
        same_rows = all(
            x["partner"] == y["partner"] and x["neurons"] == y["neurons"]
            and x["edges"] == y["edges"]
            and abs(x["synapses"] - y["synapses"]) < 1.0
            and abs(x["exc"] - y["exc"]) < 1.0
            for x, y in zip(a["rows"], b["rows"]))
        check(f"region:AL / in: {len(a['rows'])} групп совпали на обоих"
              f" устройствах ({a['total']['elapsed_ms']:.1f} мс на процессоре,"
              f" {b['total']['elapsed_ms']:.1f} мс на GPU)",
              same_rows and len(a["rows"]) == len(b["rows"]))

    return sets


# ------------------------------------------------------- матрица нейропилей


def test_matrix(index, df):
    head("6. Матрица нейропиль -> нейропиль")
    if L.MATRIX_CACHE.exists():
        L.MATRIX_CACHE.unlink()
    t0 = time.perf_counter()
    m1 = L.region_matrix(index, df)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    m2 = L.region_matrix(index, df)
    warm = time.perf_counter() - t0

    regions = m1["regions"]
    syn = np.array(m1["synapses"])
    edges = np.array(m1["edges"])
    check(f"размер матрицы {syn.shape[0]}x{syn.shape[1]}"
          f" ({len(regions)} областей)", syn.shape == (len(regions),) * 2)
    check(f"первый счёт {cold:.2f} с, из кэша {warm * 1000:.1f} мс",
          m2.get("from_cache") is True and warm < cold)
    total_syn = float(np.abs(index.in_value).sum(dtype=np.float64))
    check(f"сумма матрицы = все синапсы коннектома"
          f" ({syn.sum():.0f} против {total_syn:.0f})",
          abs(syn.sum() - total_syn) < 1.0)
    check(f"сумма рёбер матрицы = nnz ({int(edges.sum())} против {index.nnz})",
          int(edges.sum()) == index.nnz)
    check("возб + торм == синапсы",
          abs(np.array(m1["exc"]).sum() + np.array(m1["inh"]).sum()
              - syn.sum()) / total_syn < 1e-6)
    miss = regions.index(L.MISSING)
    check("корзина «нет в таблице» пуста",
          int(edges[miss].sum()) == 0 and int(edges[:, miss].sum()) == 0)

    # Сверка одной строки матрицы с summarize: исходящие из AL по областям.
    al = D.select(df, "region:AL")
    out = L.summarize(index, df, al, "out", "region", limit=None, device="cpu")
    i_al = regions.index("AL")
    ok = True
    for r in out["rows"]:
        j = regions.index(r["partner"])
        if abs(syn[i_al, j] - r["synapses"]) > 1.0 or edges[i_al, j] != r["edges"]:
            ok = False
    check("строка AL матрицы совпала с summarize по всем областям", ok)

    print("\n  куда идёт AL (верхние шесть областей):")
    order = np.argsort(-syn[i_al])[:6]
    for j in order:
        print(f"    AL -> {regions[j]:<6} {syn[i_al, j]:>12.0f} синапсов,"
              f" {edges[i_al, j]:>9} рёбер")
    return cold, warm


# ------------------------------------------------------------------ замеры


def test_speed(index, df, sets):
    head("7. Замеры времени")
    rng = np.random.default_rng(7)
    cases = [("1", np.array([int(sets['type:LC10'][0])]))]
    for n in (100, 5_000, 50_000):
        cases.append((str(n), rng.choice(index.n, n, replace=False)
                      .astype(np.int64)))
    print("  {:<8} {:<5} {:>12} {:>12} {:>12} {:>10}".format(
        "нейронов", "напр", "рёбер", "ЦП, мс", "GPU, мс", "выбор"))
    have = torch.cuda.is_available()
    for name, idx in cases:
        for direction in ("in", "out"):
            _, edges = L.pick_device(index, np.unique(idx), direction, "cpu")
            ms_cpu = []
            for _ in range(3):
                r = L.summarize(index, df, idx, direction, "region",
                                limit=30, device="cpu")
                ms_cpu.append(r["total"]["elapsed_ms"])
            ms_gpu = []
            if have:
                for _ in range(3):
                    r = L.summarize(index, df, idx, direction, "region",
                                    limit=30, device="cuda")
                    ms_gpu.append(r["total"]["elapsed_ms"])
            auto = L.summarize(index, df, idx, direction, "region",
                               device="auto")["total"]["device"]
            print("  {:<8} {:<5} {:>12} {:>12.1f} {:>12} {:>10}".format(
                name, direction, edges, min(ms_cpu),
                f"{min(ms_gpu):.1f}" if ms_gpu else "—", auto))

    # Худший случай: вся таблица сразу — столько же работы, сколько полный
    # проход по матрице, и ответ всё равно обязан прийти.
    everyone = np.arange(index.n, dtype=np.int64)
    t0 = time.perf_counter()
    r = L.summarize(index, df, everyone, "in", "region", limit=None,
                    device="cpu")
    cpu_all = time.perf_counter() - t0
    gpu_all = None
    if have:
        L.summarize(index, df, everyone[:10], "in", "region", device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r2 = L.summarize(index, df, everyone, "in", "region", limit=None,
                         device="cuda")
        torch.cuda.synchronize()
        gpu_all = time.perf_counter() - t0
        check("все 138 639 нейронов: GPU и процессор дали один итог",
              abs(r["total"]["synapses"] - r2["total"]["synapses"])
              / r["total"]["synapses"] < 1e-6)
    print(f"\n  все {index.n} нейронов, in: {r['total']['edges']} рёбер,"
          f" ЦП {cpu_all:.2f} с"
          + (f", GPU {gpu_all:.3f} с" if gpu_all else ""))
    return cpu_all, gpu_all


# -------------------------------------------------------------------- точка


def main():
    import psutil

    proc = psutil.Process()
    test_tiny()

    head("Загрузка настоящего коннектома")
    t0 = time.perf_counter()
    df = D.load_neurons()
    t_neurons = time.perf_counter() - t0
    t0 = time.perf_counter()
    w = D.load_weights("cpu")
    t_weights = time.perf_counter() - t0
    rss0 = proc.memory_info().rss
    print(f"  таблица нейронов: {len(df)} строк за {t_neurons:.2f} с")
    print(f"  матрица: {tuple(w.shape)}, ненулевых {w._nnz()},"
          f" за {t_weights:.2f} с")

    t0 = time.perf_counter()
    index = L.build_index(w)
    t_index = time.perf_counter() - t0
    rss1 = proc.memory_info().rss
    print(f"  build_index: {t_index:.2f} с (внутренний замер"
          f" {index.build_s:.2f} с)")
    print(f"  массивы индекса: {index.nbytes / 2**20:.1f} МБ;"
          f" прирост RSS процесса {(rss1 - rss0) / 2**20:.1f} МБ")
    check("индекс укладывается в бюджет 400 МБ сверх матрицы",
          index.nbytes < 400 * 2**20,
          f"{index.nbytes / 2**20:.1f} МБ")
    check("указатели согласованы с числом рёбер",
          int(index.post_ptr[-1]) == index.nnz
          and int(index.pre_ptr[-1]) == index.nnz)

    sets = test_real(index, df)
    cold, warm = test_matrix(index, df)
    cpu_all, gpu_all = test_speed(index, df, sets)

    if torch.cuda.is_available():
        print(f"\n  видеопамять под индекс:"
              f" {torch.cuda.memory_allocated() / 2**20:.1f} МБ выделено,"
              f" {torch.cuda.max_memory_allocated() / 2**20:.1f} МБ пик")

    head("Сводка")
    print(f"  build_index              {t_index:.2f} с,"
          f" {index.nbytes / 2**20:.0f} МБ")
    print(f"  region_matrix первый раз {cold:.2f} с")
    print(f"  region_matrix из кэша    {warm * 1000:.1f} мс")
    print(f"  summarize на всей таблице {cpu_all:.2f} с на ЦП"
          + (f", {gpu_all:.3f} с на GPU" if gpu_all else ""))
    print(f"\n  провалов: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
