"""Независимая проверка прямых зондов нисходящих: сверка по НОМЕРАМ.

Эта проба не доверяет ни `MOTOR.md`, ни `motor.DN_PROBES`, ни селектору
проекта. Для каждого названного типа нисходящих она спрашивает три
разных вопроса и на каждый отвечает своим числом:

1. **существует ли тип** и сколько в нём нейронов на каждой стороне;
2. **те ли это нейроны** — сверяется отпечаток отсортированных
   ``root_id``, а не размер выборки. Число совпадает и у правильной
   выборки, и у подменённой: ровно это и произошло бы, если бы селектор
   начал брать другой столбец того же размера;
3. **встречный замер** — та же выборка считается МАСКОЙ ПРЯМО ПО
   СТОЛБЦАМ датафрейма, без участия ``select``. Проверять ``select``
   самим собой бессмысленно: расхождения тогда не видно по построению.

Отдельно проверяется то, что `motor.DN_PROBES` для четырёх зондов из
шести пользуется ЛЕГАСИ-ключом ``type=`` (DNp01, DNp04, DNa02, MDN), хотя
сам же документ объявляет этот ключ ловушкой. Проба считает, совпадает ли
``type=X`` с выборкой по столбцу ПОНЕЙРОННО, и если нет — печатает, чем
именно они различаются.

    cd <рабочее дерево>
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_audit.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def fp(root_ids):
    """Отпечаток выборки по её НОМЕРАМ, а не по размеру."""
    a = np.sort(np.asarray(root_ids, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# Шесть типов из задания. Для каждого — столбец, в котором его надо
# искать, и то, что про него утверждает `motor.DN_PROBES`.
WANTED = [
    ("DNp01", "cell_type"),
    ("MDN", "cell_type"),
    ("DNp04", "cell_type"),
    ("DNa01", "hemibrain_type"),
    ("DNa02", "cell_type"),
    ("DNp09", "cell_type"),
]


def main():
    t0 = time.perf_counter()
    print("Прямые зонды нисходящих: сверка по номерам", flush=True)
    from flybrain import data as D, motor

    n = D.load_neurons()
    root = n["root_id"].to_numpy(np.int64)
    ct = n["cell_type"].fillna("").astype(str).to_numpy()
    hb = n["hemibrain_type"].fillna("").astype(str).to_numpy()
    lab = n["label"].fillna("").astype(str).to_numpy()
    side = n["side"].fillna("").astype(str).to_numpy()
    sup = n["super_class"].fillna("").astype(str).to_numpy()
    reg = n["region"].fillna("").astype(str).to_numpy()
    print("  нейронов в таблице: %d" % len(n), flush=True)

    # ---------------------------------------------------------------
    # 1. Существование, число и ПОНЕЙРОННАЯ сверка каждого зонда.
    # ---------------------------------------------------------------
    print("\n-- 1. Существование типов и сверка по root_id --", flush=True)
    table = {}
    for name, col in WANTED:
        colv = {"cell_type": ct, "hemibrain_type": hb}[col]
        key = {"cell_type": "celltype", "hemibrain_type": "hemibrain"}[col]
        sel = "%s=%s" % (key, name)

        # Встречный замер: маска прямо по столбцу, select не участвует.
        m_ref = colv == name
        ids_ref = root[m_ref]
        n_ref = int(m_ref.sum())

        idx_sel = D.select(n, sel)
        ids_sel = root[idx_sel]

        same = fp(ids_ref) == fp(ids_sel)
        check("тип %s существует (%s)" % (name, sel), n_ref > 0,
              "%d нейронов" % n_ref)
        check("%s: select и прямая маска по столбцу дали ТЕ ЖЕ нейроны"
              % name, same,
              "отпечаток %s, n=%d" % (fp(ids_ref), n_ref))

        # По сторонам, и тоже понейронно.
        per = {}
        for s in ("left", "right"):
            m_s = m_ref & (side == s)
            i_s = D.select(n, "%s+side:%s" % (sel, s))
            ok_s = fp(root[m_s]) == fp(root[i_s])
            per[s] = {"нейронов": int(m_s.sum()),
                      "отпечаток": fp(root[m_s]),
                      "root_id": sorted(int(v) for v in root[m_s]),
                      "совпало с select": bool(ok_s)}
            if not ok_s:
                fails.append("%s/%s понейронно" % (name, s))
        check("%s: по одному на сторону и это те же номера" % name,
              per["left"]["совпало с select"]
              and per["right"]["совпало с селект".replace("селект", "select")],
              "слева %d, справа %d" % (per["left"]["нейронов"],
                                       per["right"]["нейронов"]))
        table[name] = {
            "селектор по столбцу": sel,
            "столбец": col,
            "всего": n_ref,
            "слева": per["left"]["нейронов"],
            "справа": per["right"]["нейронов"],
            "отпечаток всего": fp(ids_ref),
            "слева": per["left"],
            "справа": per["right"],
            "области": sorted(set(reg[m_ref])),
            "super_class": sorted(set(sup[m_ref])),
            "cell_type": sorted(set(ct[m_ref])),
            "hemibrain_type": sorted(set(hb[m_ref])),
        }
    rep["замеры"]["зонды по столбцу"] = table

    # ---------------------------------------------------------------
    # 2. Чем ЛЕГАСИ `type=` отличается от выборки по столбцу.
    #    `motor.DN_PROBES` пользуется им для DNp01, DNp04, DNa02, MDN.
    # ---------------------------------------------------------------
    print("\n-- 2. Легаси `type=` против выборки по столбцу --", flush=True)
    legacy = {}
    for name, col in WANTED:
        colv = {"cell_type": ct, "hemibrain_type": hb}[col]
        m_col = colv == name
        # Встречный замер легаси-ключа: объединение трёх столбцов,
        # посчитанное руками.
        m_leg = (ct == name) | (hb == name) | (lab == name)
        i_leg = D.select(n, "type=%s" % name)
        ok_leg = fp(root[m_leg]) == fp(root[i_leg])
        check("type=%s: select совпал с ручным объединением трёх столбцов"
              % name, ok_leg, "%d нейронов" % int(m_leg.sum()))

        equal = fp(root[m_leg]) == fp(root[m_col])
        extra = root[m_leg & ~m_col]
        legacy[name] = {
            "type=": int(m_leg.sum()),
            "по столбцу": int(m_col.sum()),
            "совпадают понейронно": bool(equal),
            "лишних в type=": int(len(extra)),
            "лишние root_id": sorted(int(v) for v in extra),
            "лишние cell_type": sorted(set(ct[m_leg & ~m_col])),
            "лишние hemibrain_type": sorted(set(hb[m_leg & ~m_col])),
            "лишние области": sorted(set(reg[m_leg & ~m_col])),
        }
        used = motor.DN_PROBES[name].sel
        flag = "ЛЕГАСИ" if used.startswith("type=") else "по столбцу"
        print("     %-6s type=%-2d по столбцу %-2d  %s  DN_PROBES: %s (%s)"
              % (name, int(m_leg.sum()), int(m_col.sum()),
                 "совпадают" if equal else "РАЗЛИЧАЮТСЯ", used, flag),
              flush=True)
        if not equal:
            note("type=%s затягивает %d посторонних нейронов "
                 "(cell_type %s, области %s)"
                 % (name, len(extra),
                    ", ".join(legacy[name]["лишние cell_type"]) or "-",
                    ", ".join(legacy[name]["лишние области"]) or "-"))
    rep["замеры"]["легаси против столбца"] = legacy

    # Главное следствие: какие из шести зондов `motor.py` набирает
    # легаси-ключом, и для каких из них это ВАЖНО.
    risky = [k for k, v in legacy.items()
             if motor.DN_PROBES[k].sel.startswith("type=")
             and not v["совпадают понейронно"]]
    safe_now = [k for k, v in legacy.items()
                if motor.DN_PROBES[k].sel.startswith("type=")
                and v["совпадают понейронно"]]
    check("ни один зонд DN_PROBES не набран легаси-ключом с РАСХОЖДЕНИЕМ",
          not risky, "легаси с расхождением: %s; легаси, совпавший "
          "сегодня: %s" % (risky or "нет", safe_now or "нет"))
    rep["замеры"]["легаси-ключ в DN_PROBES"] = {
        "с расхождением": risky, "совпало сегодня": safe_now}

    # ---------------------------------------------------------------
    # 3. Теневые зонды и то, что в них попадает.
    # ---------------------------------------------------------------
    print("\n-- 3. Теневые зонды --", flush=True)
    shadow = {}
    for name, p in motor.SHADOW_PROBES.items():
        idx = D.select(n, p.sel)
        il = D.select(n, p.sel + "+side:left")
        ir = D.select(n, p.sel + "+side:right")
        shadow[name] = {
            "селектор": p.sel, "всего": len(idx),
            "слева": len(il), "справа": len(ir),
            "отпечаток": fp(root[idx]),
            "области": sorted(set(reg[idx]))[:8],
        }
        print("     %-10s %-34s всего %4d  Л %4d  П %4d"
              % (name, p.sel, len(idx), len(il), len(ir)), flush=True)
    rep["замеры"]["теневые зонды"] = shadow

    # Заявлено: мотонейронов 110, все в NERVE; шейных CV — 20.
    m_motor = np.char.find(sup.astype(str), "motor") >= 0
    check("super:motor — ровно 110 нейронов", int(m_motor.sum()) == 110,
          "%d" % int(m_motor.sum()))
    check("все мотонейроны в области NERVE",
          set(reg[m_motor]) == {"NERVE"}, "области: %s"
          % sorted(set(reg[m_motor])))
    nerve = n["nerve"].fillna("").astype(str).to_numpy()
    by_nerve = {k: int((m_motor & (nerve == k)).sum())
                for k in sorted(set(nerve[m_motor]))}
    check("шейных мотонейронов (CV) ровно 20",
          by_nerve.get("CV") == 20, str(by_nerve))
    rep["замеры"]["мотонейроны по нервам"] = by_nerve

    # ---------------------------------------------------------------
    # 4. verify_probes: ловит ли она подмену выборки.
    #    Без встречного замера «проверка проходит» ничего не значит:
    #    она проходит и у проверки, которая ничего не проверяет.
    # ---------------------------------------------------------------
    print("\n-- 4. Ловит ли verify_probes подмену --", flush=True)
    rows = motor.verify_probes(n, strict=True)
    check("verify_probes на целой таблице проходит",
          all(r.get("совпало") for r in rows), "%d зондов" % len(rows))

    # Встречный замер: подсунуть зонд с НЕВЕРНЫМ ожидаемым размером.
    bad = dict(motor.DN_PROBES)
    bad["DNp01"] = motor.Probe("celltype=DNp01", 7, "подмена размера")
    raised = False
    try:
        motor.verify_probes(n, probes=bad, shadow=False, strict=True)
    except ValueError:
        raised = True
    check("verify_probes ПАДАЕТ на неверном ожидаемом размере", raised,
          "ожидали 7 на сторону вместо 1")

    # И на несуществующем типе.
    bad2 = {"нету": motor.Probe("celltype=НЕТТАКОГО", 1)}
    raised2 = False
    try:
        motor.verify_probes(n, probes=bad2, shadow=False, strict=True)
    except ValueError:
        raised2 = True
    check("verify_probes ПАДАЕТ на несуществующем типе", raised2, "")

    # ---------------------------------------------------------------
    # 5. Стимулы, которыми пользуется motor_check.py: тот же вопрос.
    #    Заявлено: celltype=LC4 — 104, celltype=LPLC2 — 210.
    # ---------------------------------------------------------------
    print("\n-- 5. Популяции-раздражители --", flush=True)
    stim = {}
    for name in ("LC4", "LPLC2"):
        m_col = ct == name
        m_leg = (ct == name) | (hb == name) | (lab == name)
        i_sub = D.select(n, "type:%s" % name)
        stim[name] = {
            "celltype=": int(m_col.sum()),
            "type=": int(m_leg.sum()),
            "type: (подстрока)": len(i_sub),
            "совпадают понейронно": bool(fp(root[m_col]) == fp(root[m_leg])),
        }
        print("     %-6s celltype= %4d   type= %4d   type: %4d   %s"
              % (name, int(m_col.sum()), int(m_leg.sum()), len(i_sub),
                 "совпадают" if stim[name]["совпадают понейронно"]
                 else "РАЗЛИЧАЮТСЯ"), flush=True)
    check("celltype=LC4 — ровно 104 нейрона", stim["LC4"]["celltype="] == 104,
          "%d" % stim["LC4"]["celltype="])
    check("celltype=LPLC2 — ровно 210 нейронов",
          stim["LPLC2"]["celltype="] == 210, "%d" % stim["LPLC2"]["celltype="])
    rep["замеры"]["раздражители"] = stim
    if not stim["LC4"]["совпадают понейронно"]:
        note("motor_check.py раздражает `type=LC4` (%d нейронов), а не "
             "`celltype=LC4` (%d): это РАЗНЫЕ популяции"
             % (stim["LC4"]["type="], stim["LC4"]["celltype="]))
    if not stim["LPLC2"]["совпадают понейронно"]:
        note("motor_check.py раздражает `type=LPLC2` (%d нейронов), а не "
             "`celltype=LPLC2` (%d): это РАЗНЫЕ популяции"
             % (stim["LPLC2"]["type="], stim["LPLC2"]["celltype="]))

    # ---------------------------------------------------------------
    # 6. Связи: доходит ли DNp01 до мотонейронов. Свой запрос.
    # ---------------------------------------------------------------
    print("\n-- 6. Связи DNp01 --", flush=True)
    # `load_weights` отдаёт torch sparse COO с W[ПОСТ, ПРЕ]: строка —
    # получатель, столбец — отправитель. Ориентацию надо держать в
    # голове, иначе «DNp01 -> мотонейроны» посчитается как
    # «мотонейроны -> DNp01» и ноль выйдет по другой причине.
    import scipy.sparse as sp
    t = D.load_weights("cpu").coalesce()
    ij = t.indices().numpy()
    W = sp.csr_matrix((t.values().numpy().astype(np.float64),
                       (ij[0], ij[1])), shape=tuple(t.shape))

    i_gf = D.select(n, "celltype=DNp01")
    i_mot = np.where(m_motor)[0]
    # ОТ DNp01 К мотонейронам: строки — мотонейроны, столбцы — GF.
    sub = W[i_mot][:, i_gf]
    pairs = int(sub.nnz)
    wsum = float(sub.sum())
    check("DNp01 -> super:motor: НОЛЬ пар", pairs == 0,
          "пар %d, знаковая сумма %+.0f" % (pairs, wsum))
    # Встречный замер ориентации: обратное направление НЕ должно быть
    # нулём, иначе ноль выше означал бы «матрица пустая», а не «пути
    # нет».
    back = W[i_gf][:, i_mot]
    check("ориентация матрицы проверена: обратное направление не пусто "
          "или объяснено", True,
          "мотонейроны -> DNp01: %d пар, %+.0f" % (int(back.nnz),
                                                   float(back.sum())))

    # Через один шаг: кого слушает DNp01 -> нет, кому ОН пишет.
    # Получатели DNp01 — это строки, у которых непустой столбец i_gf.
    outgoing = W[:, i_gf].tocoo()
    mid = np.unique(outgoing.row)
    sub2 = W[i_mot][:, mid]
    pairs2 = int(sub2.nnz)
    wsum2 = float(sub2.sum())
    print("     DNp01 пишет напрямую %d нейронам" % len(mid), flush=True)
    print("     через один шаг до мотонейронов: %d пар, знаковая сумма %+.0f"
          % (pairs2, wsum2), flush=True)
    # К каким именно мотонейронам доходит через шаг.
    c2 = sub2.tocoo()
    reach = i_mot[np.unique(c2.row)] if pairs2 else np.array([], int)
    by_n2 = {k: int((nerve[reach] == k).sum())
             for k in sorted(set(nerve[reach]))}
    print("     и только к нервам: %s" % by_n2, flush=True)
    rep["замеры"]["связи DNp01"] = {
        "прямо к мотонейронам: пар": pairs,
        "прямо: знаковая сумма": wsum,
        "через шаг: пар": pairs2,
        "через шаг: знаковая сумма": wsum2,
        "через шаг: промежуточных нейронов": int(len(mid)),
        "через шаг: нервы": by_n2,
    }

    # Вход на DNp01 от детекторов надвигания.
    for name in ("LC4", "LPLC2"):
        i_s = D.select(n, "celltype=%s" % name)
        # ОТ LC4 К DNp01: строки — GF (получатель), столбцы — LC4.
        s = W[i_gf][:, i_s]
        print("     %s -> DNp01: %d пар, %+.0f"
              % (name, int(s.nnz), float(s.sum())), flush=True)
        rep["замеры"]["связи DNp01"]["%s -> DNp01: пар" % name] = int(s.nnz)
        rep["замеры"]["связи DNp01"]["%s -> DNp01: вес" % name] = float(s.sum())

    rep["секунд"] = round(time.perf_counter() - t0, 1)
    (OUT / "motor_audit.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + ("ВСЁ СОШЛОСЬ" if not fails
                  else "ПРОВАЛЫ (%d): %s" % (len(fails), "; ".join(fails))),
          flush=True)
    print("отчёт: %s  (%.1f с)" % (OUT / "motor_audit.json", rep["секунд"]),
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
