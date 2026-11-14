r"""Две пары, которых обе зовут `DNa01`: кто их питает и куда они выходят.

Спор об адресации нельзя решить именем — это уже записано в
`BANC_MOTOR.md` (поправка к §1): `cell_type` и `hemibrain_type`
называют `DNa01` РАЗНЫЕ клетки, а BANC этот спор независимо не решает,
потому что его ярлык получен переносом из FAFB через тот же матч.

Но у вопроса есть сторона, которую можно решить НАШИМИ данными:
кто в нашей же модели подаёт на каждую пару и куда она выходит. Это
важно практически, а не терминологически. Наш зонд `DNa01` молчит во
всех замеренных сценариях (`MOTOR.md` §1 и §7: 0,00 Гц на покое, на
надвигании и под обонятельной накачкой). Если у второй пары вход
такой же пустой, переадресация ничего не изменит в поведении и остаётся
чисто анатомическим решением. Если нет — это уже разница, которую
видно в прогоне.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\dna01_who.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from flybrain import data as D, links  # noqa: E402

# Номера из `BANC_MOTOR.md` §4. Пары не пересекаются ни одним нейроном.
ПАРЫ = {
    "наш зонд  (hemibrain=DNa01, cell_type=DNae001, GNG)":
        [720575940618167579, 720575940626328707],
    "выбор BANC (cell_type=DNa01, hemibrain=VES006, VES)":
        [720575940627787609, 720575940644438551],
    "для сравнения: DNa02 (оба столбца согласны)": None,
}


def индексы(neurons, ids):
    root = neurons.root_id.to_numpy()
    if root.dtype.kind in "OU":
        ids = [str(i) for i in ids]
    есть = np.isin(root, np.asarray(ids, dtype=root.dtype))
    return np.nonzero(есть)[0]


def main():
    neurons = D.load_neurons()
    weights = D.load_weights(device="cpu")
    index = links.build_index(weights)
    ПАРЫ["для сравнения: DNa02 (оба столбца согласны)"] = None

    alpn = set(np.asarray(D.select(neurons, "class:ALPN"), dtype=np.int64))
    desc = set(np.asarray(D.select(neurons, "super:descending"),
                          dtype=np.int64))

    for имя, ids in ПАРЫ.items():
        idx = (np.asarray(D.select(neurons, "celltype=DNa02"), dtype=np.int64)
               if ids is None else индексы(neurons, ids))
        if len(idx) == 0:
            print("%s: не найдено" % имя)
            continue
        вх = links.summarize(index, neurons, idx, direction="in", by="class")
        вых = links.summarize(index, neurons, idx, direction="out", by="class")
        всего_вх = float(вх["total"]["synapses"])
        всего_вых = float(вых["total"]["synapses"])
        # Прямой вход от обонятельного пути, которым мы и качаем модель.
        прямо = 0.0
        for i in idx:
            нач, кон = index.post_ptr[i], index.post_ptr[i + 1]
            партнёры = index.in_partner[нач:кон]
            значения = index.in_value[нач:кон]
            маска = np.fromiter((int(p) in alpn for p in партнёры),
                                dtype=bool, count=len(партнёры))
            прямо += float(значения[маска].sum())
        на_нисходящих = 0.0
        for i in idx:
            нач, кон = index.pre_ptr[i], index.pre_ptr[i + 1]
            партнёры = index.out_partner[нач:кон]
            значения = index.out_value[нач:кон]
            маска = np.fromiter((int(p) in desc for p in партнёры),
                                dtype=bool, count=len(партнёры))
            на_нисходящих += float(значения[маска].sum())
        print("\n%s" % имя)
        print("  нейронов %d, вход %.0f, выход %.0f" % (len(idx), всего_вх,
                                                        всего_вых))
        print("  вход прямо от class:ALPN: %.0f (%.2f %% входа)"
              % (прямо, 100.0 * прямо / max(всего_вх, 1.0)))
        print("  выход на super:descending: %.0f (%.2f %% выхода)"
              % (на_нисходящих, 100.0 * на_нисходящих / max(всего_вых, 1.0)))
        print("  главные источники: %s"
              % ", ".join("%s %.0f" % (r["partner"], r["synapses"])
                          for r in вх["rows"][:5]))
        print("  главные мишени:   %s"
              % ", ".join("%s %.0f" % (r["partner"], r["synapses"])
                          for r in вых["rows"][:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
