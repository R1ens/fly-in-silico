"""Увести прогоны проб во временный каталог.

Пробы считают настоящие опыты и настоящие же папки за собой оставляют.
Складывать их в `out/` рядом с научными результатами — значит каждый раз
потом разбирать, где чьё. Этот модуль подменяет каталог вывода до того,
как `flybrain.data` впервые прочитан, и печатает, куда именно.

    import probe._isolate   # первой строкой, до любого flybrain
"""

import os
import sys
import tempfile
from pathlib import Path

__all__ = ["OUT"]

if "flybrain.data" in sys.modules:
    raise RuntimeError("_isolate импортируется до flybrain.data, иначе "
                       "каталог уже выбран")

OUT = Path(os.environ.get("FLYBRAIN_OUT")
           or (Path(tempfile.gettempdir()) / "flybrain-probe-out"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["FLYBRAIN_OUT"] = str(OUT)
