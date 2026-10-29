#!/bin/sh
# Продолжение основного прогона: шаги 2-5. Отдельным файлом, потому что
# процесс первого шага повис на выходе (нить-кормилица очереди IPC,
# см. комментарий в bench_seven.py) и держал очередь внутри final.sh.
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out"
PY=/d/fly-brain/.venv/Scripts/python.exe
B="$ROOT/lab/bench"
cd "$ROOT/lab" || exit 1

echo "### 2/5 пары на движке LIF"
"$PY" probe/bench_pairs.py --what lif --пар 6 --tag базовая \
      > "$B/final_pairs_lif.log" 2>&1
echo "код $? (пары lif)"

echo "### 3/5 сверка с замороженным эталоном и два дерева кода"
sh "$B/gates2.sh" > "$B/final_gates2.log" 2>&1
echo "код $? (gates2)"

echo "### 4/5 пары на полном контуре: порог доверия"
"$PY" probe/bench_pairs.py --what loop --пар 4 --steps 12 --tag базовая \
      > "$B/final_pairs_loop.log" 2>&1
echo "код $? (пары loop)"

echo "### 5/5 долгий сеанс песочницы"
"$PY" probe/bench_seven.py --long --минут 8 --окно 10 --tag базовая \
      > "$B/final_long.log" 2>&1
echo "код $? (долгий сеанс)"

echo "### 6/6 пересъём двух шлюзов"
sh "$B/gates3.sh" > "$B/final_gates3.log" 2>&1
echo "код $? (gates3)"

echo "ПРОДОЛЖЕНИЕ ПРОГОНА СТЕНДА ЗАКОНЧЕНО"
