#!/bin/sh
# Основной прогон стенда. Строго последовательно: два замера
# одновременно на этой машине мерили бы друг друга.
#   sh lab/bench/final.sh
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out"
PY=/d/fly-brain/.venv/Scripts/python.exe
B="$ROOT/lab/bench"
cd "$ROOT/lab" || exit 1

echo "### 1/5 шесть уровней по 50 шагов"
"$PY" probe/bench_seven.py --levels phys gait lif closed vision live \
      --steps 50 --tag базовая > "$B/final_levels.log" 2>&1
echo "код $? (уровни 1-6)"

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

echo "ОСНОВНОЙ ПРОГОН СТЕНДА ЗАКОНЧЕН"
