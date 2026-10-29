#!/bin/sh
# Сверка кандидата с замороженным эталоном. Дерево кандидата развёрнуто
# из его ветки (`git archive`) рядом, пакет `flybrain` берётся из него.
# Каталог прогонов у каждого кандидата свой, чтобы прогоны не мешались.
#   sh lab/bench/cand.sh lab_gait кандидат-походка
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
TREE="$1"
TAG="$2"
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out-$TAG"
PY=/d/fly-brain/.venv/Scripts/python.exe
cd "$ROOT/$TREE" || exit 1
cp "$ROOT/lab/probe/bench_v2_удержание.json" probe/ 2>/dev/null
"$PY" probe/bench_v2.py --tag "$TAG" --only lif body
echo "КОД bench_v2 = $?"
"$PY" probe/bench_cmp.py удержание "$TAG"
echo "КОД bench_cmp = $?"
