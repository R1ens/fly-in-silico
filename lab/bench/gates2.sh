#!/bin/sh
# Вторая очередь шлюзов: сверка с замороженным эталоном и побитовость
# прежнего моторного пути на двух деревьях кода.
#   sh lab/bench/gates2.sh
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out"
PY=/d/fly-brain/.venv/Scripts/python.exe
LOG="$ROOT/lab/bench/gates2.log"
CODES="$ROOT/lab/bench/gates_codes.txt"
cd "$ROOT/lab" || exit 1
echo "=== вторая очередь шлюзов, старт $(date -Is) ===" > "$LOG"

run() {
  name="$1"; shift
  echo "--- $name $(date -Is) ---" >> "$LOG"
  "$PY" "$@" >> "$LOG" 2>&1
  code=$?
  echo "КОД ВОЗВРАТА $name = $code" >> "$LOG"
  echo "ИТОГ|$name|$code" >> "$CODES"
  echo "ШЛЮЗ $name код $code"
}

# Сверка выходов с замороженным эталоном: пять сценариев, три уровня
# равнозначности (bench_cmp).
run bench_v2_стенд probe/bench_v2.py --tag стенд
run bench_cmp      probe/bench_cmp.py эталон стенд
# Побитовость прежнего пути на двух деревьях кода: `lab_base` развёрнут
# из коммита-основания ccd321c, `lab` — эта ветка.
run motor_equal_base probe/motor_equal.py base
run motor_equal_now  probe/motor_equal.py now
run motor_equal_cmp  probe/motor_equal.py cmp
echo "=== вторая очередь шлюзов, конец $(date -Is) ===" >> "$LOG"
echo "ВТОРАЯ ОЧЕРЕДЬ ПРОШЛА"
