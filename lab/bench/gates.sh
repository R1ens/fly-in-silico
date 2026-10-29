#!/bin/sh
# Прогон обязательных шлюзов на базовой линии: каждый пишет свой код
# возврата в gates_codes.txt, вывод — в gates.log.
#   sh lab/bench/gates.sh
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out"
PY=/d/fly-brain/.venv/Scripts/python.exe
LOG="$ROOT/lab/bench/gates.log"
CODES="$ROOT/lab/bench/gates_codes.txt"
cd "$ROOT/lab" || exit 1
echo "=== шлюзы, старт $(date -Is) ===" > "$LOG"

run() {
  name="$1"; shift
  echo "--- $name $(date -Is) ---" >> "$LOG"
  "$PY" "$@" >> "$LOG" 2>&1
  code=$?
  echo "КОД ВОЗВРАТА $name = $code" >> "$LOG"
  echo "ИТОГ|$name|$code" >> "$CODES"
  echo "ШЛЮЗ $name код $code"
}

run five_spaces        probe/five_spaces.py
run lif_equal          probe/lif_equal.py --against эталон
run sel_dump           probe/sel_dump.py
run timeline_trap      probe/timeline_trap.py
run replay_modes       probe/replay_modes.py
run resume_state_arena probe/resume_state.py --part arena
run resume_state_body  probe/resume_state.py --part body
run integrity          probe/integrity.py
run cli_check          probe/cli_check.py
run compat_runs        probe/compat_runs.py --base ccd321c --out D:/fly-brain/out
run body_equal         probe/body_equal.py
run exp_shot           probe/exp_shot.py
run brain_shot         probe/brain_shot.py
run sandbox_shot       probe/sandbox_shot.py
run ana_shot           probe/ana_shot.py
echo "=== шлюзы, конец $(date -Is) ===" >> "$LOG"
echo "ВСЕ ШЛЮЗЫ ПРОШЛИ ПРОГОН"
