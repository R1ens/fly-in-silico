#!/bin/sh
# Третья очередь: пересъём двух шлюзов, которые в первой очереди
# отказали НЕ по вине кода.
#
# * resume_state --part body отказал потому, что проба сама запрещает
#   объявлять горизонт больше проверенного: объявлено 150 шагов, а
#   запущена она была с --tail 25 по умолчанию. Это защита пробы от
#   недопроверки, и её надо уважать, а не обходить: гоним с --tail 150,
#   как и объявлено в V2.md;
# * integrity отказал на «нашёлся прогон для отчёта — нет»: каталог
#   прогонов у стенда свой (FLYBRAIN_OUT), и в нём не было прогона в
#   теле. cli_check его там создаёт, поэтому integrity гоняется ПОСЛЕ
#   него.
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
ROOT=/d/fly-brain/.claude/worktrees/agent-a87c8ae8bec18c1d0
export FLYBRAIN_OUT="C:/Users/Graveir/AppData/Local/Temp/claude/D--fly-brain/84cf3d1c-bcf5-48f0-854e-f5753269cea5/scratchpad/bench-out"
PY=/d/fly-brain/.venv/Scripts/python.exe
LOG="$ROOT/lab/bench/gates3.log"
CODES="$ROOT/lab/bench/gates_codes.txt"
cd "$ROOT/lab" || exit 1
echo "=== третья очередь, старт $(date -Is) ===" > "$LOG"

run() {
  name="$1"; shift
  echo "--- $name $(date -Is) ---" >> "$LOG"
  "$PY" "$@" >> "$LOG" 2>&1
  code=$?
  echo "КОД ВОЗВРАТА $name = $code" >> "$LOG"
  echo "ИТОГ|$name|$code" >> "$CODES"
  echo "ШЛЮЗ $name код $code"
}

run integrity_2          probe/integrity.py
run resume_state_body150 probe/resume_state.py --part body --tail 150
echo "=== третья очередь, конец $(date -Is) ===" >> "$LOG"
echo "ТРЕТЬЯ ОЧЕРЕДЬ ПРОШЛА"
