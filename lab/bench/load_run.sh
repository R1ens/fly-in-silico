#!/usr/bin/env bash
# Ограниченная нагрузочная проверка повторной сборки движка.
#
# Зачем отдельным файлом, а не строчкой. Единичный отказ
# `CUDA illegal memory access` случился на третьей сборке движка в одном
# процессе под ЗАНЯТОЙ картой (`flybrain/KERNEL_FAULT.md`). Проверка на
# свободной карте этого состояния не ставит вовсе, поэтому нагрузка
# поднимается здесь же отдельным процессом, а число сборок и итог
# называются числами.
#
#     bash lab/bench/load_run.sh [каталог] [процессов] [сборок] [ГБ]
#
# По умолчанию 10 процессов по 12 сборок под 8 ГБ удержания — те самые
# 120 сборок, которыми проверялась правка.
#
# Коды: 0 — ни одного отказа; 1 — есть отказ, и тогда журнал каждого
# прогона остаётся в каталоге итогов.
#
# Имена переменных латиницей нарочно: bash не допускает неASCII-имён
# (см. заголовок `gate_run.sh`).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/lab/bench/нагрузка}"
PROCS="${2:-10}"
BUILDS="${3:-12}"
GB="${4:-8}"
PY="$ROOT/.venv/Scripts/python.exe"
mkdir -p "$OUT"

export PYTHONIOENCODING=utf-8
export FLYBRAIN_OUT="${FLYBRAIN_OUT:-$OUT/out}"

# Нагрузка на всё время прогона, с запасом: под нагрузкой одна сборка со
# тридцатью шагами идёт секунды, но карта делится.
SECS=$(( PROCS * 30 + 90 ))
"$PY" "$ROOT/lab/bench/load_gpu.py" "$SECS" "$GB" >"$OUT/нагрузка.log" 2>&1 &
LOADPID=$!
# Нагрузке надо дать занять память, иначе первые прогоны пойдут по
# свободной карте и проверят не то.
sleep 12
echo "нагрузка поднята на $SECS с, под удержание $GB ГБ"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  карта: /'
echo

bad=0
builds=0
for i in $(seq 1 "$PROCS"); do
    log="$OUT/прогон_$i.log"
    ( cd "$ROOT/lab" && "$PY" probe/kernel_fault.py --part сборка \
        --сборок "$BUILDS" ) >"$log" 2>&1
    code=$?
    builds=$(( builds + BUILDS ))
    if [ "$code" = "0" ]; then
        printf '  ok      прогон %2d: %2d сборок, код 0\n' "$i" "$BUILDS"
    else
        bad=$(( bad + 1 ))
        printf '  ОТКАЗ   прогон %2d: код %s\n' "$i" "$code"
        grep -nE 'ПРОВАЛ|illegal memory|Traceback|Error' "$log" | head -10 \
            | sed 's/^/            /'
    fi
done

kill "$LOADPID" 2>/dev/null || true
wait "$LOADPID" 2>/dev/null || true

echo
tail -1 "$OUT/нагрузка.log" 2>/dev/null
printf 'ИТОГ: процессов %s, сборок движка %s, отказов %s\n' \
    "$PROCS" "$builds" "$bad"
[ "$bad" = "0" ] || exit 1
