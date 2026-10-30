#!/usr/bin/env bash
# Самая близкая к обстоятельствам попытка воспроизвести единичный отказ.
#
# Нашедший описал обстановку так: третья сборка движка в одном процессе,
# карта занята, рядом считали другие прогоны. Здесь это ставится как
# было: НЕСКОЛЬКО прогонов `probe/vision_chain.py` одновременно и
# нагрузка рядом (`load_gpu.py`).
#
#     bash lab/bench/vision_storm.sh [каталог] [одновременно] [ГБ]
#
# Проба при этом НЕ ослабляется ничем: ни таймаута, ни повтора при
# отказе, ни пропуска. Она обязана остаться способной провалиться —
# иначе прогон не значит ничего.
#
# В замере разбора (`flybrain/KERNEL_FAULT.md` §4а): 3 одновременных
# прогона под 6 ГБ удержания, карта 19 236 МБ из 24 564 и 100 %
# занятости, каждый прогон 8 из 8 за ~2 500 с против 195 с в одиночку.
# Отказов 0 — не воспроизведено.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/lab/bench/шторм}"
PAR="${2:-3}"
GB="${3:-6}"
PY="$ROOT/.venv/Scripts/python.exe"
mkdir -p "$OUT"

export PYTHONIOENCODING=utf-8

"$PY" "$ROOT/lab/bench/load_gpu.py" 7200 "$GB" \
    >"$OUT/нагрузка.log" 2>&1 &
LOADPID=$!
sleep 12
echo "нагрузка поднята, под удержание $GB ГБ; пускаю $PAR прогонов сразу"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  карта: /'
echo

pids=""
for i in $(seq 1 "$PAR"); do
    (
        # Свой каталог вывода каждому: `out/` основного дерева — только
        # чтение, там сохранённые научные прогоны.
        export FLYBRAIN_OUT="$OUT/out_$i"
        ( cd "$ROOT/lab" && "$PY" probe/vision_chain.py ) \
            >"$OUT/vc_$i.log" 2>&1
        echo "$?" >"$OUT/код_$i.txt"
    ) &
    pids="$pids $!"
done
for p in $pids; do wait "$p" || true; done

kill "$LOADPID" 2>/dev/null || true
wait "$LOADPID" 2>/dev/null || true

echo
bad=0
for i in $(seq 1 "$PAR"); do
    code="$(cat "$OUT/код_$i.txt" 2>/dev/null || echo нет)"
    said="$(grep -E 'проверок' "$OUT/vc_$i.log" 2>/dev/null | tail -1)"
    if [ "$code" = "0" ]; then
        printf '  ok      прогон %d: код 0  %s\n' "$i" "$said"
    else
        bad=$(( bad + 1 ))
        printf '  ОТКАЗ   прогон %d: код %s\n' "$i" "$code"
        grep -nE 'ПРОВАЛ|illegal memory|Traceback|Error' "$OUT/vc_$i.log" \
            | head -12 | sed 's/^/            /'
    fi
done
printf 'ИТОГ: одновременных прогонов %s, отказов %s\n' "$PAR" "$bad"
[ "$bad" = "0" ] || exit 1
