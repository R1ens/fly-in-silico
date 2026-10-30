#!/usr/bin/env bash
# Прогон шлюзов с СОХРАНЁННЫМИ диагностическими данными.
#
# Зачем отдельным файлом. Прежде шлюзы гонялись строчкой вида
# `проба >/dev/null; echo код $?`, и когда `resume_state` однажды дал
# код 1, от отказа остался ТОЛЬКО код выхода: ни имён провалившихся
# проверок, ни отпечатков состояния, а `resume_state.json` затёрли
# последующие прогоны. Причину того отказа установить уже нечем — он не
# воспроизвёлся девять раз из девяти (тихо, под 100 % нагрузкой, в том
# же порядке проб, в том же каталоге вывода), и это остаётся
# НЕОБЪЯСНЁННЫМ единичным отказом, а не «помехой».
#
# Поэтому здесь каждая проба оставляет свой журнал и свой json, а при
# отказе печатаются имена провалившихся проверок. Следующий такой случай
# будет разобран, а не потерян.
#
#     bash lab/bench/gate_run.sh [каталог-итогов]
#
# Коды: 0 — все шлюзы сошлись; 1 — есть неожиданный отказ.
#
# Имена переменных здесь ЛАТИНИЦЕЙ нарочно: bash не допускает
# неASCII-идентификаторов, и `КОРЕНЬ=...` он разбирает не как
# присваивание, а как имя команды. Поймано прогоном: «КОРЕНЬ=/d/fly-brain:
# No such file or directory» и синтаксическая ошибка на `declare -A`.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/lab/bench/итоги}"
PY="$ROOT/.venv/Scripts/python.exe"
mkdir -p "$OUT"

export PYTHONIOENCODING=utf-8
# Свой каталог вывода: `out/` основного дерева — только чтение, там 28
# сохранённых научных прогонов.
export FLYBRAIN_OUT="${FLYBRAIN_OUT:-$OUT/out}"

# `body_equal` выходит с кодом 1 ПО ПОСТРОЕНИЮ: она сравнивает обычный
# эпизод с `fast_physics`, равенства между ними нет, и это
# задокументировано в SPEED.md §3 теми же числами. Смотреть надо, совпал
# ли ОБЫЧНЫЙ эпизод с эталоном — это и печатает её журнал.
expected_code() {
    case "$1" in
        body_equal) echo 1 ;;
        *)          echo 0 ;;
    esac
}

unexpected=""

run_gate() {                     # $1 — имя пробы, далее её аргументы
    name="$1"; shift
    log="$OUT/$name.log"
    ( cd "$ROOT/lab" && "$PY" "probe/$name.py" "$@" ) >"$log" 2>&1
    code=$?
    # json пробы складываем рядом с журналом: именно его затирает
    # следующий прогон, и именно он нужен при разборе отказа.
    if [ -f "$ROOT/lab/probe/$name.json" ]; then
        cp "$ROOT/lab/probe/$name.json" "$OUT/$name.json" 2>/dev/null
    fi
    want="$(expected_code "$name")"
    if [ "$code" = "$want" ]; then
        printf '  ok      %-18s код %s\n' "$name" "$code"
    else
        printf '  ОТКАЗ   %-18s код %s (ждали %s)\n' "$name" "$code" "$want"
        unexpected="$unexpected $name"
        # Имена провалившихся проверок — сразу в вывод, чтобы разбор не
        # требовал возвращаться к журналу.
        grep -E 'ПРОВАЛ' "$log" | head -8 | sed 's/^/            /'
    fi
}

echo "шлюзы: $(cd "$ROOT" && git log --oneline -1)"
echo "итоги и журналы: $OUT"
echo

# --- побитовые эталоны -------------------------------------------------
run_gate lif_equal --against эталон
run_gate resume_state --part all --tail 150

# --- семантика вмешательств и моторный путь ----------------------------
for p in lif_event semantics lesion motor_equal motor_audit motor_grade; do
    run_gate "$p"
done

# --- данные, совместимость, пространства -------------------------------
for p in compat_runs catalog_test five_spaces sel_dump select_exact \
         timeline_trap replay_modes ipc_test session_test; do
    run_gate "$p"
done

# --- интерфейс и сцена -------------------------------------------------
for p in objects objects_live sandbox_capture sandbox_shot exp_shot \
         brain_shot cli_check integrity body_equal; do
    run_gate "$p"
done

# `ana_shot` читает сохранённые прогоны из настоящего `out/`, поэтому
# зовётся БЕЗ перенаправления вывода — иначе не находит их и падает на
# отсутствующем каталоге прогона.
log="$OUT/ana_shot.log"
( unset FLYBRAIN_OUT; cd "$ROOT/lab" && "$PY" probe/ana_shot.py ) >"$log" 2>&1
code=$?
if [ "$code" = "0" ]; then
    printf '  ok      %-18s код 0\n' ana_shot
else
    printf '  ОТКАЗ   %-18s код %s\n' ana_shot "$code"
    grep -E 'ПРОВАЛ' "$log" | head -5 | sed 's/^/            /'
    unexpected="$unexpected ana_shot"
fi

echo
if [ -z "$unexpected" ]; then
    echo "все шлюзы сошлись"
    exit 0
fi
echo "неожиданные отказы:$unexpected"
echo "журналы и json каждой пробы лежат в $OUT"
exit 1
