"""Настоящая нагрузка на карту — фон для нагрузочных проверок.

Единичный отказ `CUDA illegal memory access` случился под занятой
картой, поэтому проверки, которые должны его ловить, обязаны идти под
нагрузкой, а не на свободной карте. Нагрузка держится отдельным
процессом: он и занимает память, и занимает счёт.

    python lab/bench/load_gpu.py <секунд> <ГБ под удержание>

Занятость карты смотреть `nvidia-smi`: в замерах разбора
(`flybrain/KERNEL_FAULT.md`) это 100 % и около 20 ГБ из 24,5.
"""
import sys
import time

import torch

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
HOLD_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

if not torch.cuda.is_available():
    raise SystemExit("нагрузка на карту, а torch её не видит")

dev = "cuda"
# Удержание памяти: без него карта занята счётом, но свободна по памяти,
# а отказ случился именно когда рядом считали другие прогоны.
hold = torch.empty(int(HOLD_GB * (1 << 30) / 4), dtype=torch.float32,
                   device=dev)
hold.fill_(1.0)
# Три потока: один поток карту этого класса не занимает целиком.
streams = [torch.cuda.Stream() for _ in range(3)]
mats = [(torch.randn(4096, 4096, device=dev),
         torch.randn(4096, 4096, device=dev)) for _ in streams]
free, total = torch.cuda.mem_get_info()
print("нагрузка: под удержание %.1f ГБ, свободно %d МБ из %d, "
      "три потока по 4096×4096, держу %.0f с"
      % (HOLD_GB, free // 2 ** 20, total // 2 ** 20, SECONDS), flush=True)

t0 = time.perf_counter()
rounds = 0
while time.perf_counter() - t0 < SECONDS:
    for st, (a, b) in zip(streams, mats):
        with torch.cuda.stream(st):
            for _ in range(20):
                a @ b
    rounds += 1
torch.cuda.synchronize()
print("нагрузка снята, кругов %d за %.0f с"
      % (rounds, time.perf_counter() - t0))
