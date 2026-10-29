"""Событийное умножение своим ядром: замер, а не предсказание.

Из разбора (`probe/lifperf.py`): шаг движка упирается в видеокарту, и
86 % его занимает одно ядро — `csrmv` по матрице связей. Оно читает все
15 091 983 связи на каждом тике 0,1 мс, а спайкуют 28…93 нейрона из
138 639, то есть нужны 0,04 % связей.

Здесь то же произведение считается событийно: решётка по ПРЕсинаптическим
нейронам, блок на нейрона, неспайкующие выходят сразу, спайкующие
складывают свою строку исходящих атомарными добавлениями. Ядро
компилируется на месте через `torch.utils.cpp_extension` (nvcc и MSVC в
системе есть; `.venv` при этом не трогается, сборка уходит в каталог,
заданный `TORCH_EXTENSIONS_DIR`).

**Почему атомарные добавления остаются ПОБИТОВО воспроизводимыми.**
Обычно atomicAdd по float32 недетерминирован: порядок сложения случаен, а
сложение float не ассоциативно. Здесь это не так, и причина измерена, а
не предположена (`probe/lifperf_mat.py`, пункт 1): значения матрицы
ЦЕЛЫЕ (от −2405 до 1897), спайк ровно 0.0 или 1.0, а максимальная сумма
модулей по строке 69 948 против предела точности float32 в
2^24 = 16 777 216. Сумма целых во float32 точна при любом порядке
сложения, пока частичные суммы не выходят за предел. Поэтому порядок
безразличен, пропуск нулевых слагаемых безразличен, и результат обязан
совпасть с `csrmv` до байта. Проба это и проверяет — байтами, на пяти
разных наборах настоящих спайков.

Что ещё проверяется, потому что без этого выигрыш ничего не значит:

* **пишется ли ядро в CUDA-граф.** Формы у него статические (решётка по
  всем нейронам, ранний выход), так что обязано. Если нет — выигрыш
  съедает потеря графов, а это известное трёхкратное замедление;
* **худший случай.** Залп после снятия маски `fire` поднимает число
  спайкующих в десятки раз, и событийное умножение тем дороже, чем
  больше спайкует. Поэтому замер идёт по развёртке: 100, 1 000, 10 000,
  50 000 и все 138 639 спайкующих. Точка, где оно становится дороже
  `csrmv`, названа числом, а не оставлена на удивление.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf_event.py
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe.lifperf import BRAIN_STEPS, build, med, pairs, sync  # noqa: E402

OUT = Path(__file__).parent
BASE_STEP_MS = 0.3203
THREADS = 128

SRC = r"""
#include <cuda_runtime.h>

// out[post] += sum over spiking pre of W[post, pre] * sp[pre].
//
// Блок на пресинаптический нейрон, нити блока идут по его строке
// исходящих с шагом в размер блока. Неспайкующий нейрон выходит сразу,
// прочитав одно число: решётка по всем нейронам статична, поэтому ядро
// пишется в CUDA-граф.
template <typename I>
__global__ void ev_spmv_kernel(const I* __restrict__ crow,
                               const I* __restrict__ col,
                               const float* __restrict__ val,
                               const float* __restrict__ sp,
                               float* __restrict__ out,
                               int n) {
  int pre = blockIdx.x;
  if (pre >= n) return;
  float s = sp[pre];
  if (s == 0.0f) return;
  I a = crow[pre];
  I b = crow[pre + 1];
  for (I k = a + threadIdx.x; k < b; k += blockDim.x) {
    atomicAdd(out + static_cast<int>(col[k]), val[k] * s);
  }
}

// Точки входа на обычном C: заголовки torch не нужны вовсе, и
// расширение не собирается — собирается простая DLL, которую зовут через
// ctypes сырыми указателями устройства (`tensor.data_ptr()`) и ручкой
// потока (`torch.cuda.current_stream().cuda_stream`). Так обходится
// требование ninja, которого в окружении нет, а ставить в `.venv`
// запрещено.
extern "C" __declspec(dllexport)
void ev_spmv_i32(const int* crow, const int* col, const float* val,
                 const float* sp, float* out, int n, int threads,
                 void* stream) {
  ev_spmv_kernel<int><<<n, threads, 0, (cudaStream_t)stream>>>(
      crow, col, val, sp, out, n);
}

extern "C" __declspec(dllexport)
void ev_spmv_i64(const long long* crow, const long long* col,
                 const float* val, const float* sp, float* out, int n,
                 int threads, void* stream) {
  ev_spmv_kernel<long long><<<n, threads, 0, (cudaStream_t)stream>>>(
      crow, col, val, sp, out, n);
}
"""

NVCC = (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
        r"\nvcc.exe")
VCVARS = (r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC"
          r"\Auxiliary\Build\vcvars64.bat")


def build_kernel(verbose=False):
    """Собрать ядро в DLL через nvcc. `.venv` не трогается.

    Сборка уходит в каталог из ``TORCH_EXTENSIONS_DIR`` (или во временный),
    туда же ложится и .cu, чтобы его можно было прочитать глазами.
    """
    import ctypes
    import subprocess

    where = Path(os.environ.get("TORCH_EXTENSIONS_DIR")
                 or (Path(os.environ.get("TEMP", ".")) / "flybrain-ext"))
    where.mkdir(parents=True, exist_ok=True)
    src = where / "ev_spmv.cu"
    src.write_text(SRC, encoding="utf-8")
    dll = where / "ev_spmv.dll"
    # Через .bat, а не строкой в `cmd /c`: кавычки в пути к vcvars64
    # subprocess на Windows переэкранирует, и вызов ломается.
    bat = where / "build.bat"
    bat.write_text(
        '@echo off\r\n'
        'call "%s" >nul\r\n'
        '"%s" -O3 -shared -gencode arch=compute_89,code=sm_89 -o "%s" "%s"\r\n'
        % (VCVARS, NVCC, dll, src), encoding="utf-8")
    got = subprocess.run(["cmd", "/c", str(bat)], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if verbose or got.returncode != 0:
        print(got.stdout[-3000:])
        print(got.stderr[-3000:])
    if got.returncode != 0 or not dll.exists():
        raise RuntimeError("nvcc вернул %d" % got.returncode)
    lib = ctypes.CDLL(str(dll))
    for name in ("ev_spmv_i32", "ev_spmv_i64"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p]
    return lib


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch

    from flybrain import data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"метка": args.tag, "устройство": device, "нитей в блоке": THREADS}

    print("собираю ядро...")
    t0 = time.perf_counter()
    try:
        ext = build_kernel(args.verbose)
    except Exception as exc:                                  # noqa: BLE001
        print("ЯДРО НЕ СОБРАЛОСЬ:", str(exc)[:600])
        rep["ядро собралось"] = False
        rep["почему нет"] = str(exc)[:600]
        (OUT / ("lifperf_event_%s.json" % args.tag)).write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1
    rep["ядро собралось"] = True
    rep["сборка, с"] = round(time.perf_counter() - t0, 1)
    print("ядро собрано за %.1f с" % rep["сборка, с"])

    env = build(device)
    eng, n = env["engine"], env["n"]
    rates = env["rates"]
    R = args.rounds
    rep["нейронов"] = int(n)

    # ------------------------------------------------------------- фон
    for _ in range(50):
        eng.step(rates)
    sync()
    vals = []
    for _ in range(5):
        sync()
        t0 = time.perf_counter()
        for _ in range(BRAIN_STEPS):
            eng.step(rates)
        sync()
        vals.append((time.perf_counter() - t0) * 1000.0 / BRAIN_STEPS)
    step_ms = statistics.median(vals)
    rep["шаг движка сейчас, мс"] = round(step_ms, 4)
    rep["фон, раз медленнее базовой линии"] = round(step_ms / BASE_STEP_MS, 2)
    print("шаг движка сейчас %.4f мс, фон ×%.2f"
          % (step_ms, step_ms / BASE_STEP_MS))

    # ------------------------------------- матрица по пресинаптическому
    csr = eng.w_csr
    with D._quiet_sparse():
        wt = csr.t().to_sparse_csr()
    crow_t = wt.crow_indices().contiguous()
    col_t = wt.col_indices().contiguous()
    val_t = wt.values().contiguous()
    rep["тип индексов"] = str(col_t.dtype)
    deg = (crow_t[1:] - crow_t[:-1]).to(torch.int64)
    nnz = int(deg.sum())
    rep["исходящих: макс"] = int(deg.max())
    rep["исходящих: среднее"] = round(float(deg.to(torch.float64).mean()), 1)
    print("исходящих связей: в среднем %.1f, максимум %d"
          % (rep["исходящих: среднее"], rep["исходящих: макс"]))

    out = torch.zeros(n, device=device)
    call = (ext.ev_spmv_i32 if col_t.dtype == torch.int32
            else ext.ev_spmv_i64)
    pc, pcol, pval, pout = (crow_t.data_ptr(), col_t.data_ptr(),
                            val_t.data_ptr(), out.data_ptr())

    def event_mm(sp1):
        out.zero_()
        call(pc, pcol, pval, sp1.data_ptr(), pout, n, THREADS,
             torch.cuda.current_stream().cuda_stream)
        return out

    # ------------------------------------------------ равенство побитово
    print("\n1. равенство побитово на настоящих спайках")
    rep["равенство"] = []
    ok_all = True
    for _ in range(5):
        for _ in range(37):
            eng.step(rates)
        sp2 = (eng.spikes * eng.m_out)
        sp1 = sp2[0].contiguous()
        ref = torch.sparse.mm(csr, sp2.t().contiguous()).t().reshape(-1)
        got = event_mm(sp1).clone()
        same = bool(np.array_equal(ref.cpu().numpy().view(np.uint8),
                                   got.cpu().numpy().view(np.uint8)))
        ok_all = ok_all and same
        rep["равенство"].append(
            {"спайкует": int((sp1 > 0).sum()), "побитово то же": same,
             "макс расхождение": float((ref - got).abs().max())})
        print("   спайкует %5d -> побитово то же: %s, макс расхождение %.6g"
              % (rep["равенство"][-1]["спайкует"], same,
                 rep["равенство"][-1]["макс расхождение"]))
    rep["побитово везде"] = ok_all

    # --------------------------------------------- время на рабочей точке
    print("\n2. время на рабочей точке, попарно против csrmv")
    sp2 = (eng.spikes * eng.m_out)
    sp1 = sp2[0].contiguous()
    col2 = sp2.t().contiguous()
    rep["спайкует в рабочей точке"] = int((sp1 > 0).sum())
    rep["рабочая точка"] = pairs(lambda: torch.sparse.mm(csr, col2),
                                 lambda: event_mm(sp1), rounds=R, reps=20)
    p = rep["рабочая точка"]
    print("   спайкует %d: csrmv %7.4f мс, событийно %7.4f мс, "
          "в %.2f раза (%.2f…%.2f, %d пар)"
          % (rep["спайкует в рабочей точке"], p["a, мс"], p["b, мс"],
             p["a/b медиана"], p["a/b мин"], p["a/b макс"], p["пар"]))

    # ------------------------------------------- развёртка по числу спайков
    print("\n3. развёртка: где событийное становится дороже csrmv")
    rep["развёртка"] = {}
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    perm = torch.randperm(n, device=device, generator=gen)
    for k in (100, 1000, 10000, 50000, n):
        art = torch.zeros(n, device=device)
        art[perm[:k]] = 1.0
        art2 = art.reshape(1, n).t().contiguous()
        touched = int(deg[perm[:k]].sum())
        ok = bool(np.array_equal(
            torch.sparse.mm(csr, art2).reshape(-1).cpu().numpy()
            .view(np.uint8),
            event_mm(art).cpu().numpy().view(np.uint8)))
        pr = pairs(lambda: torch.sparse.mm(csr, art2),
                   lambda: event_mm(art), rounds=max(7, R // 2), reps=10)
        rep["развёртка"][k] = {"связей задето": touched,
                               "доля связей": round(touched / nnz, 5),
                               "побитово то же": ok, **pr}
        print("   спайкует %6d (связей %9d, %6.2f %%): csrmv %7.4f, "
              "событийно %7.4f, в %.2f раза, побитово %s"
              % (k, touched, 100.0 * touched / nnz, pr["a, мс"], pr["b, мс"],
                 pr["a/b медиана"], ok))

    # ------------------------------------------------- пишется ли в граф
    print("\n4. пишется ли событийное ядро в CUDA-граф")
    try:
        static_sp = sp1.clone()
        ref = torch.sparse.mm(
            csr, static_sp.reshape(1, n).t().contiguous()).reshape(-1).clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                event_mm(static_sp)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out.zero_()
            call(pc, pcol, pval, static_sp.data_ptr(), pout, n, THREADS,
                 torch.cuda.current_stream().cuda_stream)
        out.zero_()
        g.replay()
        sync()
        same = bool(np.array_equal(ref.cpu().numpy().view(np.uint8),
                                   out.cpu().numpy().view(np.uint8)))
        ms = med(lambda: g.replay(), rounds=R, reps=20)
        rep["в граф пишется"] = True
        rep["граф: побитово то же"] = same
        rep["граф: мс"] = ms["мс"]
        print("   захват вышел, побитово то же: %s, повтор %.4f мс"
              % (same, ms["мс"]))
    except Exception as exc:                                  # noqa: BLE001
        rep["в граф пишется"] = False
        rep["граф: почему нет"] = str(exc)[:300]
        print("   НЕ пишется:", str(exc)[:300])

    # --------------------------------------- что это дало бы шагу движка
    p = rep["рабочая точка"]
    saved = p["a, мс"] - p["b, мс"]
    rep["экономия на шаге движка, мс (при нынешнем фоне)"] = round(saved, 4)
    rep["шаг движка стал бы, мс (при нынешнем фоне)"] = round(
        step_ms - saved, 4)
    rep["во сколько раз шаг движка"] = round(
        step_ms / max(step_ms - saved, 1e-9), 3)
    print("\n   шаг движка при нынешнем фоне: %.4f -> %.4f мс, в %.2f раза"
          % (step_ms, step_ms - saved, rep["во сколько раз шаг движка"]))
    print("   (это оценка по одному ядру, а не замер шага целиком — "
          "шаг надо мерить после правки в sim.py)")

    path = OUT / ("lifperf_event_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
