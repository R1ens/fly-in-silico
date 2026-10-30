// Событийное разрежённое умножение для движка LIF.
//
// Что считается:
//
//     out[b, post] += sum по pre от W[post, pre] * sp[b, pre] * mo[b, pre]
//
// Матрица лежит ПО ПРЕСИНАПТИЧЕСКОМУ нейрону: это CSR транспонированной
// `W`, то есть строка `pre` — все исходящие связи этого нейрона. Решётка
// по `pre` (и по партии вторым измерением), блок на нейрона, нити блока
// идут по строке с шагом в размер блока. Неспайкующий нейрон выходит,
// прочитав два числа, поэтому ядро читает не 15 091 983 связи, а только
// связи спайкующих — в рабочей точке это 0,04 % от матрицы.
//
// Решётка СТАТИЧНА (всегда `n` блоков, ранний выход вместо сжатия
// списка спайкующих), и это не небрежность, а условие: динамическая
// форма выбила бы шаг из CUDA-графа, а потеря графов в этом проекте
// стоит втрое.
//
// ПОЧЕМУ АТОМАРНЫЕ ДОБАВЛЕНИЯ ЗДЕСЬ ПОБИТОВО ВОСПРОИЗВОДИМЫ.
// Обычно `atomicAdd` по float32 недетерминирован: порядок сложения
// случаен, а сложение float не ассоциативно. Здесь порядок безразличен,
// и причина не в надежде, а в трёх замеренных числах (проверяются при
// сборке движка, `flybrain.evspmv.exactness`):
//
//   * значения матрицы ЦЕЛЫЕ (число синапсов связи, −2405…1897);
//   * множитель `sp * mo` — ровно 0.0 или 1.0;
//   * максимальная сумма модулей по строке результата 69 948 против
//     предела точности float32 в 2^24 = 16 777 216, запас в 240 раз.
//
// Сумма целых во float32 точна при любом порядке сложения, пока
// частичные суммы не выходят за 2^24; пропуск нулевых слагаемых тоже
// ничего не меняет (x + 0.0 == x для любого конечного x). Отсюда
// результат обязан совпасть с `csrmv` ДО БАЙТА, и это проверяется
// `probe/lif_event.py`, а не берётся на слово.
//
// Если хоть одно из трёх чисел перестанет держаться — движок откажется
// собирать событийное умножение громко, с указанием, какое именно.

#include <cuda_runtime.h>

template <typename I>
__global__ void ev_spmv_kernel(const I* __restrict__ crow,
                               const I* __restrict__ col,
                               const float* __restrict__ val,
                               const float* __restrict__ sp,
                               const float* __restrict__ mo,
                               float* __restrict__ out,
                               int n) {
  const int pre = blockIdx.x;
  if (pre >= n) return;
  const long long off = (long long)blockIdx.y * (long long)n;
  const float s = sp[off + pre] * mo[off + pre];
  if (s == 0.0f) return;
  float* o = out + off;
  const I a = crow[pre];
  const I e = crow[pre + 1];
  for (I k = a + (I)threadIdx.x; k < e; k += (I)blockDim.x) {
    atomicAdd(o + (long long)col[k], val[k] * s);
  }
}

// Точки входа на обычном C: заголовки torch не нужны вовсе, и
// расширение не собирается — собирается простая DLL, которую зовут
// через ctypes сырыми указателями устройства (`tensor.data_ptr()`) и
// ручкой потока (`torch.cuda.current_stream().cuda_stream`). Так
// обходится требование ninja, которого в окружении нет, а ставить что
// бы то ни было в `.venv` запрещено.

extern "C" __declspec(dllexport)
int ev_spmv_i32(const int* crow, const int* col, const float* val,
                const float* sp, const float* mo, float* out, int n,
                int batch, int threads, void* stream) {
  dim3 grid(n, batch);
  ev_spmv_kernel<int><<<grid, threads, 0, (cudaStream_t)stream>>>(
      crow, col, val, sp, mo, out, n);
  return (int)cudaGetLastError();
}

extern "C" __declspec(dllexport)
int ev_spmv_i64(const long long* crow, const long long* col,
                const float* val, const float* sp, const float* mo,
                float* out, int n, int batch, int threads, void* stream) {
  dim3 grid(n, batch);
  ev_spmv_kernel<long long><<<grid, threads, 0, (cudaStream_t)stream>>>(
      crow, col, val, sp, mo, out, n);
  return (int)cudaGetLastError();
}

// Подпись сборки: по ней видно, что загружена именно та DLL, а не
// оставшаяся от прежней правки источника.
extern "C" __declspec(dllexport)
int ev_spmv_abi(void) { return 2; }
