// =============================================================================
// 文件：spinloop.cpp
// 模块：硬件优化的自旋等待（Spinloop）模块
// =============================================================================
//
// 【模块功能概述】
// 本模块实现了一个高性能的自旋等待机制，用于 vLLM 内部的进程间/线程间同步。
// 它通过调用 Python 回调函数来检查某个条件是否满足，如果未满足则继续等待。
//
// 【核心设计理念】
// 1. 传统自旋等待（busy-wait）会持续占用 CPU 资源，导致功耗浪费和 CPU 过热。
// 2. 本模块利用 AMD CPU 的 MONITORX/MWAITX 指令，让 CPU 在等待期间进入低功耗状态，
//    当被监控的内存地址被修改时，CPU 会自动唤醒，从而显著降低功耗。
// 3. 对于不支持 MONITORX/MWAITX 的 CPU（如 Intel、ARM），回退到普通的自旋等待，
//    但会插入 PAUSE（x86）或 YIELD（ARM）指令来提示 CPU 当前处于忙等待状态。
//
// 【应用场景】
// 主要用于 vLLM 的多进程架构中，例如：
// - EngineCore 进程等待 Worker 完成某项操作
// - 异步任务之间的低延迟同步
// 相比传统的 sleep/poll 方式，自旋等待可以实现更低的延迟。
//
// 【使用方式】
// Python 层调用：spinloop.spinloop(buffer, callback, timeout)
//   - buffer：需要监控的内存缓冲区（bytes/bytearray）
//   - callback：回调函数，返回 True 表示条件满足，退出等待
//   - timeout：超时时间（秒），可选
//
// =============================================================================

#include <Python.h>

extern "C" {

#include <stdbool.h>
#include <time.h>

// 【平台相关头文件】
// x86/x86_64 平台需要以下头文件：
//   - cpuid.h：用于查询 CPU 特性（如是否支持 MONITORX/MWAITX）
//   - mwaitxintrin.h：提供 _mm_monitorx() 和 _mm_mwaitx() 内联函数
#if defined(__i386__) || defined(__x86_64__)
  #include <cpuid.h>
  #include <mwaitxintrin.h>
#endif

#if defined(CLOCK_MONOTONIC_RAW)
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC_RAW
#else
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC
#endif

#define CPU_SUPPORT_NONE 0
#define CPU_SUPPORT_MONITORX 1

#define MWAITX_DEFAULT_TIMEOUT_CYCLES 1000000

typedef struct {
  unsigned int cpu_support;
  unsigned int max_monitor_line_size;
} spinloop_state_t;

static void determine_cpu_support(spinloop_state_t* state) {
  state->cpu_support = CPU_SUPPORT_NONE;
  state->max_monitor_line_size = 0;

#if defined(__i386__) || defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 1) {
    // AMD CPU (possible monitorx/mwaitx support)
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
      if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 1 &&
          eax >= 0x80000001 &&
          __get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 1) {
        if ((ecx & (1 << 29)) != 0) {
          state->cpu_support = CPU_SUPPORT_MONITORX;
        }
      }
    }
  }

  if (state->cpu_support == CPU_SUPPORT_MONITORX) {
    if (__get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
      state->max_monitor_line_size = ebx & 0xff;
    }
  }
#endif
}

static PyObject* method_spinloop(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  Py_buffer buffer;
  PyObject* callback;
  double timeout = 0.;

  spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(self);
  if (state == NULL) {
    PyErr_SetString(PyExc_TypeError, "Failed to retrieve module state!");
    return NULL;
  }

  static const char* keywords[] = {"buffer", "callback", "timeout", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*O|d", (char**)keywords,
                                   &buffer, &callback, &timeout)) {
    return NULL;
  }

  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback parameter must be callable!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  struct timespec t_start;
  if (clock_gettime(TIMEOUT_CLOCK, &t_start) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  bool result = false;
  bool error = false;
  bool have_timeout = (timeout > 1e-9);
  unsigned int iteration = 0;
  const bool buffer_qualifies = (buffer.len <= state->max_monitor_line_size);

  while (true) {
    PyObject* res = PyObject_CallNoArgs(callback);
    if (res == NULL) {
      error = true;
      break;
    }
    int ok = (res == Py_True);
    Py_DECREF(res);

    if (ok) {
      result = true;
      break;
    }

    // Check timeout at most every 16 iterations to avoid clock_gettime and
    // comparison cost
    if (have_timeout && (iteration & 15u) == 0) {
      struct timespec t_now;
      if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }

      const double elapsed = (double)(t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
      if (elapsed >= timeout) {
        result = false;
        break;
      }
    }
    ++iteration;

#if defined(__i386__) || defined(__x86_64__)
    // monitorx + mwaitx with qualified buffer
    if (buffer_qualifies && state->cpu_support == CPU_SUPPORT_MONITORX) {
      _mm_monitorx(buffer.buf, 0, 0);

      // Check once more in case the buffer has been modified while we were
      // arming the monitor hardware
      res = PyObject_CallNoArgs(callback);
      if (res == NULL) {
        error = true;
        break;
      }
      ok = (res == Py_True);
      Py_DECREF(res);

      if (ok) {
        result = true;
        break;
      }

      // Run mwaitx with enabled timeout (bit 1). The actual timeout value
      // is not very important, we just want to ensure we don't lock up
      // here for too long.
      Py_BEGIN_ALLOW_THREADS _mm_mwaitx((1 << 1), 0,
                                        MWAITX_DEFAULT_TIMEOUT_CYCLES);
      Py_END_ALLOW_THREADS
    }

    // Fallback: Busy poll
    else {
#endif
      // Give other threads a chance to be scheduled
      Py_BEGIN_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield" :: : "memory");
#endif
      Py_END_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
    }
#endif
  }

  PyBuffer_Release(&buffer);

  if (error) {
    return NULL;
  }

  if (result) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyMethodDef spinloop_methods[] = {
    {"spinloop", (PyCFunction)method_spinloop, METH_VARARGS | METH_KEYWORDS,
     "Wait for store with callback"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spinloop_module = {
    PyModuleDef_HEAD_INIT, "spinloop",
    "Hardware-optimized spinloops for Python", sizeof(spinloop_state_t),
    spinloop_methods};

PyMODINIT_FUNC PyInit_spinloop(void) {
  PyObject* m = PyModule_Create(&spinloop_module);
  if (m != NULL) {
    spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(m);
    if (state != NULL) {
      determine_cpu_support(state);
    }
  }
  return m;
}

}  // extern "C"
