// SPDX-License-Identifier: Apache-2.0
// Test-only native task: proves dispatch proceeds while Python holds its GIL.
#include <pybind11/pybind11.h>
#include <atomic>
#include <chrono>
#include <dlfcn.h>
#include "kt_dispatch_api.h"
namespace py = pybind11;
struct Task { std::atomic<unsigned> executed{0}; };
static void entry(void* arg) { static_cast<Task*>(arg)->executed.fetch_add(1, std::memory_order_release); }
PYBIND11_MODULE(_native_dispatch_fixture, m) {
  m.def("task", [] {
    auto* task = new Task;
    return py::capsule(task, [](void* p) { delete static_cast<Task*>(p); });
  });
  m.def("address", [](py::capsule task) { return reinterpret_cast<uintptr_t>(task.get_pointer()); });
  m.def("entry", [] { return reinterpret_cast<uintptr_t>(&entry); });
  m.def("api", [] {
    static const KtDispatchApi api{1,
      [](uintptr_t, uintptr_t fn, uintptr_t arg) { reinterpret_cast<void (*)(void*)>(fn)(reinterpret_cast<void*>(arg)); },
      [](uintptr_t) {}, [](int) -> uint64_t { return 1; }, [](uint64_t, int) {}, [](const char*) {}};
    return py::capsule(const_cast<KtDispatchApi*>(&api));
  });
  // Intentionally no gil_scoped_release/call_guard here.
  m.def("wait_holding_gil", [](py::capsule task, unsigned target) {
    auto* ptr = static_cast<Task*>(task.get_pointer());
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (ptr->executed.load(std::memory_order_acquire) < target)
      if (std::chrono::steady_clock::now() > end) return false;
    return ptr->executed.load() == target;
  });
  // A host sentinel written by DMA after the device-ready wait: Python stays
  // inside this C++ call holding GIL until the entire pipeline completes.
  m.def("wait_value_holding_gil", [](uintptr_t address, uint64_t target) {
    auto* ptr = reinterpret_cast<uint64_t*>(address);
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (__atomic_load_n(ptr, __ATOMIC_ACQUIRE) != target)
      if (std::chrono::steady_clock::now() > end) return false;
    return true;
  });
}
