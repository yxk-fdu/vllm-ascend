// SPDX-License-Identifier: Apache-2.0
// CPU-only LRC state shared by Python and native graph admission; mutex protected.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <numeric>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "load_submit.h"

namespace py = pybind11;
using Id = int64_t;
using Counts = std::vector<std::pair<Id, Id>>;

template <typename T>
T read_scalar(const void* ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}

struct Layer {
  std::vector<Id> freq, last;
  std::vector<float> ema;
  std::vector<double> router;
  std::deque<Counts> queue;
  Id step = 0;
  explicit Layer(Id n) : freq(n, 0), last(n, -1), ema(n, 0), router(n, 0) {}
};

class Policy {
  Id experts_, window_;
  double beta_, recent_, ema_weight_, router_weight_, age_weight_;
  bool scalar_f32_;
  std::vector<Layer> layers_;

  Layer& layer(Id i) {
    if (i < 0) i += static_cast<Id>(layers_.size());
    if (i < 0 || i >= static_cast<Id>(layers_.size())) throw py::index_error("layer index out of range");
    return layers_[i];
  }
  Id expert(Id i) const {
    if (i < 0) i += experts_;
    if (i < 0 || i >= experts_) throw py::index_error("expert index out of range");
    return i;
  }
  void trim(Layer& s) {
    while (static_cast<Id>(s.queue.size()) > window_) {
      for (auto [e, count] : s.queue.front()) s.freq[e] -= count;
      s.queue.pop_front();
    }
  }
  double score(const Layer& s, Id e) const {
    Id age = s.last[e] < 0 ? 0 : s.step - s.last[e];
    if (scalar_f32_) {
      // NumPy 2 weak Python-scalar promotion, matching hotness().
      float value = static_cast<float>(recent_ * s.freq[e]) + static_cast<float>(ema_weight_) * s.ema[e];
      value = value + static_cast<float>(router_weight_ * s.router[e]);
      value = value - static_cast<float>(age_weight_ * age);
      return value;
    }
    return recent_ * s.freq[e] + ema_weight_ * static_cast<double>(s.ema[e]) + router_weight_ * s.router[e] -
           age_weight_ * age;
  }

 public:
  std::recursive_mutex state_mutex;
  Policy(Id layers, Id experts, Id window, double beta, double recent, double ema_weight, double router_weight,
         double age_weight, bool scalar_f32)
      : experts_(experts),
        window_(window),
        beta_(beta),
        recent_(recent),
        ema_weight_(ema_weight),
        router_weight_(router_weight),
        age_weight_(age_weight),
        scalar_f32_(scalar_f32) {
    if (layers < 1 || experts < 1 || window < 1 || !(beta >= 0 && beta < 1))
      throw py::value_error("Invalid LRC dimensions or EMA beta");
    for (Id i = 0; i < layers; ++i) layers_.emplace_back(experts);
  }
  Id add_layer() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    layers_.emplace_back(experts_);
    return static_cast<Id>(layers_.size()) - 1;
  }
  Id num_layers() const { return layers_.size(); }
  Id step(Id idx) { std::lock_guard<std::recursive_mutex> lock(state_mutex); return layer(idx).step; }

  std::unordered_set<Id> observe_values(Id idx, const std::vector<std::vector<Id>>& rows,
                                        const std::vector<std::vector<double>>& scores, bool has_scores) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    auto& s = layer(idx);
    std::unordered_set<Id> unique;
    for (size_t r = 0; r < rows.size(); ++r) {
      const auto& row = rows[r];
      ++s.step;
      for (auto& v : s.ema) v = v * static_cast<float>(beta_);
      Counts counts;
      for (Id e : row) {
        if (e < 0 || e >= experts_) continue;
        unique.insert(e);
        auto found = std::find_if(counts.begin(), counts.end(), [e](const auto& v) { return v.first == e; });
        if (found == counts.end()) {
          counts.emplace_back(e, 1);
          s.ema[e] = s.ema[e] + static_cast<float>(1.0 - beta_);
        } else {
          ++found->second;
        }
        ++s.freq[e];
        s.last[e] = s.step;
      }
      s.queue.push_back(counts);
      // Match the reference's error ordering (state is already updated).
      if (has_scores) {
        if (r >= scores.size()) throw py::index_error("router score row missing");
        if (scores[r].size() != row.size()) throw py::value_error("router score width does not match topk ID width");
        for (size_t c = 0; c < row.size(); ++c) {
          Id e = row[c];
          if (e >= 0 && e < experts_) s.router[e] = scores[r][c];
        }
      }
      trim(s);
    }
    return unique;
  }

  std::unordered_set<Id> observe(Id idx, const std::vector<std::vector<Id>>& rows, const py::object& score_obj) {
    bool has_scores = !score_obj.is_none();
    auto scores = has_scores ? score_obj.cast<std::vector<std::vector<double>>>() : std::vector<std::vector<double>>{};
    return observe_values(idx, rows, scores, has_scores);
  }

  py::dict plan(Id idx, const py::array& ids, const py::object& scores_obj, py::array mapping, bool prefetch, Id cap,
                bool measure) {
    // No forcecast: mapping must remain a writable view of the pinned CPU buffer.
    layer(idx);
    auto id_info = ids.request();
    auto map_info = mapping.request();
    bool ids32 = ids.dtype().is(py::dtype::of<int32_t>());
    bool ids64 = ids.dtype().is(py::dtype::of<int64_t>());
    bool map32 = mapping.dtype().is(py::dtype::of<int32_t>());
    bool map64 = mapping.dtype().is(py::dtype::of<int64_t>());
    if (id_info.ndim != 2 || (!ids32 && !ids64)) throw py::value_error("topk IDs must be a 2D int32/int64 CPU array");
    if (map_info.ndim != 1 || map_info.shape[0] != experts_ || (!map32 && !map64) || !mapping.writeable() ||
        !(mapping.flags() & py::array::c_style))
      throw py::value_error("mapping must be writable contiguous int32/int64 with shape (num_experts,)");
    if (cap < 1) throw py::value_error("prefetch cap must be positive");
    auto read_map = [&](Id e) -> Id {
      auto ptr = static_cast<char*>(map_info.ptr) + e * map_info.itemsize;
      return map32 ? read_scalar<int32_t>(ptr) : read_scalar<int64_t>(ptr);
    };
    auto write_map = [&](Id e, Id slot) {
      auto ptr = static_cast<char*>(map_info.ptr) + e * map_info.itemsize;
      if (map32) {
        int32_t value = static_cast<int32_t>(slot);
        std::memcpy(ptr, &value, sizeof(value));
      } else
        std::memcpy(ptr, &slot, sizeof(slot));
    };
    auto read_id = [&](Id r, Id c) -> Id {
      auto ptr = static_cast<char*>(id_info.ptr) + r * id_info.strides[0] + c * id_info.strides[1];
      return ids32 ? read_scalar<int32_t>(ptr) : read_scalar<int64_t>(ptr);
    };
    std::vector<Id> residents;
    std::unordered_set<Id> slots;
    py::set on_device;
    for (Id e = 0; e < experts_; ++e) {
      Id slot = read_map(e);
      if (slot < -1 || slot >= experts_) throw py::value_error("invalid physical slot");
      if (slot >= 0) {
        if (!slots.insert(slot).second) throw py::value_error("duplicate physical slot");
        residents.push_back(e);
        on_device.add(py::int_(e));
      }
    }
    Id nrows = id_info.shape[0], width = id_info.shape[1];
    std::vector<std::vector<Id>> rows(nrows, std::vector<Id>(width));
    std::vector<std::vector<double>> scores;
    bool has_scores = !scores_obj.is_none();
    if (has_scores) {
      if (!py::isinstance<py::array>(scores_obj)) throw py::value_error("scores must be a CPU array");
      auto arr = scores_obj.cast<py::array>();
      auto info = arr.request();
      bool f32 = arr.dtype().is(py::dtype::of<float>());
      if (info.ndim != 2 || info.shape != id_info.shape || (!f32 && !arr.dtype().is(py::dtype::of<double>())))
        throw py::value_error("scores must be float32/float64 with the same shape as IDs");
      scores.assign(nrows, std::vector<double>(width));
      for (Id r = 0; r < nrows; ++r)
        for (Id c = 0; c < width; ++c) {
          auto ptr = static_cast<char*>(info.ptr) + r * info.strides[0] + c * info.strides[1];
          scores[r][c] = f32 ? read_scalar<float>(ptr) : read_scalar<double>(ptr);
        }
    }
    py::set needed;
    std::unordered_set<Id> protected_ids;
    for (Id r = 0; r < nrows; ++r)
      for (Id c = 0; c < width; ++c) {
        Id e = read_id(r, c);
        rows[r][c] = e;
        if (e >= 0 && e < experts_) {
          needed.add(py::int_(e));
          protected_ids.insert(e);
        } else if (prefetch) {
          throw py::value_error("prefetch IDs must be valid experts");
        }
      }
    using Clock = std::chrono::steady_clock;
    Id observe_ns = 0, choose_ns = 0;
    auto start = measure ? Clock::now() : Clock::time_point{};
    if (!prefetch) observe_values(idx, rows, scores, has_scores);
    if (measure && !prefetch)
      observe_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();

    // Keep CPython set iteration for exact slot assignment parity, including hash
    // collisions. The per-expert loops, scoring and mapping updates run in C++.
    py::set misses;
    if (prefetch) {
      std::vector<Id> candidates;
      std::vector<double> best(experts_, -std::numeric_limits<double>::infinity());
      for (Id r = 0; r < nrows; ++r)
        for (Id c = 0; c < width; ++c) {
          Id e = rows[r][c];
          if (read_map(e) >= 0) continue;
          double value = has_scores ? scores[r][c] : -static_cast<double>(c * nrows + r);
          if (value > best[e]) {
            if (best[e] == -std::numeric_limits<double>::infinity()) candidates.push_back(e);
            best[e] = value;
          }
        }
      std::stable_sort(candidates.begin(), candidates.end(), [&](Id a, Id b) { return best[a] > best[b]; });
      for (Id i = 0; i < std::min(cap, static_cast<Id>(candidates.size())); ++i) misses.add(py::int_(candidates[i]));
    } else {
      misses = py::reinterpret_steal<py::set>(PyNumber_Subtract(needed.ptr(), on_device.ptr()));
      if (!misses) throw py::error_already_set();
    }
    auto hits = py::reinterpret_steal<py::set>(PyNumber_And(needed.ptr(), on_device.ptr()));
    if (!hits) throw py::error_already_set();
    Id reusable = 0;
    for (Id e : residents)
      if (!protected_ids.count(e)) ++reusable;
    start = measure ? Clock::now() : Clock::time_point{};
    auto victims = choose(idx, residents, protected_ids, py::len(misses));
    if (measure) choose_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    std::vector<std::pair<Id, Id>> loads;
    for (auto item : misses) {
      if (loads.size() >= victims.size()) break;
      Id e = py::cast<Id>(item), victim = victims[loads.size()], slot = read_map(victim);
      loads.emplace_back(slot, e);
    }
    py::dict result;
    result["needed"] = needed;
    result["on_device"] = on_device;  // Snapshot before eviction, for statistics.
    result["already_there"] = hits;
    result["need_to_load"] = misses;
    result["planned_loads"] = loads;
    result["reusable_count"] = reusable;
    result["observe_ns"] = observe_ns;
    result["choose_ns"] = choose_ns;
    for (size_t i = 0; i < loads.size(); ++i) {
      write_map(victims[i], -1);
      write_map(loads[i].second, loads[i].first);
    }
    return result;
  }

  std::unordered_set<Id> observe_global(Id idx, const std::vector<Id>& counts, const py::object& score_obj) {
    if (static_cast<Id>(counts.size()) != experts_) throw py::value_error("global_counts has wrong shape");
    for (auto v : counts)
      if (v < 0) throw py::value_error("global_counts must be non-negative");
    bool has_scores = !score_obj.is_none();
    std::vector<float> scores;
    if (has_scores) {
      scores = score_obj.cast<std::vector<float>>();
      if (scores.size() != counts.size()) throw py::value_error("global_router_score has wrong shape");
    }
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    auto& s = layer(idx);
    ++s.step;
    Counts snapshot;
    std::unordered_set<Id> active;
    for (Id e = 0; e < experts_; ++e) {
      s.ema[e] = s.ema[e] * static_cast<float>(beta_);
      float increment = static_cast<float>(counts[e]) * static_cast<float>(1.0 - beta_);
      s.ema[e] = s.ema[e] + increment;
      if (counts[e]) {
        active.insert(e);
        snapshot.emplace_back(e, counts[e]);
        s.freq[e] += counts[e];
        s.last[e] = s.step;
        if (has_scores) s.router[e] = scores[e];
      }
    }
    s.queue.push_back(snapshot);
    trim(s);
    return active;
  }

  std::vector<Id> choose(Id idx, const std::vector<Id>& residents, const std::unordered_set<Id>& protected_ids,
                         Id count) {
    if (count < 0) throw py::value_error("victim count must be >= 0");
    if (!count) return {};
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    const auto& s = layer(idx);
    std::vector<std::tuple<double, Id, Id>> keys;
    for (Id e : residents) {
      if (!protected_ids.count(e)) keys.emplace_back(score(s, expert(e)), s.last[expert(e)], e);
    }
    std::stable_sort(keys.begin(), keys.end());
    std::vector<Id> result;
    for (Id i = 0; i < std::min(count, static_cast<Id>(keys.size())); ++i) result.push_back(std::get<2>(keys[i]));
    return result;
  }
  double hotness(Id idx, Id e) { std::lock_guard<std::recursive_mutex> lock(state_mutex); return score(layer(idx), expert(e)); }
  std::vector<float> hotness_array(Id idx) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    const auto& s = layer(idx);
    std::vector<float> result(experts_);
    for (Id e = 0; e < experts_; ++e) {
      float last = static_cast<float>(s.last[e]);
      float age = last < 0 ? 0 : static_cast<float>(s.step) - last;
      float value =
          static_cast<float>(recent_) * static_cast<float>(s.freq[e]) + static_cast<float>(ema_weight_) * s.ema[e];
      value = value + static_cast<float>(router_weight_) * static_cast<float>(s.router[e]);
      result[e] = value - static_cast<float>(age_weight_) * age;
    }
    return result;
  }
  void seed(Id idx, Id eid, Id freq, double weight) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    auto& s = layer(idx);
    Id e = expert(eid);
    s.freq[e] = freq;
    s.router[e] = weight;
    s.last[e] = s.step;
  }
  py::dict snapshot(Id idx) {
    std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
    const auto& s = layer(idx);
    py::dict d;
    d["freq"] = s.freq;
    d["ema"] = s.ema;
    d["router_score"] = s.router;
    d["last_used"] = s.last;
    d["step"] = s.step;
    d["recent_queue"] = s.queue;
    return d;
  }
};

#ifdef __linux__
#include "native_dispatch.h"
#endif

PYBIND11_MODULE(_vllm_ascend_lrc, m) {
#ifdef __linux__
  m.attr("native_dispatch_api_version") = 1;
  py::class_<NativePlan>(m, "NativePlan")
      .def_readonly("mapping", &NativePlan::mapping).def_readonly("loads", &NativePlan::loads)
      .def_readonly("cpu_ids", &NativePlan::ids).def_readonly("cpu_weights", &NativePlan::weights);
  py::class_<NativeHostPlanner, std::shared_ptr<NativeHostPlanner>>(m, "NativeHostPlanner")
      .def(py::init<Id, Id, Id, Id, Id, Id, std::string, Policy*, std::vector<double>, Id>(),
           py::arg("experts"), py::arg("slots"), py::arg("layer"), py::arg("numerator"),
           py::arg("denominator"), py::arg("max_transfers"), py::arg("policy"),
           py::arg("lrc"), py::arg("scores"), py::arg("npu_fixed_cost") = 0, py::keep_alive<1, 9>())
      .def("plan", &NativeHostPlanner::plan)
      .def("plan_prefetch", &NativeHostPlanner::plan_prefetch)
      .def("prepare_prefetch", &NativeHostPlanner::prepare_prefetch)
      .def("prepare", &NativeHostPlanner::prepare);
  py::class_<NativeCpuDispatcher>(m, "NativeCpuDispatcher")
      .def(py::init<py::capsule, uintptr_t, uintptr_t, int, int>())
      .def("add", &NativeCpuDispatcher::add)
      .def("add_prefetch", &NativeCpuDispatcher::add_prefetch)
      .def("add_prefetch_after", &NativeCpuDispatcher::add_prefetch_after)
      .def("prefetch_stats", &NativeCpuDispatcher::prefetch_stats)
      .def("next_weight", &NativeCpuDispatcher::next_weight, py::call_guard<py::gil_scoped_release>())
      .def("next_weight_details", &NativeCpuDispatcher::next_weight_details, py::call_guard<py::gil_scoped_release>())
      .def("stats", &NativeCpuDispatcher::stats)
      .def("close", &NativeCpuDispatcher::close, py::call_guard<py::gil_scoped_release>());
#endif
  m.attr("api_version") = 3;
  py::class_<LoadSubmitMigration>(m, "LoadSubmitMigration")
      .def(py::init<py::object, py::object, py::object>())
      .def("build_tasks", &LoadSubmitMigration::build_tasks)
      .def("refresh_scale", &LoadSubmitMigration::refresh_scale)
      .def("copy_batch", &LoadSubmitMigration::copy_batch)
      .def("load", &LoadSubmitMigration::load);
  py::class_<Policy>(m, "Policy")
      .def(py::init<Id, Id, Id, double, double, double, double, double, bool>())
      .def("add_layer", &Policy::add_layer)
      .def("num_layers", &Policy::num_layers)
      .def("step", &Policy::step)
      .def("observe", &Policy::observe)
      .def("plan", &Policy::plan, py::arg("layer_idx"), py::arg("ids").noconvert(), py::arg("scores"),
           py::arg("mapping").noconvert(), py::arg("prefetch"), py::arg("cap"), py::arg("measure"))
      .def("observe_global", &Policy::observe_global)
      .def("choose", &Policy::choose)
      .def("hotness", &Policy::hotness)
      .def("hotness_array", &Policy::hotness_array)
      .def("seed", &Policy::seed)
      .def("snapshot", &Policy::snapshot);
}
