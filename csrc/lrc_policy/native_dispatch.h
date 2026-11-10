// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kt_dispatch_api.h"
#include "native_dma.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <map>
#include <sys/syscall.h>
#include <unistd.h>

struct NativePlan {
  std::vector<Id> mapping;
  std::vector<std::pair<Id, Id>> loads;
  std::vector<std::vector<Id>> ids;
  std::vector<std::vector<double>> weights;
};

// Same admission arithmetic, ordering and LRC state as HostTransferPolicy.
// One instance is shared by eager calls and every captured shape of a layer.
class NativeHostPlanner {
  Id experts_, slots_, layer_, numerator_, denominator_, max_transfers_, npu_fixed_cost_;
  std::string policy_;
  Policy* lrc_;
  std::vector<double> scores_;
  std::mutex mutex_;
 public:
  NativeHostPlanner(Id experts, Id slots, Id layer, Id numerator, Id denominator,
                    Id max_transfers, std::string policy, Policy* lrc,
                    std::vector<double> scores, Id npu_fixed_cost = 0)
      : experts_(experts), slots_(slots), layer_(layer), numerator_(numerator),
        denominator_(denominator), max_transfers_(max_transfers), npu_fixed_cost_(npu_fixed_cost), policy_(policy),
        lrc_(lrc), scores_(std::move(scores)) {
    if (experts < 1 || slots < 0 || slots > experts || denominator < 1 || numerator < 0 ||
        numerator > denominator || max_transfers < 0 || npu_fixed_cost < 0 || scores_.size() != size_t(experts))
      throw std::invalid_argument("Invalid native planner configuration");
    if (policy != "first_miss" && policy != "frequency" && policy != "load_balance")
      throw std::invalid_argument("Unsupported native admission policy");
  }

  Id experts() const { return experts_; }
  Id slots() const { return slots_; }

  NativePlan plan(std::vector<std::vector<Id>> ids,
                  std::vector<std::vector<double>> weights, std::vector<Id> mapping) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mapping.size() != size_t(experts_) || ids.size() != weights.size())
      throw std::invalid_argument("Host route/mapping shape mismatch");
    std::vector<Id> owners(slots_, -1), counts(experts_, 0), order;
    std::unordered_set<Id> protected_ids;
    for (Id e = 0; e < experts_; ++e) {
      Id s = mapping[e];
      if (s < -1 || s >= slots_ || (s >= 0 && owners[s] >= 0))
        throw std::invalid_argument("Invalid or duplicate resident slot");
      if (s >= 0) owners[s] = e;
    }
    for (size_t r = 0; r < ids.size(); ++r) {
      if (ids[r].size() != weights[r].size()) throw std::invalid_argument("Route width mismatch");
      for (size_t c = 0; c < ids[r].size(); ++c) {
        Id e = ids[r][c];
        if (e < 0 || e >= experts_ || weights[r][c] == 0) {
          ids[r][c] = -1; weights[r][c] = 0; continue;
        }
        if (counts[e]++ == 0) order.push_back(e);
        protected_ids.insert(e);
      }
    }
    for (Id e = 0; e < experts_; ++e) scores_[e] = scores_[e] * 0.99 + counts[e];
    std::vector<Id> residents, cold, targets, missing;
    for (Id s = 0; s < slots_; ++s) if (owners[s] < 0) targets.push_back(s);
    // Python owners are inserted in expert-ID order, not slot order.
    for (Id e = 0; e < experts_; ++e) if (mapping[e] >= 0) {
      residents.push_back(e);
      if (!protected_ids.count(e)) cold.push_back(e);
    }
    if (lrc_) {
      std::lock_guard<std::recursive_mutex> state_lock(lrc_->state_mutex);
      lrc_->observe_values(layer_, ids, weights, true);
      cold = lrc_->choose(layer_, residents, protected_ids, cold.size());
    } else {
      std::stable_sort(cold.begin(), cold.end(), [&](Id a, Id b) {
        return scores_[a] == scores_[b] ? a < b : scores_[a] < scores_[b];
      });
    }
    for (Id e : cold) targets.push_back(mapping[e]);
    for (Id e : order) if (mapping[e] < 0) missing.push_back(e);
    Id n = 1;
    if (policy_ == "load_balance") {
      std::stable_sort(missing.begin(), missing.end(), [&](Id a, Id b) {
        if (counts[a] != counts[b]) return counts[a] > counts[b];
        return scores_[a] == scores_[b] ? a < b : scores_[a] > scores_[b];
      });
      // A repeated expert still consumes one CPU computation per token,
      // while uploading that expert's weights is paid only once.
      Id remaining_routes = 0;
      for (Id e : missing) remaining_routes += counts[e];
      const Id copy_unit = denominator_ - numerator_;
      __int128 best_cost = std::max(static_cast<__int128>(remaining_routes) * numerator_,
                                    static_cast<__int128>(npu_fixed_cost_));
      n = 0;
      const Id limit = std::min<Id>({Id(missing.size()), Id(targets.size()), max_transfers_});
      for (Id copied = 1; copied <= limit; ++copied) {
        remaining_routes -= counts[missing[copied - 1]];
        __int128 cpu_cost = static_cast<__int128>(remaining_routes) * numerator_;
        __int128 copy_cost = static_cast<__int128>(copied) * copy_unit + npu_fixed_cost_;
        __int128 cost = std::max(cpu_cost, copy_cost);
        if (cost < best_cost) { best_cost = cost; n = copied; }
      }
    } else if (policy_ == "frequency") {
      std::stable_sort(missing.begin(), missing.end(), [&](Id a, Id b) { return scores_[a] > scores_[b]; });
    }
    NativePlan out{mapping, {}, std::move(ids), std::move(weights)};
    for (Id i = 0; i < std::min<Id>({n, Id(missing.size()), Id(targets.size())}); ++i) {
      Id e = missing[i], slot = targets[i], victim = owners[slot];
      if (policy_ == "frequency" && (scores_[e] < 2.0 ||
          scores_[e] <= (victim >= 0 ? scores_[victim] : 0.0) * 1.2)) continue;
      if (victim >= 0) out.mapping[victim] = -1;
      out.mapping[e] = slot; out.loads.emplace_back(slot, e);
    }
    for (size_t r = 0; r < out.ids.size(); ++r)
      for (size_t c = 0; c < out.ids[r].size(); ++c) {
        Id e = out.ids[r][c];
        if (e < 0 || out.mapping[e] >= 0) { out.ids[r][c] = -1; out.weights[r][c] = 0; }
      }
    return out;
  }

  // Predictions read the target layer's history; they are NOT observations.
  NativePlan plan_prefetch(const std::vector<std::vector<Id>>& ids,
                           const std::vector<std::vector<double>>& weights,
                           std::vector<Id> mapping, Id cap) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cap < 1 || mapping.size() != size_t(experts_) || ids.size() != weights.size())
      throw std::invalid_argument("Invalid native prefetch shape/cap");
    std::vector<Id> owners(slots_, -1), residents, targets, candidates, counts(experts_, 0);
    std::vector<double> best(experts_, -std::numeric_limits<double>::infinity());
    std::unordered_set<Id> predicted;
    for (Id e = 0; e < experts_; ++e) {
      Id slot = mapping[e];
      if (slot < -1 || slot >= slots_ || (slot >= 0 && owners[slot] >= 0))
        throw std::invalid_argument("Invalid or duplicate resident slot");
      if (slot >= 0) { owners[slot] = e; residents.push_back(e); }
    }
    for (size_t r = 0; r < ids.size(); ++r) {
      if (ids[r].size() != weights[r].size()) throw std::invalid_argument("Route width mismatch");
      for (size_t c = 0; c < ids[r].size(); ++c) {
        Id e = ids[r][c];
        if (e < 0 || e >= experts_) continue;
        predicted.insert(e);
        if (counts[e]++ == 0 && mapping[e] < 0) candidates.push_back(e);
        best[e] = std::max(best[e], weights[r][c]);
      }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](Id a, Id b) {
      if (counts[a] != counts[b]) return counts[a] > counts[b];
      return best[a] > best[b];
    });
    for (Id slot = 0; slot < slots_; ++slot) if (owners[slot] < 0) targets.push_back(slot);
    std::vector<Id> cold;
    if (lrc_) cold = lrc_->choose(layer_, residents, predicted, cap);
    else for (Id e : residents) if (!predicted.count(e)) cold.push_back(e);
    for (Id e : cold) targets.push_back(mapping[e]);
    NativePlan out{std::move(mapping), {}, ids, weights};
    for (Id i = 0; i < std::min<Id>({cap, Id(candidates.size()), Id(targets.size())}); ++i) {
      Id slot = targets[i], victim = owners[slot], e = candidates[i];
      if (victim >= 0) out.mapping[victim] = -1;
      out.mapping[e] = slot; out.loads.emplace_back(slot, e);
    }
    return out;
  }

  std::vector<std::pair<Id, Id>> prepare(uintptr_t ids, uintptr_t weights, uintptr_t mapping,
                                       Id tokens, Id topk, int id_size, int map_size) {
    return prepare_impl(ids, weights, mapping, tokens, topk, id_size, map_size, -1);
  }
  std::vector<std::pair<Id, Id>> prepare_prefetch(uintptr_t ids, uintptr_t weights, uintptr_t mapping,
                                       Id tokens, Id topk, int id_size, int map_size, Id cap) {
    if (cap < 1) throw std::invalid_argument("Prefetch cap must be positive");
    return prepare_impl(ids, weights, mapping, tokens, topk, id_size, map_size, cap);
  }
  std::vector<std::pair<Id, Id>> prepare_impl(uintptr_t ids, uintptr_t weights, uintptr_t mapping,
                                       Id tokens, Id topk, int id_size, int map_size, Id prefetch_cap) {
    if (!ids || !weights || !mapping || tokens < 1 || topk < 1 ||
        (id_size != 4 && id_size != 8) || (map_size != 4 && map_size != 8))
      throw std::invalid_argument("Invalid native pinned route buffers");
    auto read = [](uintptr_t p, Id i, int size) -> Id {
      return size == 4 ? read_scalar<int32_t>(reinterpret_cast<void*>(p + i * size))
                       : read_scalar<int64_t>(reinterpret_cast<void*>(p + i * size));
    };
    auto write = [](uintptr_t p, Id i, int size, Id value) {
      if (size == 4) { auto v = int32_t(value); std::memcpy(reinterpret_cast<void*>(p + i * size), &v, size); }
      else std::memcpy(reinterpret_cast<void*>(p + i * size), &value, size);
    };
    std::vector<std::vector<Id>> rows(tokens, std::vector<Id>(topk));
    std::vector<std::vector<double>> scores(tokens, std::vector<double>(topk));
    std::vector<Id> map(experts_);
    for (Id e = 0; e < experts_; ++e) map[e] = read(mapping, e, map_size);
    for (Id r = 0; r < tokens; ++r) for (Id c = 0; c < topk; ++c) {
      rows[r][c] = read(ids, r * topk + c, id_size);
      scores[r][c] = read_scalar<float>(reinterpret_cast<void*>(weights + (r * topk + c) * sizeof(float)));
    }
    auto out = prefetch_cap >= 0 ? plan_prefetch(rows, scores, std::move(map), prefetch_cap)
                                : plan(std::move(rows), std::move(scores), std::move(map));
    for (Id e = 0; e < experts_; ++e) write(mapping, e, map_size, out.mapping[e]);
    if (prefetch_cap < 0) for (Id r = 0; r < tokens; ++r) for (Id c = 0; c < topk; ++c) {
      write(ids, r * topk + c, id_size, out.ids[r][c]);
      float w = out.weights[r][c];
      std::memcpy(reinterpret_cast<void*>(weights + (r * topk + c) * sizeof(float)), &w, sizeof(float));
    }
    return out.loads;
  }
};

class NativeCpuDispatcher {
  static constexpr unsigned kMaxChannels = 4096;
  struct Channel {
    uint32_t* flag; uint32_t seen = 0;
    bool prefetch = false; Id prefetch_cap = 0;
    Id dependency = -1;
    uintptr_t cpu, entry, args, ids, weights, mapping, ready, weight_ready;
    Id tokens, topk, layer; int id_size, map_size;
    std::shared_ptr<NativeHostPlanner> planner;
    std::shared_ptr<const NativeDmaLayer> dma_layer;
    std::atomic<bool> dma_bound{false};
    uintptr_t mapping_device = 0; size_t mapping_bytes = 0;
  };
  struct Completion { Channel* channel; uint64_t task; };
  using WeightJob = std::pair<unsigned, std::vector<std::pair<Id, Id>>>;
  const KtDispatchApi* api_;
  std::array<std::unique_ptr<Channel>, kMaxChannels> channels_;
  std::atomic<unsigned> count_{0};
  std::atomic<bool> alive_{true};
  std::atomic<uint64_t> submitted_{0}, completed_{0}, prefetched_{0}, prefetch_loads_{0};
  std::mutex registration_, completion_mutex_, weight_mutex_;
  std::condition_variable completion_cv_, weight_cv_;
  std::deque<Completion> completions_;
  struct TimedWeightJob { WeightJob job; uint64_t task, enqueued_ns; };
  std::deque<TimedWeightJob> weights_;
  std::thread poll_, completion_, dma_;
  std::array<std::shared_ptr<const NativeDmaLayer>, kMaxChannels> dma_layers_;
  unsigned dma_layer_count_ = 0;
  std::atomic<bool> native_dma_{false}, draining_{false};
  std::atomic<uint64_t> dma_jobs_{0}, dma_experts_{0}, dma_fields_{0};
  uintptr_t dma_stream_ = 0;
  int (*memcpy_async_)(void*, size_t, const void*, size_t, int, void*) = nullptr;
  int (*synchronize_stream_)(void*) = nullptr;
  std::mutex close_mutex_;
  bool closed_ = false;
  static constexpr int kHostToDevice = 1;
  static constexpr size_t kDmaTraceCapacity = 4096;
  using DmaTrace = std::array<uint64_t, 14>;
  std::array<DmaTrace, kDmaTraceCapacity> dma_trace_{};
  std::atomic<bool> dma_trace_enabled_{false};
  std::atomic<size_t> dma_trace_size_{0}, dma_trace_active_{0}, dma_trace_dropped_{0};
  std::mutex dma_trace_control_;
  void* acl_lib_ = nullptr; void* context_ = nullptr; uintptr_t stream_, weight_stream_;
  int (*set_context_)(void*) = nullptr;
  // Torch NPU 2.10 ExternalEvent is a 32-byte device value, not an aclrtEvent.
  // Publish 1 using the same aclrtValueWrite protocol as Event.record().
  int (*record_event_)(void*, uint64_t, uint32_t, void*) = nullptr;

  static void bind(int cpu, const char* name, bool explicit_dma_core = false) {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (cpu < 0 || cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) || (!explicit_dma_core && !CPU_ISSET(cpu, &allowed)))
      throw std::runtime_error("Native dispatcher CPU is outside inherited affinity");
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) throw std::runtime_error("Native dispatcher affinity failed");
    pthread_setname_np(pthread_self(), name);
  }
  [[noreturn]] static void fail(const std::exception& e) {
    std::fprintf(stderr, "Native CPU MoE dispatcher failed: %s\n", e.what());
    // Never release an NPU consumer with stale pinned output.
    std::abort();
  }
  static const char* demand_transfer_phase(size_t count) {
    // Trace records retain the phase pointer until asynchronous flush.
    static const auto names = [] {
      std::array<std::string, kMaxChannels + 1> result;
      for (size_t i = 0; i < result.size(); ++i) result[i] = "demand_transfers_" + std::to_string(i);
      return result;
    }();
    return count < names.size() ? names[count].c_str() : "demand_transfers_unknown";
  }
  static uint64_t clock_ns(clockid_t clock) {
    timespec t; clock_gettime(clock, &t);
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
  }
  void queue_weight(unsigned index, std::vector<std::pair<Id, Id>> loads, uint64_t task) {
    { std::lock_guard<std::mutex> l(weight_mutex_);
      weights_.push_back({{index, std::move(loads)}, task, clock_ns(CLOCK_MONOTONIC)}); }
    weight_cv_.notify_one();
  }
  void poll(int cpu) {
    try {
      bind(cpu, "kt_native_poll");
      if (set_context_(context_)) throw std::runtime_error("Cannot set ACL context on input thread");
      while (alive_.load(std::memory_order_acquire)) {
        unsigned n = count_.load(std::memory_order_acquire);
        // Demand notifications take priority over speculative planning.
        for (int pass = 0; pass < 2; ++pass) for (unsigned i = 0; i < n; ++i) {
          Channel& c = *channels_[i];
          if (int(c.prefetch) != pass) continue;
          uint32_t generation = __atomic_load_n(c.flag, __ATOMIC_ACQUIRE);
          if (generation == c.seen) continue;
          if (generation != uint32_t(c.seen + 1)) throw std::runtime_error("NPU input sequence skipped");
          // Both captured channels advance once per replay. A flag may become
          // visible between the demand and prefetch scans, so priority alone
          // does not enforce FIFO. Admit this generation's demand first.
          if (c.prefetch && c.dependency >= 0 &&
              channels_[c.dependency]->seen != generation) continue;
          c.seen = generation;
          if (c.prefetch) {
            auto loads = c.planner->prepare_prefetch(c.ids, c.weights, c.mapping,
                c.tokens, c.topk, c.id_size, c.map_size, c.prefetch_cap);
            prefetched_.fetch_add(1, std::memory_order_relaxed);
            prefetch_loads_.fetch_add(loads.size(), std::memory_order_relaxed);
            if (loads.empty()) {
              if (record_event_(reinterpret_cast<void*>(c.weight_ready), 1, 0,
                                reinterpret_cast<void*>(weight_stream_)))
                throw std::runtime_error("Native prefetch-ready record failed");
            } else {
              queue_weight(i, std::move(loads), 0);
            }
            continue;
          }
          // The historical callback_enter marker now means native input observed.
          // Begin before admission so the full CPU startup path is measurable.
          uint64_t task = api_->begin(c.layer);
          auto loads = c.planner->prepare(c.ids, c.weights, c.mapping, c.tokens, c.topk, c.id_size, c.map_size);
          if (task) api_->mark(demand_transfer_phase(loads.size()));
          api_->mark("admission_end");
          api_->submit(c.cpu, c.entry, c.args);
          api_->mark("submit_return");
          submitted_.fetch_add(1, std::memory_order_relaxed);
          { std::lock_guard<std::mutex> l(completion_mutex_); completions_.push_back({&c, task}); }
          completion_cv_.notify_one();
          api_->mark("completion_enqueued");
          // No admission means the device mapping is unchanged: no Python,
          // no redundant mapping copy, and no dependency on CPU completion.
          if (loads.empty()) {
            if (record_event_(reinterpret_cast<void*>(c.weight_ready), 1, 0, reinterpret_cast<void*>(weight_stream_)))
              throw std::runtime_error("Native weight-ready record failed");
            api_->mark("dma_ready_recorded");
          } else {
            queue_weight(i, std::move(loads), task);
          }
        }
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
      }
    } catch (const std::exception& e) { fail(e); }
  }
  void dma(int cpu) {
    try {
      // DMA may intentionally use a separate NUMA node's physical core. The
      // syscall still enforces the actual cgroup/cpuset restrictions.
      bind(cpu, "kt_native_dma", true);
      if (set_context_(context_)) throw std::runtime_error("Cannot set ACL context on DMA thread");
      for (;;) {
        TimedWeightJob item;
        {
          std::unique_lock<std::mutex> lock(weight_mutex_);
          weight_cv_.wait(lock, [&] { return !weights_.empty() || draining_.load(); });
          if (weights_.empty()) break;
          item = std::move(weights_.front()); weights_.pop_front();
        }
        dma_trace_active_.fetch_add(1, std::memory_order_acq_rel);
        const bool timed = dma_trace_enabled_.load(std::memory_order_acquire);
        DmaTrace timing{};
        if (timed) timing = {item.job.first, item.task, 0, 0, item.job.second.size(), item.enqueued_ns,
                            clock_ns(CLOCK_MONOTONIC), 0, 0, 0, 0,
                            uint64_t(syscall(SYS_gettid)), clock_ns(CLOCK_THREAD_CPUTIME_ID), 0};
        const Channel& c = *channels_[item.job.first];
        if (timed) { timing[2] = c.layer; timing[3] = c.prefetch; }
        if (!c.dma_bound.load(std::memory_order_acquire))
          throw std::runtime_error("Native DMA channel has no registered byte buffers");
        auto copy = [&](uintptr_t dst, uintptr_t src, size_t bytes) {
          if (timed && !timing[7]) timing[7] = clock_ns(CLOCK_MONOTONIC);
          int error = memcpy_async_(reinterpret_cast<void*>(dst), bytes,
              reinterpret_cast<const void*>(src), bytes, kHostToDevice,
              reinterpret_cast<void*>(dma_stream_));
          if (error) throw std::runtime_error("Native aclrtMemcpyAsync failed: " + std::to_string(error));
        };
        // Preserve per-expert pipelining and FIFO: submit all fields of one
        // expert immediately, then the next. Never synchronize each expert.
        for (const auto& load : item.job.second) {
          if (load.first < 0 || size_t(load.first) >= c.dma_layer->slots ||
              load.second < 0 || size_t(load.second) >= c.dma_layer->experts)
            throw std::runtime_error("Native DMA expert/slot outside registered table");
          for (const auto& field : c.dma_layer->fields) {
            copy(field.destinations[load.first], field.sources[load.second], field.bytes);
            dma_fields_.fetch_add(1, std::memory_order_relaxed);
          }
        }
        if (timed) timing[8] = clock_ns(CLOCK_MONOTONIC);
        copy(c.mapping_device, c.mapping, c.mapping_bytes);
        if (timed) timing[9] = clock_ns(CLOCK_MONOTONIC);
        // This queues the same value-based Torch NPU external-ready protocol
        // after weights, scale and mapping on the very same DMA stream.
        if (record_event_(reinterpret_cast<void*>(c.weight_ready), 1, 0,
                          reinterpret_cast<void*>(dma_stream_)))
          throw std::runtime_error("Native DMA weight-ready publication failed");
        dma_experts_.fetch_add(item.job.second.size(), std::memory_order_relaxed);
        dma_jobs_.fetch_add(1, std::memory_order_release);
        if (timed) {
          timing[10] = clock_ns(CLOCK_MONOTONIC); timing[13] = clock_ns(CLOCK_THREAD_CPUTIME_ID);
          size_t index = dma_trace_size_.load(std::memory_order_relaxed);
          if (index < kDmaTraceCapacity) {
            dma_trace_[index] = timing; dma_trace_size_.store(index + 1, std::memory_order_release);
          } else dma_trace_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        dma_trace_active_.fetch_sub(1, std::memory_order_release);
      }
    } catch (const std::exception& e) { fail(e); }
  }
  void complete(int cpu) {
    try {
      bind(cpu, "kt_native_done");
      if (set_context_(context_)) throw std::runtime_error("Cannot set ACL context on completion thread");
      for (;;) {
        Completion item;
        {
          std::unique_lock<std::mutex> l(completion_mutex_);
          completion_cv_.wait(l, [&] { return !completions_.empty() || draining_.load(); });
          if (completions_.empty()) break;
          item = completions_.front(); completions_.pop_front();
        }
        Channel& c = *item.channel;
        api_->context(item.task, c.layer);
        api_->mark("completion_wait_enter");
        api_->sync(c.cpu);
        api_->mark("completion_wait_end");
        if (record_event_(reinterpret_cast<void*>(c.ready), 1, 0, reinterpret_cast<void*>(stream_)))
          throw std::runtime_error("Native ACL external output-ready record failed");
        api_->mark("output_ready_recorded");
        completed_.fetch_add(1, std::memory_order_release);
      }
    } catch (const std::exception& e) { fail(e); }
  }
 public:
  NativeCpuDispatcher(py::capsule api, uintptr_t stream, uintptr_t weight_stream, int poll_cpu, int completion_cpu) : stream_(stream), weight_stream_(weight_stream) {
#if !defined(__x86_64__)
    throw std::runtime_error("Native NPU-to-CPU dispatch currently requires coherent x86 host memory");
#endif
    api_ = static_cast<const KtDispatchApi*>(api.get_pointer());
    if (!api_ || api_->version != 1 || !api_->submit || !api_->sync || !api_->begin || !api_->context || !api_->mark)
      throw std::invalid_argument("kt-kernel native dispatch API mismatch");
    acl_lib_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (!acl_lib_) throw std::runtime_error("Cannot load libascendcl.so");
    auto get = reinterpret_cast<int (*)(void**)>(dlsym(acl_lib_, "aclrtGetCurrentContext"));
    set_context_ = reinterpret_cast<int (*)(void*)>(dlsym(acl_lib_, "aclrtSetCurrentContext"));
    record_event_ = reinterpret_cast<int (*)(void*, uint64_t, uint32_t, void*)>(dlsym(acl_lib_, "aclrtValueWrite"));
    if (!get || !set_context_ || !record_event_ || !stream || !weight_stream || get(&context_) || !context_)
      throw std::runtime_error("Native dispatcher requires an initialized ACL context and value-based external events");
    demand_transfer_phase(0);  // Allocate stable trace labels before monitoring starts.
    poll_ = std::thread([this, poll_cpu] { poll(poll_cpu); });
    completion_ = std::thread([this, completion_cpu] { complete(completion_cpu); });
  }
  ~NativeCpuDispatcher() {
    try { close(); } catch (const std::exception& e) { fail(e); }
    if (acl_lib_) dlclose(acl_lib_);
  }
  void configure_dma(int cpu, uintptr_t stream) {
    std::lock_guard<std::mutex> lock(registration_);
    if (!stream || cpu < 0 || cpu >= CPU_SETSIZE || count_.load() || native_dma_.load() || !alive_.load())
      throw std::invalid_argument("Configure native DMA before channel registration");
    memcpy_async_ = reinterpret_cast<decltype(memcpy_async_)>(dlsym(acl_lib_, "aclrtMemcpyAsync"));
    synchronize_stream_ = reinterpret_cast<decltype(synchronize_stream_)>(dlsym(acl_lib_, "aclrtSynchronizeStream"));
    if (!memcpy_async_ || !synchronize_stream_)
      throw std::runtime_error("CANN asynchronous memcpy/stream synchronization API unavailable");
    dma_stream_ = stream;
    native_dma_.store(true, std::memory_order_release);
    dma_ = std::thread([this, cpu] { dma(cpu); });
  }
  unsigned add_dma_layer(std::vector<std::vector<uintptr_t>> sources,
                         std::vector<std::vector<uintptr_t>> destinations,
                         std::vector<size_t> bytes, size_t experts, size_t slots) {
    auto layer = std::make_shared<NativeDmaLayer>(std::move(sources), std::move(destinations),
                                                std::move(bytes), experts, slots);
    std::lock_guard<std::mutex> lock(registration_);
    if (!native_dma_.load() || !alive_.load() || dma_layer_count_ == kMaxChannels)
      throw std::runtime_error("Native DMA layer registration is closed/full");
    unsigned index = dma_layer_count_++;
    dma_layers_[index] = std::move(layer); return index;
  }
  void bind_dma_channel(unsigned channel, unsigned layer, uintptr_t mapping_device, size_t bytes) {
    std::lock_guard<std::mutex> lock(registration_);
    if (!alive_.load() || !native_dma_.load() || channel >= count_.load() || layer >= dma_layer_count_ ||
        !mapping_device || bytes != dma_layers_[layer]->experts * channels_[channel]->map_size ||
        dma_layers_[layer]->experts != size_t(channels_[channel]->planner->experts()) ||
        dma_layers_[layer]->slots != size_t(channels_[channel]->planner->slots()))
      throw std::invalid_argument("Invalid native DMA channel mapping");
    auto& c = *channels_[channel];
    if (c.dma_layer) throw std::invalid_argument("Native DMA channel already bound");
    c.mapping_device = mapping_device; c.mapping_bytes = bytes; c.dma_layer = dma_layers_[layer];
    c.dma_bound.store(true, std::memory_order_release);
  }
  void set_dma_capture(bool enabled) {
    std::lock_guard<std::mutex> lock(dma_trace_control_);
    dma_trace_enabled_.store(false, std::memory_order_release);
    while (dma_trace_active_.load(std::memory_order_acquire)) std::this_thread::yield();
    if (enabled) {
      dma_trace_size_.store(0); dma_trace_dropped_.store(0);
      dma_trace_enabled_.store(true, std::memory_order_release);
    }
  }
  std::vector<std::map<std::string, uint64_t>> dma_timings() const {
    if (dma_trace_enabled_.load())
      throw std::runtime_error("Stop native DMA timing before reading records");
    static const std::array<const char*, 14> names{{"channel", "task_id", "layer", "prefetch", "experts",
      "cpp_enqueue_ns", "cpp_dequeue_ns", "first_copy_call_enter_ns", "expert_fields_return_ns",
      "mapping_return_ns", "ready_record_return_ns", "native_tid", "dequeue_cpu_ns", "return_cpu_ns"}};
    std::vector<std::map<std::string, uint64_t>> result;
    size_t n = dma_trace_size_.load(std::memory_order_acquire); result.reserve(n + 1);
    for (size_t i = 0; i < n; ++i) {
      std::map<std::string, uint64_t> row;
      for (size_t j = 0; j < names.size(); ++j) row.emplace(names[j], dma_trace_[i][j]);
      result.push_back(std::move(row));
    }
    result.push_back({{"written", n}, {"dropped", dma_trace_dropped_.load()}});
    return result;
  }
  std::tuple<uint64_t, uint64_t, uint64_t> dma_stats() const {
    return {dma_jobs_.load(std::memory_order_acquire), dma_experts_.load(), dma_fields_.load()};
  }
  unsigned add(uintptr_t flag, uintptr_t cpu, std::pair<uintptr_t, uintptr_t> task,
               uintptr_t ids, uintptr_t weights, uintptr_t mapping, uintptr_t ready, uintptr_t weight_ready,
               Id tokens, Id topk, Id layer, int id_size, int map_size,
               std::shared_ptr<NativeHostPlanner> planner) {
    if (!flag || flag % alignof(uint32_t) || !cpu || !task.first || !task.second || !ids || !weights || !mapping || !ready || !weight_ready || !planner ||
        tokens < 1 || topk < 1 || (id_size != 4 && id_size != 8) || (map_size != 4 && map_size != 8))
      throw std::invalid_argument("Invalid native task registration");
    std::lock_guard<std::mutex> l(registration_);
    unsigned i = count_.load();
    if (i == kMaxChannels || !alive_.load()) throw std::runtime_error("Native task registration is closed/full");
    auto c = std::make_unique<Channel>();
    c->flag = reinterpret_cast<uint32_t*>(flag); c->seen = __atomic_load_n(c->flag, __ATOMIC_ACQUIRE);
    c->cpu = cpu; c->entry = task.first; c->args = task.second; c->ids = ids; c->weights = weights;
    c->mapping = mapping; c->ready = ready; c->weight_ready = weight_ready; c->tokens = tokens; c->topk = topk; c->layer = layer;
    c->id_size = id_size; c->map_size = map_size; c->planner = std::move(planner);
    channels_[i] = std::move(c); count_.store(i + 1, std::memory_order_release); return i;
  }
  unsigned add_prefetch(uintptr_t flag, uintptr_t ids, uintptr_t weights, uintptr_t mapping,
                        uintptr_t ready, Id tokens, Id topk, Id layer, int id_size, int map_size,
                        Id cap, std::shared_ptr<NativeHostPlanner> planner) {
    return add_prefetch_after(flag, ids, weights, mapping, ready, tokens, topk,
                              layer, id_size, map_size, cap, std::move(planner), -1);
  }
  unsigned add_prefetch_after(uintptr_t flag, uintptr_t ids, uintptr_t weights, uintptr_t mapping,
                        uintptr_t ready, Id tokens, Id topk, Id layer, int id_size, int map_size,
                        Id cap, std::shared_ptr<NativeHostPlanner> planner, Id dependency) {
    if (!flag || flag % alignof(uint32_t) || !ids || !weights || !mapping || !ready || !planner ||
        tokens < 1 || topk < 1 || cap < 1 || (id_size != 4 && id_size != 8) || (map_size != 4 && map_size != 8))
      throw std::invalid_argument("Invalid native prefetch registration");
    std::lock_guard<std::mutex> l(registration_);
    unsigned i = count_.load();
    if (i == kMaxChannels || !alive_.load()) throw std::runtime_error("Native task registration is closed/full");
    auto c = std::make_unique<Channel>();
    c->flag = reinterpret_cast<uint32_t*>(flag); c->seen = __atomic_load_n(c->flag, __ATOMIC_ACQUIRE);
    if (dependency < -1 || dependency >= Id(i) ||
        (dependency >= 0 && channels_[dependency]->prefetch))
      throw std::invalid_argument("Prefetch dependency must name an existing demand channel");
    c->dependency = dependency;
    c->prefetch = true; c->prefetch_cap = cap; c->ids = ids; c->weights = weights; c->mapping = mapping;
    c->weight_ready = ready; c->tokens = tokens; c->topk = topk; c->layer = layer;
    c->id_size = id_size; c->map_size = map_size; c->planner = std::move(planner);
    channels_[i] = std::move(c); count_.store(i + 1, std::memory_order_release); return i;
  }
  std::pair<uint64_t, uint64_t> prefetch_stats() const { return {prefetched_.load(), prefetch_loads_.load()}; }
  WeightJob next_weight() {
    if (native_dma_.load()) throw std::runtime_error("Python DMA consumer disabled in native DMA mode");
    std::unique_lock<std::mutex> l(weight_mutex_);
    weight_cv_.wait(l, [&] { return !weights_.empty() || !alive_.load(); });
    if (weights_.empty()) return {kMaxChannels, {}};
    auto item = std::move(weights_.front()); weights_.pop_front(); return std::move(item.job);
  }
  std::tuple<unsigned, std::vector<std::pair<Id, Id>>, uint64_t, uint64_t, uint64_t, uint64_t> next_weight_details() {
    if (native_dma_.load()) throw std::runtime_error("Python DMA consumer disabled in native DMA mode");
    std::unique_lock<std::mutex> l(weight_mutex_);
    weight_cv_.wait(l, [&] { return !weights_.empty() || !alive_.load(); });
    if (weights_.empty()) return {kMaxChannels, {}, 0, 0, 0, 0};
    auto item = std::move(weights_.front()); weights_.pop_front();
    return {item.job.first, std::move(item.job.second), item.task, item.enqueued_ns,
            clock_ns(CLOCK_MONOTONIC), clock_ns(CLOCK_THREAD_CPUTIME_ID)};
  }
  void close() {
    std::lock_guard<std::mutex> close_lock(close_mutex_);
    if (closed_) return;
    alive_.store(false, std::memory_order_release);
    if (poll_.joinable()) poll_.join();
    draining_.store(true, std::memory_order_release);
    completion_cv_.notify_all(); weight_cv_.notify_all();
    if (completion_.joinable()) completion_.join();
    if (dma_.joinable()) dma_.join();
    if (native_dma_.load() && synchronize_stream_(reinterpret_cast<void*>(dma_stream_)))
      throw std::runtime_error("Cannot drain native DMA stream before freeing buffers");
    closed_ = true;
  }
  std::pair<uint64_t, uint64_t> stats() const { return {submitted_.load(), completed_.load()}; }
};
