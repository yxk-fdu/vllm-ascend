# CPU LRC and DMA for hybrid experts

> **Draft / 草稿版。** The cumulative CPU MoE changes on this branch have
> yet to be verified on Ascend hardware. This draft is for review and hardware validation.

The single-card hybrid path uses CPU admission and LRC, host-submitted weight
DMA, and mapped-host CPU result combine. Native dispatch supports optional
next-layer prefetch. Remove `backend`, `transfer_copy_library`,
`transfer_trace_capacity`, and `transfer_trace_mode` from old CPU MoE
configurations. Remove `transfer_policy` and `transfer_scheduler` as well. The CPU MoE host planner always balances CPU work and weight DMA.

CPU MoE prefetch supports `expert_predictor="fate"`. Selecting a trained
predictor with CPU MoE prefetch enabled fails configuration validation.

## Runtime dependencies

Build the CPU-only LRC extension before starting the service:

```bash
python -m pip install pybind11 setuptools wheel
python -m pip install ./csrc/lrc_policy --no-build-isolation --no-deps
```

The C++ implementation is adapted from
[haodi19/vllm-ascend at 268b963c](https://github.com/haodi19/vllm-ascend/tree/268b963ca365b03fccd70d41e49ce4bca1e852fc/csrc/lrc_policy).
Its existing hotness, protected-expert and deterministic victim ordering are
retained. Host hybrid admission uses this implementation when the cache policy
is enabled. Native NPU-only paging also uses this policy when enabled.

Graph CPU submission polls an ordered D2H sequence and starts the C++
dispatcher with the CPU backend. The installed kt-kernel must expose the
native dispatch ABI and pinned input/output buffers. Mapped combine and
value-based `ExternalEvent` support remain required.

Select `h2d_backend="torch"` for the host transfer transport. Shared NZ
graph replay submits the prepared DMA copies through the native dispatcher.
Set `cpu_moe.max_num_tokens` to cover every captured decode/verification batch;
capture sizes are token counts. For single-card MRV1 full-decode-only mode, the
worker checks the requested sizes before loading weights, including speculative
decode alignment and the scheduler's sequence limit. MRV1 also validates the
resolved target-model graph descriptors before capture. Eager batches above the
CPU MoE capacity retain ordinary paging.

### NPU-only decode with native DMA

For a single A5 device with the V1 model runner and W4A8MXFP experts, set
`expert_offload_config.native_expert_dma=true` and `cpu_moe.enabled=false`.
This uses the same native planner and DMA queue as hybrid graph execution.
All required misses go to the NPU. There is no kt-kernel runtime or CPU FFN
completion event in this mode. Build the value-wait extension as well:

```bash
(cd csrc/lrc_policy && python setup_npu_wait.py build_ext --inplace)
export PYTHONPATH="$PWD/csrc/lrc_policy${PYTHONPATH:+:$PYTHONPATH}"
```

Use `h2d_backend="torch"` and FATE, with route substitution and pruning
disabled. Both captured and eager prefetch work queue behind current-layer
demand via the native channel dependency. A batch whose
token count exceeds the single-card paging limit (`min(device_slots) // topk`)
uses the eager full-layer pool. The option defaults to false.

`KT_MOE_POLL_CPU` and `cpu_moe.native_dispatch_dma_cpu` can pin the two
threads. By default the runtime selects distinct CPUs from the process
affinity. The dispatcher exposes separate demand and prefetch counters.

### CPU and NUMA affinity

CPU MoE startup requires a kt-kernel build that exports
`available_worker_pool_cpus` and accepts explicit placement in
`WorkerPoolConfig.subpool_cpus`. Before loading weights, the CPU MoE worker
resolves the bound NPU's physical device ID, reads `npu-smi info -t topo`,
and maps its CPU affinity to a NUMA node using Linux sysfs. If that node can
support the compute pool, it is preferred for compute, shared weights and
DMA submission. The planner reserves DMA first, then poll and completion.
It moves service roles off-node as needed to preserve the requested compute
thread count, or at least one compute core when no count was specified.

When the topology is missing, ambiguous, or the device-local node cannot
satisfy the configured placement, the worker selects the allowed NUMA node
with the most compute cores after reserving services. This fallback uses
service cores outside the compute pool first. Set `cpuinfer_threads` or
`numa_nodes` to override the automatic thread count or node.

Omit `KT_MOE_POLL_CPU`, `KT_MOE_COMPLETION_CPU`, and
`cpu_moe.native_dispatch_dma_cpu` for automatic service placement. Explicit role assignments
use Linux logical CPU IDs. Explicit DMA placement may use a core outside the
inherited affinity if the worker's effective cpuset permits it. Compute and
service roles take distinct physical cores, including when SMT is enabled.
Shared weights prefer the selected compute node unless
`cpu_moe.host_weight_numa_node` overrides it. When a container blocks memory
policy syscalls, automatic placement falls back to allocating from a CPU on
the selected node. The worker logs the plan before loading weights. The reference CPU MoE
launcher sets `enable_cpu_binding=false` to use the CPU MoE thread bindings
without the ARM worker's post-warmup process-wide binding.

### Shared NZ CPU weights

Use vLLM 0.30.0 and the matching kt-kernel build with `NZMOE` and the
affinity preflight export above. `cpu_moe.enabled=true` attaches the shared
NZ backend to every eligible MoE layer and requires one NUMA pool. Place the
shared weight buffers and compute pool on the same NUMA node.
Set `KT_MOE_MATCH_NPU_MXFP8=1` before loading the backend. The graph dispatcher
uses the CPU backend's native ABI directly. The ARM shared NZ path was
verified on Ascend 950DT with CANN 9.2.1. On ARM, graph results are written
into a 64 KiB-aligned, ACL-registered host buffer that the NPU reads directly.
ARM eager execution uses the existing non-blocking result H2D merge.

The offload manager converts expert weights and scales during model loading
and owns those buffers. It passes the four finalized fields to kt-kernel's
`NZMOE`. Native registration verifies that the CPU backend and DMA use the
same addresses. Graph replay submits CPU work and weight DMA through C++ and
signals each completion on its own stream.

## Per-layer execution

1. NPU routing produces final logical expert IDs and route weights. Invalid and
   pruned routes have no compute owner.
2. The input stream resets the output and weight external events, copies the
   current mapping and complete CPU inputs, then publishes the next sequence.
3. The input poller observes that sequence. CPU admission updates C++ LRC once,
   chooses which missing experts to upload, and protects all current routed
   resident experts from eviction. Free slots are used before cold victims.
4. The planner masks kt-kernel's pinned inputs to the remaining CPU routes and
   enqueues a weight request. CPU compute is independently enqueued on the
   existing submit worker; that worker never waits for weight DMA.
5. The DMA worker uploads selected experts and quantization attributes, writes
   back the mapping, and records **weights_ready on its own DMA stream**.
6. NPU route splitting/mapping and expert compute wait for weights_ready only.
7. The independent CPU drain worker records **cpu_output_ready on its existing
   completion stream** immediately after CPU output writes finish. It does not
   wait for DMA, mapping, or NPU expert computation.
8. Combine waits for CPU output readiness and reads pinned host results directly.
   There is no separate CPU-result H2D staging copy.

```text
input ready -> CPU admission/LRC
                 |                     |
                 v                     v
            weight DMA             CPU computation
            mapping H2D            output writes
                 |                     |
           weights_ready          cpu_output_ready
                 |                     |
            NPU computation            |
                 +-------- combine ----+
```

There is no ordering between weight completion and CPU output completion.
The first version computes the NPU-owned routes together after weights_ready;
it does not split existing hits and newly uploaded experts into two NPU calls.
The planner ranks missing experts by the number of routed tokens and checks
each candidate transfer count up to `balance_max_transfers` and the available
slots. It minimizes the larger of estimated CPU time and the estimated DMA
time plus NPU layer cost. Startup calibration measures one CPU route, two
routes to the same expert, and the H2D cost of one expert. The planner charges
first and additional routes separately. LRC ranks eviction victims independently.

## Graph and lifetime rules

Each captured site owns a distinct pinned mapping and weight event. Its reset,
input D2H, sequence publication, and external waits are graph nodes, replayed
for each invocation. CPU planning executes on each observed sequence, never
once at capture time. Site objects remain owned by the backend's graph tasks.
Eager contexts retain their pinned buffers across asynchronous H2D. Draft/MTP
layers registered after finalization use their manager ordinal for LRC/weight
storage and their model layer index for CPUInfer.

Python DMA worker initialization errors fail startup. If an accepted DMA
request fails on the Python worker or native dispatcher, the worker process
exits and leaves the failed request's weight event unsignalled. The service
supervisor must replace the failed worker before forwarding requests again.
Shutdown drains live device work while poll, compute and DMA workers are alive,
then retires DMA resources.

## Validation

CPU regression tests include route conservation, protected slots, free slots,
pruned routes, repeated plans, Python/C++ LRC parity, deferred graph planning,
per-site events, and both possible weight/output completion orders:

```bash
pytest -q tests/ut/expert_offload/test_host_hybrid_transfer.py \
  tests/ut/expert_offload/test_lrc_policy.py \
  tests/ut/expert_offload/test_hybrid_executor.py \
  tests/ut/expert_offload/test_cpu_moe_wiring.py \
  tests/ut/expert_offload/test_nz_mxfp4_backend.py \
  tests/ut/expert_offload/test_shared_nz_owner.py \
  tests/ut/expert_offload/test_cpu_weight_load_order.py
```

On an Ascend machine with the dependencies above, run the mapped combine
operator test and the native dispatch, demand DMA and next-layer prefetch
checks in `tests/ut/expert_offload/`.

```bash
pytest -q tests/e2e/cpu_moe/test_mapped_combine_operator.py
```
