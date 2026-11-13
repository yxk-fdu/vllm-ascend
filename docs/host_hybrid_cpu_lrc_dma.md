# CPU LRC and DMA for hybrid experts

The single-card hybrid path uses CPU admission and LRC, host-submitted weight
DMA, and the existing mapped-host CPU result combine. Expert prefetch is
disabled for this path. The old NPU admission setting is rejected when CPU MoE
is enabled; use `cpu_moe.transfer_scheduler="cpu"`.

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
is enabled; ordinary non-hybrid paging keeps its original Python policy.

Graph CPU submission polls an ordered D2H sequence. With
`cpu_moe.native_dispatch=true` shared NZ starts the C++ dispatcher with the CPU backend.
The older x86 Python dispatcher uses `KT_MOE_POLL_LIBRARY` to load its
`register_flag` and `poll_next` implementation before backend creation.
The installed kt-kernel must expose pinned input/output buffers and the existing
completion runtime. Mapped combine and value-based `ExternalEvent` support
remain required.

Select `h2d_backend="torch"` for the host transfer transport. Shared NZ
graph replay submits the prepared DMA copies through the native dispatcher.
Set `cpu_moe.max_num_tokens` to cover every captured decode/verification batch;
an oversized hybrid graph raises rather than entering legacy callback paging.
Eager batches above that capacity retain ordinary paging.

### Shared NZ CPU weights

Use vLLM 0.30.0 and build kt-kernel from
[cpu_moe_compute at 599e66a](https://github.com/yxk-fdu/cpu_moe_compute/tree/599e66a4f3155f21e02b2a924bd8a372430cabd7/cpu_moe).
This revision includes the AArch64 build fix for the packaged ready-poll helper.
Set `cpu_moe.weight_format="nz_mxfp4"`, `native_dispatch=true`, and
`threadpool_count=1`. Place the host DMA buffers and compute pool on the
same NUMA node. Leave `cpu_moe.weight_path` empty.
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
The proportional load-balance rule remains
`floor(missing * cpu_cost / (cpu_cost + copy_cost))`, bounded by available slots
and `balance_max_transfers`. LRC selects victims; it does not choose how many
experts the CPU computes. `first_miss` and frequency admission are also retained.

## Graph and lifetime rules

Each captured site owns a distinct pinned mapping and weight event. Its reset,
input D2H, sequence publication, and external waits are graph nodes, replayed
for each invocation. CPU planning executes on each observed sequence, never
once at capture time. Site objects remain owned by the backend's graph tasks.
Eager contexts retain their pinned buffers across asynchronous H2D. Draft/MTP
layers registered after finalization use their manager ordinal for LRC/weight
storage and their model layer index for CPUInfer.

DMA errors are retained and raised on subsequent host submissions. A failed
request never publishes weight readiness. An already waiting graph follows the
runtime task-timeout/error path; do not continue inference with that graph.
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
  tests/ut/expert_offload/cpu_backend/test_async_completion_handoff.py
```

On an Ascend machine with the dependencies above, run:

```bash
pytest -q tests/e2e/cpu_moe/test_host_hybrid_graph.py \
  tests/e2e/cpu_moe/test_mapped_result_combine.py
```

The new hardware test uses deterministic small expert stand-ins, changing
routes across two captured graphs and 30 replays, with either CPU or DMA
artificially delayed. It covers real NPU DMA/events and mapped combine; full
model accuracy and performance still need a model-level server run.
