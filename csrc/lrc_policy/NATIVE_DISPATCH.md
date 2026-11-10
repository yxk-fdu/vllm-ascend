# Native CPU MoE graph dispatch

`cpu_moe.native_dispatch=true` replaces the per-layer Python input producer,
submission queue and result-completion worker during captured graph replay.
The default remains false, preserving the original path for rollback.

The input stream copies hidden states, routes and the resident mapping before
publishing a monotonically increasing sequence to coherent pinned host memory.
A C++ thread acquire-loads that sequence, runs the same host admission policy,
masks the CPU routes and invokes a pre-registered CPUInfer task through a
versioned C interface. Task arguments, tensors, events, planners and backends
are retained by the Python registration owner until graphs retire and native
threads join. No Python callable or GIL acquisition occurs in this path.

A separate C++ thread drains CPUInfer and publishes the result-ready value.
Torch NPU 2.10 ExternalEvent uses a device allocation and value wait/write;
its `npu_event` must **not** be passed to `aclrtRecordEvent`. Native completion
uses `aclrtValueWrite(address, 1, 0, stream)`, matching Torch NPU's SyncLaunchStream
implementation. The recorded graph consumes and clears that value.

CPU results and expert weights use independent readiness values and streams.
When admission produces no loads, the mapping is unchanged, so C++ publishes
weight readiness directly without a redundant mapping copy. Non-empty expert
loads are consumed by a separate C++ DMA thread; it never returns through
pybind or reacquires GIL between jobs. It never gates native CPU submission
or CPU completion. Eager/prefill retains its existing execution
path, using the same native planner and shared LRC state.

## Native prefetch and demand-first DMA

Prefetch notifications use the same C++ monitor as CPU compute, with distinct
registered channels and sequence buffers. Graph replay does not call an ACL
Python host callback to start either path. Predicted next-layer routes read
that layer's existing C++ LRC history; only actual demand routes update LRC.
CPU computation, CPU completion and DMA submission remain separate workers.

After publishing current-layer input, the NPU may predict and stage next-layer
prefetch while demand DMA is in flight. The C++ monitor admits this generation's
demand before its dependent prefetch and queues both on the same FIFO DMA
worker/stream. Prefetch preparation need not wait for demand DMA completion;
its actual copies cannot overtake the current demand. Zero-load demand still
advances the dependency and publishes ready through its separate stream. A load
builds and submits one expert's weights/scales before preparing the next expert,
rather than constructing every expert before the first copy can start. Publish
mapping and readiness only after every required weight and scale operation.
The next layer waits for its prefetched weights and mapping before using them.
The prefetch cap is `expert_prefetch_num`; use `1` for the measured configuration.

Prefetch readiness owns a 32-byte device flag. Reset, signal and wait consistently
use ACL value operations; the bridge drains Torch's host launch queue before
issuing raw ACL operations. Every real graph capture gets distinct site owners,
including captures that occur outside the Python capture-context flag. This
avoids mixing native zero-copy notifications with Torch external-event record
bookkeeping across eager prewarm and multiple captured shapes.

## Route-weighted load balancing

`load_balance` counts expert-token routes: one expert selected by two tokens is
two CPU work units but one weight transfer. Missing experts are ranked by route
frequency, then candidate transfer prefixes are evaluated with a simple model:

```text
CPU(k) = remaining expert-token routes * balance_cpu_expert_us
NPU(k) = k * balance_copy_us + balance_npu_fixed_us
choose k to minimize max(CPU(k), NPU(k))
```

The fixed NPU estimate applies once per layer, including the no-transfer
baseline. It accounts for NPU compute after weights arrive; it is not multiplied
by the number of experts. The default is zero for backward compatibility; the
measured setup explicitly uses 200 us. LRC chooses eligible eviction victims,
while route frequency chooses which missing experts are worth moving to NPU.
Python and native planning share the same integer-scaled cost comparison and
prefer fewer transfers when costs tie.

## Build and configuration

The matching kt-kernel must export `native_dispatch_api()` and
`native_cpu_handle()`. `kt_dispatch_api.h` defines the versioned C interface;
the functions must be compiled into kt-kernel itself so no private class
layout or hidden C++ symbol is guessed across extensions. Keep the existing
mathematical kernel objects unchanged when adding this interface.

Install this extension from source:

```sh
python -m pip install -e ./csrc/lrc_policy --no-build-isolation --no-deps
# Required additionally when native expert prefetch is enabled:
(cd csrc/lrc_policy && python setup_npu_wait.py build_ext --inplace)
```

The optional prefetch bridge links Torch NPU and CANN; the CPU-only LRC build
remains independent of those libraries. Build it in the same Python/Torch NPU
environment as the service. Third-party kt-kernel sources and binaries are not
vendor-copied here: its versioned API exports above remain a prerequisite.

This implementation requires Linux, coherent x86 pinned host memory, the
value-based Torch NPU external-event API and the matching kt-kernel build.
With a cache policy enabled, use C++ LRC. Admission state is shared between
eager execution and all captured shapes, and protected by a state mutex.

Choose CPUs from the actual NPU/NUMA topology. `KT_MOE_POLL_CPU` controls the
native input/planning thread; `KT_MOE_COMPLETION_CPU` controls the native
completion thread. `cpu_moe.native_dispatch_dma_cpu` controls weight DMA;
its default uses the completion CPU. All must be outside the computation
workers' physical cores, including SMT siblings. The verified 153 setup uses
compute 32–61, poll 62, completion 63 and DMA 31 (SMT sibling 95, outside
compute/communication physical cores). The DMA thread explicitly binds its
configured core even if it lies outside the inherited NUMA-1 process affinity;
the OS still enforces cgroup/cpuset constraints.

## Validation

`test_native_submission.py` compares repeated native plans against the
reference for all three admission policies, with and without seeded LRC,
and checks in-place route conservation for 32/64-bit IDs and mappings.

`native_dispatch_fixture.cpp` and `native_dispatch_npu_check.py` provide a small
real NPU graph check. The test deliberately holds Python's GIL while waiting
for a native task, checks five exactly-once generations, and verifies that
both independent external values release the captured graph. Run this only
with the intended NPU explicitly selected; then validate actual model outputs
and benchmark with profiling disabled.

## Native DMA ownership and profiling

Python loads/pins weights, prepares their final physical layout, creates the
streams and events, registers pointer/byte tables, and retains all owners.
The native worker copies each expert's fields, then log2phy, then publishes
ready on the same DMA stream using `aclrtMemcpyAsync`/`aclrtValueWrite`. It
neither converts tensors nor calls Python. The C++ submit return means queued,
not device completion; the graph's ready wait remains required. Eager/prefill
keeps the existing Torch path. Unsupported prepared layouts fail registration
explicitly rather than silently changing the graph execution path.

Registered plans reject loader writes, buffer replacement and reformatting.
Shutdown first retires graph consumers, stops/joins the monitor, drains the
completion and DMA queues, and synchronizes the DMA stream before releasing
buffer owners. The manager closes native DMA before clearing prepared views.

Profile-only timestamps go into a bounded C++ array (4096 jobs, with an explicit
dropped count). There is no Python logger or shared trace mutex in the DMA hot
path. Stop-profile converts records and writes `cpu_native_dma_submit_PID.jsonl`
with monotonic/wall-clock anchors. These are native submission records, not the
old Python `cpu_dma_submit` schema; consumers must use the new event name and
must not interpret submit timestamps as device start/end times.

`native_dma_cpu_check.py` uses an isolated fake ACL DSO to check ordering,
GIL independence, invalid registrations, resident-only notification and close
queue draining. `native_dma_npu_check.py` verifies actual DMA, strided scale,
log2phy and external-ready joins across eight graph replays for demand,
prefetch, resident-only, all-CPU and multi-expert pipeline cases. Its CPU expert task is a test fixture,
not a numerical MoE kernel or an end-to-end throughput measurement.
