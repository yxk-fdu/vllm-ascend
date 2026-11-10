"""Actual dispatcher with a test-only ACL DSO; run in an isolated process."""
import ctypes
import os
import numpy as np
import _native_dispatch_fixture as fixture
from _vllm_ascend_lrc import NativeCpuDispatcher, NativeHostPlanner, Policy

acl=ctypes.CDLL('libascendcl.so');acl.fake_order_size.restype=ctypes.c_size_t
acl.fake_order_at.argtypes=[ctypes.c_size_t];acl.fake_order_at.restype=ctypes.c_size_t
addr=lambda x:x.ctypes.data
sources=[np.full((4,64),13,dtype=np.uint8),np.full((4,64),29,dtype=np.uint8)]
for field in sources:
 for e in range(4):field[e]+=e
destinations=[np.zeros((1,64),dtype=np.uint8) for _ in sources]
engine=NativeCpuDispatcher(fixture.api(),1,2,62,63)
engine.configure_dma(31,3)
layer=engine.add_dma_layer([[addr(x[e]) for e in range(4)] for x in sources],[[addr(x[0])] for x in destinations],[64,64],4,1)
owners=[]
def channel(e,prefetch=False,dependency=None):
 flag=np.zeros(1,np.uint32);ids=np.array([[e]],np.int64);weights=np.ones((1,1),np.float32)
 mapping=np.array([0,-1,-1,-1],np.int32);dmapping=mapping.copy();ready=np.zeros(1,np.uint64);done=np.zeros(1,np.uint64)
 state=Policy(1,4,32,.9,1.,.5,.3,.01,False)
 planner=NativeHostPlanner(4,1,0,0,1,1,'first_miss',state,[0.]*4)
 task=fixture.task();taskaddr=fixture.address(task)
 if prefetch:
  args=(addr(flag),addr(ids),addr(weights),addr(mapping),addr(ready),1,1,0,8,4,1,planner)
  index=engine.add_prefetch(*args) if dependency is None else engine.add_prefetch_after(*args,dependency)
 else:index=engine.add(addr(flag),taskaddr,(fixture.entry(),taskaddr),addr(ids),addr(weights),addr(mapping),addr(done),addr(ready),1,1,0,8,4,planner)
 engine.bind_dma_channel(index,layer,addr(dmapping),dmapping.nbytes)
 result=(index,flag,ids,weights,mapping,dmapping,ready,done,state,planner,task)
 owners.append(result);return result
try:
 demand=channel(2);prefetch=channel(3,True,demand[0])
 engine.set_dma_capture(True)
 # Intentionally advance prefetch first: this generation must still wait for demand.
 prefetch[1][0]=1;demand[1][0]=1
 assert fixture.wait_value_holding_gil(addr(prefetch[6]),1),'DMA required Python/GIL'
 assert fixture.wait_value_holding_gil(addr(demand[7]),1)
 assert demand[5].tolist()==[-1,-1,0,-1];assert prefetch[5].tolist()==[-1,-1,-1,0]
 for src,dst in zip(sources,destinations):assert np.array_equal(src[3],dst[0])
 engine.set_dma_capture(False)
 assert engine.dma_stats()==(2,2,4),engine.dma_stats()
 order=[acl.fake_order_at(i) for i in range(acl.fake_order_size())]
 assert order.index(addr(demand[6]))<order.index(addr(prefetch[6]))
 engine.set_dma_capture(False);records=engine.dma_timings();assert records[-1]=={'written':2,'dropped':0}
 for record in records[:-1]:
  times=[record[k] for k in ('cpp_enqueue_ns','cpp_dequeue_ns','first_copy_call_enter_ns','expert_fields_return_ns','mapping_return_ns','ready_record_return_ns')]
  assert times==sorted(times),record
  assert 'python_resume_ns' not in record
 print('GIL_HELD_DMA_WEIGHTS_SCALE_MAPPING_FIFO_READY_OK',records,flush=True)
 # Resident-only demand uses the existing independent zero-transfer ready path.
 demand[6][0]=0;demand[7][0]=0;demand[1][0]=2
 assert fixture.wait_value_holding_gil(addr(demand[6]),1)
 assert fixture.wait_value_holding_gil(addr(demand[7]),1)
 assert engine.dma_stats()==(2,2,4)
 try:engine.next_weight();raise AssertionError('Python stole a native DMA job')
 except RuntimeError as e:assert 'disabled' in str(e)
 # Invalid pointer/table is rejected on cold registration before any submission.
 for args in (([[0]*4],[[1]],[64],4,1),([[1]*4],[[1]],[0],4,1),([],[],[],4,1)):
  try:engine.add_dma_layer(*args);raise AssertionError('bad table accepted')
  except ValueError:pass
 # Queue one final transfer, then close while the poll thread is still active.
 demand[2][0,0]=1;demand[3].fill(1);demand[6][0]=0;demand[7][0]=0;demand[1][0]=3
 assert fixture.wait_holding_gil(demand[10],3)
 engine.close();engine.close()
 assert demand[6][0]==1 and demand[7][0]==1
 assert engine.dma_stats()==(3,3,6)
 print('ZERO_TRANSFER_VALIDATION_CLOSE_DRAIN_IDEMPOTENT_OK',engine.stats(),engine.dma_stats(),flush=True)
finally:engine.close()
