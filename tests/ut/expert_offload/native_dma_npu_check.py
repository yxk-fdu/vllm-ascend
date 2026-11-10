"""Small real NPU graph: native DMA completes while the caller holds GIL."""
import torch
import torch_npu
import _native_dispatch_fixture as fixture
from _vllm_ascend_lrc import NativeCpuDispatcher, NativeHostPlanner, Policy

torch.npu.set_device(0)
complete=torch_npu.npu.SyncLaunchStream(device=0)
zero=torch_npu.npu.SyncLaunchStream(device=0)
dma=torch_npu.npu.SyncLaunchStream(device=0)
engine=NativeCpuDispatcher(fixture.api(),int(complete.npu_stream),int(zero.npu_stream),62,63)
engine.configure_dma(31,int(dma.npu_stream));owners=[]

def channel(expert,prefetch=False,dependency=None,slots=1,multiple=False):
 # The scale has dense, non-contiguous final physical strides, shared with H2D.
 source_w=[torch.full((1024,),20+e,dtype=torch.uint8,pin_memory=True) for e in range(4)]
 source_s=[torch.empty_strided((2,32,2),(2,4,1),dtype=torch.uint8,pin_memory=True) for _ in range(4)]
 for e,x in enumerate(source_s):x.copy_((torch.arange(128).reshape(2,32,2)+e).to(torch.uint8))
 dest_w=torch.zeros((slots,1024),dtype=torch.uint8,device='npu')
 dest_s=torch.zeros((slots,32,2,2),dtype=torch.uint8,device='npu').transpose(1,2)
 layer=engine.add_dma_layer([[x.data_ptr() for x in source_w],[x.data_ptr() for x in source_s]],
                          [[dest_w[s].data_ptr() for s in range(slots)],[dest_s[s].data_ptr() for s in range(slots)]],[1024,128],4,slots)
 dflag=torch.zeros(1,dtype=torch.int32,device='npu');hflag=torch.zeros(1,dtype=torch.int32,pin_memory=True)
 ids=torch.tensor([[1,2,3]] if multiple else [[expert]],dtype=torch.int64,pin_memory=True);dids=ids.to('npu')
 weights=torch.ones(ids.shape,dtype=torch.float32,pin_memory=True);dweights=weights.to('npu')
 mapping=torch.tensor([0,-1,-1,-1] if slots else [-1]*4,dtype=torch.int32,pin_memory=True)
 device_mapping=mapping.to('npu');original_mapping=device_mapping.clone()
 ready=torch_npu.npu.ExternalEvent();ready.reset();done=torch_npu.npu.ExternalEvent();done.reset()
 state=Policy(1,4,32,.9,1.,.5,.3,.01,False)
 planner=NativeHostPlanner(4,slots,0,100 if multiple else 0,101 if multiple else 1,2 if multiple else 1,'load_balance' if multiple else 'first_miss',state,[0.]*4)
 task=fixture.task();taskaddr=fixture.address(task)
 if prefetch:
  args=(hflag.data_ptr(),ids.data_ptr(),weights.data_ptr(),mapping.data_ptr(),int(ready.npu_event),1,ids.shape[1],0,8,4,1,planner)
  index=engine.add_prefetch(*args) if dependency is None else engine.add_prefetch_after(*args,dependency)
 else:index=engine.add(hflag.data_ptr(),taskaddr,(fixture.entry(),taskaddr),ids.data_ptr(),weights.data_ptr(),mapping.data_ptr(),int(done.npu_event),int(ready.npu_event),1,ids.shape[1],0,8,4,planner)
 engine.bind_dma_channel(index,layer,device_mapping.data_ptr(),mapping.numel()*mapping.element_size())
 token=torch.zeros(1,dtype=torch.int64,device='npu');host_token=torch.zeros(1,dtype=torch.int64,pin_memory=True)
 # Capture without running: each replay starts from the same known cache map.
 torch.npu.synchronize();graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):
  ready.reset();done.reset();device_mapping.copy_(original_mapping)
  ids.copy_(dids,non_blocking=True);weights.copy_(dweights,non_blocking=True)
  mapping.copy_(device_mapping,non_blocking=True);dflag.add_(1);hflag.copy_(dflag,non_blocking=True)
  ready.wait()
  if not prefetch:done.wait()
  token.add_(1);host_token.copy_(token,non_blocking=True)
 result=(index,graph,host_token,source_w,source_s,dest_w,dest_s,device_mapping,task)
 owners.append((result,dflag,hflag,ids,dids,weights,dweights,mapping,original_mapping,ready,done,state,planner,token))
 return result
try:
 demand=channel(2);prefetch=channel(3,True,demand[0]);resident=channel(0);cpu_only=channel(2,slots=0);multi=channel(1,slots=2,multiple=True)
 engine.set_dma_capture(True)
 for generation in range(1,9):
  # Both signals may be visible together; FIFO must admit demand first.
  demand[1].replay();prefetch[1].replay()
  for result in (demand,prefetch):
   assert fixture.wait_value_holding_gil(result[2].data_ptr(),generation),'Native DMA stalled under GIL'
  for result,e in ((demand,2),(prefetch,3)):
   assert torch.equal(result[5][0].cpu(),result[3][e])
   assert torch.equal(result[6][0].cpu(),result[4][e])
   assert result[7].cpu().tolist()==[-1,-1,0,-1] if e==2 else result[7].cpu().tolist()==[-1,-1,-1,0]
  for result in (resident,cpu_only,multi):
   result[1].replay();assert fixture.wait_value_holding_gil(result[2].data_ptr(),generation)
  for slot,e in ((1,1),(0,2)):
   assert torch.equal(multi[5][slot].cpu(),multi[3][e])
   assert torch.equal(multi[6][slot].cpu(),multi[4][e])
  assert multi[7].cpu().tolist()==[-1,1,0,-1]
 torch.npu.synchronize();engine.set_dma_capture(False)
 assert engine.dma_stats()==(24,32,64),engine.dma_stats()
 assert engine.stats()==(32,32),engine.stats()
 assert engine.prefetch_stats()==(8,8)
 records=engine.dma_timings();assert records[-1]=={'written':24,'dropped':0}
 for i in range(0,24,3):assert records[i]['prefetch']==0 and records[i+1]['prefetch']==1 and records[i+2]['experts']==2
 print('REAL_NPU_GIL_HELD_DEMAND_PREFETCH_STRIDED_SCALE_MAPPING_8_REPLAYS_OK',engine.stats(),engine.dma_stats(),flush=True)
 print('REAL_NPU_ZERO_TRANSFER_AND_ALL_CPU_8_REPLAYS_OK',flush=True)
 print('REAL_NPU_TWO_EXPERT_PIPELINE_8_REPLAYS_OK',flush=True)
 print('DMA_TIMINGS',records,flush=True)
finally:
 torch.npu.synchronize();engine.close();engine.close()
