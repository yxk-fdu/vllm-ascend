"""Exercise real Python registration against the native dispatcher/fake ACL."""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import _native_dispatch_fixture as fixture

root=Path(sys.argv[1]);os.environ['KT_MOE_POLL_CPU']='62';os.environ['KT_MOE_COMPLETION_CPU']='63'
class Stream:
 next=10
 def __init__(self,**kwargs):Stream.next+=1;self.npu_stream=Stream.next
class Event:
 def __init__(self):self.buffer=torch.zeros(1,dtype=torch.int64);self.npu_event=self.buffer.data_ptr()
torch.npu=NS(synchronize=lambda *args:None)
sys.modules['torch_npu']=NS(npu=NS(SyncLaunchStream=Stream))
kt=types.ModuleType('kt_kernel');kt.kt_kernel_ext=NS(native_dispatch_api=fixture.api,native_cpu_handle=lambda _:1)
sys.modules['kt_kernel']=kt
class Buffers:
 buffer_depth=1
 get_buffer=None
module=types.ModuleType('kt_kernel.experts_base');module.KExpertsCPUBuffer=Buffers;sys.modules[module.__name__]=module
spec=importlib.util.spec_from_file_location('tested_native_submission',root/'vllm_ascend/expert_offload/cpu_backend/native_submission.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
worker=module.NativePollWorker(0);dma=Stream();manager=NS(_prepared_expert_h2d={});owners=[]
class Plan:
 native_dma_live=False
 def __init__(self):
  self.src=torch.arange(256,dtype=torch.int32).reshape(4,64);self.dst=torch.zeros((1,64),dtype=torch.int32)
 def native_dma_table(self):return [[x.data_ptr() for x in self.src]],[[self.dst[0].data_ptr()]],[256],4,1
class Context:
 def before_submit(self):pass
try:
 for prefetch in (False,True):
  index=int(prefetch);plan=Plan();manager._prepared_expert_h2d[index]=plan
  from _vllm_ascend_lrc import NativeHostPlanner
  planner=NativeHostPlanner(4,1,index,0,1,1,'first_miss',None,[0.]*4)
  dest_mapping=torch.tensor([0,-1,-1,-1],dtype=torch.int32)
  transfer=NS(worker=NS(manager=manager,stream=dma),layer_idx=index,layer=NS(log2phy=dest_mapping),native_dma_cpu=31,native_planner=planner)
  context=Context();context.transfer=transfer;context.mapping=dest_mapping.clone();context.ready=Event()
  ids=torch.tensor([[index+2]],dtype=torch.int64);weights=torch.ones((1,1));flag=torch.zeros(1,dtype=torch.int32);done=Event();hidden=torch.zeros((1,8));task=fixture.task()
  if not prefetch:
   batches=torch.tensor([1],dtype=torch.int32)
   Buffers.get_buffer=staticmethod(lambda *args:([hidden],[ids],None,[weights],[hidden],[batches],None))
   backend=NS(config=NS(top_k=1,layer_idx=0),_wrapper=NS(max_deferred_experts_per_token=0,layer_idx=0,cpu_infer=1,moe=NS(forward_task=lambda *args:(fixture.entry(),fixture.address(task)))))
   worker.register(backend,hidden,None,done,flag,context.before_submit)
   assert context.native_channel==0
  else:
   site=NS(transfer=transfer,ids=ids,weights=weights,mapping=context.mapping,ready=context.ready,host_flag=flag,dependency=0)
   worker.register_prefetch(site,1)
  owners.append((plan,ids,weights,flag,done,context,task,dest_mapping))
  assert plan.native_dma_live and manager._native_dma_worker is worker
 assert not hasattr(worker,'dma_thread') and not hasattr(worker,'_dma')
 with tempfile.TemporaryDirectory(dir=sys.argv[2]) as directory:
  worker.set_dma_capture(True,directory)
  owners[1][3].fill_(1);owners[0][3].fill_(1)
  assert fixture.wait_value_holding_gil(owners[1][5].ready.npu_event,1)
  worker.set_dma_capture(False,directory)
  files=list(Path(directory).glob('*.jsonl'));assert len(files)==1
  records=[json.loads(x) for x in files[0].read_text().splitlines()];assert len(records)==3
  assert records[-1]['written']==2 and all('reference_mono_ns' in x and 'wall_ns' in x for x in records)
  worker.set_dma_capture(False,directory);assert len(files[0].read_text().splitlines())==3
 for e,(plan,*_) in enumerate(owners):assert torch.equal(plan.dst[0],plan.src[e+2])
 worker.close();worker.close();assert all(not x[0].native_dma_live for x in owners)
 print('REAL_PYTHON_DEMAND_PREFETCH_REGISTRATION_OWNERS_CAPTURE_CLOSE_OK',flush=True)
finally:worker.close()
