"""CPU-only regression of actual manager/worker method bodies; no NPU runtime."""
import ast,contextlib,copy,dataclasses,enum,json,logging,os,sys,threading,time,types,typing,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch as real_torch

ROOT=Path(sys.argv.pop(1));D=ROOT/'vllm_ascend/expert_offload'
for name in ('vllm_ascend','vllm_ascend.expert_offload'):
 module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
class NPU:
 def __init__(self):self.events=[]
 @contextlib.contextmanager
 def stream(self,s):yield
 def synchronize(self,*args):self.events.append('device-sync')
class TorchProxy:
 npu=NPU()
 def __getattr__(self,name):return getattr(real_torch,name)
torch=TorchProxy()
class Phase:
 def __getattr__(self,name):return name
phase=Phase()
def stamp(record,name):
 if record is not None:record[name+'_ns']=time.monotonic_ns();record[name+'_cpu_ns']=time.thread_time_ns()
module=types.ModuleType('vllm_ascend.expert_offload.dma_submit_timing');module.stamp=stamp;sys.modules[module.__name__]=module

def definitions(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert len(nodes)==len(names),(path,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

env=dict(globals(),__package__='vllm_ascend.expert_offload',dataclass=dataclasses.dataclass,Enum=enum.Enum,Any=typing.Any,CpuPrefetchPhase=phase)
definitions(D/'h2d_transfer.py',{'CopyDirection','H2DCopyTask','TorchCopyH2DTransport'},env)
module=types.ModuleType('vllm_ascend.expert_offload.h2d_transfer');module.H2DCopyTask=env['H2DCopyTask'];sys.modules[module.__name__]=module
module=types.ModuleType('vllm_ascend.expert_offload.prepared_h2d');module.__package__='vllm_ascend.expert_offload';sys.modules[module.__name__]=module;exec(compile((D/'prepared_h2d.py').read_text(),str(D/'prepared_h2d.py'),'exec'),module.__dict__)
plan_type=module.PreparedLayerH2D
env['prepare_mapping_copy']=module.prepare_mapping_copy
source=ast.parse((D/'expert_offload_manager.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ExpertOffloadManager')
names={'_assert_native_dma_reload_safe','_init_layer_scale_buffers','_copy_scale_assembled','_copy_scale_direct','_load_scale_shard','_get_h2d_transport','_cached_expert_h2d_view','_cached_expert_h2d_task','_expert_src_storage','_build_quant_attr_h2d_tasks','_build_expert_h2d_tasks','_build_prefill_h2d_tasks','_load_expert_weights_into_slots','_refresh_expert_fp32_scale','process_weights_after_loading','_synchronize_h2d'}
if any(isinstance(n,ast.FunctionDef) and n.name=='_prepare_expert_h2d_layer' for n in cls.body):names.add('_prepare_expert_h2d_layer')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(methods)==len(names)
env['logger']=logging.getLogger('h2d-regression')
env['torch_npu']=NS(get_npu_format=lambda _: (_ for _ in ()).throw(AssertionError('NPU queried')))
definitions(D/'expert_offload_manager.py',{'_expert_weight','_prefill_scale_copy_views'},env)
proxy=ast.ClassDef(name='Manager',bases=[],keywords=[],body=methods,decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[proxy],type_ignores=[])),str(D/'expert_offload_manager.py'),'exec'),env)
Manager=env['Manager'];Transport=env['TorchCopyH2DTransport']
# Load the real quant-layout helper without importing an NPU runtime.
for name in ['vllm_ascend.quantization','vllm_ascend.quantization.methods','vllm_ascend.quantization.methods.w4a8']:
 q=types.ModuleType(name);q.__path__=[];sys.modules[name]=q
quant=types.ModuleType('vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4');quant.torch=real_torch
sys.modules[quant.__name__]=quant
definitions(ROOT/'vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py',{'apply_mxfp4_weight_scale_layout'},quant.__dict__)
layout=quant.apply_mxfp4_weight_scale_layout

worker_cls=next(n for n in ast.parse((D/'host_transfer.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HostWeightWorker');process=next(n for n in worker_cls.body if isinstance(n,ast.FunctionDef) and n.name=='_process')
exec(compile(ast.Module(body=[process],type_ignores=[]),str(D/'host_transfer.py'),'exec'),env)
process=env['_process']
env['HostTransferPolicy']=lambda **kwargs:NS(**kwargs)
definitions(D/'host_transfer.py',{'HostExpertTransfer'},env)
Transfer=env['HostExpertTransfer']

class Layer:
 def __init__(self,**attrs):self.__dict__.update(attrs)

class Fixture:
 def __init__(self,slots=2,strided=False,cast=False):
  m=self.manager=Manager();m.offload_config=NS(shard_per_rank=False,cpu_moe=NS(weight_format='gguf'));m.num_total_experts=3;m.ep_size=1
  m._allocate_expert_host_tensor=lambda shape,dtype:real_torch.empty(shape,dtype=dtype)
  m.w13_expert_size_bytes=8;m.w2_expert_size_bytes=4;m._shard_local=lambda e:e
  m.w13_weights_cpu=[[real_torch.full((8,),11+e,dtype=real_torch.uint8) for e in range(3)]]
  m.w2_weights_cpu=[[real_torch.full((4,),21+e,dtype=real_torch.uint8) for e in range(3)]]
  scales=[real_torch.arange(4,dtype=real_torch.float32).reshape(2,2)+e for e in range(3)]
  if strided:scales=[s.t() for s in scales]
  m.scale_cpu_buffers={'scale':[scales]};m.offset_cpu_buffers={};m.scale_bias_cpu_buffers={}
  self.layer=Layer(w13_weight=real_torch.zeros((slots,8),dtype=real_torch.uint8),w2_weight=real_torch.zeros((slots,4),dtype=real_torch.uint8),scale=real_torch.zeros((slots,4) if strided else (slots,2,2),dtype=real_torch.float64 if cast else real_torch.float32))
  m._prefetch_trace=NS(mark=lambda *args:None);m.h2d_transport=Transport()
  m._prefill_w13=[real_torch.zeros((3,8),dtype=real_torch.uint8)];m._prefill_w2=[real_torch.zeros((3,4),dtype=real_torch.uint8)]
  for attr in ('_prefill_w13_scale','_prefill_w2_scale','_prefill_w13_offset','_prefill_w2_offset','_prefill_w13_scale_bias','_prefill_w2_scale_bias'):setattr(m,attr,[])
 def prepare(self):
  if hasattr(self.manager,'_prepare_expert_h2d_layer'):self.manager._prepare_expert_h2d_layer(self.layer,0)
  else:self.manager._prepared_expert_h2d={0:plan_type(self.manager,self.layer,0,env['_expert_weight'])}

class Tests(unittest.TestCase):
 def test_native_table_owns_exact_existing_buffers(self):
  f=Fixture();f.prepare();plan=f.manager._prepared_expert_h2d[0]
  owners=tuple(plan.owners)
  src,dst,sizes,experts,slots=plan.native_dma_table()
  self.assertEqual((experts,slots),(3,2))
  self.assertEqual(src,[[s.data_ptr() for s in field[0]] for field in plan.fields])
  self.assertEqual(dst,[[s.data_ptr() for s in field[1]] for field in plan.fields])
  self.assertEqual(sizes,[8,4,16]);self.assertEqual(tuple(map(id,plan.owners)),tuple(map(id,owners)))
 def test_native_live_buffers_reject_reload_and_rebuild(self):
  f=Fixture();f.prepare();plan=f.manager._prepared_expert_h2d[0];plan.native_dma_live=True
  with self.assertRaisesRegex(RuntimeError,'reloading'):f.manager._assert_native_dma_reload_safe(0)
  with self.assertRaisesRegex(RuntimeError,'rebuilding'):f.prepare()
  with self.assertRaisesRegex(RuntimeError,'reformatting'):f.manager.process_weights_after_loading()
  self.assertIs(f.manager._prepared_expert_h2d[0],plan)
  plan.native_dma_live=False;f.prepare()

 def test_cold_hook_covers_initial_and_late_layers(self):
  f=Fixture();m=f.manager;m.moe_layers=[f.layer];m.offload_config.cpu_moe=NS(native_dispatch=False);m.num_device_experts_for_layer=lambda i:2;m.cache_policy=None
  Transfer(m,f.layer,NS(),NS());self.assertIs(m._prepared_expert_h2d[0].layer,f.layer)
  late=copy.deepcopy(f.layer);m.moe_layers.append(late)
  for buffers in (m.w13_weights_cpu,m.w2_weights_cpu,*m.scale_cpu_buffers.values()):buffers.append([t.clone() for t in buffers[0]])
  Transfer(m,late,NS(),NS());self.assertIs(m._prepared_expert_h2d[1].layer,late)
 def test_actual_prefetch_callback_gate_order_and_cleanup(self):
  tree=ast.parse((D/'routed_experts.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare_prefetch')
  outer=ast.parse('def make(mgr,layer,hidden_states):\n prefetch_started_early=False\n def placeholder(): pass\n return prepare_prefetch,lambda:prefetch_started_early\n').body[0];outer.body[1]=node
  scope=dict(env);exec(compile(ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[])),str(D/'routed_experts.py'),'exec'),scope)
  events=[];old=object();context=NS(native_channel=7);published=object();task=NS(_host_context=context,_input_published_event=published)
  torch.npu.current_stream=lambda:NS(wait_event=lambda event:events.append(('input-join',event)))
  for enabled in (False,True):
   events.clear();mgr=NS(offload_config=NS(expert_prefetch_enabled=enabled),_prefetch_demand_context=old)
   mgr.trigger_next_layer_prefetch=lambda *args:events.append(('predict',mgr._prefetch_demand_context))
   callback,started=scope['make'](mgr,object(),object());callback(task)
   self.assertEqual(events,[('input-join',published),('predict',context)] if enabled else []);self.assertEqual(started(),enabled);self.assertIs(mgr._prefetch_demand_context,old)
  # Even an exception must not leak the demand context into another layer.
  mgr.trigger_next_layer_prefetch=lambda *args: (_ for _ in ()).throw(RuntimeError('prediction failed'))
  callback,_=scope['make'](mgr,object(),object())
  with self.assertRaisesRegex(RuntimeError,'prediction failed'):callback(task)
  self.assertIs(mgr._prefetch_demand_context,old)
 def test_actual_executor_stages_before_weight_wait(self):
  namespace=dict(env,CPUExpertTask=object)
  definitions(D/'hybrid_executor.py',{'HybridExpertPlan'},namespace)
  cls=next(n for n in ast.parse((D/'hybrid_executor.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HybridExpertExecutor')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prepare');exec(compile(ast.Module(body=[method],type_ignores=[]),str(D/'hybrid_executor.py'),'exec'),namespace)
  events=[];task=NS(wait_for_weights=lambda:events.append('weight-wait'))
  def submit(**kwargs):events.append('submit');return task
  executor=NS(layer_idx=0,backend=NS(_host_transfer=object(),submit=submit))
  result=namespace['prepare'](executor,hidden_states=real_torch.zeros((1,4)),topk_ids=real_torch.tensor([[0,2]]),topk_weights=real_torch.ones((1,2)),log2phy=real_torch.tensor([0,-1,-1]),before_weight_wait=lambda t:events.append('prefetch'))
  self.assertEqual(events,['submit','prefetch','weight-wait']);self.assertIs(result.cpu_task,task)
 def test_worker_reaches_prepared_decode_and_pipelines(self):
  for timed in (False,True):
   with self.subTest(timed=timed):
    f=Fixture();f.prepare();m=f.manager;events=[];plan=m._prepared_expert_h2d[0];build=plan.tasks
    def selected(e,s):events.append(('build',e));return build(e,s)
    plan.tasks=selected
    def forbidden(*args):raise AssertionError('decode fell back to tensor metadata queries')
    m._cached_expert_h2d_view=forbidden
    class Recording(Transport):
     def copy_batch(self,tasks,**kwargs):events.append(('mapping' if tasks[0].name=='log2phy' else 'copy',tasks[0].name));super().copy_batch(tasks,**kwargs)
    m.h2d_transport=Recording();mapping=real_torch.zeros(3,dtype=real_torch.int32);expected=real_torch.tensor([1,-1,0],dtype=real_torch.int32)
    class Mapping:
     def copy_(self,src,**kwargs):events.append(('mapping',));mapping.copy_(src)
    class Ready:
     def record(self,stream):
      assert real_torch.equal(mapping,expected);assert real_torch.all(f.layer.w13_weight[0]==13).item();assert real_torch.all(f.layer.w2_weight[1]==21).item();events.append(('ready',))
    req=NS(transfer=NS(layer=f.layer,layer_idx=0),loads=[(0,2),(1,0)],mapping_host=expected,mapping_device=mapping,mapping_task=module.prepare_mapping_copy(expected,mapping),ready=Ready(),timing={} if timed else None)
    process(NS(manager=m,stream=object()),req)
    self.assertEqual([x[0] for x in events],['build','copy','build','copy','mapping','ready'])
    if timed:self.assertEqual(len(req.timing['expert_submissions']),2)
 def test_prefill_uses_pool_and_never_decode_cache(self):
  f=Fixture();f.prepare();m=f.manager;plan=m._prepared_expert_h2d[0]
  plan.tasks=lambda *_: (_ for _ in ()).throw(AssertionError('prefill used decode plan'))
  tasks=m._build_prefill_h2d_tasks(0,0,2,1);m.h2d_transport.copy_batch(tasks)
  self.assertTrue(real_torch.all(m._prefill_w13[0][1]==13));self.assertTrue(real_torch.all(m._prefill_w2[0][1]==23));self.assertTrue(real_torch.all(f.layer.w13_weight==0))
 def test_zero_slot_initialization(self):
  f=Fixture(slots=0);f.prepare()
  with self.assertRaises(ValueError):f.manager._prepared_expert_h2d[0].tasks(0,0)
 def test_quant_dtype_conversion_and_live_updates(self):
  f=Fixture(cast=True);m=f.manager;legacy=m._build_expert_h2d_tasks(f.layer,0,2,0);f.prepare();current=m._build_expert_h2d_tasks(f.layer,0,2,0)
  self.assertEqual([t.name for t in legacy],[t.name for t in current]);self.assertEqual(current[-1].nbytes,32)
  for value in (3.25,17.5):
   m.scale_cpu_buffers['scale'][0][2].fill_(value);f.prepare();m.h2d_transport.copy_batch(m._build_expert_h2d_tasks(f.layer,0,2,0));self.assertTrue(real_torch.all(f.layer.scale[0]==value))
 def test_strided_source_is_packed_after_cold_rebuild(self):
  f=Fixture(strided=True);f.prepare();m=f.manager;self.assertIn(0,m._prepared_expert_h2d)
  for value in (7,19):
   m.scale_cpu_buffers['scale'][0][1].fill_(value);f.prepare();m.h2d_transport.copy_batch(m._build_expert_h2d_tasks(f.layer,0,1,0));self.assertTrue(real_torch.all(f.layer.scale[0]==value))
 def test_reformat_rejects_live_shared_nz_owner(self):
  f=Fixture();f.prepare();m=f.manager;torch.npu.events=[]
  m.offload_config.cpu_moe.weight_format='nz_mxfp4';m._cpu_weights_finalized=True
  with self.assertRaisesRegex(RuntimeError,'Retire the NZ CPU backend'):
   m.process_weights_after_loading()
  self.assertEqual(torch.npu.events,[]);self.assertIn(0,m._prepared_expert_h2d)
 def test_sharded_path_does_not_build_plain_cache(self):
  f=Fixture();m=f.manager;m.ep_size=2;m.offload_config.shard_per_rank=True
  m._prepare_expert_h2d_layer(f.layer,0)
  self.assertFalse(getattr(m,'_prepared_expert_h2d',{}))

 def test_single_rank_default_shard_uses_prepared_storage(self):
  f=Fixture(cast=True);m=f.manager;m.offload_config.shard_per_rank=True;f.prepare()
  self.assertIn(0,m._prepared_expert_h2d)
  tasks=m._build_expert_h2d_tasks(f.layer,0,2,1)
  self.assertTrue(all(isinstance(t.source,real_torch.UntypedStorage) and isinstance(t.destination,real_torch.UntypedStorage) for t in tasks))
  m.h2d_transport.copy_batch(tasks);self.assertTrue(real_torch.equal(f.layer.scale[1],m.scale_cpu_buffers['scale'][0][2].double()))


 def test_transposed_destination_dtype_and_storage_offsets(self):
  f=Fixture();m=f.manager;base=real_torch.full((4,2,2),-91.,dtype=real_torch.float64);f.layer.scale=base[1:3].transpose(1,2);f.prepare()
  for eid,slot in [(2,0),(1,1)]:
   tasks=m._build_expert_h2d_tasks(f.layer,0,eid,slot)
   self.assertTrue(all(not isinstance(t.source,real_torch.Tensor) and not isinstance(t.destination,real_torch.Tensor) for t in tasks))
   m.h2d_transport.copy_batch(tasks);self.assertTrue(real_torch.equal(f.layer.scale[slot],m.scale_cpu_buffers['scale'][0][eid].reshape(2,2).double()))
  self.assertTrue(real_torch.all(base[0]==-91));self.assertTrue(real_torch.all(base[3]==-91))
 def test_packed_source_mutation_requires_safe_rebuild(self):
  f=Fixture(cast=True);f.prepare();f.manager.scale_cpu_buffers['scale'][0][0].add_(1)
  with self.assertRaisesRegex(RuntimeError,'drain DMA'):f.manager._build_expert_h2d_tasks(f.layer,0,0,0)
 def test_mapping_storage_copy_is_live_and_offset_safe(self):
  source=real_torch.arange(12,dtype=real_torch.int32)[2:5];owner=real_torch.full((10,),-1,dtype=real_torch.int32);destination=owner[4:7]
  task=module.prepare_mapping_copy(source,destination)
  for value in [2,7]:
   source.fill_(value);Transport().copy_batch([task]);self.assertTrue(real_torch.equal(source,destination));self.assertEqual(int(owner[:4].sum()),-4);self.assertEqual(int(owner[7:].sum()),-3)
 def test_derived_scale_matches_two_stage_cast(self):
  f=Fixture();m=f.manager;m.scale_cpu_buffers={'w13_weight_scale':m.scale_cpu_buffers['scale']};f.layer.w13_weight_scale=real_torch.zeros((2,2,2),dtype=real_torch.uint8);f.layer.w13_weight_scale_fp32=real_torch.zeros((2,2,2),dtype=real_torch.float32)
  m.scale_cpu_buffers['w13_weight_scale'][0][1].fill_(3.75);f.prepare();self.assertTrue(m._prepared_expert_h2d[0].provides_fp32_scale)
  m._load_expert_weights_into_slots(f.layer,0,[(0,1)])
  self.assertTrue(real_torch.all(f.layer.w13_weight_scale_fp32[0]==3))
 def test_overlapping_or_holey_quant_layout_falls_back(self):
  f=Fixture();f.layer.scale=real_torch.zeros((2,2,4))[:,:,:2];f.prepare();self.assertNotIn(0,f.manager._prepared_expert_h2d)


 def make_single_buffer_fixture(self,shard=False,ep=1):
  f=Fixture();m=f.manager;m.scale_cpu_buffers={};m.ep_size=ep;m.offload_config.shard_per_rank=shard
  f.layer.w13_weight_scale=real_torch.empty((2,4,8),dtype=real_torch.uint8)
  f.layer.w2_weight_scale=real_torch.empty((2,2,8),dtype=real_torch.uint8)
  calls=[]
  def allocate(shape,dtype):
   value=real_torch.empty(shape,dtype=dtype);calls.append(value.untyped_storage().data_ptr());return value
  m._allocate_expert_host_tensor=allocate;m._init_layer_scale_buffers(f.layer,0,3)
  f.layer.w13_weight_scale=layout(f.layer.w13_weight_scale);f.layer.w2_weight_scale=layout(f.layer.w2_weight_scale)
  m._weight_load_calls=0;m._scale_shard_temp={};m._get_load_pool=lambda:NS(submit=lambda fn,*args:fn(*args));m._track_load_future=lambda _:None
  expected={}
  for eid in range(len(m.scale_cpu_buffers['w13_weight_scale'][0])):
   a=((real_torch.arange(16).reshape(2,8)+eid*31)%251).to(real_torch.uint8)
   b=((real_torch.arange(16).reshape(2,8)+83+eid*29)%251).to(real_torch.uint8)
   down=((real_torch.arange(16).reshape(2,8)+147+eid*17)%251).to(real_torch.uint8)
   order=[('w1',a),('w3',b)] if eid%2==0 else [('w3',b),('w1',a)]
   for name,value in order:m._load_scale_shard(0,eid,'w13_weight_scale',name,value)
   m._load_scale_shard(0,eid,'w2_weight_scale','w2',down)
   expected[eid]={'w13_weight_scale':layout(real_torch.cat([a,b])),'w2_weight_scale':layout(down)}
  self.assertFalse(m._scale_shard_temp)
  return f,calls,expected

 def test_single_buffer_actual_shard_loader_produces_final_bytes(self):
  f,calls,expected=self.make_single_buffer_fixture();m=f.manager
  self.assertEqual(len(calls),6)
  for eid,fields in expected.items():
   for name,value in fields.items():
    source=m.scale_cpu_buffers[name][0][eid]
    self.assertEqual(source.stride(),getattr(f.layer,name)[0].stride())
    self.assertTrue(real_torch.equal(source,value))
    self.assertEqual(source.untyped_storage().nbytes(),source.numel()*source.element_size())
    # Both have equal logical values and identical physical bytes.
    self.assertTrue(real_torch.equal(real_torch.tensor(list(source.untyped_storage())),real_torch.tensor(list(value.untyped_storage()))))

 def test_single_buffer_preparation_reuses_original_allocation(self):
  for shard in (False,True):
   f,calls,expected=self.make_single_buffer_fixture(shard=shard);m=f.manager
   owners={t.untyped_storage().data_ptr() for fields in m.scale_cpu_buffers.values() for t in fields[0]}
   m._allocate_expert_host_tensor=lambda *args: (_ for _ in ()).throw(AssertionError('Allocated a duplicate scale slab'))
   for _ in range(2):
    f.prepare();plan=m._prepared_expert_h2d[0]
    self.assertEqual(len(plan.quant_fields),2)
    for field in plan.quant_fields:
     self.assertIsNone(field.versions)
     self.assertTrue(all(t.untyped_storage().data_ptr() in owners for t in field.logical_sources))
    for eid,slot in [(2,0),(0,1),(1,0)]:
     tasks=m._build_expert_h2d_tasks(f.layer,0,eid,slot)
     self.assertTrue(all(isinstance(t.source,real_torch.UntypedStorage) and isinstance(t.destination,real_torch.UntypedStorage) for t in tasks))
     m.h2d_transport.copy_batch(tasks)
     for name,value in expected[eid].items():self.assertTrue(real_torch.equal(getattr(f.layer,name)[slot],value))
   self.assertEqual(len(calls),6)

 def test_single_buffer_prefill_and_decode_use_same_sources(self):
  f,_,expected=self.make_single_buffer_fixture();m=f.manager;f.prepare()
  m._prefill_w13_scale=[layout(real_torch.zeros((3,4,8),dtype=real_torch.uint8))]
  m._prefill_w2_scale=[layout(real_torch.zeros((3,2,8),dtype=real_torch.uint8))]
  for eid,slot in [(2,1),(0,2),(1,0)]:
   m.h2d_transport.copy_batch(m._build_prefill_h2d_tasks(0,0,eid,slot))
   self.assertTrue(real_torch.equal(m._prefill_w13_scale[0][slot],expected[eid]['w13_weight_scale']))
   self.assertTrue(real_torch.equal(m._prefill_w2_scale[0][slot],expected[eid]['w2_weight_scale']))
   m.h2d_transport.copy_batch(m._build_expert_h2d_tasks(f.layer,0,eid,slot%2))
   self.assertTrue(real_torch.equal(f.layer.w13_weight_scale[slot%2],expected[eid]['w13_weight_scale']))

 def test_single_buffer_strided_storage_offsets_are_raw_copy_safe(self):
  f=Fixture();m=f.manager;base=real_torch.full((4,3,2),-91.,dtype=real_torch.float32);f.layer.scale=base[1:3].transpose(1,2)
  shape=f.layer.scale[0].shape;strides=f.layer.scale[0].stride();sources=[]
  for eid in range(3):
   owner=real_torch.full((10,),-33.,dtype=real_torch.float32);src=owner.as_strided(shape,strides,2);src.copy_(real_torch.arange(6).reshape(shape)+eid*10);sources.append(src)
  m.scale_cpu_buffers={'scale':[sources]};m._allocate_expert_host_tensor=lambda *args: (_ for _ in ()).throw(AssertionError('Duplicate strided scale'))
  f.prepare()
  for eid,slot in [(2,0),(0,1),(1,0)]:
   m.h2d_transport.copy_batch(m._build_expert_h2d_tasks(f.layer,0,eid,slot));self.assertTrue(real_torch.equal(f.layer.scale[slot],sources[eid]))
  self.assertTrue(real_torch.all(base[0]==-91));self.assertTrue(real_torch.all(base[3]==-91))

 def test_multirank_scale_allocation_keeps_existing_layout(self):
  f,calls,expected=self.make_single_buffer_fixture(shard=True,ep=2)
  self.assertEqual(len(calls),2)
  for name,value in expected[0].items():
   source=f.manager.scale_cpu_buffers[name][0][0];self.assertTrue(source.is_contiguous());self.assertTrue(real_torch.equal(source,value))

if __name__=='__main__':
 suite=unittest.TestSuite(Tests(name) for name in ('test_worker_reaches_prepared_decode_and_pipelines','test_prefill_uses_pool_and_never_decode_cache')) if os.environ.get('H2D_REVIEW_OLD')=='1' else unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
 assert 'torch_npu' not in sys.modules, 'Unexpected NPU backend autoload'
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'NPU_runtime_initialized':False}))
 sys.exit(not result.wasSuccessful())
