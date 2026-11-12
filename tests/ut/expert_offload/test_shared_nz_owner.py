"""CPU-only ownership/layout regressions; run in the target Ascend environment."""
import ast
from pathlib import Path
import unittest

import torch
from kt_kernel.utils.nz_mxfp4 import NZWeights


class SharedNZOwnerTests(unittest.TestCase):
    def owner(self):
        experts,hidden,intermediate=2,128,64
        w13=tuple(torch.zeros(hidden*intermediate,dtype=torch.uint8) for _ in range(experts))
        w2=tuple(torch.zeros(hidden*intermediate//2,dtype=torch.uint8) for _ in range(experts))
        def scale(n,g):
            return tuple(torch.empty_strided((g//2,n,2),(2,g,1),dtype=torch.uint8).fill_(127)
                         for _ in range(experts))
        return NZWeights(w13,w2,scale(2*intermediate,hidden//32),scale(hidden,intermediate//32))

    def test_borrowed_owners_and_pointers_remain_identical(self):
        owner=self.owner();before=owner.pointers();objects=[[id(t) for t in field] for field in owner.fields]
        self.assertIs(owner.validate(2,128,64),owner)
        self.assertEqual(owner.pointers(),before)
        self.assertEqual([[id(t) for t in field] for field in owner.fields],objects)

    def test_reject_wrong_scale_layout(self):
        owner=self.owner();bad=NZWeights(owner.w13,owner.w2,tuple(t.contiguous() for t in owner.scale13),owner.scale2)
        with self.assertRaisesRegex(ValueError,'final-stride'):bad.validate(2,128,64)

    def test_reject_reserved_e8m0_scale(self):
        owner=self.owner();owner.scale13[0][0,0,0]=255
        with self.assertRaisesRegex(ValueError,'Reserved'):owner.validate(2,128,64)

    def test_reject_expert_count_and_dtype(self):
        owner=self.owner()
        with self.assertRaisesRegex(ValueError,'count'):owner.validate(3,128,64)
        bad=NZWeights(tuple(t.float() for t in owner.w13),owner.w2,owner.scale13,owner.scale2)
        with self.assertRaisesRegex(ValueError,'uint8'):bad.validate(2,128,64)

    def test_npu_slot_nz_helper_survives_old_mode_removal(self):
        root=Path(__file__).resolve().parents[3]
        tree=ast.parse((root/'vllm_ascend/expert_offload/expert_offload_manager.py').read_text())
        manager=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExpertOffloadManager')
        methods={n.name:n for n in manager.body if isinstance(n,ast.FunctionDef)}
        self.assertIn('_cast_device_slots_to_mxfp4_nz',methods)
        self.assertNotIn('native_mxfp4_enabled',methods)
        call=methods['process_weights_after_loading']
        self.assertTrue(any(isinstance(n,ast.Attribute) and n.attr=='_cast_device_slots_to_mxfp4_nz'
                            for n in ast.walk(call)))


if __name__=='__main__':unittest.main()
