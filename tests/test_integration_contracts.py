"""CPU-only cross-module contracts; no ComfyUI import, model load or CUDA."""
import ast,importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
session=load('moss_contract_session','runtime/session.py')
plans=load('moss_contract_plan','runtime/profile_plan.py')
assets=load('moss_contract_assets','runtime/assets.py')
def literal(name,relative):
    tree=ast.parse((ROOT/relative).read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise AssertionError(name)
class IntegrationContracts(unittest.TestCase):
    def test_fp8_dispatch_does_not_inherit_integer_optimizations(self):
        for candidate in session.FP8_CANDIDATES:
            cfg=session.RuntimeConfig(candidate=candidate,profile='4g',qkv_backend='auto',ffn_tile=-1,rope_inplace=True,qkv_reuse_input=True,projection_tile=2048,ffn_decode_reuse=True,writer_epilogue=True)
            effective,_=plans.kernel_plan(cfg,(64,64),266,19,89)
            self.assertEqual(effective.kernels,'baseline');self.assertEqual(effective.ffn_tile,2048)
            for field in ('rope_inplace','qkv_reuse_input','projection_tile','ffn_decode_reuse','writer_epilogue'):self.assertFalse(getattr(effective,field))
            with self.assertRaises(ValueError):plans.kernel_plan(cfg,(64,64),266,19,75)
    def test_live_shape_change_requires_new_effective_plan(self):
        cfg=session.RuntimeConfig(candidate='base_fp8',profile='8g',ffn_tile=-1)
        short,_=plans.kernel_plan(cfg,(64,64),266,19,89)
        long,_=plans.kernel_plan(cfg,(104,78),2094,19,89)
        self.assertEqual(short.ffn_tile,0);self.assertEqual(long.ffn_tile,2048)
        self.assertNotEqual(short,long)  # Owner must apply this change to a reused core.
        self.assertEqual(cfg.ffn_tile,-1)  # Request planning does not mutate config.
    def test_base_integer_plan_preserves_model_identity(self):
        cfg=session.RuntimeConfig(candidate='base_w8_int8c',profile='4g',qkv_backend='auto',ffn_tile=-1)
        effective,receipt=plans.kernel_plan(cfg,(64,64),266,19,89)
        self.assertEqual(effective.candidate,'base_w8_int8c');self.assertEqual(receipt['checkpoint_variant'],'base')
    def test_legacy_widget_values_and_original_precision_are_distinct(self):
        modes=literal('_TEXT','nodes.py');attention=literal('_ATTENTION','nodes.py')
        self.assertEqual(modes['INT8 ConvRot'],('te_int8','auto','fp16'))
        self.assertEqual(modes['W4A8 GPTQ'],('te_w4','auto','fp16'))
        self.assertEqual(modes['Original BF16'],('te_original','original','bf16'))
        self.assertEqual(modes['Original FP16'],('te_original','original','fp16'))
        self.assertEqual(attention['Sage FP8 fast + H128'],'sage_fp8_rot')
        self.assertEqual(attention['Flash FP16'],'flash')
    def test_manifest_assets_resolve_to_maintained_local_names(self):
        self.assertEqual(assets.ASSETS['sft_w5'][0],'diffusion_models')
        self.assertIn('q5_ridge_gptq.safetensors',assets.ASSETS['sft_w5'][1])
        self.assertIn('q6_direct.safetensors',assets.ASSETS['sft_w6'][1])
        self.assertIn('MOSS-Image-Base/text_encoder',assets.ASSETS['te_original'][1])
if __name__=='__main__':unittest.main()
