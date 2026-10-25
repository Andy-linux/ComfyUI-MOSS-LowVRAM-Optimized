"""CPU-only launcher decisions: no subprocess, imports of Comfy, or GPU probes."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('allocator_launcher',TOOLS/'launch_comfy_allocator.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
class LauncherTests(unittest.TestCase):
    def fixture(self,root,flags):
        main=Path(root)/'main.py';main.write_text('raise AssertionError("must never execute")')
        cli=main.parent/'comfy/cli_args.py';cli.parent.mkdir()
        cli.write_text('raise AssertionError("must never import")\n'+''.join('parser.add_argument('+repr(flag)+')\n' for flag in flags))
        return main
    def test_existing_mechanism_prefers_async(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,['--cuda-malloc','--disable-cuda-malloc'])
            with patch.object(launch,'run_probe',return_value={'status':'PASS'}) as probe:
                row=launch.select_backend(main)
                self.assertEqual(probe.call_args.args[1],'cudaMallocAsync')
                self.assertEqual(row['selected_backend'],'cudaMallocAsync');self.assertEqual(row['comfy_allocator_flags'],['--cuda-malloc'])
    def test_missing_async_uses_native_and_known_disable(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,['--disable-cuda-malloc'])
            with patch.object(launch,'run_probe',return_value={'status':'PASS'}) as probe:
                row=launch.select_backend(main)
                self.assertEqual(probe.call_args.args[1],'native');self.assertEqual(probe.call_count,1)
                self.assertIn('mechanism unavailable',row['fallback']['reason'])
                self.assertEqual(row['comfy_allocator_flags'],['--disable-cuda-malloc'])
    def test_missing_both_flags_native_env_only(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,[])
            with patch.object(launch,'run_probe',return_value={'status':'PASS'}):row=launch.select_backend(main)
            self.assertEqual(row['selected_backend'],'native');self.assertEqual(row['comfy_allocator_flags'],[])
    def test_failed_correctness_never_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,['--cuda-malloc'])
            with patch.object(launch,'run_probe',return_value={'status':'FAILED'}) as probe:row=launch.select_backend(main)
            self.assertEqual(row['status'],'FAILED');self.assertEqual(probe.call_count,1)
    def test_unsupported_falls_back_in_new_process(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,['--cuda-malloc','--disable-cuda-malloc'])
            with patch.object(launch,'run_probe',side_effect=[{'status':'UNSUPPORTED'},{'status':'PASS'}]) as probe:row=launch.select_backend(main)
            self.assertEqual([call.args[1] for call in probe.call_args_list],['cudaMallocAsync','native'])
            self.assertTrue(row['new_process']);self.assertEqual(row['selected_backend'],'native')
    def test_explicit_native_never_probes_async(self):
        with tempfile.TemporaryDirectory() as d:
            main=self.fixture(d,['--cuda-malloc','--disable-cuda-malloc'])
            with patch.object(launch,'run_probe',return_value={'status':'PASS'}) as probe:row=launch.select_backend(main,'native')
            self.assertEqual(probe.call_count,1);self.assertEqual(probe.call_args.args[1],'native')
            self.assertEqual(row['requested_backend'],'native');self.assertIsNone(row['fallback'])
if __name__=='__main__':unittest.main()
