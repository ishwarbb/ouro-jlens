import unittest
from unittest.mock import patch
import run8_relocate_gpu as relocation
import run_deception_run8 as run8
import run8_recovery as base


class RelocationTests(unittest.TestCase):
    def test_only_operational_gpu_sites_change(self):
        original=run8.run
        updated=relocation.relocated_runner(original,'1')
        self.assertEqual(updated.__globals__['relocation_gpu'],'1')
        self.assertIs(updated.__globals__['benchmark_episode'],original.__globals__['benchmark_episode'])
        self.assertIs(updated.__globals__['tool_episode'],original.__globals__['tool_episode'])
        self.assertIs(updated.__globals__['generate'],original.__globals__['generate'])
        self.assertIs(updated.__globals__['verify_frozen'],original.__globals__['verify_frozen'])

    def test_original_environment_is_preserved(self):
        from pathlib import Path
        runner=relocation.relocated_runner(run8.run,'1')
        old={'cuda_visible_devices':'2','dtype':'bfloat16','gpu':'NVIDIA L40S'}
        new={**old,'cuda_visible_devices':'1'}
        with patch.object(relocation,'read',return_value=old),patch.object(relocation,'immutable') as save:
            runner.__globals__['relocation_environment'](Path('/test/run'),new)
            self.assertEqual(save.call_args.args[0],Path('/test/run/environment-gpu1-20261013.json'))
            with self.assertRaises(ValueError):
                runner.__globals__['relocation_environment'](Path('/test/run'),{**new,'dtype':'float32'})

    def test_device_patch_restores_after_error(self):
        original=run8.run;numeric=base.preserve_numerical_receipt
        with self.assertRaises(RuntimeError):
            with relocation.device_context('1'):
                self.assertIsNot(run8.run,original)
                raise RuntimeError('test')
        self.assertIs(run8.run,original);self.assertIs(base.preserve_numerical_receipt,numeric)


if __name__=='__main__':unittest.main()
