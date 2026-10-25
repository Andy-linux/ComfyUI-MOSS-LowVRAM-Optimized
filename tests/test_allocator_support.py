"""CPU-only tests for backend policy; GPU evidence comes from allocator_probe."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


alloc = module("moss_allocator", ROOT / "runtime/allocator.py")
probe = module("allocator_probe", ROOT / "tools/allocator_probe.py")


class PolicyTests(unittest.TestCase):
    def test_unlimited_async_fraction_is_restorable(self):
        self.assertEqual(alloc.restorable_fraction(1074299169.1500106), 1.0)
        self.assertEqual(alloc.restorable_fraction(0.5), 0.5)
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                alloc.restorable_fraction(invalid)

    def test_environment_aliases_and_array_options(self):
        env = probe.allocator_env(
            {
                "PYTORCH_ALLOC_CONF": "backend:native,roundup_power2_divisions:[256:1,512:2],pinned_use_cuda_host_register:True"
            },
            "cudaMallocAsync",
        )
        self.assertEqual(env["PYTORCH_ALLOC_CONF"], env["PYTORCH_CUDA_ALLOC_CONF"])
        self.assertEqual(
            env["PYTORCH_ALLOC_CONF"],
            "pinned_use_cuda_host_register:True,backend:cudaMallocAsync",
        )

    def test_async_reserved_does_not_consume_fraction_twice(self):
        cuda = NS(
            memory=NS(get_allocator_backend=lambda: "cudaMallocAsync"),
            memory_allocated=lambda d: 20,
            memory_reserved=lambda d: 80,
            max_memory_allocated=lambda d: 25,
            max_memory_reserved=lambda d: 90,
            get_device_properties=lambda d: NS(total_memory=100),
            get_per_process_memory_fraction=lambda d: 0.5,
            mem_get_info=lambda d: (10, 100),
        )
        row = alloc.admission(NS(cuda=cuda), 0, external_bytes=5)
        self.assertEqual(row["effective_available_bytes"], 25)
        self.assertFalse(row["reserved_is_fraction_capped"])
        self.assertEqual(row["fraction_scope"], "pytorch_requested_bytes")

    def test_native_semantics_remain_labeled(self):
        cuda = NS(
            memory=NS(get_allocator_backend=lambda: "native"),
            memory_allocated=lambda d: 20,
            memory_reserved=lambda d: 40,
            max_memory_allocated=lambda d: 25,
            max_memory_reserved=lambda d: 45,
        )
        row = alloc.observation(NS(cuda=cuda), 0)
        self.assertTrue(row["reserved_is_fraction_capped"])
        self.assertTrue(row["snapshot_supported"])


if __name__ == "__main__":
    unittest.main()
