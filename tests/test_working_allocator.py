"""Bounded working windows with fake accelerator APIs; no CUDA/Torch required."""

import importlib.util, sys, types, unittest
from pathlib import Path
from unittest.mock import patch


class Device:
    def __init__(self, value, index=None):
        self.type = (
            value.type if isinstance(value, Device) else str(value).split(":")[0]
        )
        self.index = (
            value.index
            if isinstance(value, Device)
            else index
            if index is not None
            else int(str(value).split(":")[1])
            if ":" in str(value)
            else None
        )

    def __str__(self):
        return self.type + (":" + str(self.index) if self.index is not None else "")


p = (
    Path(__file__).resolve().parents[1]
    / "runtime/vendor/kohaku_src/kohakumoss_kernels/hardware/memory.py"
)
spec = importlib.util.spec_from_file_location("working_memory", p)
m = importlib.util.module_from_spec(spec)
with patch.dict(
    sys.modules,
    {
        "torch": types.SimpleNamespace(device=Device),
        "kohakumoss_kernels.hardware.current": types.SimpleNamespace(
            device_module=lambda d: None
        ),
    },
):
    spec.loader.exec_module(m)


class Accelerator:
    def __init__(self, backend="cudaMallocAsync", fraction=1.0, fail_at=None):
        self.memory = types.SimpleNamespace(get_allocator_backend=lambda: backend)
        self.fraction = fraction
        self.calls = []
        self.fail_at = fail_at

    def get_per_process_memory_fraction(self, d):
        return self.fraction

    def set_per_process_memory_fraction(self, x, d):
        self.calls.append(x)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("injected setter error")
        self.fraction = x

    def memory_allocated(self, d):
        return 20

    def memory_reserved(self, d):
        return 80

    def max_memory_allocated(self, d):
        return 30

    def max_memory_reserved(self, d):
        return 90

    def reset_peak_memory_stats(self, d):
        pass

    def mem_get_info(self, d):
        return 800, 1000


class WorkingTests(unittest.TestCase):
    def run_with(self, mod):
        return patch.object(m, "_accel", return_value=mod)

    def tearDown(self):
        self.assertFalse(any(m._OPEN.values()))

    def test_async_unset_fraction_and_used_basis(self):
        mod = Accelerator(fraction=1074299169.1500106)
        with self.run_with(mod), m.working("cuda:0", 100) as w:
            self.assertTrue(w.enforced)
        self.assertAlmostEqual(mod.calls[0], 0.12)
        self.assertEqual(mod.calls[-1], 1.0)

    def test_outer_cap_never_relaxed(self):
        mod = Accelerator(fraction=0.05)
        with self.run_with(mod), m.working("cuda:0", 100):
            pass
        self.assertEqual(mod.calls, [0.05, 0.05])

    def test_native_and_xpu_use_reserved_basis(self):
        for device in ("cuda:0", "xpu:0"):
            mod = Accelerator(backend="native")
            with self.run_with(mod), m.working(device, 100):
                pass
            self.assertAlmostEqual(mod.calls[0], 0.18)

    def test_setter_and_restore_errors_close_window(self):
        for fail_at in (1, 2):
            mod = Accelerator(fail_at=fail_at)
            with self.run_with(mod):
                with self.assertRaisesRegex(RuntimeError, "injected setter error"):
                    with m.working("cuda:0", 100):
                        pass
            self.assertFalse(any(m._OPEN.values()))

    def test_invalid_fraction_closes_window_without_setting(self):
        for invalid in (-1, float("nan"), float("inf")):
            mod = Accelerator(fraction=invalid)
            with self.run_with(mod):
                with self.assertRaises(ValueError):
                    with m.working("cuda:0", 100):
                        pass
            self.assertEqual(mod.calls, [])

    def test_without_getter_does_not_relax_unknown_outer_cap(self):
        mod = Accelerator(fraction=0.05)
        mod.get_per_process_memory_fraction = None
        with self.run_with(mod), m.working("cuda:0", 100) as w:
            self.assertFalse(w.enforced)
        self.assertEqual(mod.calls, [])


if __name__ == "__main__":
    unittest.main()
