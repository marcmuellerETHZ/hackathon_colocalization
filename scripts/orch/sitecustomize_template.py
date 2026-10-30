"""orch per-job sitecustomize (copied to runs/<job>/site/, first on PYTHONPATH).

If ORCH_VRAM_GB is set (GPU jobs), cap PyTorch's caching allocator at the declared
VRAM: when torch is imported, wrap torch.cuda._lazy_init so that right after CUDA
initialises, torch.cuda.set_per_process_memory_fraction(vram_gb / total) is applied
on every visible device. Allocations beyond the cap raise torch.OutOfMemoryError
("CUDA out of memory"), which orch classifies as CUDA OOM. Non-torch jobs are untouched.
"""
import os
import sys

_VRAM = os.environ.get("ORCH_VRAM_GB")

if _VRAM:
    import importlib.abc
    import importlib.machinery

    def _orch_apply(torch):
        cuda = torch.cuda
        orig = cuda._lazy_init
        state = {"done": False}

        def _lazy_init():
            orig()
            if not state["done"]:
                state["done"] = True
                for d in range(cuda.device_count()):
                    total = cuda.get_device_properties(d).total_memory
                    frac = min(1.0, float(_VRAM) * 2**30 / total)
                    cuda.set_per_process_memory_fraction(frac, d)
                os.environ["ORCH_VRAM_FRACTION_SET"] = "1"

        cuda._lazy_init = _lazy_init

    class _OrchTorchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "torch":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _orch_apply(module)
                except Exception as e:  # never break the job because of the cap
                    print(f"[orch sitecustomize] could not install VRAM cap: {e}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _OrchTorchFinder())
