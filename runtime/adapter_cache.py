"""Byte-bounded exact CPU cache of frozen TextAdapter evaluations.

Entries belong to one session. Fingerprints select entries; full input equality
is checked before reuse. No CUDA tensor, model module, or conditioning view is
retained. Different prompts still run the normal text encoder.
"""
from collections import OrderedDict
import hashlib
import torch


class ExactAdapterCache:
    def __init__(self, max_bytes=64 * 1024**2):
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.nbytes = 0
        self.hits = self.misses = self.evictions = self.oversized = 0

    def key(self, context, text):
        if text.device.type != "cpu" or text.dtype not in (torch.float16,torch.bfloat16) or not text.is_contiguous():
            raise ValueError("Adapter cache uses the actual contiguous FP16/BF16 CPU input snapshot")
        digest = hashlib.sha256(memoryview(text.view(torch.uint8).numpy())).digest()
        return context, tuple(text.shape), str(text.dtype), digest

    def get(self, key, text):
        entry = self.entries.get(key)
        if entry is None or not torch.equal(entry[0], text):
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return entry[1]

    def put(self, key, text, output):
        if output.device.type != "cpu":
            raise ValueError("Adapter output cache must stay on CPU")
        size = text.numel() * text.element_size() + output.numel() * output.element_size()
        if size > self.max_bytes:
            self.oversized += 1
            return
        if key in self.entries:
            self.nbytes -= self.entries.pop(key)[2]
        while self.entries and self.nbytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.nbytes -= old[2]
            self.evictions += 1
        self.entries[key] = (text.clone(), output.contiguous().clone(), size)
        self.nbytes += size

    def clear(self):
        self.entries.clear()
        self.nbytes = 0

    def receipt(self):
        return dict(max_CPU_bytes=self.max_bytes, resident_CPU_bytes=self.nbytes,
                    entries=len(self.entries), hits=self.hits, misses=self.misses,
                    evictions=self.evictions, oversized=self.oversized,
                    key="checkpoint revision, DiT numeric mode, shape/positions/bands, exact input bytes and dtype",
                    equality="full torch.equal after hash lookup", CUDA_tensor_retained=False)
