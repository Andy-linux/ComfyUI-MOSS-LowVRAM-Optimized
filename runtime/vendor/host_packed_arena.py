"""Request-independent pageable CPU packed arena; no CUDA at import/capture.

Service owns HostPackedArena/HostArenaCache. Request owns only registered GPU
views plus a weak ArenaHandle. CPU payload is private and never attached to models.
A hit avoids repacking; every fresh CUDA model still needs its packed H2D copy.
The source identity is an explicit checkpoint revision + observed file identity;
it is not a replacement for auditing a new loader or authenticating file content.
"""

import hashlib
import json
import threading
import weakref
import time
from collections import OrderedDict
from pathlib import Path

import torch


def checkpoint_identity(path, *, revision):
    """Pin a caller-audited content revision and current stat/header identity.

    revision must be a nonempty immutable identifier supplied by the loader
    (prefer a full-file SHA256 or repository commit + artifact digest).
    No model payload is hashed here; safetensors header is read without tensors.
    """
    p = Path(path).resolve()
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Explicit immutable checkpoint revision required")
    st = p.stat()
    with p.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        n = int.from_bytes(raw, "little")
        if not 2 <= n <= min(st.st_size - 8, 64 * 2**20):
            raise ValueError("Invalid safetensors header length")
        header = f.read(n)
    return dict(
        path=p.as_posix(),
        revision=revision,
        bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,
        ctime_ns=st.st_ctime_ns,
        header_sha256=hashlib.sha256(header).hexdigest(),
        header_hash_scope="JSON header bytes excluding 8-byte length prefix",
    )


def _identity(value):
    if not isinstance(value, dict) or not value.get("revision"):
        raise ValueError("source_identity must include explicit immutable revision")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _check_file(identity):
    if "path" not in identity:
        return
    p = Path(identity["path"])
    st = p.stat()
    if any(
        identity.get(k) != v
        for k, v in (
            ("bytes", st.st_size),
            ("mtime_ns", st.st_mtime_ns),
            ("ctime_ns", st.st_ctime_ns),
        )
    ):
        raise RuntimeError("Checkpoint changed since host arena capture")


def _digest(t):
    return hashlib.sha256(memoryview(t.detach().numpy())).hexdigest()


def _plans(bb, alignment, base_type=None):
    if base_type is None:
        from ..storage import FrozenPackedBase

        base_type = FrozenPackedBase
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    plans, end = [], 0
    for name, base in bb.named_modules():
        if type(base) is not base_type:
            continue
        q, s = base.qweight, base.wscales
        if (
            q.device.type not in ("cpu", "meta")
            or q.dtype != torch.uint8
            or not q.is_contiguous()
        ):
            raise ValueError(
                "Expected original CPU/meta contiguous packed payload: " + name
            )
        if base.bits not in (5, 6) or tuple(q.shape) != (
            (base.k * base.bits + 7) // 8,
            s.shape[0],
        ):
            raise ValueError("Original packed layout mismatch: " + name)
        if (
            s.device.type != "cpu"
            or s.dtype != torch.float32
            or tuple(s.shape) != (q.shape[1], 1)
        ):
            raise ValueError("Original row scale contract mismatch: " + name)
        end = (end + alignment - 1) // alignment * alignment
        key = (name[:-5] if name.endswith(".base") else name) + ".qweight"
        plans.append(
            dict(
                module=name,
                state_key=key,
                offset=end,
                shape=list(q.shape),
                stride=list(q.stride()),
                bytes=q.numel(),
                bits=base.bits,
                k=base.k,
                scale_shape=list(s.shape),
                scale_sha256=_digest(s),
            )
        )
        end += q.numel()
    if not plans:
        raise ValueError("No exact FrozenPackedBase buffers")
    return plans, end


class RequestArenaHandle:
    """Weak model reference; no Tensor/device slab retained by handle."""

    def __init__(self, owner, names, receipt):
        self._owner = weakref.ref(owner)
        self._names = tuple(names)
        self.receipt = receipt

    def storage_receipt(self):
        bb = self._owner()
        if bb is None:
            return dict(owner_alive=False, unique_storage_bytes=0, devices=[])
        storages = {}
        logical = 0
        for name in self._names:
            q = bb.get_submodule(name).qweight
            logical += q.numel() * q.element_size()
            if q.device.type != "meta":
                storage = q.untyped_storage()
                storages[(str(q.device), storage.data_ptr())] = storage.nbytes()
        return dict(
            owner_alive=True,
            logical_qweight_bytes=logical,
            unique_storage_bytes=sum(storages.values()),
            unique_storage_count=len(storages),
            devices=sorted({key[0] for key in storages}),
        )

    def release(self):
        bb = self._owner()
        if bb is not None:
            for name in self._names:
                base = bb.get_submodule(name)
                base.qweight = torch.empty_like(base.qweight, device="meta")
        self.receipt["explicit_release_called"] = True


class HostPackedArena:
    """One immutable host revision; explicit close and no retained GPU Tensor."""

    def __init__(self):
        raise TypeError("Use HostPackedArena.capture")

    @classmethod
    def capture(cls, bb, *, source_identity, alignment=256, audit=True, base_type=None):
        started = time.perf_counter()
        key = _identity(source_identity)
        _check_file(source_identity)
        if base_type is None:
            from ..storage import FrozenPackedBase

            base_type = FrozenPackedBase
        plans, end = _plans(bb, alignment, base_type)
        if any(
            bb.get_submodule(p["module"]).qweight.device.type != "cpu" for p in plans
        ):
            raise ValueError("Capture requires actual original CPU payloads")
        # A normal Tensor version counter is needed even if caller uses inference_mode.
        planned = time.perf_counter()
        with torch.inference_mode(False):
            host = torch.empty(end, dtype=torch.uint8, device="cpu", pin_memory=False)
        allocated = time.perf_counter()
        payload = hashlib.sha256()
        cursor = 0
        copy_s = auth_s = 0.0
        for p in plans:
            q = bb.get_submodule(p["module"]).qweight
            view = host.narrow(0, p["offset"], p["bytes"]).view(p["shape"])
            t = time.perf_counter()
            if p["offset"] > cursor:
                host[cursor : p["offset"]].zero_()
            view.copy_(q)
            cursor = p["offset"] + p["bytes"]
            copy_s += time.perf_counter() - t
            t = time.perf_counter()
            p["sha256"] = _digest(q) if audit else None
            if audit:
                assert _digest(view) == p["sha256"]
                payload.update(memoryview(q.numpy()))
            auth_s += time.perf_counter() - t
        obj = object.__new__(cls)
        obj._host = host
        obj._version = host._version
        obj._identity = json.loads(key)
        obj._key = key
        obj._plans = plans
        obj._base_type = base_type
        obj._alignment = alignment
        obj._lock = threading.RLock()
        obj._closed = False
        obj.receipt = dict(
            version=1,
            host_storage_bytes=end,
            logical_bytes=sum(p["bytes"] for p in plans),
            projection_count=len(plans),
            payload_sha256=payload.hexdigest() if audit else None,
            full_payload_authenticated=audit,
            capture_s=time.perf_counter() - started,
            capture_plan_s=planned - started,
            host_allocation_s=allocated - planned,
            host_copy_s=copy_s,
            full_payload_auth_s=auth_s,
            host_pinned=False,
            gpu_tensor_retained=False,
            immutable_host_private=True,
            source_identity=json.loads(key),
            uploads=0,
            upload_bytes=0,
        )
        return obj

    @property
    def nbytes(self):
        return 0 if self._closed else self.receipt["host_storage_bytes"]

    def _validate(self, source_identity):
        if self._closed:
            raise RuntimeError("Host packed arena closed")
        if _identity(source_identity) != self._key:
            raise ValueError("Host arena checkpoint revision/identity mismatch")
        _check_file(self._identity)
        if self._host._version != self._version:
            raise RuntimeError("Immutable host arena was mutated")

    def upload_into(
        self, bb, *, source_identity, device, release_state=None, verify_model=True
    ):
        """Validate all modules, then copy one slab and bind original buffer views.

        verify_model=True authenticates every input qweight byte against capture.
        After a loader is qualified, False can accept CPU/meta skeleton payloads
        from that exact source_identity; caller must not attach arbitrary edited
        weights under the same identity. Scales are NEVER cached or replaced.
        """
        with self._lock:
            self._validate(source_identity)
            target = torch.device(device)
            if target.type not in ("cpu", "cuda") or (
                target.type == "cuda" and target.index is None
            ):
                raise ValueError("Use explicit cuda:N or CPU validation")
            if release_state is not None and not isinstance(release_state, dict):
                raise TypeError("release_state must be original state dict")
            if getattr(bb, "_weight_arena_v4_metadata", None) is not None:
                raise RuntimeError("A request arena is already installed")
            plans, end = _plans(bb, self._alignment, self._base_type)
            template = [
                {k: v for k, v in p.items() if k != "sha256"} for p in self._plans
            ]
            if plans != template or end != self.nbytes:
                raise ValueError("Model layout/packing differs from host revision")
            for p, original in zip(plans, self._plans):
                q = bb.get_submodule(p["module"]).qweight
                if verify_model:
                    if original["sha256"] is None:
                        raise ValueError("verify_model requires audit=True at capture")
                    if q.device.type != "cpu" or _digest(q) != original["sha256"]:
                        raise ValueError(
                            "Model packed bytes differ from immutable host revision: "
                            + p["module"]
                        )
                if (
                    release_state is not None
                    and p["state_key"] in release_state
                    and release_state[p["state_key"]] is not q
                ):
                    raise ValueError(
                        "Original state tensor identity mismatch: " + p["state_key"]
                    )
            started = time.perf_counter()
            # CPU clone ensures CPU tests cannot expose writable views of private host.
            slab = (
                self._host.to(target, non_blocking=False)
                if target.type == "cuda"
                else self._host.clone()
            )
            released = 0
            for p in plans:
                base = bb.get_submodule(p["module"])
                old = base.qweight
                base.qweight = slab.narrow(0, p["offset"], p["bytes"]).view(p["shape"])
                if (
                    release_state is not None
                    and release_state.get(p["state_key"]) is old
                ):
                    del release_state[p["state_key"]]
                    released += p["bytes"]
            self.receipt["uploads"] += 1
            self.receipt["upload_bytes"] += end if target.type == "cuda" else 0
            receipt = dict(
                version="host-v1",
                target=str(target),
                host_arena_reused=True,
                host_repacking_this_request=False,
                arena_storage_bytes=end,
                projection_count=len(plans),
                packed_H2D_copy_calls=1 if target.type == "cuda" else 0,
                copy_scope="one Tensor.to; no claim about driver DMA",
                upload_and_bind_s=time.perf_counter() - started,
                source_state_qweight_bytes_released=released,
                source_identity=json.loads(self._key),
                CPU_input_payload_hash_checked=verify_model,
                gpu_tensor_retained_by_host=False,
                entries=[dict(p) for p in self._plans],
            )
            bb._weight_arena_v4_metadata = receipt
            handle = RequestArenaHandle(bb, [p["module"] for p in plans], receipt)
            del slab
            receipt["storage_after_install"] = handle.storage_receipt()
            return handle

    def close(self):
        with self._lock:
            self._host = None
            self._closed = True


class HostArenaCache:
    """Explicit service-owned byte-bounded LRU; no process-global hidden cache."""

    def __init__(self, *, max_bytes):
        if max_bytes <= 0:
            raise ValueError("Positive host budget required")
        self.max_bytes = int(max_bytes)
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def get_or_capture(
        self, bb, *, source_identity, alignment=256, audit=True, base_type=None
    ):
        key = (_identity(source_identity), alignment, bool(audit), base_type)
        with self._lock:
            if key in self._entries:
                arena = self._entries[key]
                arena._validate(source_identity)
                self._entries.move_to_end(key)
                return arena, True
            _, needed = _plans(bb, alignment, base_type)
            if needed > self.max_bytes:
                raise MemoryError("Host arena exceeds cache budget")
            while (
                sum(a.nbytes for a in self._entries.values()) + needed > self.max_bytes
            ):
                _, old = self._entries.popitem(last=False)
                old.close()
            arena = HostPackedArena.capture(
                bb,
                source_identity=source_identity,
                alignment=alignment,
                audit=audit,
                base_type=base_type,
            )
            self._entries[key] = arena
            return arena, False

    def close(self):
        with self._lock:
            for arena in self._entries.values():
                arena.close()
            self._entries.clear()
