"""Direct MOSS inference session shared by benchmarks and ComfyUI nodes.

No CLI execution, subprocess generation, or source transformation. CUDA admission
and per-process allocator caps belong to the launcher. Session owns model/cache
lifetimes; callbacks are optional local stage measurements owned by the caller.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
import os
import gc
import importlib.util
import json
import sys
import time

INT8_CANDIDATES = ("w8_int8c", "base_w8_int8c")
FP8_CANDIDATES = ("base_fp8", "sft_fp8")
PACKED_CANDIDATES = ("q5_ridge_gptq", "q6_direct")
CANDIDATE_ASSETS = {
    "q5_ridge_gptq": "sft_w5",
    "q6_direct": "sft_w6",
    "w8_int8c": "sft_w8",
    "base_w8_int8c": "base_w8",
    "base_fp8": "base_fp8",
    "sft_fp8": "sft_fp8",
}


@dataclass(frozen=True)
class RuntimeConfig:
    candidate: str = "q5_ridge_gptq"
    profile: str = "6g"
    precision: str = "fp16_fixed"
    te_weights: str = "auto"
    te_asset: str = "te_int8"
    te_mode: str = "auto"
    te_compute_dtype: str = "fp16"
    dit_weights: str = "auto"
    attention: str = "auto"
    attention_head_group: int = 0
    resident_blocks: int = -1
    payload_pinning: str = "staging"
    qkv_backend: str = "shared"
    ffn_tile: int = 0
    kernels: str = "v3_shared"
    host_arena: bool = False
    projection_tile: int = 0
    sdpa_capacity_fallback: bool = True
    qkv_reuse_input: bool = False
    resident_vae_stage_budget: bool = False
    strict_full_vae: bool = False
    resident_vae_discard: bool = False
    resident_vae_keep_dit: bool = False
    qualified_vae_budget: bool = False
    single_frame: bool = False
    rope_inplace: bool = False
    ffn_lifetime: bool = False
    ffn_decode_reuse: bool = False
    conditioning_schedule: bool = False
    adapter_cache: bool = False
    writer_epilogue: bool = False
    te_memory: str = "ring"
    te_resident_gib: float = 3.0
    vae_name: str = "moss-image-vae.safetensors"
    vae_precision: str = "fp32"
    vae_admission: bool = False
    device: str = "cuda:0"


def bootstrap_paths():
    vendor = Path(__file__).resolve().parent / "vendor"
    expected = (vendor / "kohaku_src").resolve()
    for path in (expected,):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    # A normal Comfy process may already host other custom nodes. Never silently
    # reuse or replace another Kohaku module identity in that process.
    for name in ("kohakumoss", "kohakumoss_kernels", "kohakumoss_common"):
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None) if module is not None else None
        if module is None:
            spec = importlib.util.find_spec(name)
            origin = None if spec is None else spec.origin
        if origin is not None and expected not in Path(origin).resolve().parents:
            raise RuntimeError(
                "Conflicting already loaded or discoverable "
                + name
                + "; expected packaged vendor, found "
                + str(origin)
            )


def _official_loaders():
    from .vendor.official_moss import loaders

    return loaders


class MossSession:
    """One immutable config; close releases resources and next call lazy reopens.

    Model patchers disposed to meta are never reused. Resident patchers remain
    ordinary CPU-offloadable Comfy objects. Host arena survives disposable requests
    only while this session is open. This first implementation supports frozen SFT
    projections and does not accept LoRA patches.
    """

    def __init__(self, config=RuntimeConfig(), stage=None):
        from .resources import BUDGET_GIB
        import torch

        if config.profile not in BUDGET_GIB:
            raise ValueError(config.profile)
        if config.precision not in ("fp16_fixed", "bf16"):
            raise ValueError(config.precision)
        if config.vae_precision not in ("fp32", "bf16"):
            raise ValueError(config.vae_precision)
        if config.vae_precision == "bf16" and (
            config.resident_vae_stage_budget or config.qualified_vae_budget
        ):
            raise ValueError(
                "FP32-specific VAE memory tables cannot be used with the BF16 decoder"
            )
        if not -1 <= config.resident_blocks <= 40:
            raise ValueError("resident_blocks must be -1 (automatic) or 0..40")
        if config.payload_pinning not in ("staging", "suffix"):
            raise ValueError(config.payload_pinning)
        if config.attention_head_group not in (-1, 0, 8, 12, 24):
            raise ValueError(
                "attention_head_group must be -1 (automatic), 0, 8, 12 or 24"
            )
        self.compute_dtype = (
            torch.float16 if config.precision == "fp16_fixed" else torch.bfloat16
        )
        if config.candidate not in CANDIDATE_ASSETS:
            raise ValueError(config.candidate)
        if config.attention not in (
            "auto",
            "flash",
            "cudnn",
            "sage_fp16",
            "sage_fp8",
            "sage_fp8_fast",
            "sage_fp16_rot",
            "sage_fp8_rot",
            "ck_int8",
        ):
            raise ValueError(config.attention)
        if config.qkv_backend not in (
            "auto",
            "shared",
            "native",
            "register",
            "register3",
        ):
            raise ValueError(config.qkv_backend)
        if config.kernels not in ("baseline", "native_shared", "v3_shared"):
            raise ValueError(config.kernels)
        if config.ffn_tile not in (-1, 0, 2048):
            raise ValueError(config.ffn_tile)
        if config.te_memory not in ("ring", "resident", "partial", "auto"):
            raise ValueError(config.te_memory)
        if not 0.25 <= config.te_resident_gib <= 6.0:
            raise ValueError("Partial TE prefix budget must be 0.25..6.0 GiB")
        if config.rope_inplace and (
            config.qkv_backend != "register3"
            or config.candidate not in PACKED_CANDIDATES
        ):
            raise ValueError("In-place RoPE requires independent register3 Q/K outputs")
        if config.qkv_reuse_input and config.qkv_backend != "register3":
            raise ValueError("Destructive norm input reuse requires register3")
        if config.projection_tile not in (0, 2048) or (
            config.projection_tile and config.profile != "4g"
        ):
            raise ValueError(
                "Attention-output projection tiling is a bounded 4g candidate"
            )
        if config.resident_vae_stage_budget and (
            config.profile != "unlimited"
            or config.te_memory not in ("resident", "auto")
            or (
                not (config.resident_vae_discard or config.resident_vae_keep_dit)
                and config.te_memory != "auto"
            )
            or config.qualified_vae_budget
        ):
            raise ValueError(
                "Resident VAE budget requires unlimited resident TE, explicit DiT discard and original estimator"
            )
        if config.resident_vae_keep_dit and (
            config.resident_vae_discard or config.profile != "unlimited"
        ):
            raise ValueError(
                "Explicit resident DiT policy is unlimited-only and mutually exclusive with discard"
            )
        if config.writer_epilogue and config.precision != "fp16_fixed":
            raise ValueError(
                "Experimental writer epilogue has only the FP16 rounding contract"
            )
        if config.ffn_decode_reuse and config.writer_epilogue:
            raise ValueError(
                "Decode reuse keeps the original down writer; experimental writer is separate"
            )
        self.config = config
        import logging

        logging.info(
            "[KohakuMOSS runtime] CPU_intraop_threads=%d precision=%s profile=%s",
            torch.get_num_threads(),
            config.precision,
            config.profile,
        )
        self.stage = stage
        self.lock = RLock()
        self.te = self.bb = self.wrapper = self.patcher = self.vae = None
        self.original_adapter = None
        self.te_lease = None
        self.retired_te_leases = []
        self.arena_cache = None
        self.adapter_cache = None
        self.arena_handle = None
        self.payload_cache = None
        self.payload_owner = None
        self.restores = []
        self.scratch = None
        self.closed = False
        self.counts = {
            "dit_builds": 0,
            "dit_uploads": 0,
            "dit_reuses": 0,
            "kernel_installs": 0,
            "host_arena_hits": 0,
            "host_arena_misses": 0,
            "disposals": 0,
        }
        self.last = {}

    def _mark(self, phase):
        self.current_phase = phase
        if self.stage:
            self.stage(phase)

    def _sync(self):
        import torch

        torch.cuda.synchronize(self.config.device)

    def _ready(self, event="request_entry"):
        bootstrap_paths()
        self.closed = False
        # Normal local lifecycle evidence; no UI endpoint or global MM hook.
        import logging
        import torch
        import comfy.model_management as mm

        allocated = (
            torch.cuda.memory_allocated(self.config.device)
            if torch.cuda.is_initialized()
            else None
        )
        core_devices = (
            sorted(
                {
                    str(t.device)
                    for t in list(self.bb.parameters()) + list(self.bb.buffers())
                }
            )
            if self.bb is not None
            else []
        )
        logging.info(
            "[KohakuMOSS resource] event=%s profile=%s device=%s loaded_models=%d allocated_bytes=%s core_devices=%s",
            event,
            self.config.profile,
            self.config.device,
            len(mm.current_loaded_models),
            allocated,
            core_devices,
        )

    def _offload_vae(self):
        if self.vae is None:
            return
        import torch
        import comfy.model_management as mm

        matching = [
            model
            for model in mm.current_loaded_models
            if model.model is not None and model.model.model is self.vae.patcher.model
        ]
        if matching:
            self._mark("VAE_offload_for_next_request")
            mm.unload_model_and_clones(self.vae.patcher, unload_additional_models=False)
            self.counts["vae_offloads"] = self.counts.get("vae_offloads", 0) + 1
            gc.collect()
            torch.cuda.empty_cache()

    def encode_pair(self, positive, negative="", *, cache_negative=False):
        with self.lock:
            self._ready("encode_entry")
            self._check_residency("before_TE")
            if self.config.profile == "4g":
                self._offload_vae()
            from .vendor.paired_session import PairedTextSession, SessionConfig
            from .assets import resolve_asset

            te_path = resolve_asset(self.config.te_asset, self.config.te_weights)
            memory = self.config.te_memory
            partial_gib = self.config.te_resident_gib
            if memory == "auto":
                import torch

                capability = tuple(torch.cuda.get_device_capability(self.config.device))
                qualifies = (
                    self.config.te_mode != "original"
                    and self.config.profile in ("16g", "unlimited")
                    and (
                        capability == (8, 9)
                        or (
                            capability == (7, 5)
                            and torch.cuda.get_device_properties(
                                self.config.device
                            ).total_memory
                            >= 21 * 1024**3
                        )
                    )
                )
                memory = "resident" if qualifies else "ring"
                known_int8 = (
                    te_path.name == "moss-image-qwen35-int8-convrot.safetensors"
                    and self.config.te_mode in ("auto", "w8a16")
                )
                if (
                    capability == (8, 9)
                    and known_int8
                    and self.config.vae_precision == "bf16"
                    and self.config.vae_admission
                ):
                    if self.config.profile == "12g":
                        memory = "partial"
                        partial_gib = {"q5_ridge_gptq": 4.0, "q6_direct": 3.5}.get(
                            self.config.candidate, 2.5
                        )
                    elif (
                        self.config.profile == "16g"
                        and self.config.candidate not in PACKED_CANDIDATES
                    ):
                        memory = "partial"
                        partial_gib = 5.5
            self.last["TE_memory_plan"] = {
                "requested": self.config.te_memory,
                "effective": memory,
                "partial_prefix_budget_GiB": partial_gib
                if memory == "partial"
                else None,
                "automatic_ring": self.config.te_memory == "auto" and memory == "ring",
                "unqualified_architecture_uses_ring": self.config.te_memory == "auto"
                and memory == "ring"
                and capability not in ((8, 9), (7, 5)),
            }
            if self.te is None:
                self._mark("TE_load")
                self.te = PairedTextSession.load(
                    SessionConfig(
                        str(te_path),
                        device=self.config.device,
                        memory="ring",
                        te_mode=self.config.te_mode,
                        compute_dtype=self.config.te_compute_dtype,
                    )
                )
                if memory in ("resident", "partial"):
                    try:
                        from .resident_text import promote_resident, promote_partial

                        self._mark("TE_promote_" + memory)
                        self.last["TE_resident_load"] = (
                            promote_resident(self.te.encoder)
                            if memory == "resident"
                            else promote_partial(
                                self.te.encoder, max_bytes=int(partial_gib * 1024**3)
                            )
                        )
                        from .te_lease import register_text_lease

                        register_text_lease(self)
                    except BaseException:
                        if self.te is not None:
                            self.te.close()
                            self.te = None
                        self.te_lease = None
                        raise
            self._mark("TE_encode")
            result = self.te.encode_pair(
                positive, negative, cache_negative=cache_negative
            )
            self.last["TE"] = result.receipt
            if self.config.profile == "4g":
                import torch

                if torch.cuda.get_device_capability(self.config.device) == (7, 5):
                    # Failed build exceptions no longer have persistent cache roots.
                    # Collect dead traceback cycles before the tight DiT stage.
                    import gc

                    self._mark("TE_release_dead_errors")
                    before_gc = torch.cuda.memory_allocated(self.config.device)
                    gc_started = time.perf_counter()
                    gc.collect()
                    self.last["TE_dead_error_gc"] = {
                        "wall_s": time.perf_counter() - gc_started,
                        "allocated_before": before_gc,
                        "allocated_after": torch.cuda.memory_allocated(
                            self.config.device
                        ),
                    }
            self._mark("TE_complete")
            return result.positive, result.negative

    def _core_signature(self):
        if self.bb is None:
            return {}
        return {
            name: (
                tensor.data_ptr(),
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
            )
            for name, tensor in list(self.bb.named_parameters())
            + list(self.bb.named_buffers())
            if tensor.numel()
            and not name.startswith("text_embedder.")
            and name != "rope.bands"
        }

    def _check_residency(self, phase):
        if self.bb is None:
            return
        current = self._core_signature()
        valid = current == self.core_signature
        self.last.setdefault("residency", []).append(
            {
                "phase": phase,
                "core_storage_unchanged": valid,
                "core_devices": sorted({value[-1] for value in current.values()}),
            }
        )
        if not valid:
            self.counts["core_invalidations"] = (
                self.counts.get("core_invalidations", 0) + 1
            )
            self._mark("DiT_invalidated_rebuild_required")
            self._dispose_dit(discard=True)

    def _build_dit(self, positive, negative, height, width):
        import torch

        if (
            self.compute_dtype == torch.bfloat16
            and torch.cuda.get_device_capability(self.config.device)[0] < 8
        ):
            raise ValueError("BF16 DiT requires SM80 or newer")
        from safetensors.torch import load_file
        from kohakumoss.model.backbone import MossBackbone
        from kohakumoss.model.opset import OpSet, PRIMITIVES, FUSED
        from kohakumoss.model.numerics.fp16fix import apply_fp16_fix
        from kohakumoss.types.arch import PRESETS
        from .storage import attach_frozen, FrozenPackedBase
        from .owned_scratch import SessionScratch
        from .packed_decode import unpack_to_int8
        from .vendor.bridge import KohakuComfyBridge
        import comfy.model_patcher

        if self.config.profile == "unlimited" and self.te_lease is not None:
            # The previous VAE can leave large disjoint allocator cache blocks.
            # Drop this session's VAE placement before allocating the next contiguous
            # DiT arena. Standard MM unload deliberately keeps unrelated additions.
            lease_before, te_before = self.te_lease, self.te
            self._mark("DiT_prepare_upload_headroom")
            before = {
                "allocated_bytes": torch.cuda.memory_allocated(self.config.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.config.device),
            }
            self._offload_vae()
            gc.collect()
            torch.cuda.empty_cache()
            if self.te_lease is not lease_before or self.te is not te_before:
                raise RuntimeError("Owned VAE unload unexpectedly evicted resident TE")
            self.last["fresh_DiT_upload_headroom"] = {
                "before": before,
                "after": {
                    "allocated_bytes": torch.cuda.memory_allocated(self.config.device),
                    "reserved_bytes": torch.cuda.memory_reserved(self.config.device),
                },
                "TE_lease_preserved": True,
                "TE_resident_bytes": self.te.encoder.placement.resident_bytes(),
                "scope": "standard unload of owned VAE and release unused allocator cache only",
            }
            self._mark("DiT_upload_headroom_ready")
        self._mark("DiT_load")
        t = time.perf_counter()
        candidate = self.config.candidate
        use_stream = self.config.resident_blocks >= 0 or (
            self.config.profile == "4g" and candidate != "q5_ridge_gptq"
        )
        use_resident_arena = (
            candidate in INT8_CANDIDATES + FP8_CANDIDATES
            and self.config.profile in ("6g", "8g")
            and self.config.host_arena
            and not use_stream
            and torch.cuda.get_device_capability(self.config.device) == (8, 9)
        )
        if use_stream and torch.cuda.get_device_capability(self.config.device) != (
            8,
            9,
        ):
            raise ValueError(
                "Partial original-payload streaming is currently qualified only on SM89; use the established W5 path or a larger budget on SM75"
            )
        adapter_first = (
            self.config.profile == "4g"
            or use_stream
            or (
                self.config.profile == "6g"
                and candidate in INT8_CANDIDATES + FP8_CANDIDATES
            )
        )
        self._adapter_prepared_for_build = adapter_first
        from .assets import resolve_asset

        path = resolve_asset(CANDIDATE_ASSETS[candidate], self.config.dit_weights)
        source_stat = path.stat()
        self.adapter_revision = (
            str(path),
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        )
        state = retained = {}
        if candidate in PACKED_CANDIDATES:
            state = load_file(str(path), device="cpu")
            retained = {
                k: v.to(self.compute_dtype)
                if v.is_floating_point() and k != "rope.bands"
                else v
                for k, v in state.items()
                if not k.endswith((".qweight", ".wscales"))
            }
            bb = MossBackbone(PRESETS["sft"], device="meta", dtype=self.compute_dtype)
            coverage = bb.load_state_dict(retained, strict=False, assign=True)
            self.scratch = SessionScratch(
                decoder_kernel=unpack_to_int8,
                device=self.config.device,
                maximum_elements=50331648,
            )
            stems = attach_frozen(
                bb, state, 5 if candidate == "q5_ridge_gptq" else 6, self.scratch
            )
            if (
                set(coverage.missing_keys) != {s + ".weight" for s in stems}
                or coverage.unexpected_keys
            ):
                raise ValueError("Packed SFT checkpoint coverage mismatch")
            self.last["checkpoint"] = {
                "path": str(path),
                "format": "packed_w5" if candidate == "q5_ridge_gptq" else "packed_w6",
                "variant": "sft",
            }
        else:
            from .checkpoints import load_checkpoint

            bb, receipt = load_checkpoint(
                path,
                variant="base" if candidate.startswith("base_") else "sft",
                dtype=self.compute_dtype,
                device="cpu",
                keep_fp32=(),
                kernels={n: "eager" for n in PRIMITIVES + FUSED},
            )
            if candidate in INT8_CANDIDATES and (
                receipt["format"] != "int8_convrot"
                or len(receipt["w8a8_layers"]) != 252
            ):
                raise ValueError(
                    "Selected W8 mode requires all 252 joint ConvRot256 INT8 projections"
                )
            if candidate in FP8_CANDIDATES and receipt["format"] not in (
                "fp8_compute",
                "mxfp8",
                "fp8_weight_only",
            ):
                raise ValueError(
                    "Selected FP8 mode requires an explicitly described FP8 checkpoint"
                )
            self.last["checkpoint"] = receipt
        self.bb = bb
        if adapter_first:
            bb.eval().requires_grad_(False)
            bb.bind_ops(
                OpSet.resolve(
                    self.config.device,
                    self.compute_dtype,
                    {n: "eager" for n in PRIMITIVES + FUSED},
                )
            )
            if self.config.precision == "fp16_fixed":
                apply_fp16_fix(bb, "online", c=1 / 16)
            self.original_adapter = bb.text_embedder
            bb.rope.to(self.config.device)
            self._prepare_adapter(positive, negative, height, width)
        if use_stream or use_resident_arena:
            from .payload_stream import (
                PayloadCache,
                FrozenPayloadStream,
                FrozenPayloadResident,
            )

            if self.payload_cache is None:
                self.payload_cache = PayloadCache()
            code_path = path
            stat = code_path.stat()
            identity = (
                str(code_path),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            self._mark("DiT_host_payload_capture")
            hit = self.payload_cache.capture(bb, identity)
            for key in list(state):
                if key.endswith(".qweight"):
                    del state[key]
            if use_resident_arena:
                self._mark("DiT_resident_payload_upload")
                self.payload_owner = FrozenPayloadResident(
                    bb,
                    self.payload_cache,
                    self.config.device,
                    pinning=self.config.payload_pinning,
                )
            else:
                prefix = self.kernel_config.resident_blocks
                if prefix < 0:
                    prefix = 33 if candidate == "q6_direct" else 23
                    if (
                        self.config.attention.startswith("sage_")
                        or self.config.attention == "ck_int8"
                    ):
                        prefix -= 2
                self._mark("DiT_partial_upload")
                self.payload_owner = FrozenPayloadStream(
                    bb,
                    self.payload_cache,
                    self.config.device,
                    resident_blocks=prefix,
                    pinning=self.config.payload_pinning,
                )
            self.payload_owner.place()
            self.last["payload_host_cache"] = {
                "hit": hit,
                "capture_s": 0.0 if hit else self.payload_cache.capture_s,
                "host_bytes": self.payload_cache.arena.numel(),
            }
        elif self.config.host_arena and candidate in PACKED_CANDIDATES:
            from .vendor.host_packed_arena import HostArenaCache, checkpoint_identity

            if self.arena_cache is None:
                self.arena_cache = HostArenaCache(max_bytes=4 * 1024**3)
            identity = checkpoint_identity(path, revision="frozen-round2-" + candidate)
            self._mark("DiT_host_arena_capture")
            arena, hit = self.arena_cache.get_or_capture(
                bb, source_identity=identity, audit=False, base_type=FrozenPackedBase
            )
            self.counts["host_arena_hits" if hit else "host_arena_misses"] += 1
            self._mark("DiT_upload")
            self.arena_handle = arena.upload_into(
                bb,
                source_identity=identity,
                device=self.config.device,
                release_state=state,
                verify_model=False,
            )
            self.last["packed_upload"] = {
                k: v for k, v in self.arena_handle.receipt.items() if k != "entries"
            }
        if not (use_stream or use_resident_arena):
            bb = bb.to(self.config.device)
        bb = bb.eval().requires_grad_(False)
        self.counts["dit_builds"] += 1
        self.counts["dit_uploads"] += 1
        if not adapter_first:
            bb.bind_ops(
                OpSet.resolve(
                    self.config.device,
                    self.compute_dtype,
                    {n: "eager" for n in PRIMITIVES + FUSED},
                )
            )
            if self.config.precision == "fp16_fixed":
                apply_fp16_fix(bb, "online", c=1 / 16)
            self.original_adapter = bb.text_embedder
        loaders = _official_loaders()
        cfg = loaders.MOSSImageConfig(
            {
                "dim": 3072,
                "num_heads": 24,
                "num_layers": 42,
                "ffn_dim": 8192,
                "latent_channels": 128,
                "text_dim": 4096,
                "freq_dim": 256,
                "rope_base": 10000.0,
            }
        )
        cfg.supported_inference_dtypes = [self.compute_dtype]
        cfg.set_inference_dtype(self.compute_dtype, None)
        wrapper = cfg.get_model({}, device=torch.device("meta"))

        class CountedBridge(KohakuComfyBridge):
            def __init__(self, backbone):
                super().__init__(backbone)
                self.evaluations = 0

            def forward(self, x, *args, **kwargs):
                self.evaluations += len(x)
                return super().forward(x, *args, **kwargs)

        wrapper.diffusion_model = CountedBridge(bb)
        wrapper.device = torch.device(self.config.device)
        if self.payload_owner is not None:
            wrapper._moss_payload_owner = self.payload_owner
        self.bb, self.wrapper = bb, wrapper
        from .core_patcher import make_core_patcher

        self.patcher = make_core_patcher(
            wrapper,
            load_device=torch.device(self.config.device),
            offload_device=torch.device("cpu"),
        )
        if self.te_lease is not None:
            self.te_lease.attach(self.patcher)
        del state, retained
        self._sync()
        self.core_signature = self._core_signature()
        self.last["DiT_load_s"] = (
            time.perf_counter()
            - t
            - (self.last.get("adapter_s", 0) if adapter_first else 0)
        )
        self.last["DiT_precision"] = {
            "mode": self.config.precision,
            "compute_dtype": str(self.compute_dtype),
            "fp16_fix": bb.fp16_fix,
            "checkpoint_format": self.last["checkpoint"]["format"],
            "integer_accumulator": "int32" if candidate not in FP8_CANDIDATES else None,
        }
        if candidate in FP8_CANDIDATES:
            from .fp8_support import bind_fp8_inference

            self.last["FP8_binding"] = bind_fp8_inference(
                bb, dtype=self.compute_dtype, device=self.config.device
            )
        self._mark("DiT_ready")

    def _prepare_adapter(self, positive, negative, height, width):
        import torch
        from kohakumoss.types.seqinfo import build_seqinfo

        self._mark("adapter_prepare")
        t = time.perf_counter()
        old = self.original_adapter
        cache = None
        if self.config.adapter_cache:
            from .adapter_cache import ExactAdapterCache

            if self.adapter_cache is None:
                self.adapter_cache = ExactAdapterCache(max_bytes=64 * 1024**2)
            cache = self.adapter_cache
            context = (
                self.config.precision,
                self.adapter_revision,
                height,
                width,
                str(self.config.device),
                tuple(self.bb.rope.bands.detach().cpu().reshape(-1).tolist()),
            )
        loaded = cache is None
        if loaded:
            old = old.to(self.config.device)
            if self.config.candidate in FP8_CANDIDATES:
                from .fp8_support import bind_fp8_inference

                self.last["FP8_adapter_binding"] = bind_fp8_inference(
                    old, dtype=self.compute_dtype, device=self.config.device
                )
        entries = []
        cache_hits = []
        for cond in (positive, negative):
            text = cond[0][0][0].to(device=self.config.device, dtype=self.compute_dtype)
            mask = cond[0][1].get("attention_mask")
            if mask is not None:
                text = text[mask[0].to(device=self.config.device, dtype=torch.bool)]
            seq = build_seqinfo(
                torch.tensor([[height, width]]), torch.tensor([len(text)])
            ).to(self.config.device)
            cached = None
            if cache is not None:
                probe = text.detach().cpu().contiguous()
                key = cache.key(context, probe)
                cached = cache.get(key, probe)
            cache_hits.append(cached is not None)
            if cached is None:
                if not loaded:
                    old = old.to(self.config.device)
                    if self.config.candidate in FP8_CANDIDATES:
                        from .fp8_support import bind_fp8_inference

                        self.last["FP8_adapter_binding"] = bind_fp8_inference(
                            old, dtype=self.compute_dtype, device=self.config.device
                        )
                    loaded = True
                out = old(text, seq, self.bb.rope.bands).detach().clone()
                if cache is not None:
                    cache.put(key, probe, out.cpu())
            else:
                out = cached.to(self.config.device)
            entries.append((text.detach().clone(), out))
        if cache is not None:
            self.last["adapter_cache"] = dict(
                cache.receipt(),
                request_hits_positive_negative=cache_hits,
                original_adapter_uploaded=loaded,
            )
        effective = getattr(self, "kernel_config", self.config)
        cpu_guard = (
            self.config.qkv_backend == "auto"
            and effective.profile == "4g"
            and effective.attention == "auto"
            and effective.qkv_reuse_input
            and effective.rope_inplace
            and effective.sdpa_capacity_fallback
            and torch.cuda.get_device_capability(self.config.device) == (7, 5)
        )

        class RequestAdapter(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                # CPU-only exact guard snapshots are ordinary owner metadata;
                # Module._apply must not migrate them back to CUDA.
                self._cpu_guards = (
                    [text.detach().cpu().clone() for text, out in values]
                    if cpu_guard
                    else None
                )
                for i, (text, out) in enumerate(values):
                    if not cpu_guard:
                        self.register_buffer("text" + str(i), text)
                    self.register_buffer("out" + str(i), out)

            def bind(self, *args):
                pass

            def forward(self, text, seq, bands):
                dtype = (
                    self._cpu_guards[0].dtype
                    if self._cpu_guards is not None
                    else self.text0.dtype
                )
                text = text.to(dtype=dtype)
                probe = text.detach().cpu() if self._cpu_guards is not None else text
                for i in range(2):
                    ref = (
                        self._cpu_guards[i]
                        if self._cpu_guards is not None
                        else getattr(self, "text" + str(i))
                    )
                    if (
                        ref.shape == probe.shape
                        and ref.dtype == probe.dtype
                        and torch.equal(ref, probe)
                    ):
                        return getattr(self, "out" + str(i)).clone()
                raise RuntimeError(
                    "Conditioning changed after request adapter preparation"
                )

        self.bb.text_embedder = RequestAdapter(entries)
        self.last["adapter_CPU_exact_guard"] = {
            "enabled": cpu_guard,
            "GPU_bytes_removed": sum(t.numel() * t.element_size() for t, o in entries)
            if cpu_guard
            else 0,
            "comparison": "same dtype/shape and torch.equal; per-NFE D2H sync included in DiT time",
        }
        self.original_adapter.cpu()
        self._sync()
        self.last["adapter_s"] = time.perf_counter() - t
        self._mark("adapter_ready")

    def _ensure_scratch(self):
        if self.config.candidate not in PACKED_CANDIDATES:
            return
        from .storage import FrozenPackedBase
        from .owned_scratch import SessionScratch
        from .packed_decode import unpack_to_int8

        if self.scratch is None:
            self.scratch = SessionScratch(
                decoder_kernel=unpack_to_int8,
                device=self.config.device,
                maximum_elements=50331648,
            )
        self.scratch.reserve(50331648)
        for module in self.bb.modules():
            if type(module) is FrozenPackedBase:
                module.decoder = self.scratch
        if getattr(self, "qkv_handle", None) is not None and hasattr(
            self.qkv_handle, "workspace"
        ):
            self.qkv_handle.workspace.scratch["buffer"] = self.scratch.buffer

    def _release_scratch(self):
        if getattr(self, "qkv_handle", None) is not None and hasattr(
            self.qkv_handle, "workspace"
        ):
            self.qkv_handle.workspace.scratch.clear()
        if self.scratch is not None:
            self.scratch.close()
            self.scratch = None

    def _install_kernels(self):
        config = getattr(self, "kernel_config", self.config)
        import torch
        from kohakumoss.model.opset import OpSet, PRIMITIVES, FUSED

        choices = {n: "eager" for n in PRIMITIVES + FUSED}
        for name in (
            "rms_norm",
            "rms_norm_mod",
            "qk_norm_rope",
            "situ_glu",
            "gated_residual",
        ):
            choices[name] = "triton"
        ops = OpSet.resolve(config.device, self.compute_dtype, choices)
        private_capacity = (
            self.config.qkv_backend == "auto"
            and config.profile == "4g"
            and config.attention == "auto"
            and config.qkv_reuse_input
            and config.rope_inplace
            and torch.cuda.get_device_capability(config.device) == (7, 5)
        )
        # Query tiling consumes only the qualified private Q and keeps full K/V.
        # Keep CPU text guards/private embedding on their separately qualified SM75 path.
        query_tile = (
            self.config.qkv_backend == "auto"
            and config.profile == "4g"
            and config.attention == "auto"
            and config.qkv_reuse_input
            and config.rope_inplace
            and config.sdpa_capacity_fallback
            and torch.cuda.get_device_capability(config.device) in ((7, 5), (8, 9))
        )
        self.last["query_M_SDPA_tile"] = 2048 if query_tile else 0
        if private_capacity:
            from .vendor.embed_private import install as embed_install

            record, restore = embed_install(self.bb)
            self.restores.append(restore)
            self.last["private_embed"] = record

        def single_attention(q, k, v, cu_seqlens, max_seqlen, scale):
            if cu_seqlens.numel() != 2 or int(max_seqlen) != len(q):
                raise ValueError("Current bridge owns one complete attention segment")
            if query_tile:
                from .vendor.query_tile_sdpa import attention as tiled_sdpa

                # Only this session's private norm-input Q is consumed here.
                # Exceptions propagate: a partially overwritten Q is never reused.
                return tiled_sdpa(q, k, v, scale, overwrite_private_q=True)
            return (
                torch.nn.functional.scaled_dot_product_attention(
                    *(x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)), scale=scale
                )
                .squeeze(0)
                .transpose(0, 1)
            )

        ops.attention_varlen = single_attention
        self.bb.bind_ops(ops)
        if config.candidate in FP8_CANDIDATES:
            from .fp8_support import bind_fp8_inference

            self.last["FP8_binding"] = bind_fp8_inference(
                self.bb, dtype=self.compute_dtype, device=config.device
            )
        self.last["effective_FFN_backend"] = (
            "kohaku_fp8"
            if config.candidate in FP8_CANDIDATES
            else "fused_v3"
            if config.kernels == "v3_shared"
            else "native_original"
        )
        self.last["effective_qkv_backend"] = (
            "kohaku_fp8_separate"
            if config.candidate in FP8_CANDIDATES
            else "stored_int8_separate"
            if config.candidate in INT8_CANDIDATES
            else config.qkv_backend
        )
        self.last["host_arena_effective"] = (
            config.host_arena and config.candidate in PACKED_CANDIDATES
        ) or (
            self.payload_owner is not None
            and self.payload_owner.receipt().get("mode") == "full_resident_arena"
        )
        if config.candidate not in FP8_CANDIDATES and config.kernels in (
            "v3_shared",
            "native_shared",
        ):
            if config.candidate in PACKED_CANDIDATES:
                if config.qkv_backend == "shared":
                    from .vendor.projection_share_v2 import install as qkv_install

                    qkv = qkv_install(self.bb)
                else:
                    if config.qkv_backend == "register3":
                        if config.qkv_reuse_input:
                            from .vendor.qkv_reuse_input import install as qkv_install
                        else:
                            from .vendor.qkv_three_output import install as qkv_install
                    else:
                        from .vendor.qkv_fused_v1 import install as qkv_install
                    qkv = qkv_install(
                        self.bb,
                        backend="register"
                        if config.qkv_backend == "register3"
                        else config.qkv_backend,
                        scratch={"buffer": self.scratch.buffer},
                        experimental=True,
                        config={"bm": 64, "bn": 128, "bk": 64, "warps": 4, "stages": 2},
                    )
                self.qkv_handle = qkv
                self.restores.append(qkv.restore)
            if config.kernels == "v3_shared":
                if config.ffn_lifetime:
                    from .vendor.ffn_lifetime import install as ffn_install
                else:
                    from .vendor.ffn_fused_v3 import install as ffn_install
                record, restore = ffn_install(self.bb, tile=config.ffn_tile)
                self.restores.append(restore)
                self.last["ffn"] = record
                if (
                    config.ffn_decode_reuse
                    and config.ffn_tile
                    and config.candidate in PACKED_CANDIDATES
                ):
                    from .tiled_ffn import install as install_tiled_decode

                    record, restore = install_tiled_decode(
                        self.bb, tile=config.ffn_tile
                    )
                    self.restores.append(restore)
                    self.last["ffn_decode_reuse"] = record
        if (
            config.kernels != "v3_shared" or config.candidate in FP8_CANDIDATES
        ) and config.ffn_tile:
            self.bb._cap_tile = config.ffn_tile
            # No CAP_FUSED rebind: the final ops above already own dispatch.
            from kohakumoss.types.context import SHIFT_FFN, SCALE_FFN, GATE_FFN
            from types import MethodType

            tile = config.ffn_tile

            def ffn(block, x, ctx):
                mod = ctx.mod
                output = torch.empty_like(x)
                for start in range(0, len(x), tile):
                    xx = x[start : start + tile]
                    seg = ctx.seq.seg_joint[start : start + tile]
                    y = block._cast(
                        block.ops.rms_norm_mod(
                            xx,
                            block.norm2.weight,
                            mod[:, SHIFT_FFN],
                            mod[:, SCALE_FFN],
                            seg,
                            block.norm2.eps,
                        )
                    )
                    output[start : start + tile] = block._post(
                        block._ffn_out(block._situ(y), xx, mod[:, GATE_FFN], seg, 1.0)
                    )
                return output

            for block in self.bb.layers:
                block.ffn_part = MethodType(ffn, block)
        if config.attention == "flash":
            capability = tuple(torch.cuda.get_device_capability(config.device))
            if capability == (7, 5):
                from .vendor.turing_adapter import install
            else:
                from .vendor.flash_adapter import install
            record, restore = install(self.bb)
            self.restores.append(restore)
            self.last["attention"] = record
        elif config.attention != "auto":
            from .attention import install

            record, restore = install(
                self.bb,
                backend=config.attention,
                head_group=config.attention_head_group,
            )
            self.restores.append(restore)
            self.last["attention"] = record
        if config.projection_tile:
            from .storage import FrozenPackedBase, FrozenInt8Base

            for block in self.bb.layers:
                if type(block.proj["o"].base) not in (FrozenPackedBase, FrozenInt8Base):
                    raise ValueError(
                        "Projection tile requires canonical frozen W5/W6/W8 o"
                    )
                block.proj["o"].base.row_tile = config.projection_tile
            self.last["attention_output_projection"] = {
                "M_tile": config.projection_tile,
                "quantize_full_input_once": True,
                "decode_weight_once": True,
                "compute_boundary_before_original_writer": str(self.compute_dtype),
            }
        if config.rope_inplace:
            from .vendor.rope_inplace import qk_norm_rope_inplace

            old_rope = ops.qk_norm_rope
            ops.qk_norm_rope = qk_norm_rope_inplace
            self.restores.append(lambda: setattr(ops, "qk_norm_rope", old_rope))
            self.last["inplace_rope"] = {
                "original_kernel": True,
                "fixed_ROWS": 16,
                "fixed_warps": 4,
                "autotune": False,
                "QK_storages": 2,
            }
        if config.writer_epilogue:
            from .vendor.writer_epilogue import install as install_writer
            from .storage import unpack_base
            from .vendor.inference_contracts_v3 import projection_errors

            record, restore = install_writer(
                self.bb,
                unpack=unpack_base,
                fp_fusion=True,
                experimental=True,
                projection_validator=projection_errors,
            )
            self.restores.append(restore)
            self.last["writer_epilogue"] = record
        if config.single_frame:
            from .vendor.single_frame_forward import install

            record, restore = install(self.bb)
            self.restores.append(restore)
            self.last["single_frame_forward"] = record
        self.counts["kernel_installs"] += 1
        self.installed_kernel_config = config
        import logging

        logging.info(
            "[KohakuMOSS dispatch] precision=%s attention=%s head_group=%d QKV=%s FFN_tile=%d",
            config.precision,
            config.attention,
            config.attention_head_group,
            config.qkv_backend,
            config.ffn_tile,
        )

    def sample(
        self,
        positive,
        negative,
        latent,
        seed,
        *,
        noise=None,
        steps=16,
        cfg=3.0,
        sampler_name="euler",
        scheduler="sgm_uniform",
        shift=5.0,
        slg_layer=22,
        callback_factory=None,
    ):
        with self.lock:
            self._ready("sample_entry")
            import torch
            import comfy.sample
            from comfy_extras.nodes_model_advanced import ModelSamplingSD3
            from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

            self._check_residency("before_sample")
            initial = latent["samples"]
            if (
                self.config.profile == "4g"
                or initial.shape[-2] * initial.shape[-1] > 4096
            ):
                self._offload_vae()
            if initial.shape[0] != 1 or initial.shape[1] != 128:
                raise ValueError(
                    "MOSS runtime currently supports one Flux2 latent [1,128,H,W]"
                )
            from .profile_plan import kernel_plan

            major, minor = torch.cuda.get_device_capability(self.config.device)
            conditioning_eligible = (
                self.config.conditioning_schedule
                and self.config.profile == "4g"
                and sampler_name == "euler"
                and 1 <= steps <= 64
            )
            self.kernel_config, self.last["kernel_plan"] = kernel_plan(
                self.config,
                initial.shape[-2:],
                positive[0][0].shape[-2],
                negative[0][0].shape[-2],
                major * 10 + minor,
                conditioning_eligible=conditioning_eligible,
            )
            if (
                self.bb is not None
                and getattr(self, "installed_kernel_config", None) != self.kernel_config
            ):
                self._mark("DiT_rebuild_for_changed_execution_plan")
                self._dispose_dit(discard=True)
                self.last["kernel_plan"]["rebuild_for_changed_execution_plan"] = True
            if (
                self.bb is not None
                and self.config.profile == "6g"
                and self.config.candidate in INT8_CANDIDATES + FP8_CANDIDATES
                and positive[0][0].shape[-2] > 1024
            ):
                # Long text adapter intermediates do not fit beside all W8 codes.
                # Prepare it first, then rebuild/upload the unchanged frozen core.
                self._mark("DiT_dispose_for_long_adapter_capacity")
                self._dispose_dit(discard=True)
            if self.bb is None:
                try:
                    self._build_dit(
                        positive, negative, initial.shape[-2], initial.shape[-1]
                    )
                except BaseException:
                    self._dispose_dit(discard=True)
                    raise
                new = True
            else:
                self.counts["dit_reuses"] += 1
                self.last["DiT_load_s"] = 0.0
                new = False
            complete = False
            schedule_restore = None
            schedule_validate = None
            try:
                with torch.inference_mode():
                    if not (new and self._adapter_prepared_for_build):
                        self._prepare_adapter(
                            positive, negative, initial.shape[-2], initial.shape[-1]
                        )
                    self._ensure_scratch()
                    if new:
                        self._install_kernels()
                    qkv_before = (
                        dict(self.qkv_handle.stats)
                        if getattr(self, "qkv_handle", None) is not None
                        else {}
                    )
                    sampled = ModelSamplingSD3().patch(self.patcher, shift=shift)[0]
                    sampled = SkipLayerGuidanceDiTSimple.execute(
                        sampled, 0, 1, double_layers="", single_layers=str(slg_layer)
                    )[0]
                    if self.config.conditioning_schedule:
                        eligible = conditioning_eligible
                        self.last["conditioning_schedule"] = {
                            "enabled": eligible,
                            "reason": "Euler batch1 request, 4G lifecycle"
                            if eligible
                            else "Original path: profile, sampler or step count outside qualification",
                        }
                        if eligible:
                            import comfy.samplers
                            from .schedule_conditioning import (
                                install as install_schedule,
                            )

                            self._mark("conditioning_schedule_prepare")
                            prep_started = time.perf_counter()
                            schedule = comfy.samplers.KSampler(
                                sampled,
                                steps,
                                self.config.device,
                                sampler=sampler_name,
                                scheduler=scheduler,
                                denoise=1.0,
                                model_options=sampled.model_options,
                            )
                            record, schedule_restore, schedule_validate = (
                                install_schedule(
                                    self.bb,
                                    sampled.get_model_object("model_sampling"),
                                    schedule.sigmas,
                                )
                            )
                            self._sync()
                            record["prepare_s"] = time.perf_counter() - prep_started
                            record["enabled"] = True
                            self.patcher.size = sampled.size = 0
                            self.last["conditioning_schedule"] = record
                            self.core_signature = self._core_signature()
                            del schedule
                    if noise is None:
                        noise = comfy.sample.prepare_noise(
                            initial, seed, latent.get("batch_index")
                        )
                    self.wrapper.diffusion_model.evaluations = 0
                    self._mark("DiT_sample")
                    self._sync()
                    started = time.perf_counter()
                    from contextlib import nullcontext

                    scope = (
                        self.scratch.execution()
                        if self.scratch is not None
                        else nullcontext()
                    )
                    callback = (
                        callback_factory(sampled, steps)
                        if callback_factory is not None
                        else None
                    )
                    with scope:
                        output = comfy.sample.sample(
                            sampled,
                            noise,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            positive,
                            negative,
                            initial,
                            denoise=1.0,
                            seed=seed,
                            callback=callback,
                        )
                    self._sync()
                    if schedule_validate is not None:
                        schedule_validate()
                    self.last["DiT_s"] = time.perf_counter() - started
                    self.last["NFE"] = self.wrapper.diffusion_model.evaluations
                    if self.payload_owner is not None:
                        self.last["payload_stream"] = self.payload_owner.receipt()
                    if getattr(self, "qkv_handle", None) is not None:
                        after = dict(self.qkv_handle.stats)
                        self.last["qkv"] = {
                            "before": qkv_before,
                            "after": after,
                            "delta": {
                                k: after[k] - qkv_before.get(k, 0) for k in after
                            },
                            "receipt": dict(getattr(self.qkv_handle, "receipt", {})),
                        }
                    self._mark("DiT_complete")
                    del sampled
                complete = True
            finally:
                if schedule_restore is not None:
                    schedule_restore()
                self._release_scratch()
                if self.config.profile == "4g" or not complete:
                    self._dispose_dit(discard=True)
            return {"samples": output.cpu()}

    def decode(self, latent):
        with self.lock:
            self._ready("decode_entry")
            import torch
            import nodes

            if self.vae is None:
                self._mark("VAE_load")
                started = time.perf_counter()
                from .assets import resolve_asset

                vae_path = resolve_asset("vae", self.config.vae_name)
                self.vae_artifact_name = vae_path.name
                if self.config.vae_precision == "bf16":
                    if torch.cuda.get_device_capability(self.config.device)[0] < 8:
                        raise ValueError("BF16 VAE requires SM80 or newer")
                    from .vae_precision import load_bf16_vae

                    self.vae = load_bf16_vae(str(vae_path))
                else:
                    import comfy.sd
                    import comfy.utils

                    state, metadata = comfy.utils.load_torch_file(
                        str(vae_path), return_metadata=True
                    )
                    self.vae = comfy.sd.VAE(sd=state, metadata=metadata)
                    self.vae.throw_exception_if_invalid()
                    self.vae.patcher.cached_patcher_init = (
                        comfy.sd.load_vae_patcher,
                        (str(vae_path), metadata, None),
                    )
                if self.vae.vae_dtype != torch.float32:
                    raise ValueError("MOSS VAE requires its FP32 latent input boundary")
                for name in (
                    "decode_tiled_",
                    "_decode_tiled_owned",
                    "decode_tiled",
                    "decode_tiled_1d",
                    "decode_tiled_3d",
                ):
                    if not hasattr(self.vae, name):
                        continue
                    original = getattr(self.vae, name)
                    if self.config.strict_full_vae:

                        def no_tile(*args, **kwargs):
                            self.last["VAE_effective_mode"] = (
                                "full_capacity_failed_strict_benchmark"
                            )
                            raise RuntimeError(
                                "Full FP32 VAE capacity failed; strict benchmark disables tiled fallback"
                            )

                        setattr(self.vae, name, no_tile)
                    else:

                        def tile_call(*args, _original=original, _name=name, **kwargs):
                            self.last["VAE_effective_mode"] = "official_tiled_fallback"
                            self.last.setdefault("VAE_tile_calls", []).append(_name)
                            return _original(*args, **kwargs)

                        setattr(self.vae, name, tile_call)
                if self.config.qualified_vae_budget:
                    if (
                        self.config.profile != "unlimited"
                        or self.vae_artifact_name != "moss-image-vae.safetensors"
                    ):
                        raise ValueError(
                            "Qualified VAE estimate is an explicit same-artifact unlimited experiment"
                        )
                    from .vae_budget import install as install_vae_budget

                    self.vae_budget_records = install_vae_budget(
                        self.vae, device=self.config.device
                    )
                if self.config.vae_admission:
                    if (
                        self.config.qualified_vae_budget
                        or self.config.resident_vae_stage_budget
                    ):
                        raise ValueError("Use only one VAE admission policy")
                    from .vae_admission import install as install_admission

                    self.vae_admission_records = install_admission(
                        self.vae,
                        device=self.config.device,
                        precision=self.config.vae_precision,
                        artifact_name=self.vae_artifact_name,
                    )
                self.last["VAE_load_s"] = time.perf_counter() - started
                import logging

                logging.info(
                    "[KohakuMOSS VAE] decoder=%s public_input=%s patcher=%s dynamic=%s",
                    self.config.vae_precision,
                    self.vae.vae_dtype,
                    type(self.vae.patcher).__name__,
                    self.vae.patcher.is_dynamic(),
                )
            self.last["VAE_precision"] = getattr(
                self.vae,
                "_moss_precision",
                {"decoder": "torch.float32", "latent_denormalization": "torch.float32"},
            )
            from .vae_stage_budget import KNOWN_FULL_SHAPES

            capability = tuple(torch.cuda.get_device_capability(self.config.device))
            stage_budget_qualified = tuple(
                latent["samples"].shape[-2:]
            ) in KNOWN_FULL_SHAPES and capability in ((8, 9), (7, 5))
            auto_resident_policy = (
                self.config.profile == "unlimited" and self.config.te_memory == "auto"
            )
            auto_keep_dit = (
                auto_resident_policy
                and stage_budget_qualified
                and capability == (7, 5)
                and torch.cuda.get_device_properties(self.config.device).total_memory
                >= 21 * 1024**3
            )
            keep_dit_for_budget = (
                (self.config.resident_vae_keep_dit or auto_keep_dit)
                and stage_budget_qualified
                and capability == (7, 5)
                and torch.cuda.get_device_properties(self.config.device).total_memory
                >= 21 * 1024**3
            )
            discard_dit_for_resident = self.config.resident_vae_discard or (
                auto_resident_policy and not auto_keep_dit
            )
            if (
                discard_dit_for_resident
                and self.config.profile == "unlimited"
                and self.te_lease is not None
                and self.bb is not None
            ):
                self._mark("DiT_dispose_before_VAE_preserve_TE")
                self._dispose_dit(discard=True)
                self.last["resident_VAE_stage"] = {
                    "DiT_disposed": True,
                    "TE_lease_present": self.te_lease is not None,
                    "estimate": "qualified"
                    if self.config.qualified_vae_budget
                    else "upstream",
                }
            if self.config.resident_vae_keep_dit or auto_resident_policy:
                self.last["resident_DiT_VAE_policy"] = {
                    "requested_keep": self.config.resident_vae_keep_dit,
                    "automatic_policy": auto_resident_policy,
                    "effective_keep": keep_dit_for_budget,
                    "effective_discard": discard_dit_for_resident,
                    "scope": "SM75 >=21GiB exact measured VAE shapes keep both; SM89 discards DiT before VAE; unknowns use upstream safe eviction",
                }
            estimate = self.vae.memory_used_decode(
                latent["samples"].shape, self.vae.vae_dtype
            )
            from .allocator import admission

            allocator_plan = admission(
                torch,
                self.config.device,
                include_physical=self.config.qualified_vae_budget
                or self.config.vae_admission,
            )
            allocator_available = allocator_plan["available_allocator_bytes"]
            available = allocator_plan["effective_available_bytes"]
            physical_free = allocator_plan["physical_free_bytes"]
            reusable_cache = allocator_plan["reusable_allocator_cache_bytes"]
            if self.config.qualified_vae_budget:
                self.last["qualified_VAE_estimate"] = list(self.vae_budget_records)
            missing_vae_bytes = self.vae.patcher.model_size()
            target_vae_bytes = missing_vae_bytes
            loaded_vae_bytes = self.vae.patcher.loaded_size()
            external_mapped_bytes = 0
            if self.config.vae_admission:
                target_vae_bytes = getattr(
                    self.vae, "_moss_admission_target_bytes", missing_vae_bytes
                )
                missing_vae_bytes = max(0, target_vae_bytes - loaded_vae_bytes)
                if self.vae.patcher.is_dynamic():
                    # Dynamic VBAR uses external mapped storage. Its ordinary
                    # torch-backed buffers are already in memory_allocated;
                    # mapped weights must count once toward the logical budget.
                    ordinary_loaded = getattr(
                        self.vae.patcher.model, "model_loaded_weight_memory", 0
                    )
                    external_mapped_bytes = max(0, loaded_vae_bytes - ordinary_loaded)
                    allocator_plan = admission(
                        torch,
                        self.config.device,
                        external_bytes=external_mapped_bytes,
                        include_physical=True,
                    )
                    allocator_available = allocator_plan["available_allocator_bytes"]
                    available = allocator_plan["effective_available_bytes"]
            needed = estimate + missing_vae_bytes
            if self.config.vae_admission:
                self.last["VAE_admission"] = list(self.vae_admission_records)
                self.last["VAE_admission_qualification"] = dict(
                    self.vae._moss_admission_qualification
                )
            self.last["VAE_memory_plan"] = {
                "estimated_activation_bytes": int(estimate),
                "estimated_model_bytes": self.vae.patcher.model_size(),
                "allocator": allocator_plan,
                "target_model_bytes": target_vae_bytes,
                "loaded_target_bytes": loaded_vae_bytes,
                "external_dynamic_mapped_bytes": external_mapped_bytes,
                "missing_model_bytes": missing_vae_bytes,
                "available_allocator_bytes": int(allocator_available),
                "physical_free_bytes": int(physical_free),
                "reusable_allocator_cache_bytes": int(reusable_cache),
                "effective_available_bytes": int(available),
            }
            bounded_large_or_heavy = latent["samples"].shape[-2] * latent[
                "samples"
            ].shape[-1] > 4096 or (
                self.config.profile == "6g" and self.config.candidate != "q5_ridge_gptq"
            )
            if (
                self.bb is not None
                and bounded_large_or_heavy
                and needed > available
                and not keep_dit_for_budget
            ):
                self._mark("DiT_dispose_for_VAE_capacity")
                self._dispose_dit(discard=True)
                self.last["VAE_memory_plan"]["DiT_disposed_before_decode"] = True
            self.last["VAE_effective_mode"] = "full"
            self._mark("VAE_decode")
            self._sync()
            started = time.perf_counter()
            with torch.inference_mode():
                if (
                    self.config.resident_vae_stage_budget
                    and self.te_lease is not None
                    and not stage_budget_qualified
                ):
                    self.last["VAE_stage_budget_fallback"] = (
                        "Unknown shape or architecture: upstream memory manager owns safe text eviction; no estimate override"
                    )
                if (
                    self.config.resident_vae_stage_budget
                    and self.te_lease is not None
                    and stage_budget_qualified
                ):
                    from .vae_stage_budget import resident_vae_budget

                    with resident_vae_budget(
                        self.vae, latent["samples"].shape, device=self.config.device
                    ) as stage_budget:
                        self.last["VAE_stage_budget"] = stage_budget
                        image = nodes.VAEDecode().decode(self.vae, latent)[0]
                else:
                    image = nodes.VAEDecode().decode(self.vae, latent)[0]
            self._sync()
            self.last["VAE_s"] = time.perf_counter() - started
            if self.config.vae_admission:
                self.last["VAE_memory_plan"]["loaded_target_bytes_after_decode"] = (
                    self.vae.patcher.loaded_size()
                )
            self._mark("VAE_complete")
            self._check_residency("after_VAE")
            return image

    def generate(
        self,
        positive,
        negative,
        width,
        height,
        seed,
        *,
        noise=None,
        initial_latent=None,
        **sampling,
    ):
        with self.lock:
            import torch

            started = time.perf_counter()
            self.last = {}
            pos, neg = self.encode_pair(positive, negative)
            if initial_latent is None:
                initial_latent = torch.zeros((1, 128, height // 16, width // 16))
            latent = self.sample(
                pos, neg, {"samples": initial_latent}, seed, noise=noise, **sampling
            )
            image = self.decode(latent)
            self.last["request_s"] = time.perf_counter() - started
            self.last["counts"] = dict(self.counts)
            return image, latent, pos, neg, dict(self.last)

    def _dispose_dit(self, discard=False):
        import torch
        import comfy.model_management as mm

        if self.bb is None:
            return
        disposal = {"started_epoch": time.time(), "discard": discard}
        began = time.perf_counter()
        if self.payload_owner is not None:
            self.last["payload_stream"] = self.payload_owner.receipt()
            self.payload_owner.close()
            self.payload_owner = None
        if discard:
            # These objects are request-disposable, never cached/reused after meta.
            matching = [
                m.model
                for m in mm.current_loaded_models
                if m.model is not None and m.model.model is self.wrapper
            ]
            for patcher in matching + (
                [self.patcher] if self.patcher is not None else []
            ):
                patcher.offload_device = torch.device("meta")
        if self.patcher is not None:
            mm.unload_model_and_clones(self.patcher, unload_additional_models=False)
        elif discard:
            self.bb.to("meta")
        for restore in reversed(self.restores):
            restore()
        self.restores = []
        self.bb = self.wrapper = self.patcher = self.original_adapter = None
        self.arena_handle = None
        self.qkv_handle = None
        self._release_scratch()
        self.counts["disposals"] += 1
        disposal["detach_and_unload_s"] = time.perf_counter() - began
        began = time.perf_counter()
        gc.collect()
        disposal["gc_s"] = time.perf_counter() - began
        began = time.perf_counter()
        torch.cuda.empty_cache()
        disposal["empty_device_cache_s"] = time.perf_counter() - began
        self.last["disposal"] = disposal

    def close(self):
        with self.lock:
            self._dispose_dit()
            from .te_lease import close_text_lease

            close_text_lease(self)
            if self.te is not None:
                self.te.close()
                self.te = None
            if self.vae is not None:
                import comfy.model_management as mm

                mm.unload_model_and_clones(
                    self.vae.patcher, unload_additional_models=False
                )
                self.vae = None
            if self.arena_cache is not None:
                self.arena_cache.close()
                self.arena_cache = None
            if self.payload_cache is not None:
                self.payload_cache.close()
                self.payload_cache = None
            if self.adapter_cache is not None:
                self.adapter_cache.clear()
                self.adapter_cache = None
            self._release_scratch()
            import torch

            # A closed tensor does not return its pinned pages to the OS: the
            # process host allocator otherwise retains all previous size bins.
            # This is an explicit unload/configuration-switch boundary only.
            gc.collect()
            from .resources import release_idle_host_cache

            self.last["host_cache_release"] = release_idle_host_cache()
            torch.cuda.empty_cache()
            self.closed = True
        from .resources import deactivate

        deactivate(self)
