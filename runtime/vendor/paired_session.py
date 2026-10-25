"""Paired Qwen3.5 conditioning service; no CLI, node registration, or CUDA on import.

Caller owns process-wide GPU admission and ComfyUI model-manager coordination.
Kohaku imports must be available on sys.path. All timings are seconds; byte counts
are weight H2D, not PCIe transaction measurements. Different stored quantizations
and different layouts require independent numerical/quality validation.
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import RLock
import time

TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"

def token_rows(tokenizer, texts, max_length=4096):
    """Full input tokenization, fail closed instead of silently truncating."""
    rows = [tokenizer.encode(TEMPLATE.format(text=s), add_special_tokens=False,
                             truncation=False) for s in texts]
    if any(not r or len(r) > max_length for r in rows):
        raise ValueError(f"full token lengths {[len(r) for r in rows]} exceed supported {max_length}; no text truncated")
    return rows

def choose_groups(lengths, layout="packed", max_pack_tokens=4096):
    """Bounded calls under one layer-major pass. Auto is conservative, uncalibrated."""
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("nonempty token sequences required")
    if max(lengths) > max_pack_tokens:
        raise ValueError("individual sequence exceeds max_pack_tokens")
    if layout == "auto":
        # No speculative padded policy: packed until device-specific measurements.
        layout = "packed"
    if layout not in ("single", "packed", "padded"):
        raise ValueError(layout)
    if layout == "single":
        return [[i] for i in range(len(lengths))], False
    groups, current = [], []
    for i in sorted(range(len(lengths)), key=lambda x: (-lengths[x], x)):
        trial = current + [i]
        cost = max(lengths[j] for j in trial)*len(trial) if layout == "padded" else sum(lengths[j] for j in trial)
        if current and cost > max_pack_tokens:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups, layout == "padded"

@dataclass(frozen=True)
class SessionConfig:
    weights: str
    device: str = "cuda:0"
    memory: str = "ring"
    w4a8: str = "packed"
    ring_slots: int = 2
    max_length: int = 4096
    max_pack_tokens: int = 4096
    # auto preserves the historical weight-only FP16 numerical path.
    te_mode: str = "auto"
    compute_dtype: str = "fp16"

@dataclass
class PairResult:
    positive: list
    negative: list
    receipt: dict

def _key(key):
    if key.startswith(("visual.", "model.visual.", "mtp.", "lm_head.")):
        return None
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    # Also accept a text-only single file, retaining strict unknown-key checks.
    plain = key.removeprefix("model.")
    if plain.startswith(("layers.", "embed_tokens.", "norm.", "rotary_emb.")):
        return plain
    raise KeyError(f"unsupported TE key: {key}")

class PairedTextSession:
    """Immutable weights with serial encoding, explicit one-entry negative cache.

    encode_pair returns two ordinary Comfy CONDITIONING values and a receipt.
    No positive cache. Cache is session-local; close invalidates all references.
    The loader preserves every stored base, including mixed W4A8/INT8 formats.
    """
    def __init__(self, encoder, config):
        self.encoder, self.config = encoder, config
        self.lock = RLock()
        self.negative_cache = None
        self.closed = False
        self.requests = 0

    @classmethod
    def load(cls, config):
        import torch
        from kohakumoss.encoders.text.files import TextFiles
        from kohakumoss.encoders.text.comfy_files import packaged_config, packaged_tokenizer
        from kohakumoss.encoders.text.load import TextCheckpoint, load_placed_text, COMFY_VALUES, hf_checkpoint
        from kohakumoss.encoders.text.encoder import QwenTextEncoder
        from kohakumoss.quant.comfy_format import comfy_layers, layer_base
        from kohakumoss.types.plans import TextEncoderPlan
        if __package__:
            from .te_precision import validate_mode, configure_base, DTYPES, TEA8Weight, TEA8PackedWeight
        else:
            from te_precision import validate_mode, configure_base, DTYPES, TEA8Weight, TEA8PackedWeight
        validate_mode(config.te_mode, config.compute_dtype)
        if config.memory not in ("ring", "resident"):
            raise ValueError("explicit ring/resident placement required")
        if config.w4a8 not in ("packed", "int8"):
            raise ValueError("explicit W4A8 representation required")
        if config.ring_slots < 2:
            raise ValueError("prefetching requires at least two ring slots")
        started = time.perf_counter()
        source = Path(config.weights)
        tokenizer = packaged_tokenizer()
        if source.is_dir():
            if config.te_mode != "original":
                raise ValueError("HF directories require explicit original TE mode")
            # Accept either OpenMOSS pipeline root or its text_encoder directory.
            model_dir = source / "text_encoder" if (source / "text_encoder").is_dir() else source
            ck = hf_checkpoint(model_dir)
            # Build only the language model, never even instantiate visual meta modules.
            ck.config = getattr(ck.config, "text_config", ck.config)
            files = ck.files
            from transformers import AutoTokenizer
            token_dir = model_dir.parent / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(str(token_dir if token_dir.is_dir() else model_dir),
                                                     local_files_only=True)
        else:
            files = TextFiles([source], _key)
            ck = TextCheckpoint(config.weights, packaged_config().text_config, files,
                                values=COMFY_VALUES, strict=True)
        layers = comfy_layers(files.keys(), files.get, files.dtype)
        formats = sorted(set(layer.fmt for layer in layers.values()))
        if config.te_mode == "original" and layers:
            raise ValueError("original TE requires original floating weights; dequantized INT8 is not original")
        if config.te_mode == "original" and any(files.dtype(k) not in COMFY_VALUES + (torch.float32,)
                                                 for k in files.keys()):
            raise ValueError("original TE contains non-floating tensors")
        if config.te_mode == "w4a8" and "w4a8" not in formats:
            raise ValueError("w4a8 TE requires a W4A8 checkpoint")
        if config.te_mode in ("w8a8", "w8a16") and formats != ["int8"]:
            raise ValueError("w8a8/w8a16 TE requires an INT8 checkpoint, not W4 or dense")
        def make_stored(layer):
            base=layer_base(layer,files.get)
            # Factories run in load_placed_text after the device capability is set.
            if sm==75:
                if __package__:
                    from .w4_eager_decode import attach
                else:
                    from w4_eager_decode import attach
                attach(base)
            return configure_base(base, config.te_mode)
        for path, layer in layers.items():
            ck.owned.update(layer.keys)
            ck.owned.update(k for k in files.keys() if k.startswith(path+"."))
            ck.kept[path] = (partial(make_stored, layer),
                            path+".bias" if path+".bias" in files else None)
        kernels = {n: ("eager" if n in ("causal_conv_varlen", "rms_norm_gated") else "triton")
                   for n in ("rms_norm", "causal_conv_varlen", "gated_delta_rule_varlen",
                             "rms_norm_gated", "attention_varlen_causal")}
        # Explicit overrides bypass upstream automatic probe fallback. Turing cannot
        # serve these SM80+ kernels; select the supported reference before bind.
        dev = torch.device(config.device)
        sm = None
        if dev.type == "cuda":
            major, minor = torch.cuda.get_device_capability(dev)
            sm = major * 10 + minor
            if config.compute_dtype == "bf16" and sm < 80:
                raise ValueError("BF16 TE compute requires CUDA SM80+; select fp16")
            if config.te_mode in ("w4a8", "w8a8") and sm < 80:
                raise ValueError("experimental A8 TE requires CUDA SM80+; no silent A16 fallback")
            if sm < 80:
                kernels["gated_delta_rule_varlen"] = "eager"
                kernels["attention_varlen_causal"] = "eager"
        stored_precision = "w4a8" if any(layer.fmt == "w4a8" for layer in layers.values()) else "compute"
        plan = TextEncoderPlan(memory=config.memory, precision=stored_precision, dtype=config.compute_dtype,
                               w4a8=config.w4a8, stream_resident=config.ring_slots,
                               layout="packed", forward="kernels", batch=2,
                               max_tokens=config.max_pack_tokens, host="file",
                               pinned=False, kernels=kernels)
        model, placement, report, plan = load_placed_text(
            ck, plan, config.device, lambda *a, **k: config.memory,
            lambda *a, **k: config.w4a8)
        enc = QwenTextEncoder(model, tokenizer, config.device, plan,
                              max_length=config.max_length, output_device="cpu",
                              placed=(placement, report, plan), dtype=DTYPES[config.compute_dtype],
                              layout="stored-bases")
        session = cls(enc, config)
        session._a8_bases = [m for m in model.modules() if isinstance(m, (TEA8Weight, TEA8PackedWeight))]
        if config.te_mode in ("w4a8", "w8a8") and not session._a8_bases:
            placement.release()
            raise ValueError("A8 TE mode attached no A8 linear bases")
        session.load_receipt = {"load_s": time.perf_counter()-started,
                                "plan": vars(plan), "ops": enc.op_names, "sm": sm,
                                "requested_kernels": dict(kernels),
                                "W4_decoder_policy": "eager original decode_codes; no torch.compile" if sm==75 else "upstream compiled device decoder",
                                "quant_report": report}
        session.precision_receipt = {
            "requested_te_mode": config.te_mode,
            "stored_formats": formats or ["original_floating"],
            "source_kind": "HF sharded directory" if source.is_dir() else "single safetensors",
            "effective_weight_mode": ("W4 packed->INT8 grid" if "w4a8" in formats else "W8 INT8") if formats else "original floating",
            "a8_linear_bases": len(session._a8_bases),
            "linear_activation_bits": 8 if config.te_mode in ("w4a8", "w8a8") else 16,
            "activation_input_dtype": config.compute_dtype,
            "output_feature_dtype": "torch.float16",
            "gemm": "torch._int_mm INT8xINT8->INT32; FP32 scale/bias" if config.te_mode in ("w4a8", "w8a8") else "upstream weight-only/dense 16-bit GEMM",
            "activation_quantization": "symmetric dynamic per token after ConvRot; round/clamp [-127,127]" if config.te_mode in ("w4a8", "w8a8") else "none",
            "weight_codes_scales_rotation_preserved": bool(layers),
            "original_weights": config.te_mode == "original",
            "quality_status": "experimental; GPU and quality validation required" if config.te_mode in ("w4a8", "w8a8") else "existing A16 path; validate new source/dtype",
        }
        session.load_receipt["te_precision"] = session.precision_receipt
        return session

    def encode_pair(self, positive, negative="", *, layout="packed", cache_negative=False):
        import torch
        from kohakumoss.encoders.text.forward import make_pack
        from kohakumoss_kernels.hardware.current import synchronize
        with self.lock, torch.inference_mode():
            if self.closed:
                raise RuntimeError("text session is closed")
            started = time.perf_counter()
            enc = self.encoder
            rows = token_rows(enc.tokenizer, [positive, negative], self.config.max_length)
            # Include layout in cache identity: shape changes may change rounding.
            cache_key = (tuple(rows[1]), layout, self.config.max_pack_tokens)
            hit = bool(cache_negative and self.negative_cache is not None and self.negative_cache[0] == cache_key)
            active = rows[:1] if hit else rows
            groups, padded = choose_groups([len(r) for r in active], layout, self.config.max_pack_tokens)
            packs = [make_pack([active[i] for i in g], enc.device, enc.pad_id, padded) for g in groups]
            before = enc.placement.h2d_bytes
            a8_before = sum(getattr(w, "_te_a8_calls", 0) for w in getattr(self, "_a8_bases", []))
            encode_start = time.perf_counter()
            enc.placement.begin()
            try:
                hidden = enc.forward(packs, lambda *a: None)
                outputs = [None]*len(active)
                for g, p, h in zip(groups, packs, hidden):
                    for i, part in zip(g, h.split(p.lengths, 0)):
                        outputs[i] = part.to(device="cpu", dtype=torch.float16)
                del hidden
            finally:
                enc.placement.end()
            synchronize(enc.device)
            encode_s = time.perf_counter()-encode_start
            pos = outputs[0]
            neg = self.negative_cache[1] if hit else outputs[1]
            if not hit and cache_negative:
                self.negative_cache = (cache_key, neg)
            self.requests += 1
            receipt = {"request": self.requests, "requested_layout": layout,
                       "te_precision": getattr(self, "precision_receipt", {}),
                       "a8_linear_calls": sum(getattr(w, "_te_a8_calls", 0) for w in getattr(self, "_a8_bases", [])) - a8_before,
                       "effective_layout": "padded" if padded else ("single" if all(len(g)==1 for g in groups) else "packed"),
                       "groups": groups, "token_lengths_positive_negative": [len(r) for r in rows],
                       "processed_tokens": sum(len(r) for r in active),
                       "kernel_tokens": sum(max(len(active[i]) for i in g)*len(g) if padded else sum(len(active[i]) for i in g) for g in groups),
                       "positive_cache_hit": False, "negative_cache_hit": hit,
                       "layer_weight_passes": 1, "weight_h2d_bytes": enc.placement.h2d_bytes-before,
                       "encode_s": encode_s, "request_s": time.perf_counter()-started,
                       "output_shapes": [list(pos.shape), list(neg.shape)],
                       "output_dtype": str(pos.dtype), "output_device": "cpu"}
            return PairResult([[pos.unsqueeze(0), {}]], [[neg.unsqueeze(0), {}]], receipt)

    def close(self):
        # Avoid the old runtime's unavailable accelerator.empty_host_cache API.
        with self.lock:
            if self.closed:
                return
            self.encoder.placement.release()
            self.encoder.model = self.encoder.forward = None
            self.negative_cache = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
