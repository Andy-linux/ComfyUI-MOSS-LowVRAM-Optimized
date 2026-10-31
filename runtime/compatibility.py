"""Runtime compatibility ranges; installed patch releases are not hardware gates."""
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion

TRITON_SOURCE = "https://github.com/triton-lang/triton-windows#3-pytorch"
TORCH_TRITON = {
    (2, 7): (3, 3), (2, 8): (3, 4), (2, 9): (3, 5),
    (2, 10): (3, 6), (2, 11): (3, 6), (2, 12): (3, 7),
    (2, 13): (3, 7), (2, 14): (3, 8), (2, 15): (3, 9),
}


def satisfies(version, requirement):
    try:
        return Version(str(version)) in Requirement(requirement).specifier
    except InvalidVersion:
        return False


def windows_triton_requirement(report):
    sm = (report.get("device") or {}).get("sm", 0)
    torch = report.get("packages", {}).get("torch", {})
    version = Version(str(torch.get("version")))
    if sm == 75:
        # MOSS uses standalone triton.jit and eager PyTorch here, not Inductor.
        return "triton-windows>=3.2.0.post13,<3.3"
    if sm < 80:
        raise ValueError("默认推理需要 SM75 或 SM80+ 的 NVIDIA 显卡。")
    if sm >= 120 and (version < Version("2.7") or Version(str(report.get("cuda_build") or "0")) < Version("12.8")):
        raise ValueError("SM120 需要支持 Blackwell 的 Torch>=2.7 和 CUDA build>=12.8；不会自动替换宿主 Torch。")
    minor = TORCH_TRITON.get(version.release[:2])
    if minor is None:
        raise ValueError(f"Torch {version} 尚无上游确认的 Triton 配对；请更新 MOSS 插件，不要降级 ComfyUI。")
    major, minor = minor
    return f"triton-windows>={major}.{minor}.0.post13,<{major}.{minor + 1}"


def architecture_policy(report):
    sm = (report.get("device") or {}).get("sm", 0)
    if sm == 75:
        return {"architecture": "SM75 / Turing", "triton": ">=3.2.0.post13,<3.3",
                "execution": "standalone Triton JIT + eager PyTorch; no torch.compile",
                "default": "W5 INT8 GEMM / TE W8A16 / FP16 boundary / SDPA / FP32 VAE",
                "note": "Triton 3.3 已移除 SM75。保留 3.2 系列，不因此降级 ComfyUI 或 Torch；其他节点的 torch.compile 不在该路线内。"}
    if sm == 89:
        return {"architecture": "SM89 / Ada", "triton": "match Torch minor, accept compatible patch/post releases",
                "execution": "INT8 GEMM + Triton JIT; FP8 optional",
                "default": "W5 INT8 GEMM / TE W8A16 / FP16 boundary / SDPA / FP32 VAE",
                "note": "保留当前宿主；匹配范围内的 Triton 不重复安装。"}
    if sm >= 120:
        return {"architecture": f"SM{sm} / Blackwell", "triton": "match Torch minor; Triton>=3.3",
                "execution": "INT8 GEMM + Triton JIT; no Turing binaries",
                "default": "W5 INT8 GEMM / TE W8A16 ring / FP16 boundary / SDPA / FP32 VAE",
                "note": "默认使用 ring 编码与 SDPA；可显式选择 Sage、公共 Flash 和 CK INT8。"}
    return {"architecture": f"SM{sm}", "triton": "match Torch minor",
            "execution": "hardware-supported arithmetic", "default": "W5 / W8A16 / SDPA",
            "note": "按实际算子能力检查，不把版本号相等当成硬件能力。"}
