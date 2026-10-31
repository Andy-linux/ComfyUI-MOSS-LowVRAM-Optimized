"""Install the exact locally qualified optional wheel into this node only.

No pip dependency resolution, global site-packages replacement or GPU execution.
Supply a previously downloaded wheel; its identity is verified before extraction.
"""

import argparse, hashlib, importlib.metadata, io, json, shutil, stat, sys, tempfile, zipfile
from pathlib import Path
from packaging.version import Version

import importlib.util
spec = importlib.util.spec_from_file_location("moss_sage_artifacts", Path(__file__).resolve().parents[1] / "runtime/sage_artifacts.py")
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


def prepare(wheel, node_root):
    """Validate the qualified archive and destination without writing/importing CUDA."""
    if sys.platform != "win32" or sys.version_info < (3, 10):
        raise RuntimeError("This qualified wheel requires Windows / Python >=3.10")
    import torch
    version = Version(importlib.metadata.version("torch"))
    if not Version("2.10") <= version < Version("3"):
        raise RuntimeError("Sage stable ABI requires Torch>=2.10,<3")
    record = artifacts.artifact(torch.version.cuda)
    if not record:
        raise RuntimeError("Sage wheel requires CUDA build >=12,<14")
    URL, SHA256, BYTES = record["url"], record["sha256"], record["bytes"]
    data = wheel.read_bytes()
    if len(data) != BYTES:
        raise ValueError("Unexpected wheel byte size")
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise ValueError("Unexpected wheel SHA256")
    vendor = node_root.resolve() / "runtime/vendor"
    target = vendor / "sageattn220"
    marker = "SAGE_WHEEL_PROVENANCE.json"
    provenance = {
        "url": URL,
        "sha256": SHA256,
        "wheel_bytes": BYTES,
        "torch_compatibility": ">=2.10,<3",
        "scope": "Optional CUDA attention; shared stable ABI",
    }
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        extensions = [name for name in names if name.endswith(".pyd")]
        if "sageattention/core.py" not in names or not extensions:
            raise ValueError("Missing expected source or compiled extensions")
        for member in archive.infolist():
            if (
                Path(member.filename).is_absolute()
                or stat.S_ISLNK(member.external_attr >> 16)
                or not (target / member.filename)
                .resolve()
                .is_relative_to(target.resolve())
            ):
                raise ValueError("Unsafe archive member: " + member.filename)
        bad = archive.testzip()
        if bad:
            raise ValueError("Corrupt archive member: " + bad)
    required = ["sageattention/core.py", *extensions]
    plan = {
        "status": "READY_TO_INSTALL",
        "target": str(target),
        "python": sys.executable,
        "provenance": provenance,
        "required_members": required,
    }
    if target.exists():
        recorded = target / marker
        if (
            not target.is_dir()
            or not recorded.is_file()
            or json.loads(recorded.read_text(encoding="utf-8")).get("sha256") != provenance["sha256"]
            or not all((target / name).is_file() for name in required)
        ):
            raise FileExistsError(
                "Refuse to replace an unrecognized or incomplete dependency directory: "
                + str(target)
            )
        plan["status"] = "ALREADY_INSTALLED"
    return plan, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument(
        "--node-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    plan, data = prepare(a.wheel, a.node_root)
    if a.check_only or plan["status"] == "ALREADY_INSTALLED":
        print(json.dumps(plan))
        return
    target = Path(plan["target"])
    vendor = target.parent
    vendor.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="sageattn220-staging-", dir=vendor))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(staging)
        if not all((staging / name).is_file() for name in plan["required_members"]):
            raise ValueError("Missing expected extracted package files")
        (staging / "SAGE_WHEEL_PROVENANCE.json").write_text(
            json.dumps(plan["provenance"], indent=2), encoding="utf-8"
        )
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    plan["status"] = "INSTALLED"
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
