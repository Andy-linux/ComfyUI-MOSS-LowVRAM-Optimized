"""Read-only local release gates; never publishes, installs or loads torch."""

import argparse, ast, hashlib, json, re, sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    tomllib = None
from download_models import checked_file


def validate(root, *, publication=False):
    errors = []
    warnings = []
    if tomllib is None:
        return {
            "status": "BLOCKED",
            "errors": [
                "Use Python >=3.11 for TOML validation (no dependency auto-install)"
            ],
            "warnings": [],
        }
    data = tomllib.loads((root / "pyproject.toml").read_text())
    project = data.get("project", {})
    comfy = data.get("tool", {}).get("comfy", {})
    release = data.get("tool", {}).get("moss_release", {})
    name = project.get("name", "")
    version = project.get("version", "")
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*", name)
        or len(name) >= 100
    ):
        errors.append("Invalid Registry project.name")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append("Version must be X.Y.Z")
    unresolved = []
    if not re.fullmatch(
        r"https://github\.com/[^/\s]+/[^/\s]+/?",
        project.get("urls", {}).get("Repository", ""),
    ):
        unresolved.append("GitHub Repository URL")
    if not comfy.get("PublisherId", "").strip():
        unresolved.append("Registry PublisherId")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", release.get("ModelRepository", "")):
        unresolved.append("HF namespace/repository for user-exported weights")
    for item in unresolved:
        (errors if publication else warnings).append("Missing identity: " + item)
    if release.get("LicenseReview") != "component-inventory-reviewed":
        (errors if publication else warnings).append(
            "Current component license inventory has not been reviewed"
        )
    provenance_path = root / "LICENSE_PROVENANCE.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("status") != "CURRENT_COMPONENT_INVENTORY_REVIEWED":
            errors.append("License provenance inventory status incomplete")
        for component in provenance.get("licenses", []):
            path = root / component["saved_license"]
            if not path.is_file():
                errors.append(
                    "Missing preserved license/notice: " + component["saved_license"]
                )
                continue
            raw = path.read_bytes()
            if (
                len(raw) != component["bytes"]
                or hashlib.sha256(raw).hexdigest() != component["raw_sha256"]
            ):
                errors.append(
                    "License/notice bytes changed: " + component["saved_license"]
                )
        license_path = root / "LICENSE"
        original = root / "licenses/KohakuMoss-LICENSE.txt"
        if (
            not license_path.is_file()
            or not original.is_file()
            or license_path.read_bytes() != original.read_bytes()
        ):
            errors.append("Root LICENSE must equal preserved KohakuMoss License")
        for excluded in (
            "runtime/vendor/unpack_sm75.py",
            "licenses/Lowbit-GPLv3-LICENSE.txt",
        ):
            if (root / excluded).exists():
                errors.append("Historical excluded component reappeared: " + excluded)
        if not (root / "runtime/packed_decode.py").is_file():
            errors.append("Independent packed decoder missing")
    sources_path = root / "runtime/vendor/SOURCES.json"
    if not sources_path.is_file():
        errors.append("Missing current vendor SOURCES.json")
    else:
        sources = json.loads(sources_path.read_text())
        for section in ("known_vendor_files", "local_adapter_hashes"):
            for row in sources.get(section, []):
                path = (sources_path.parent / row["path"]).resolve()
                if not path.is_relative_to(sources_path.parent.resolve()):
                    errors.append(
                        "Source inventory path escapes vendor: " + row["path"]
                    )
                    continue
                if not path.is_file():
                    errors.append("Inventoried vendor source missing: " + row["path"])
                elif hashlib.sha256(path.read_bytes()).hexdigest() != row.get(
                    "current_sha256"
                ):
                    errors.append("Current vendor source hash stale: " + row["path"])
        for row in sources.get("known_assets_not_locally_reverified", []):
            path = sources_path.parent / row["path"]
            if not path.is_file():
                warnings.append(
                    "Staging-only asset not reverified: "
                    + row["path"]
                    + "; final package must include and verify it"
                )
        for component in sources.get("components", []):
            if not (sources_path.parent / component["license"]).is_file():
                errors.append(
                    "Broken component license reference: " + component["name"]
                )
    if provenance_path.is_file():
        independent = provenance.get("independent_packed_decoder", {})
        path = root / independent.get("path", "runtime/packed_decode.py")
        if path.is_file() and hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != independent.get("source_sha256"):
            errors.append("Independent packed decoder provenance hash stale")
    if not (root / "RELEASE_VALIDATION.md").is_file():
        warnings.append(
            "README links RELEASE_VALIDATION.md; owner must supply actual final runtime evidence before artifact freeze"
        )
    required = [
        "LICENSE",
        "NOTICE",
        "__init__.py",
        "nodes.py",
        "requirements.txt",
        "model_manifest.json",
        "LICENSE_PROVENANCE.json",
        "THIRD_PARTY_NOTICES.md",
        "MODEL_FILES.md",
        "RELEASE_CHECKLIST.md",
    ]
    for file in required:
        if not (root / file).is_file():
            errors.append("Missing " + file)
    manifest = json.loads((root / "model_manifest.json").read_text())
    ids = [x["id"] for x in manifest["assets"]]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate asset IDs")
    for asset in manifest["assets"]:
        if asset["status"] != "public":
            warnings.append(
                asset["id"] + ": local_only/publish_pending; download blocked"
            )
            continue
        if not asset.get("files"):
            errors.append(asset["id"] + ": empty public file set")
        for file in asset.get("files", []):
            try:
                checked_file(file, root / "_validation_models_root")
            except ValueError as e:
                errors.append(asset["id"] + ": " + str(e))
    for dep in project.get("dependencies", []):
        if re.match(
            r"(?i)^(torch|torchvision|torchaudio|triton|triton-windows|nvidia-)", dep
        ):
            errors.append("Do not overwrite environment/ABI automatically: " + dep)
    for path in root.rglob("*.py"):
        if any(
            part in (".git", "__pycache__", ".venv", "venv")
            for part in path.relative_to(root).parts
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeError) as error:
            errors.append(
                "Invalid Python source "
                + path.relative_to(root).as_posix()
                + ": "
                + str(error)
            )
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("eval", "exec")
            for node in ast.walk(tree)
        ):
            errors.append(
                "Registry forbids builtin eval/exec: "
                + path.relative_to(root).as_posix()
            )
    for wf in (
        (root / ".github/workflows").glob("*.y*ml")
        if (root / ".github/workflows").exists()
        else []
    ):
        text = wf.read_text()
        if "workflow_dispatch" not in text or re.search(
            r"^\s*(push|pull_request|schedule)\s*:", text, re.M
        ):
            errors.append("CI must be dispatch-only: " + wf.name)
    return {
        "status": "BLOCKED" if errors else "PASS",
        "publication_mode": publication,
        "errors": errors,
        "warnings": warnings,
        "identity_inputs_remaining": unresolved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--publication", action="store_true")
    a = ap.parse_args()
    result = validate(a.root, publication=a.publication)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
