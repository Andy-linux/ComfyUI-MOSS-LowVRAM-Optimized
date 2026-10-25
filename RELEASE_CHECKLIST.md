# Release checklist

## Confirmed identities and remaining publication boundary

- GitHub target: [Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized).
  This checklist does not assert that a push or visibility change occurred.
- Registry package: `kohaku-moss-lowvram-optimized`; display name:
  **MOSS LowVRAM Optimized**, using KohakuMoss inference components.
- Registry publisher: `xianbei`, public status **Active**. The local DPAPI-encrypted
  credential can be read/decrypted successfully. Registry service authentication
  with that PAT remains unverified; local decryption is not an authorization test.
  No suitable official read-only PAT verification endpoint has been established.
- A GitHub Actions secret is optional for a future CI publishing path and is not
  currently set by this preparation. Manual publication can use the private local
  credential; never commit it, print it or put it in workflow JSON.
- Only the destination HF namespace/repository for the **three new FP8 files**
  remains user-unselected. `tool.moss_release.ModelRepository` stays empty until a
  real HF destination is chosen; a GitHub URL is not an HF model repository.

Existing public W5/W6 already live in
[Andylinux/MOSS-Image-SFT-W5A8-W6A8](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8),
revision `bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`. Their exact public bytes/hash
match the measured local artifacts. They are indexed, not uploaded again. Existing
Base/SFT INT8, TE and VAE likewise retain pinned upstream sources.

## Local checks before any publication

```text
python tools/validate_release.py
python tools/validate_release.py --publication
```

The package requires Python >=3.12 and includes `accelerate` among its declared
runtime dependencies. Use the same Python that launches ComfyUI. The validator
does not install dependencies or publish. Publication mode must not manufacture a
missing HF identity or bypass a structural/metadata failure. Record actual output.

- Preserve root KohakuMoss LICENSE, NOTICE, component licenses and provenance.
  The independent packed decoder replaces the removed historical GPL decoder;
  do not restore that source or publish private baseline history. This inventory
  is not a formal legal clean-room certification.
- Verify model revisions, bytes and SHA256, including the versioned local TE-v2
  destination. Keep upstream filename unchanged and do not accept legacy v1 merely
  because its old basename matches.
- Keep Torch/CUDA outside automatic replacement. Check platform-specific Triton,
  optional Sage and exact hardware/precision boundaries. The unlicensed historical
  Turing bundle is not distributable and is not required by auto attention.
- Completed: all seven exact current UI/API graphs passed normal-step HTTP
  execution, including exact public SFT 01, Original TE directory 04, W5/W6 and
  versioned TE-v2 selectors. See `docs/PUBLICATION_VALIDATION.md` and its full JSON.
  Windows tests 62/62 and official CLI 1.23.0 passed; E702/S102/S307 findings were
  zero. Fifteen public files totaling 44,190,095,264 bytes were SHA-verified.
  This used an existing Windows environment with isolated directories, not a fresh
  dependency installation. Actual Windows/Chrome v4 finally passed final-Note 06 import
  and button/dialog checks: 9 nodes, 20 persistent plus two temporary widgets,
  settled status, refreshed GPU/allocator report, six dialog checks true, dialog
  removed after close, unchanged serialized input values, no console errors and
  no POST /prompt. Its label/displayName both read “环境状态：依赖检查通过”, visibly
  confirmed in a 2x screenshot. All prior checks plus the new label check passed.
  See `docs/validation/frontend-final-20261025.json`; the unchanged v3 receipt
  preserves its earlier button checks and disabled-value visibility issue. Only
  frontend label display changed; GPU/runtime, workflow SHA and 20 input values
  remain unchanged. This adds no unit-test or GPU-run count. The other
  six graphs are not individually browser-certified; GPU execution is separate.
  No new-image visual check is claimed. Old release evidence and old PNG samples
  remain historical; the earlier v2 browser receipt is unchanged.
- Preserve the original execution ledger hashes. The seven release UI files
  subsequently changed only Note text; `docs/validation/workflow-note-lineage.json`
  links tested/release UI SHA256 and unchanged API SHA256, with execution settings,
  links, node IDs and model metadata unchanged. Do not rewrite old run hashes.
- Keep templates flat under `example_workflows/`; retain the six stable class IDs.
  Do not enable legacy `moss-optimized`/`ComfyUI-KohakuMOSS` and the new Registry
  pack simultaneously. Update/migrate one installation; the existing folder name
  may remain for launcher compatibility. Preserve any local `models.local.json`
  privately when migrating, never in the published archive.
  Insert genuine `cnr_id`/`ver` only after publishing and installing the real
  Registry version, then re-export and validate the resulting JSON/PNG artifacts.
- Check clean install and missing-model behavior separately for Desktop and
  self-hosted deployment. Original TE requires the explicit multi-file downloader;
  ordinary model download metadata does not install all tokenizer/config files.
- Inspect git-tracked archive contents and `.comfyignore`: no weights, local model
  mappings, credentials, private receipts, environments, transient caches or
  unlicensed binaries. Preserve tokenizer/runtime resources and license texts.
- Use valid X.Y.Z versioning; do not attempt to overwrite an already published
  Registry version.

## Explicit publishing only

GitHub source upload, model upload and Registry publication are separate steps.
This package has no automatic publication trigger or scheduled publishing task.
If CI publishing is later added, use an explicit `workflow_dispatch`, not push,
pull_request or a schedule. A manually dispatched CI path may require a secret;
that is an optional deployment choice, not a prerequisite for local credential use.

After all authorized targets and credentials are concrete, execute the explicit
publishing action, retain its actual response and immutable identifiers, and only
then describe the artifact as published. Never test a PAT by issuing an otherwise
unnecessary write or publication.

Official references:

- [Registry publishing and package identity](https://docs.comfy.org/registry/publishing)
- [Metadata, dependencies and compatibility](https://docs.comfy.org/registry/specifications)
- [Publisher retrieval](https://docs.comfy.org/registry/api-reference/registry/retrieve-a-publisher-by-id)
- [Manager installation lifecycle](https://docs.comfy.org/custom-nodes/backend/manager)
- [Security standards: no runtime pip](https://docs.comfy.org/registry/standards)
- [Templates and Desktop/browser model placement](https://docs.comfy.org/interface/features/template)

Preparation and current execution validation are complete. The operator records
actual source-upload progress only after receiving its result; this checklist
does not predict a push. Repository public visibility, the three FP8 destination
and Registry publication remain distinct decisions.
