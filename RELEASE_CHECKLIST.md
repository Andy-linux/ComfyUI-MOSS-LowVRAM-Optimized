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
- The FP8 destination is now [Andylinux/MOSS-Image-Kohaku-FP8](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8).
  Only **two Base/SFT row checkpoints** are in the public upload scope. Their
  upload is in progress; require actual immutable commit/revision, bytes and SHA256
  before enabling downloads. One row file supports the FP8 / 16-bit compute choice.
  The old separate Base weight-only export stays legacy local, not a third upload.

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
  optional Sage and exact hardware/precision boundaries. The historical Turing
  bundle is separately user-authorized for a GitHub Release, with actual archive
  publication still pending. Preserve its unconfirmed upstream license status and
  actual source/build/component-license/NOTICE inventory. Source-code packages
  exclude all such binaries; the fixed optional manifest drives install-stage
  preparation only. Auto does not require it.
- Historical archived evidence: all seven earlier exact UI/API graphs passed normal-step HTTP
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
- Keep templates flat under `example_workflows/`; retain the six legacy class IDs
  and the separately added unified configuration node. Make `00_moss_lowvram.json`
  the sole user template. Old01–07 belong only to tests/historical_workflows,
  excluded from Registry packaging; do not link them as active share templates.
  Do not enable legacy `moss-optimized`/`ComfyUI-KohakuMOSS` and the new Registry
  pack simultaneously. Update/migrate one installation; the existing folder name
  may remain for launcher compatibility. Preserve any local `models.local.json`
  privately when migrating, never in the published archive.
  Insert genuine `cnr_id`/`ver` only after publishing and installing the real
  Registry version, then re-export and validate the resulting JSON/PNG artifacts.
- Validate the new 00 entry independently: changing Base/SFT, DiT storage/compute,
  TE format/compute, budget and VAE must create the correct selected asset plan.
  Its GUI preparation must fetch only the selected assets and prepare the complete
  Original TE directory, including tokenizer/config. Old `properties.models`
  metadata alone does not promise that complete directory. Changes in model family
  must not silently change prompt/seed/sampling; suggested sampling is explicit.
- Direct hook baseline check passed on Windows Python3.12.13 / Torch2.10+cu130 /
  Manager4.2.2, initially missing Triton and accelerate: requirements processing
  and install.py both returned exit 0. Keep this separate from UI-discovered
  clean Manager installation, which remains pending its own complete record.
- Check clean Manager installation and missing-model behavior separately for
  Desktop and self-hosted deployment. Start with missing plugin and missing Triton,
  not pre-seeded candidate files. For Manager4.2.2 on Windows, record UI discovery,
  queue state, next-start prestartup requirements/install.py logs, final versions,
  node registration, GUI preparation and restart state. A local staged hook test
  is useful separate evidence, never a fresh Manager installation claim.
- Confirm genuine Registry/Manager indexing before claiming workflow import →
  missing-node install. The default new Manager has no arbitrary Git URL route;
  a public repository alone is insufficient. Avoid promising a Fix Dependencies
  button: old UI uses Try fix/Reinstall, while new UI comes from the frontend.
  Default normal policy can reject fix (high+); never lower security for the demo.
- Keep dependency automation separate from runtime support: Windows auto-preparation
  covers the exact x64 Python3.12/Torch2.10.0+cu130/SM89 stack and the explicitly
  historical SM75 policy. SM75 is not newly qualified; RTX50 remains pending and
  RTX30 has no full-workflow promise. Unknown combinations get truthful diagnostics,
  not blind latest installs or Torch/CUDA/Python replacement. Dependency checks
  and inference must never execute pip; Manager install.py is the preparation phase.
- Inspect git-tracked archive contents and `.comfyignore`: no weights, local model
  mappings, credentials, private receipts, environments, transient caches or
  any embedded optional binaries. Separately authorized release assets must retain
  their real source/license/build inventory and may not leak private paths or GPU
  identifiers. Preserve tokenizer/runtime resources and license texts.
- Use valid X.Y.Z versioning; do not attempt to overwrite an already published
  Registry version.

## New-path validation scope

- Shared FP8 probe03 passed eight actual GPU requests: Base/SFT × FP8/16-bit × two,
  each 512²/two steps/decode on 4070TiSUPER with cudaMallocAsync. Eight CPU tests had
  zero skips; 263-layer q/scale pointers, hashes and bytes stayed unchanged. This
  is a shared-load/compute contract test, not normal-step image quality or speed.
- Recorded delta04 Windows 137/137 unit tests passed with zero skips; GUI unit 32
  passed. The frozen Turing helper separately passed 8 focused tests and the actual
  17-member ZIP pure validation on Windows. Do not describe that focused run as
  a new137-test full-suite execution or as SM75 GPU workload qualification.
- Candidate-service GUI 19 checks passed; job c0f6b40e6dc84e04bc19dc374ac81572 was
  ready by reusing renamed/subfolder W5, subfolder TE-v2 alias and registered
  extra-root VAE: zero model network downloads. Prompt 94fd4b29-a191-4ec0-977b-2f01326b2d6d
  succeeded at 1024²/16 steps/default SFT W5/TE W8A16/seed314159265 and saved
  MOSS/00_moss_lowvram_00001_.png. Log execution interval 25.76s is one observation,
  not a general speed guarantee. See docs/GUIDED_SETUP_VALIDATION.md.
- Earlier independent Chrome context run03 GUI 16 checks covered relationships,
  Original TE nine files, metadata and 24-value serialization without download or
  prompt. Neither UI round performed a genuine missing-plugin Manager install.
  Keep clean-Python preparation, candidate reused-model generation and actual
  empty-baseline Manager installation separate; the last remains pending.
- Turing draft ZIP/inventory assets uploaded and remote digest checked, but remain
  private draft; do not enable the optional manifest before public verification.
- Public HF target exists but two-row file uploads are still in progress with no
  remote completion record for both weights. GitHub
  remains nonpublic and Registry publication has not occurred. Record actual
  immutable responses before updating those states.

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

The earlier seven-graph execution evidence remains complete within its recorded
scope. New unified-node and clean Manager lifecycle validation require their own
actual results. The operator records source upload, two-row FP8 upload, repository
visibility and Registry publication only after receiving their real responses;
this checklist does not predict their outcome.
