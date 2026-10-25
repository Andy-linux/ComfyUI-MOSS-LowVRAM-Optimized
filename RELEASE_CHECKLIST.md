# Release checklist

## Three owner identity inputs still required

1. Actual GitHub repository URL (`project.urls.Repository`).
2. Created Registry PublisherId (`tool.comfy.PublisherId`), with a publishing API key
   stored privately as a GitHub Secret, never in this repository.
3. Actual HF namespace/repository for user-produced W5/W6 and FP8 exports
   (`tool.moss_release.ModelRepository`). Public Base/TE/VAE use pinned upstream
   repositories; the missing identity applies to the user's own exports.

Run `python tools/validate_release.py` for local structure/manifest checks, then
`python tools/validate_release.py --publication` for the strict publication gate.
The latter remains blocked on the three real owner identities. The current
component inventory and preserved-license hashes are checked; this is not formal
legal clean-room certification.
Python >=3.11 is needed for this stdlib-only TOML validator; no dependency auto-install.

## Before publishing

- Preserve the current root KohakuMoss LICENSE, NOTICE, component license texts
  and provenance. The independent mathematical packed decoder replaces the removed
  historical GPL decoder; do not restore that source or publish the private
  baseline git history. Create clean release history, preserving existing upstream
  attributions. Do not label all vendored code Apache-2.0 by inference.
- Verify public model metadata against the captured live HF API; pin full revisions
  and exact bytes/SHA256. For user exports, upload only when authorized, then
  replace pending entries with actual immutable metadata. Never invent URLs.
- Keep Torch/CUDA out of automatic requirements, describe tested Triton platforms,
  optional Sage and exact supported hardware/precision boundaries.
- Runtime dependency ranges start from the validated transformers 5.20.1,
  einops 0.8.2, comfy-kitchen 0.2.37 and safetensors 0.8.0. Transformers is <6;
  later versions still require compatibility checks. Torch/CUDA remain supplied
  by ComfyUI and are not automatically replaced.
- Run a clean-install check with default public Base INT8, model verification,
  workflow import and real output. Retain actual backend-tagged allocator evidence.
- Keep example JSON flat under `example_workflows/`; use stable node type mappings
  and genuine Registry `cnr_id`/`ver` from the installed published build.
- Inspect git-tracked archive contents and `.comfyignore`: no weights, local model
  mappings, credentials, benchmark dumps, local environments or transient caches.
- Increment X.Y.Z before publishing; Registry published versions are immutable.

## Explicit publishing only

No publish workflow is created or enabled by this preparation. If CI is added,
its only trigger must be `workflow_dispatch`; never `push`, pull_request or a
schedule. Run the local validator in publication mode before a manual
`comfy node publish` or manually dispatched official publishing action. Publishing
requires the owner's final authorization; no publication has occurred here.
