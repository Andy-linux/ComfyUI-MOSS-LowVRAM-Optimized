# Main MOSS workflow

Import **`example_workflows/00_moss_lowvram.json`** in ComfyUI. This is the sole
published workflow. It covers model family, stored weights, executed arithmetic,
text encoder, attention and allocation budget through one configuration node.
Historical graph JSON files and their old evidence notes are retained only under
`tests/historical_workflows/`, excluded from the published package by `.comfyignore`.

The default graph uses SFT packed W5 storage with INT8 × INT8 → INT32 operations,
Sumomo ConvRot v2 TE with W8A16 operations, attention auto, FP32 VAE and a 6 GB
process allocation profile. The latent is 1024 × 1024 and sampling uses 16 steps.
The process profile is not a physical-card qualification claim. Inspect the
reported environment scope before queueing.

## Prepare and reuse existing models

Use **Prepare / check selected configuration** on the main node. The dialog
shows runtime compatibility separately from Manager installation-stage preparation
scope. Complete installation in the available Manager interface and restart when
requested; Windows Manager may defer dependency preparation until restart.

The model plan inspects ComfyUI's standard registered model categories, including
`extra_model_paths.yaml` roots and subdirectories. It narrows existing ordinary
files by expected size and prioritizes known filenames. An upstream filename or
user rename is a candidate, not proof of model identity: no multi-gigabyte SHA256
read occurs during ordinary inspection. The plan clearly marks candidates awaiting
verification.

Confirm preparation to verify complete SHA256 values in a background worker.
Matching files are reused in place without duplicate downloads; inference resolves
to the verified actual file, including an upstream Sumomo TE filename without the
recommended `-v2` alias. Verification is cached against file identity and metadata;
changed files require verification again. Same-size files with different content
are preserved and never silently treated as the selected model variant.

If no candidate matches, preparation downloads only the missing fixed manifest
files. A wrong file at the exact recommended destination requires a refreshed plan
and explicit preserve-and-download confirmation. Its bytes are retained in the
same-directory `.moss_lowvram_rejected` folder before replacement is downloaded.
Candidate files under other names are never moved or overwritten.

Original float TE remains a coherent complete HF bundle: all four shards, the
index/config and three tokenizer files must match within one shared bundle root.
The loader does not combine unrelated partial directories. Model URLs, hashes and
sizes come from the fixed manifest, and custom files are not assigned invented
provenance. No model download starts merely by importing a workflow.

## Change settings

Changing model family or weight storage updates compatible computation choices and
recommended checkpoint names. It preserves prompt, seed and sampling settings.
Use **Apply recommended sampling parameters** explicitly when desired: SFT uses
16 steps and Base uses 50; only connected MOSS samplers are changed, and seeds and
prompts are preserved. Review unavailable-option reasons in the dialog if a choice
is filtered by compatibility policy.

Keep attention auto when optional Sage or Turing Flash preparation is unavailable
or unsuccessful. Installed optional files do not establish GPU qualification.

The API-form main graph is retained at
`tests/integration_workflows/00_moss_lowvram.json` for private validation; tests and
historical workflows do not ship in the release package.

## Observed main-workflow result

The Windows candidate service passed 19 GUI checks and completed preparation by
reusing three existing verified assets: renamed/subfolder W5, subfolder TE-v2
alias and an extra registered-root VAE. Model network downloads were zero.
The default SFT W5/TE W8A16 graph then generated 1024² at 16 steps/seed 314159265
successfully. This candidate reused-model result is separate from clean-Python
package preparation and from the still-pending empty-baseline Manager install.
See [GUIDED_SETUP_VALIDATION.md](GUIDED_SETUP_VALIDATION.md) for IDs and scope.
