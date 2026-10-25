# Publishing-round validation — 2026-10-25

> Historical archived01–07 release-candidate evidence. These graphs now live only in
> tests/historical_workflows and do not ship as user templates. This record does
> not qualify the new00 GUI, shared-row activation switch, or clean Manager install.
> Original measurements, counts and docs/validation receipts below remain unchanged.

**Preparation validation completed: all seven exact workflow JSON pairs passed
normal-step ComfyUI HTTP execution.** This is an execution result, not a statement
that GitHub code was pushed, the repository became public, an HF model repository
was selected or the Registry pack was published. Source-upload progress and final
publication results must be recorded by the operator after those actions occur.

Full sanitized records:

- [All execution and environment observations](validation/publication-20261025.json)
- [E702 formatting-only proof](validation/e702-formatting-proof.json)
- [Post-execution Note-text UI lineage](validation/workflow-note-lineage.json)
- [Earlier Windows/Chrome v3 button/dialog checks](validation/frontend-dialog-20261025.json)
- [Final Windows/Chrome v4 visible-status/dialog checks](validation/frontend-final-20261025.json)

The records preserve all supplied data, including the first and repeated W5
execution. They contain no machine-specific absolute model paths or credentials.
Earlier `../RELEASE_VALIDATION.md` and `validation/results.json` remain unchanged
historical evidence. Existing five sample PNGs are not relabeled as this round's
new images.

## Environment and isolation

- Windows, RTX 4070 Ti SUPER SM89, reported physical VRAM 17,170,956,288 bytes.
- Existing Python 3.12 Windows environment, Torch 2.10.0+cu130, CUDA build 13.0,
  triton-windows 3.6.0.post26, comfy-kitchen 0.2.37, transformers 5.20.1,
  safetensors 0.8.0, accelerate 1.15.0 and einops 0.8.2.
- Isolated standard custom-node/model directories; no `models.local.json` and no
  optional private attention directories. xFormers, SageAttention and flash_attn
  were absent from the reported package inventory. Attention selection was auto.
- cudaMallocAsync allocator. The 4 GB/6 GB/8 GB profiles are software allocation
  policies on this same physical 16 GB-class Ada card, not physical small-card tests.

This was **not a fresh OS or clean Python dependency installation**. Isolation
checks remove dependence on private model mappings/attention packages for these
runs; they do not prove an unattended installation on every host.

## Model verification

The operator verified all SHA256 values for 14 existing public model files,
**38,776,396,306 bytes**, then downloaded and verified the exact public SFT W8
file, **5,413,698,958 bytes**. The combined inventory is **15 public files,
44,190,095,264 bytes**. This is a file-inventory total, not peak RAM/VRAM or a
claim that every optional checkpoint is included.

The six quantized-TE workflows selected the versioned local v2 basename. Its
upstream filename, revision and bytes remain fixed by `../model_manifest.json`;
the alias avoids legacy v1 name-only matches. Original TE used the explicit
`MOSS-Image-Base/text_encoder` directory and complete companion assets. The Base
FP8 graph used its already produced local FP8 conversion; its new HF destination
remains unselected.

## Exact workflows

All rows used batch 1, 1024×1024, fixed seed 314159265 and the normal HTTP prompt
API. Each API submission included the UI workflow for PNG metadata. UI/API schema
and widget checks were separate. HTTP execution success alone is not browser
import evidence; the separate actual Chrome check below covers only workflow 06.

| Workflow | DiT / TE | Profile | Steps | Result |
|---|---|---|---:|---|
| 01_sft_w8_legacy_1024 | Exact public SFT fused W8 / historical W8A16 TE semantics | 6 GB | 16 | PASS |
| 02_base_w8_standard_1024 | Base W8 / W8A16 TE | 8 GB | 50 | PASS |
| 03_base_fp8_row_standard_1024 | Base stored FP8 row / W8A16 TE | 8 GB | 50 | PASS |
| 04_base_w8_original_bf16_te_1024 | Base W8 / explicit Original BF16 TE directory | 8 GB | 50 | PASS |
| 05_base_w8_real_w8a8_te_1024 | Base W8 / explicit W8A8 TE | 8 GB | 50 | PASS |
| 06_sft_w5_lowvram_1024 | Public W5 / explicit W8A16 TE, FP32 VAE | 4 GB | 16 | PASS; also rerun after formatting |
| 07_sft_w6_lowvram_1024 | Public W6 / explicit W8A16 TE, FP32 VAE | 6 GB | 16 | PASS |

The ledger contains **eight successful execution rows covering seven distinct
workflows**, because 06 has an initial run and a post-formatting rerun. Every row
has zero recorded errors and two complete 1024×1024 output PNG records. Their
bytes and hashes are retained in the JSON. Elapsed seconds are integration
observations, **not a controlled throughput benchmark**. The W5 repeat reproduced
the two recorded PNG hashes; this is not a general deterministic-quality guarantee.

The original execution record preserves the tested UI/API SHA256 values. After
execution, only the unexecuted Note node's `widgets_values` text was replaced with
final user instructions in all seven UI templates. The release UI SHA256 therefore
differs from its tested UI SHA256; the API SHA256 is unchanged. The linked Note
lineage records both UI hashes and the unchanged API hash for every graph, with
node IDs, links, settings and model metadata unchanged. This documentation check
verified all seven current release UI hashes against that lineage and all API
hashes against the original execution ledger. No API/GPU execution record is
rewritten to pretend it tested a later UI hash. Old sample PNGs may contain older
graph selectors and earlier Note text.

## Separate actual Windows/Chrome browser check — v4 final PASS

At **2026-10-25T18:29:15.503122Z**, the operator completed the actual Chrome v4 check of
`06_sft_w5_lowvram_1024.json`, using the final Note-text release UI SHA256:

```text
4feadbce19c6a9759833e227e2781809b71f6857f2992fa5e7c8bf3a1eca00a3
```

The configured graph contained **9 nodes**. The Session retained **20 persistent
widgets for its original inputs**, plus **two temporary frontend widgets**; these
are not extra saved API inputs. The status settled. The actual button callback
refreshed and opened the report dialog, which reported RTX 4070 Ti SUPER and
cudaMallocAsync. All **six dialog checks** were true. Clicking close removed the
dialog, and serialization retained the same 20 input values. There were **no
console errors and no POST /prompt** requests.

The final environment label and displayName both read **“环境状态：依赖检查通过”**.
The operator inspected a 2x screenshot and confirmed that the status was visible.
All earlier v3 checks plus the new label check passed. Only
`web/environment.js` changed to use the officially supported label for the short
status; readonly/name/value behavior and serialized 20 inputs were retained.
The UI workflow SHA above did not change. No GPU/runtime changed and the 62-test
and seven-GPU-workflow counts are not increased by this frontend correction.

The [v4 final receipt](validation/frontend-final-20261025.json) records the final
outcome. The unchanged [v3 receipt](validation/frontend-dialog-20261025.json)
preserves earlier successful data/button/dialog checks and the display finding:
a disabled text widget's value was hidden by frontend design. The v4 label fix
addresses that visibility issue. Earlier receipts are not overwritten to
manufacture a later result. Browser scope remains **workflow 06
only**. This does not individually certify the other six browser imports,
missing-node/model installation UX, new image generation or visual image quality.
The seven-workflow GPU/HTTP execution ledger is separate; this browser check is
not counted as an additional sampling run.

## Post-formatting evidence chain

The official Comfy CLI **1.23.0** validation completed with exit code 0. The
separate selected Ruff checks **E702, S102 and S307** reported zero findings.
The proof records 32 changed Python files with exact equality of
`ast.dump(ast.parse(source), include_attributes=False)` before and after formatting.
This editing task also verified each proof's after-SHA against current source.
Preserved license hashes and the packed decoder were not changed by that cleanup.

01 with the exact public SFT file and the repeated 06 W5 request ran after the
formatting cleanup. Previously completed graph runs remain connected by the
32-file exact-AST proof; the record does **not** claim that every graph was rerun
again after formatting. AST equality is the stated semantic-preservation check,
not a new performance or hardware certification.

The Windows suite passed **62/62 tests, zero skips/failures**. The formatting
proof also includes a distinct cloud run: 54 passes plus 8 skips for missing
torch/safetensors. Do not merge these into a misleading single test count or
describe cloud skips as numerical passes.

## What is not established

- Actual Chrome import/button/dialog evidence is limited to workflow 06 v4 above;
  the other six graphs have not been individually browser-certified. New-image
  visual inspection and universal frontend-layout behavior are not claimed.
- No quantitative image-quality, BF16-equivalence, universal FP8-GEMM or speedup
  conclusion follows from these executions.
- Current hardware evidence is SM89. Historical SM75 W5 receipts and pending
  Blackwell/Linux work retain their separate scope.
- No Registry publication was attempted in these records. Secure handoff tests
  used fake-token cases; local encrypted-credential readability is not PAT service
  authorization. The public publisher state is separate from token authentication.
- Successful HTTP execution does not validate Desktop/self-hosted missing-node,
  model-download or dependency-install UX on every distribution.

Official mechanism references: [Registry validation/publishing](https://docs.comfy.org/registry/publishing),
[security standards](https://docs.comfy.org/registry/standards),
[template model metadata and host differences](https://docs.comfy.org/interface/features/template).
These explain the integration contracts; the actual PASS evidence is the dated
local execution ledger linked above.
