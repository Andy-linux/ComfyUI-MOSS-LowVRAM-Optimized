# Release validation

**Status: component, short runtime and the required five normal-step HTTP
workflows passed on the tested SM89 system.** Five saved images received limited
visual inspection. The separate exact-public-TE-v2 GPU/API check also passed.
All observations have `performance_eligible=false`; no speed improvement or
quantified image-quality certification is claimed.

## Scope and environment

The recorded runtime checks used one NVIDIA GeForce RTX 4070 Ti SUPER with
16 GB physical VRAM (Ada SM89), PyTorch 2.10.0+cu130 and CUDA build 13.0.
Both the native and cudaMallocAsync allocator configurations were checked.
The normal-step run reported cudaMallocAsync in system statistics and used the
CUDA allocator selection flag. Its dynamic-VRAM log recorded
`ModelPatcherDynamic=True`. Versions were ComfyUI 0.41.0, frontend 1.55.17,
comfy-kitchen 0.2.37 and comfy-aimdo 0.5.6 on Windows/Python 3.12.13.
Explicit low-memory profiles are **process allocation budgets on this same
16 GB card**, not tests on physical 4 GB, 6 GB or 8 GB cards. Reserved memory,
driver/context memory and allocations outside the tracked budget remain distinct.

Most execution rows used auto attention, BF16 VAE and auto TE placement. The
short executions used two sampler steps. They do not replace the public SFT
16-step and Base 50-step workflows, and do not establish broad visual fidelity.
Checks for one model format, allocator or optional backend do not automatically
qualify another format, GPU, operating system or binary build.

## Completed checks

| Validation layer | Recorded result | What the result establishes |
| --- | --- | --- |
| Component test suite | 41 tests PASS | The checked component contracts pass in the tested environment. |
| cudaMallocAsync allocator probe | PASS | The requested allocator configuration and checked budget behavior were observed. |
| Native allocator probe | PASS | The native control configuration and checked budget behavior were observed. |
| Packed decoder vs legacy black-box decoder | 10 cases; 169,939,328 decoded values exact | The checked decoding agrees exactly with the legacy black-box decoder for those cases. |
| FP8 dispatch row probe v2 | 6 selected cases PASS | Selected Triton dispatch plus profiler-row / scaled-operation evidence for the probe; not whole-model hardware-instruction certification. |
| Short end-to-end runtime executions | 17 completed two-step rows PASS | Positive/negative conditioning, latent and decoded image were finite for the completed recorded rows. |
| Shape transition regression | W5 and Base W8 short → long → short PASS | The returning short case reproduced its conditioning and latent hashes within each tested model path. |
| Native/async SFT W8 control | Conditioning and latent hashes match | This checked same-input allocator comparison reproduced those tensors; it is not a timing claim. |
| Normal-step HTTP workflows | 5/5 success, 10 complete PNGs | SFT 16-step and four Base 50-step graphs each produced Preview and Save PNGs with workflow/prompt metadata. |

The decoder exactness result concerns the decoding operation and tested values.
It does not assert that lossy weight quantization matches original BF16 weights.
Likewise, finite output is a numerical validity check rather than a visual quality
assessment. Detailed smoke-v4 TE fields recorded **200 attached A8 linear bases
and 200 executed A8 linear calls for each of W8A8 and W4A8**. The recorded GEMM
was `torch._int_mm`, INT8×INT8→INT32 with FP32 scale/bias, and CPU FP16
conditioning output. These are actual call counts for those checked requests,
not a claim about every other run or whole-model hardware instruction
certification. Original BF16/FP16 TE rows are listed separately.

### Completed runtime rows

| Run group | Completed two-step rows | Cases | Group status |
| --- | ---: | --- | --- |
| smoke-v3 | 2 | SFT W8 legacy; Base W8 | **Interrupted by overall tool timeout**, despite the two completed rows passing. |
| smoke-v4-remaining | 4 | Base FP8; W8A8 TE; W4A8 TE; original BF16 TE | PASS |
| smoke-v5-formats | 4 | SFT FP8 row; Base FP8 weight-only; original FP16 TE; Base FP8 with 4g process profile | PASS |
| smoke-v7-transitions | 6 | W5 and Base W8, each short → long → short under 4g process profile | PASS |
| native-control-v1 | 1 | SFT W8 legacy with native allocator | PASS |
| **Completed total** | **17** | All listed completed rows used two steps. | **Row count only; not an overall normal-workflow qualification.** |

### Hash regression details

Both transition sequences used maximum joint token counts **4,140 → 9,474 →
4,140**, with image shapes 1024×1024 → 1536×1280 → 1024×1024. For each model,
the first and returning short request matched both conditioning hashes and the
latent hash. This checks that changing sequence size and the selected capacity
plan did not leave stale state that altered the returning short result.

| Comparison | Conditioning | Latent SHA256 |
| --- | --- | --- |
| W5 short request before/after long request | Exact hash match | `d0c80b66b5bcdd3b47923eb2414bd9b4a75c0dc5d8fecea570042233ec9164ca` |
| Base W8 short request before/after long request | Exact hash match | `88d1805e0a131c5203c645ff2193670a9e863dc0610d308afb33e5549c689edd` |
| Native vs cudaMallocAsync SFT W8 short request | Exact hash match | `cfe7ab120c64b79d9a10e7c792e49452cfb07e638528a4e6e6052786d34bb87a` |

These are comparisons within the stated cases, not equality assertions across
different model weights, TE precisions or samplers. A private local checkpoint
and a public manifest checkpoint must not be treated as byte-identical unless
their actual hashes agree.

## Interrupted, rejected and unverified work

| Item | Recorded condition | Treatment |
| --- | --- | --- |
| smoke-v1 | Private runner placed the plugin ahead of ComfyUI on sys.path, so plugin nodes.py shadowed ComfyUI nodes | Runner changed to Comfy-first import order. Original failed attempt remains outside successful execution counts. |
| smoke-v2 | Checkpoint loader left a zero-length weight Parameter, violating the FFN contract after replacement by FrozenInt8Base | Removed that replaced weight Parameter and initialized `_held` as an empty typed tensor. Original failed attempt is not counted as successful. |
| smoke-v3 overall run | Background tool timeout was set to 60 seconds; interruption was not a recorded model error | Remains interrupted, not relabeled PASS. Subsequent long-timeout runs completed the remaining cases. |
| smoke-v3 Base FP8 third row | INTERRUPTED; no completed dimensions, steps, tensor hashes or image evidence in that row | Excluded from the 17 completed rows. Base FP8 later completed separately in smoke-v4-remaining. |
| smoke-v6 public Flash selection | Current Windows torch build has no FlashAttention support | Correct explicit refusal, with no silent ordinary SDPA substitution. Same short/long transition sequence later passed with auto attention in v7. This does not qualify public Flash on that build. |
| FP8 dispatch row probe v1 | Upstream working() did not normalize an initial async allocator fraction above one | Added fraction normalization, outer cap and finally cleanup. v2 passed six selected probe cases; not counted as six additional end-to-end image executions. |
| Initial public SFT workflow schema check (http-normal-v1) | Hardcoded checkpoint filename absent from the running server's filename enum | No prompt was submitted. Changed only the UI/API DiT selector to `auto`; the subsequent five-workflow normal-step run succeeded. |
| Public fallback Flash and alternative Sage builds | Build-specific integration qualification is not established by the rows above | Remain separate backend qualification work. |

The SFT filename repair does not rename, rehash or silently replace a weight
file. `auto` uses the existing asset resolver, which accepts the two documented
SFT filename aliases and explicit local configuration. The workflow's model
download metadata retains the actual recommended manifest name, immutable URL
and SHA256. The resolved local source must still be recorded when executing.

## Public normal-step HTTP workflows — completed

One group of all five normal-step workflows succeeded after the SFT alias repair.
This group covers the required normal-workflow gate; duplicate runs are not a
requirement. Each result was RGB 1024×1024 and produced one Preview PNG and one
Save PNG, ten PNGs total. Every PNG contained prompt and workflow metadata, with
nine nodes in each embedded workflow. Static and two-step checks remain separate.

| Public workflow | Expected steps | HTTP execution result | PNG evidence |
| --- | ---: | --- | --- |
| 01_sft_w8_legacy_1024 | 16 | SUCCESS — alias resolver used | Preview + Save; RGB 1024×1024; metadata present |
| 02_base_w8_standard_1024 | 50 | SUCCESS | Preview + Save; RGB 1024×1024; metadata present |
| 03_base_fp8_row_standard_1024 | 50 | SUCCESS — local exported FP8 checkpoint | Preview + Save; RGB 1024×1024; metadata present |
| 04_base_w8_original_bf16_te_1024 | 50 | SUCCESS | Preview + Save; RGB 1024×1024; metadata present |
| 05_base_w8_real_w8a8_te_1024 | 50 | SUCCESS — experimental numerical mode | Preview + Save; RGB 1024×1024; metadata present |

The HTTP integration driver records the submitted API graph and UI metadata,
prompt ID, terminal execution status, errors or outputs, server system statistics
and complete downloaded PNGs. It refuses a nonempty existing queue and does not
cancel other tasks. An absent HTTP attention dispatch field stays unknown;
requesting `auto` is not proof of a specific actual attention kernel.

The maintainer individually inspected the five full-resolution Save PNGs. They
showed recognizable waterside villages, rivers, buildings, willow trees and sky,
with no blank output or obvious broken tile seams observed. This is a limited
visual check of those five images, not quantified quality certification. SFT and
Base compositions differ; these images do not isolate quantization's effect on
quality. Integration elapsed times are not benchmarks.

Sanitized results, including all ten PNG SHA256 values, are in
`docs/validation/results.json`. Original failed attempts remain in the ledger.

## Exact public TE v2 consistency

The exact public v2 header was combined with the existing v1 payload to create
a separate file; original v1 was untouched. All **1,377 tensor metadata entries
and offsets matched after removing the key prefix**. The complete new file
matches the public manifest:

```text
SHA256 511379fd690d59f11f49c062a8a687b9a985e92faa48a098eb7c66ba505574fe
Bytes  8452959908
Commit 16364ffd306244f33be05d64b34b47f6e15d0328
```

**Separate GPU/API two-step check of the exact-public-v2 file: PASS.** The actual ComfyUI server loaded the verified file, encoded both prompts, sampled and returned Preview/Save PNGs successfully. This
additional check is not included in the 17 short rows or five normal-step graphs;
those prior runs must not be relabeled as tests of this new exact-hash file.

## Remaining scope limits and follow-up

1. Public publication still requires the real GitHub repository, Registry publisher and HF repository identities.
2. Retain exact model provenance and distinguish public, private-local and
   publish-pending FP8 assets.
3. Qualify optional backend builds and any additional hardware independently.
4. Keep process-budget observations separate from physical low-VRAM-card claims.

This document intentionally omits machine-specific source paths and private raw
receipt locations. The maintainer retains the underlying evidence separately.

## Installed service cleanup

The ordinary HTTP Controls unload action completed successfully after validation.
The server remained on cudaMallocAsync with an empty queue. The pinned-host API
reported `allocated_bytes.current=0` and `allocations.current=0` after releasing
the idle pool. These are the reported pool counters, not a claim that every
process allocation or physical driver/context allocation is zero.

The five original Save PNGs, with their embedded workflows, are included under
`docs/samples/`. Their SHA256 values match the validation manifest.
