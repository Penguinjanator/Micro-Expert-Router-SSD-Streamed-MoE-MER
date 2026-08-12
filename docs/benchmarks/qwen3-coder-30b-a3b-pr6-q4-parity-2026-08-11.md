# Qwen3-Coder 30B-A3B PR6 Q4_0 numerical parity, 2026-08-11

## 1. Result and scope

MER's strict Hybrid Q4_0 numerical qualification passed on a hardware NVIDIA
L4 through WGPU/Vulkan. The authoritative report used schema
`mer.strict-hybrid-q4-parity.v1`, returned `status=pass` with `failure=null`,
and exited with code 0. All 18 required PASS checks were true.

This is three distinct kinds of evidence:

1. **Raw WGSL block parity:** eight deterministic canonical
   `ggml-standard-v1` Q4_0 fixtures compared MER's authoritative CPU decoder
   with the existing WGSL Q4_0 matmul pipeline. The eighth fixture directly
   exercised multi-row output indexing and row stride at a nonzero Q4 block
   offset.
2. **Complete checkpoint-expert parity:** three deterministic hidden vectors
   ran through global expert 0 using the authoritative CPU Q4_0 expert forward
   and the production GPU routed-expert path.
3. **Execution and residency evidence:** per-vector snapshots proved one
   physical install, stable generation reuse, complete GPU I/O, and no CPU
   execution, fallback, degraded substitution, eviction, or stale retirement.

The result qualifies numerical correctness for this command, commit, adapter,
expert, and checkpoint artifact. It is not a model-quality evaluation, does
not validate generated-token quality, and makes no throughput or production
TPS claim. PR7 batching has not started.

The authoritative hardened run was performed from a clean detached checkout of
`08c05ff1079b7676623642d354d13c15994af1ea`, after the multi-row fixture and
review hardening were present. The earlier seven-case run at `dac1d213...` is
superseded historical evidence and does not qualify the hardened implementation.

## 2. Provenance and environment

| Field | Qualified value |
|---|---|
| MER commit | `08c05ff1079b7676623642d354d13c15994af1ea` |
| Git worktree | clean (`dirty=false`) |
| Package version | `0.1.0` |
| Report schema | `mer.strict-hybrid-q4-parity.v1` |
| Qualification mode | `strict-hybrid-q4-parity` |
| Report status | `pass` |
| Failure | `null` |
| Process exit code | `0` |
| Report filename | `pr6-q4-parity-08c05ff.json` |
| Report SHA-256 | `f50e0915288de6eb0847ea77a0f3685e91966aa54d178c8d8d6ebb63b9326beb` |
| Progress watchdog | 300 seconds |
| Cloud VM | GCP `g2-standard-32` |
| Adapter | `NVIDIA L4` |
| Vendor/device | `4318` (`0x10de`) / `10168` (`0x27b8`) |
| Device type | `DiscreteGpu` |
| NVIDIA driver/version | `NVIDIA` / `580.173.02` |
| WGPU backend | `vulkan` |
| Compute plane | `wgpu-vulkan` |
| Software adapter | `false` |
| Linux release build | `cargo build --release --features "avx512,blas,tokenizer,io_uring"` passed |

The exact adapter-name gate required `NVIDIA L4`; a missing GPU, software
adapter, different adapter name, incompatible layout, malformed geometry,
dispatch failure, nonfinite result, or tolerance failure would have produced a
failed report and nonzero command exit.

## 3. Execution contract

| Component | Requested/resolved placement |
|---|---|
| Requested plan | `hybrid` |
| Resolved plan | `hybrid-cpu-attention-gpu-experts` |
| Embeddings | CPU |
| LM head | CPU |
| Dense projections | CPU |
| Attention | CPU |
| KV | CPU |
| Router | CPU |
| Routed experts | GPU |
| Routed-expert dtype | `q4_0` |
| Fallback occurred | `false` |
| GPU failure policy | strict fail-closed |

The selected global expert ID was 0. For the checkpoint's 48 layers and 128
experts per layer, the global namespace is `48 * 128 = 6144` experts and the
identity mapping was:

| Identity | Value |
|---|---:|
| Global expert ID | 0 |
| Layer index (`global / 128`) | 0 |
| Layer-local expert ID (`global % 128`) | 0 |

The expert geometry was `d_model=2048` and `d_ff=768`. The canonical Q4_0
payload, aligned checkpoint slot, and physical device allocation were each
2,654,208 bytes; this particular expert required zero alignment padding. The
payload SHA-256 was
`705583308419366429619a8840cf5997e2b1523f97b59405d32771ceacdd4948`.

## 4. Fixed comparison contract

The tolerances are schema constants, not values fitted to this run:

| Comparison | Absolute | Relative | Reference | Formula |
|---|---:|---:|---|---|
| Raw Q4_0 shader | `1e-5` | `1e-4` | authoritative CPU f32 | `abs_error <= absolute + relative * abs(cpu_reference)` |
| Complete expert | `2e-3` | `5e-3` | authoritative CPU f32 rounded to f16 | `abs_error <= absolute + relative * abs(cpu_f16_reference)` |

At the complete-expert production boundary, the primary comparison is the
authoritative CPU result rounded to f16 versus the GPU-returned f16. The typed
report also retains the original CPU f32 output, rounded CPU f16 output, GPU
f16 output, per-element absolute and relative errors, allowed error, and the
worst comparison index. Any nonfinite CPU or GPU value fails before tolerance
evaluation.

Schema v1 serializes `f32::MAX` as the finite sentinel for an undefined relative
error when the reference is zero and absolute error is nonzero. PASS remains
controlled by the combined absolute-plus-relative allowance, which uses only
the absolute component at a zero reference.

## 5. Raw WGSL block parity

All eight canonical cases passed with exactly zero observed absolute error,
zero observed relative error, and worst index 0:

| Case | Projection | Rows | Columns | `w_block_off` | Byte offset | Unaligned 18-byte start | Max abs | Max rel | Worst index | Result |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `zero-scale-extrema` | standalone | 1 | 32 | 0 | 0 | no | 0 | 0 | 0 | PASS |
| `positive-scale-extrema` | standalone | 1 | 32 | 0 | 0 | no | 0 | 0 | 0 | PASS |
| `negative-scale-sign` | standalone | 1 | 32 | 0 | 0 | no | 0 | 0 | 0 | PASS |
| `multiple-blocks-nontrivial-hidden` | standalone | 1 | 64 | 0 | 0 | no | 0 | 0 | 0 | PASS |
| `gate-projection-offset-zero` | gate | 1 | 32 | 0 | 0 | no | 0 | 0 | 0 | PASS |
| `up-projection-offset-one-unaligned` | up | 1 | 32 | 1 | 18 | yes | 0 | 0 | 0 | PASS |
| `down-projection-offset-two` | down | 1 | 32 | 2 | 36 | no | 0 | 0 | 0 | PASS |
| `multi-row-offset-row-stride` | standalone | 2 | 32 | 1 | 18 | yes | 0 | 0 | 0 | PASS |

The fixtures cover zero scale, positive and negative scale/sign behavior,
extremal nibbles, multiple blocks, nontrivial hidden vectors, all three expert
projection offsets, nonzero block offsets, and the unaligned 18-byte Q4_0
boundary. The eighth case directly qualifies two-row output indexing and row
stride using distinct row blocks at nonzero `w_block_off=1`. The raw dispatch
used ephemeral qualification buffers. Before and after raw dispatch,
production expert residency and all production GPU-I/O counters remained zero,
so the raw phase could not pre-populate the complete expert.

## 6. Complete checkpoint-expert parity

Three deterministic vectors passed against the extracted complete checkpoint
expert:

| Vector | Max absolute error | Max relative error | Worst index | Expert upload | Hidden upload | Submit | Map | Completed readback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `7.6293945e-06` | `0.0009451796` | 1831 | 1 / 2,654,208 B | 1 / 8,192 B | 1 | 1 | 1 / 8,192 B |
| 1 | `9.536743e-07` | `0.0010615712` | 1500 | 0 / 0 B | 1 / 8,192 B | 1 | 1 | 1 / 8,192 B |
| 2 | `0` | `0` | 0 | 0 / 0 B | 1 / 8,192 B | 1 | 1 | 1 / 8,192 B |

The worst absolute error across all vectors was `7.6293945e-06`; the worst
reported relative error was `0.0010615712`. Both are observations, not changed
tolerances. Every vector recorded one GPU dispatch attempt and success, with
zero dispatch failures, CPU routed-expert dispatches, GPU-to-CPU fallbacks, and
degraded substitutions.

Each vector's `routed_delta.selected_routed_experts` is zero by design. This
qualification explicitly selects global expert 0 and invokes the existing
production `forward_moe_resident` boundary; it does not run the model router and
therefore must not increment the router-selection counter. This does not weaken
dispatch evidence: every vector recorded GPU attempts/successes/failures of
`1/1/0`, while CPU expert executions, GPU-to-CPU fallbacks, and degraded
substitutions all remained zero.

## 7. Physical residency and capacity evidence

The first vector began with no logical admission and no physical registry
entry. It installed expert 0 exactly once at generation 1 and uploaded exactly
2,654,208 expert-weight bytes. Vectors 1 and 2 began and ended with the same
generation-1 logical admission and the same physical entry, and each added
zero expert-weight uploads and zero expert-weight upload bytes.

After installation, the capacity ledger remained stable:

| Ledger/counter | Observed value |
|---|---:|
| Logical admitted bytes | 2,654,208 |
| Physical live bytes | 2,654,208 |
| Physical registry bytes | 2,654,208 |
| Routed-expert workspace bytes | 1,310,720 |
| Total tracked bytes | 3,964,928 |
| Expert capacity bytes | 12,884,901,888 |
| Physical entries | 1 |
| Physical installs | 1 |
| Physical evictions | 0 |
| Stale retirements | 0 |

Every vector also recorded one 8,192-byte hidden-state upload, one queue
submission, one map request, and one completed 8,192-byte readback. The
per-vector snapshots were continuous: each vector's `before` snapshot equaled
the preceding vector's `after` snapshot.

## 8. PASS contract

All 18 independently derived checks were true:

| Required check | Observed |
|---|---|
| `clean_build` | true |
| `strict_hybrid_preflight` | true |
| `canonical_q4_0_layout` | true |
| `exact_execution_plan` | true |
| `hardware_gpu_adapter` | true |
| `expected_adapter_exact_match` | true |
| `strict_gpu_failure_policy` | true |
| `global_expert_identity_valid` | true |
| `exact_expert_payload_size` | true |
| `raw_shader_cases_passed` | true |
| `raw_dispatch_isolated_from_expert_registry` | true |
| `initial_physical_install_exactly_once` | true |
| `subsequent_dispatches_reused_generation` | true |
| `subsequent_dispatches_uploaded_zero_weight_bytes` | true |
| `every_dispatch_completed_gpu_io` | true |
| `zero_evictions_or_stale_retirements` | true |
| `zero_cpu_fallback_or_degraded_execution` | true |
| `complete_expert_vectors_passed` | true |

The successful process exit and a separate typed `jq -e` validation both
confirmed the fail-closed PASS contract.

## 9. Qualified checkpoint and report checksum

The checkpoint artifact exercised by the qualification is published at
[Amalgafy/Qwen3-Coder-30B-A3B-Instruct-MER-Q4-0](https://huggingface.co/Amalgafy/Qwen3-Coder-30B-A3B-Instruct-MER-Q4-0).

| Artifact | SHA-256 |
|---|---|
| MER archive `qwen3-coder-30b-a3b-mer-q4_0-v1.tar.zst` | `659b8d31d0a83292c632aa109c8edb5301f4041b1a60ef43c6f23ec0404061fe` |
| Pure Q4_0 GGUF `Qwen3-Coder-30B-A3B-Instruct-pure-Q4_0.gguf` | `8ddf61cadd354a5095905cc5ce535c44b777d0313ac241abcd2ceafa3362551b` |
| Hardened typed report `pr6-q4-parity-08c05ff.json` | `f50e0915288de6eb0847ea77a0f3685e91966aa54d178c8d8d6ebb63b9326beb` |

This is the current PR6 qualification artifact. Its pure Q4_0 GGUF was
requantized from an already quantized GGUF using llama.cpp
`--allow-requantize --pure`, so quantization error may compound. This
numerical parity run proves MER CPU/GPU agreement on those canonical Q4_0
bytes; it does not evaluate the artifact's model quality. A future
release-grade Amalgafy artifact derived directly from BF16/FP16 source weights
must be identified and qualified separately rather than treated as equivalent
to this requantized artifact.

## 10. Software validation

The qualified implementation had the following pre-publication validation:

| Command | Result |
|---|---|
| `cargo test q4_parity` | 19 passed |
| `cargo test q4_0_shader_logic_tests` | 4 passed |
| `cargo test qualification` | 28 passed |
| `cargo test strict` | 40 passed |
| `cargo test` | 932 passed, 0 failed, 2 ignored |
| `cargo clippy --all-targets` | passed with existing warnings |
| `git diff --check` | passed |
| Local macOS `cargo build --release` | passed with existing warnings |
| Linux release build with `avx512,blas,tokenizer,io_uring` | passed |

The first commit on the branch is a separate test-only fix that gives server
tests unique atomic temporary-directory sequences. It changes no production
server behavior. The second commit adds the strict parity command and its
qualification-only seams while preserving the existing `qualify-hybrid-q4`
command and serving fallback behavior.

## 11. Reproduction

Start from the exact qualified commit and require a clean checkout:

```bash
git fetch origin feat/pr6-q4-numerical-correctness
git switch --detach 08c05ff1079b7676623642d354d13c15994af1ea
test "$(git rev-parse HEAD)" = \
  08c05ff1079b7676623642d354d13c15994af1ea
test -z "$(git status --porcelain)"

cd rust-engine
cargo build --release \
  --features "avx512,blas,tokenizer,io_uring"

./target/release/micro-expert-router \
  --progress-timeout-secs 300 \
  qualify-hybrid-q4-parity \
  --config "$HOME/mer-pr6/qwen3-coder-strict-hybrid-q4.toml" \
  --expert-id 0 \
  --expected-adapter-name "NVIDIA L4" \
  --report-out "$HOME/mer-pr6/pr6-q4-parity-08c05ff.json"
```

The config must select strict Hybrid execution and the canonical converted
checkpoint represented by the checksums above. The command intentionally
requires the global expert ID and exact adapter name; it does not silently
select or wrap either value.

Validate the typed report:

```bash
REPORT="$HOME/mer-pr6/pr6-q4-parity-08c05ff.json"

jq -e '
  .schema_version == "mer.strict-hybrid-q4-parity.v1" and
  .mode == "strict-hybrid-q4-parity" and
  .status == "pass" and
  .failure == null and
  .provenance.git_sha ==
    "08c05ff1079b7676623642d354d13c15994af1ea" and
  (.provenance.dirty | not) and
  .provenance.package_version == "0.1.0" and
  .device.name == "NVIDIA L4" and
  .device.vendor_id == 4318 and
  .device.device_id == 10168 and
  .device.wgpu_backend == "vulkan" and
  (.device.software_adapter | not) and
  .execution_plan.requested == "hybrid" and
  .execution_plan.resolved ==
    "hybrid-cpu-attention-gpu-experts" and
  .execution_plan.embeddings == "cpu" and
  .execution_plan.lm_head == "cpu" and
  .execution_plan.dense_projections == "cpu" and
  .execution_plan.attention == "cpu" and
  .execution_plan.kv == "cpu" and
  .execution_plan.router == "cpu" and
  .execution_plan.routed_experts == "gpu" and
  .execution_plan.routed_expert_dtype == "q4_0" and
  (.execution_plan.fallback_occurred | not) and
  (.raw_cases | length == 8) and
  (all(.raw_cases[];
    .passed and .max_absolute_error == 0 and
    .max_relative_error == 0 and .worst_index == 0)) and
  (any(.raw_cases[];
    .name == "multi-row-offset-row-stride" and
    .rows == 2 and .columns == 32 and
    .w_block_off == 1)) and
  .complete_expert.identity.global_expert_id == 0 and
  .complete_expert.identity.layer_index == 0 and
  .complete_expert.identity.layer_local_expert_id == 0 and
  (.complete_expert.vectors | length == 3) and
  (all(.complete_expert.vectors[];
    .passed and
    .routed_delta.selected_routed_experts == 0 and
    .routed_delta.gpu_dispatch_attempts == 1 and
    .routed_delta.gpu_dispatch_successes == 1 and
    .routed_delta.gpu_dispatch_failures == 0 and
    .routed_delta.cpu_routed_expert_dispatches == 0 and
    .routed_delta.gpu_cpu_fallbacks == 0 and
    .routed_delta.degraded_expert_substitutions == 0)) and
  (.checks | to_entries | length == 18 and
    all(.value == true))
' "$REPORT"

printf '%s  %s\n' \
  f50e0915288de6eb0847ea77a0f3685e91966aa54d178c8d8d6ebb63b9326beb \
  "$REPORT" | sha256sum --check
```

## 12. Remaining PR6 diagnostics and exclusions

Raw-block shader parity and extracted full-expert parity are completed by this
qualification. The remaining PR6 diagnostic work is separate:

- fixed-corpus CPU/Hybrid greedy token parity;
- physical-capacity and eviction qualification;
- injected GPU-dispatch-failure qualification; and
- repeated-run stability qualification.

Those items are not implicitly passed by this report. Likewise, this report
does not redesign CUDA, KV, attention, dense execution, scheduling, or model
quantization; does not qualify model quality; and does not make a production
TPS or 10-TPS claim. Layer/top-k batching and profile-led performance work
remain PR7 scope, and PR7 has not started.
