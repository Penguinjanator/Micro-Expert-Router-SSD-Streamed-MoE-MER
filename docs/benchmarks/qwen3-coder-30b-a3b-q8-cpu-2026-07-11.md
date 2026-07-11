# Qwen3-Coder 30B A3B Q8_0 CPU benchmark, 2026-07-11

## 1. Executive summary

This report records a narrow, completed CPU-only validation of
`Qwen3-Coder-30B-A3B-Instruct` through MER's strict full-transformer
`bench-real` path. In this configuration the converted checkpoint loaded
435 of 435 required dense tensors with `strict_weights=true`,
`fallback_seeded=false`, and `seeded_fallback_remained=false`, then ran
real learned `LinearGate` routing, SSD-streamed expert execution, and
full autoregressive inference.

The current measured Qwen real-inference performance baseline is the
1,536-slot run: 0.502 prompt tokens/s, 0.551 generated tokens/s, 61.780 s
TTFT p50, and 292.135 s mean total runtime for 31 prompt tokens plus 128
completion tokens. This verifies strict execution, but it is heavily
limited by foreground expert I/O and is not production-ready throughput.

The 768-slot run is a lower-memory operating tier, not the preferred
performance configuration. It reduced RSS by approximately 44% relative
to the 1,536-slot baseline, while decode throughput fell by
approximately 40% and TTFT increased by approximately 64%.

Synthetic `run` cache results are included separately. In those results,
`sustained_tps` means synthetic benchmark iterations per second, not
generated tokens per second. The synthetic 768-slot winner did not
transfer to real learned Qwen routing.

## 2. Verification scope

Verified checkpoint and execution scope:

| Field | Value |
|---|---|
| Model | `Qwen3-Coder-30B-A3B-Instruct` |
| GGUF source repository | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` |
| GGUF file | `Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf` |
| Architecture | `qwen3_moe` |
| Expert quantization | `Q8_0` |
| Execution | CPU-only |
| Benchmark path | Full real transformer inference through `bench-real` |
| Checkpoint mode | Strict converted checkpoint |
| Dense tensor status | 435 of 435 required dense tensors loaded |
| Strict flags | `strict_weights=true`, `fallback_seeded=false`, `seeded_fallback_remained=false` |
| Router | Real learned `LinearGate` |
| Layers | 48 transformer layers |
| Experts | 128 experts per layer; 6,144 layer-qualified experts |
| Routing | Top-8 expert routing |
| Dimensions | `d_model=2048`, `d_ff=768` |
| Weight residency | Dense transformer weights resident in RAM |
| Expert residency | Expert weights managed by MER's SSD-backed expert cache |
| Output parity | Generated-output parity across measured runs: true |

This report does not claim that all Qwen3-MoE checkpoints, all Qwen
models, or all GGUF quantization recipes have been validated.

## 3. Source of benchmark data

This report records user-supplied output from completed VM benchmark
runs. The values below are treated as measured data supplied by the user
and are the source of truth for this documentation update.

Codex did not have access to the benchmark VM or its raw log files and
did not independently inspect VM logs. Missing metrics are not inferred.

## 4. Tested commit and local working-tree note

The benchmarked base commit supplied for the VM runs was:

```text
78be8cfad78707ef864d87fbf1c9073ec97730d2
```

During this documentation update, the local checkout was at:

```text
f5ba1c7453e9582d76280e7dcd360744658b415d
```

Before editing, `git status --short`, `git diff`, and
`git diff -- rust-engine/src/config.rs` were empty in this checkout. No
local `rust-engine/src/config.rs` validation adjustment was present for
Codex to inspect here. If a narrow Qwen validation patch was present
during VM testing, its contents are not visible in this working tree and
are not described by this report. The benchmark should therefore be
attributed to the user-supplied base commit and benchmark context, not to
this later documentation checkout.

This documentation update does not modify runtime code.

## 5. Hardware and storage environment

Observed environment:

| Field | Observed value |
|---|---|
| Cloud VM | GCP `g2-standard-32` |
| vCPUs | 32 |
| System RAM | 128 GB |
| CPU | Intel Xeon reported at 2.20 GHz |
| CPU feature | AVX-512 available |
| Execution | CPU-only |
| Build features | `tokenizer,io_uring,blas,avx512,tui` |
| Dense matvec backend | `rayon-matrixmultiply` |
| Expert dtype | `Q8_0` |
| Expert payload | 5,013,504 bytes |
| Physical expert slot | 5,017,600 bytes |
| Converted model directory | approximately 31 GB |

Storage caveat: the Qwen model was tested from its current
boot-disk-backed path. It was not tested from the GCP local NVMe path
used by earlier Mixtral experiments. These Qwen numbers should not be
described as local-NVMe performance.

## 6. Model geometry

The verified Qwen3-Coder configuration used 48 transformer layers with
128 experts per layer, producing 6,144 layer-qualified experts. Routing
used top-8 expert selection. The model dimensions were `d_model=2048`
and `d_ff=768`.

Dense transformer weights were resident in RAM. Expert weights remained
SSD-backed and were managed by MER's expert cache.

## 7. Conversion and strict-weight status

The run used a strict converted checkpoint. The supplied benchmark output
reported 435 of 435 required dense tensors loaded, with:

```text
strict_weights=true
fallback_seeded=false
seeded_fallback_remained=false
```

The validation therefore covers real checkpoint execution rather than a
seeded fallback model.

## 8. Benchmark methodology

Two real `bench-real` configurations were measured:

| Configuration | Purpose |
|---|---|
| 1,536 cache slots | Current measured real-inference performance baseline |
| 768 cache slots | Lower-memory comparison tier |

Both real runs used greedy decoding, cache reset policy `keep`, 31 prompt
tokens, 128 completion tokens, 158 full transformer forwards per run, one
warmup run, three measured runs, neural speculator disabled, and
generated-output parity checks across measured runs.

A separate synthetic `run` cache matrix was also measured. Its
`sustained_tps` values are synthetic benchmark iterations per second.
They are not autoregressive generated tokens per second.

## 9. Real 1,536-slot baseline

Configuration:

| Field | Value |
|---|---:|
| Cache slots | 1,536 of 6,144 |
| Expert residency | 25% |
| Approx. raw expert-cache capacity | 7.18 GiB |
| Compute pool | Automatically sized 30 threads |
| Neural speculator | disabled |
| Decoding | greedy |
| Cache reset policy | keep |
| Prompt tokens | 31 |
| Completion tokens | 128 |
| Full transformer forwards per run | 158 |
| Warmup runs | 1 |
| Measured runs | 3 |
| Output parity | true |

Important correction: a helper script originally described this test as
using a saved Rayon autotune profile. Startup instead reported
`CPU Rayon threads: default source=rayon_default` and compute pool 30
threads with source `auto`. This is documented as an automatically sized
30-thread real-inference run, not an autotuned real-inference result.

Measured aggregate:

| Metric | Observed result |
|---|---:|
| Prompt throughput mean | 0.502 tokens/s |
| Decode throughput mean | 0.551 generated tokens/s |
| TTFT p50 | 61.780 s |
| Mean total runtime | 292.135 s |
| Cache hit rate | approximately 71.9% |
| Expert misses per measured run | approximately 17,043 |
| Expert data read per measured run | approximately 85.7 GB |
| Foreground expert SSD stall per run | approximately 231.9 s |
| RSS | approximately 17.8-18.1 GiB |
| Expert compute per run | approximately 29.7 s |
| Output parity | true |

Interpretation: this is the current measured Qwen real-inference
performance baseline. It verifies correct strict full-transformer
execution, but it is heavily limited by foreground expert I/O. It is not
production-ready throughput. The 61.780-second TTFT is part of the
headline result and should not be hidden.

## 10. Real 768-slot lower-memory run

Configuration:

| Field | Value |
|---|---:|
| Cache slots | 768 of 6,144 |
| Expert residency | 12.5% |
| Approx. raw expert-cache capacity | 3.59 GiB |
| Rayon threads | explicit 16 |
| Neural speculator | disabled |
| Locality | enabled |
| Affinity | enabled |
| Prefetch governor | enabled |
| Predict fanout | 4 |
| Pipeline depth | 4 |
| Decoding | greedy |
| Cache reset policy | keep |
| Prompt tokens | 31 |
| Completion tokens | 128 |
| Full transformer forwards per run | 158 |
| Warmup runs | 1 |
| Measured runs | 3 |
| Output parity | true |

Measured aggregate:

| Metric | Observed result |
|---|---:|
| Prompt throughput mean | 0.306 tokens/s |
| Decode throughput mean | 0.330 generated tokens/s |
| TTFT p50 | 101.513 s |
| Mean total runtime | 486.922 s |
| Cache hit rate | approximately 53.04% |
| Expert misses per measured run | approximately 28,492 |
| Expert data read per measured run | approximately 142.96 GB |
| Foreground expert SSD stall per run | approximately 429.98 s |
| RSS | approximately 9.9-10.2 GiB |
| Expert compute per run | approximately 28.2 s |
| Output parity | true |

The 768-slot setup is a lower-memory operating tier. It is not presented
as the preferred performance configuration.

## 11. Direct real-inference comparison

Comparison with the 1,536-slot baseline:

| Metric | 768-slot change |
|---|---:|
| Total RSS | reduced by approximately 44% |
| Decode throughput | reduced by approximately 40% |
| Prompt throughput | reduced by approximately 39% |
| TTFT | increased by approximately 64% |
| Expert misses | increased by approximately 67% |
| Foreground SSD stall | increased by approximately 85% |
| Expert compute | improved slightly |

The performance regression is primarily associated with cache capacity
and foreground I/O, not slower expert computation.

## 12. Synthetic cache matrix

Shared synthetic setup:

| Field | Value |
|---|---|
| Synthetic iterations | 10,000 |
| Layer-qualified experts | 6,144 |
| Layers | 48 |
| Experts per layer | 128 |
| Routing | Top-8 |
| Dtype | `Q8_0` |
| Rayon threads | 16 selected by autotune in every final case |
| Neural speculator | disabled |
| Locality | enabled |
| Affinity | enabled |
| Prefetch governor | enabled |
| I/O | `io_uring`, `force_ssd` |
| Workload | skewed synthetic workload |
| Zipf | `s=1.2` |
| Workload correlation | `0.7` |
| Seed | `42` |

Measured results:

| Cache slots | Residency | Approx. raw cache | Synthetic iterations/s | Hit rate | Misses | I/O share | SSD stall |
| ----------: | --------: | ----------------: | ---------------------: | -------: | -----: | --------: | --------: |
| 384 | 6.25% | 1.79 GiB | 101.502 | 92.79875% | 5,761 | 65.58% | 63.5101 s |
| 768 | 12.50% | 3.59 GiB | 125.761 | 94.68625% | 4,251 | 49.05% | 31.9680 s |
| 1,536 | 25.00% | 7.18 GiB | 76.703 | 96.13625% | 3,091 | 29.54% | 14.1372 s |
| 2,976 | 48.44% | 13.91 GiB | 60.212 | 96.6575% | 2,674 | 27.27% | 12.9498 s |

The 125.761 result is synthetic benchmark iterations/s. It is not
generated-token throughput and should never be labeled as generated-token
TPS.

## 13. Synthetic-versus-real routing interpretation

At 768 slots, synthetic routing produced a 94.68625% hit rate while real
Qwen routing produced approximately 53.04%. Real Qwen routing therefore
exhibited a much broader working set than the synthetic Zipf workload.

Increasing synthetic cache capacity reduced misses and foreground SSD
stall, but synthetic throughput declined beyond 768 slots. The cause of
that larger-cache slowdown has not been proven. Memory locality,
resident prepared-state retention, scheduling, VM behavior, or
cache-management overhead may be investigated, but none should be
presented as an established cause.

The synthetic winner did not transfer to real learned Qwen routing.
Synthetic cache tuning cannot replace `bench-real` validation.

## 14. Neural speculator observation

The neural speculator has the highest configured weighting in the
optional unified predictor. That weighting is an architectural design
choice, not benchmark proof that it improves every model or workload.

In the tested Qwen3-Coder configuration, the current global-output
neural speculator introduced substantial overhead. No end-to-end
throughput improvement was established during this benchmark work, so
the reported Qwen benchmarks keep the neural speculator disabled.

The feature remains experimental and workload-dependent. A future
layer-local design may be evaluated, but that redesign is not currently
implemented. This report does not claim that the neural speculator is
universally useless.

## 15. Limitations

Limitations of these measurements:

* Real Qwen throughput is currently dominated by foreground expert I/O.
* Prompt ingestion and TTFT remain major bottlenecks.
* The supplied Qwen tests used boot-disk-backed storage.
* Local NVMe performance remains unmeasured.
* Intermediate real cache capacities between 768 and 1,536 remain to be
  tested.
* A controlled 1,536-slot, explicit 16-thread real run remains to be
  performed.
* Lightweight prefetch and governor behavior require direct ablation.
* The larger-cache synthetic throughput regression requires profiling.
* The current global neural speculator is not recommended for this Qwen
  benchmark configuration.
* Real coding-agent task evaluation remains future work.

Do not compare these CPU results directly with GPU throughput without a
controlled benchmark. Do not draw unsupported production-readiness or
energy-efficiency conclusions from these runs.

## 16. Reproduction guidance

Use a repository-relative config path and a repository-relative converted
checkpoint path when documenting reproduction commands. Example command
shape:

```bash
cd rust-engine
cargo build --release --features "tokenizer,io_uring,blas,avx512,tui"

./target/release/micro-expert-router bench-real \
  --config ../bench-configs/qwen3-coder-q8-1536.toml \
  --prompt "Write a small Rust function that checks whether a string is a palindrome." \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measured-runs 3 \
  --cache-reset keep \
  --greedy

./target/release/micro-expert-router bench-real \
  --config ../bench-configs/qwen3-coder-q8-768.toml \
  --prompt "Write a small Rust function that checks whether a string is a palindrome." \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measured-runs 3 \
  --cache-reset keep \
  --greedy
```

The corresponding TOML files should set `[real_transformer].enabled =
true`, `[real_transformer].strict_weights = true`, `architecture =
"qwen3_moe"`, CPU execution, and the appropriate `[storage].cache_slots`
value. They should point `[model].data_dir`, `[real_transformer].weights_dir`,
and `[tokenizer].path` at repository-relative converted checkpoint assets,
for example under `../models/qwen3-coder-q8/`, rather than embedding
machine-specific absolute paths.

Do not run expensive model benchmarks as part of a documentation-only
update.

## 17. Next experiments

Recommended next experiments:

| Area | Experiment |
|---|---|
| Storage | Repeat the Qwen real baseline from GCP local NVMe and compare with the boot-disk-backed result. |
| Cache capacity | Measure intermediate real cache sizes between 768 and 1,536 slots. |
| Threading | Run a controlled 1,536-slot, explicit 16-thread `bench-real` comparison. |
| Prefetch | Directly ablate locality, affinity, and prefetch-governor behavior on real Qwen routing. |
| Synthetic regression | Profile the larger-cache synthetic throughput drop without claiming a cause in advance. |
| Speculator | Evaluate a layer-local neural speculator design if implemented; keep the current global-output speculator disabled for this Qwen configuration. |
| Task quality | Run real coding-agent task evaluation once throughput and correctness instrumentation are stable. |
