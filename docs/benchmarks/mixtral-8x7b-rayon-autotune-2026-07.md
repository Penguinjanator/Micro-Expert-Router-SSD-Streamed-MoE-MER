# Mixtral 8x7B Rayon Autotune Validation, July 2026

This note records supplied July 2026 observations for the CPU-only
Mixtral `run` expert-streaming benchmark after the Rayon/autotune work
merged. Treat these as observed VM data, not universal performance
claims.

The `run` benchmark executes real Q4_0 SwiGLU expert FFN work while
exercising the expert cache, SSD reads, routing, prefetching, and
run-summary telemetry. `sustained_tps` means benchmark iterations per
second. It is not full autoregressive LLM generated tokens per second.

## Environment

| Field | Value |
|---|---|
| Cloud | GCP |
| Machine | `g2-standard-32` |
| CPU allocation | 32 vCPU / 16 physical cores |
| RAM | 128 GB |
| Storage | GCP local NVMe SSD at `/mnt/localssd` |
| Execution path | CPU-only |
| Model | Mixtral 8x7B Instruct Q4_0 GGUF converted with `gguf-convert --native-quant` |
| Data directory | `/mnt/localssd/data/mixtral-q4` |
| Neural speculator | off |

Model shape:

| Field | Value |
|---|---:|
| `num_experts` | 256 |
| `num_experts_per_layer` | 8 |
| `num_layers` | 32 |
| `d_model` | 4096 |
| `d_ff` | 14336 |
| `top_k` | 2 |

## Command And Config

Common command traits:

```bash
env -u RAYON_NUM_THREADS \
./target/release/micro-expert-router \
  --cpu-mask 0-24 \
  --progress-timeout-secs 300 \
  run \
  --data-dir /mnt/localssd/data/mixtral-q4 \
  --dtype q4_0 \
  --cache-slots 124 \
  --tokens 3000 \
  --autotune-rayon \
  --autotune-coarse-tokens 512 \
  --autotune-tokens 1000 \
  --autotune-repeats 2 \
  --autotune-top-candidates 3 \
  --autotune-slow-p95-ms 180 \
  --autotune-slow-p99-ms 320 \
  --autotune-print-table \
  --predict-fanout 4 \
  --pipeline-depth 4 \
  --io-uring \
  --locality \
  --affinity \
  --affinity-neighbors-k 2 \
  --affinity-decay-epoch 4096 \
  --num-layers 32 \
  --num-experts-per-layer 8 \
  --workload skewed \
  --zipf-s 1.2 \
  --workload-correlation 0.7 \
  --prefetch-governor \
  --seed 42 \
  --force-ssd
```

The 10,000-iteration profile-reuse run used the same shape with
`--tokens 10000`. `--io-uring` requires a Linux build with the
`io_uring` cargo feature.

## Measured Results

| Run | Rayon pool source | Threads | Confidence/profile behavior | Iterations | Sustained benchmark iterations/s | Hit rate | Compute p50 | Compute p95 | Compute p99 | I/O share |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fast fallback/default 3k | fallback/default auto sizing | 23 | Low-confidence autotune result was rejected, not used, and not saved. | 3,000 | 15.848835491203294 | 96.23333333333333% | 54,495 us | 56,703 us | 58,143 us | 12.67% |
| High-confidence autotune 3k | autotune | 21 | High-confidence profile saved. | 3,000 | 15.60197624613278 | 96.23333333333333% | 55,807 us | 56,959 us | 58,111 us | 12.48% |
| Profile reuse 3k slow run | profile | 21 | Loaded high-confidence profile. | 3,000 | 11.429525914115017 | 96.23333333333333% | 95,551 us | 97,599 us | 98,431 us | 9.36% |
| Profile reuse 10k run | profile | 21 | Loaded high-confidence profile. | 10,000 | 15.667079831089065 | 96.675% | 54,047 us | 95,167 us | 96,639 us | 11.08% |
| High-confidence autotune final-run divergence | autotune | 22 | High-confidence probes selected 22 threads, but the final run landed in a slow regime. | 3,000 | 10.012079866778747 | 96.23333333333333% | 97,727 us | 99,391 us | 100,735 us | 8.26% |

Additional supplied detail for the 10,000-iteration profile-reuse run:

| Metric | Value |
|---|---:|
| Cycle p50 | 54,079 us |
| Cycle p95 | 149,247 us |
| Cycle p99 | 180,735 us |

Additional supplied detail for the high-confidence autotune final-run
divergence:

| Metric | Value |
|---|---:|
| Probe median p50 | around 55.4675 ms |
| Probe median sustained benchmark iterations/s | around 14.4313 |

## Interpretation

The Rayon/autotune work is valuable as safer thread-count discovery,
confidence handling, profile persistence/reuse, and observability. These
observations show that it can land in the fast compute regime and can
safely reject low-confidence selections.

The same observations also show that high-confidence probes do not
guarantee stable final-run throughput on a noisy VM. On this VM, the
fast compute regime was around 55-60 ms and the slow compute regime was
around 95-100 ms. VM scheduler and placement variability may still
produce fast and slow compute regimes under otherwise similar commands.

## Limitations

- These are CPU-only observations from one GCP VM shape and local NVMe
  setup.
- They are `run` benchmark iterations, not full autoregressive decoder
  generation tokens.
- The neural speculator was off, so these runs do not validate neural
  speculator behavior.
- Do not present the best observed VM run as a universal MER throughput
  claim.
- A saved high-confidence Rayon profile is useful evidence for a
  placement/configuration key, but it is not a guarantee that every later
  run on a noisy VM will stay in the same compute regime.

## Historical Notes

The 2026-06-27 Mixtral cache-scaling report remains useful historical
cache-scaling evidence, including its warning about a bimodal FFN compute
anomaly. Rayon/autotune improves thread-count discovery, confidence
handling, and observability; it does not by itself prove that VM
throughput is stable or that the bimodal compute behavior is eliminated.

See
[`mixtral-8x7b-cpu-cache-scaling-2026-06-27.md`](mixtral-8x7b-cpu-cache-scaling-2026-06-27.md)
and
[`mixtral-8x7b-cpu-cache-scaling-2026-06-25.md`](mixtral-8x7b-cpu-cache-scaling-2026-06-25.md)
for earlier cache-scaling results.
