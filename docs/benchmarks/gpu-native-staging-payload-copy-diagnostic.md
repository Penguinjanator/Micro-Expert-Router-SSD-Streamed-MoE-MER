# GPU-native physical staging payload-copy attribution

`diagnose-gpu-native-physical-staging-payload-copy` emits
`mer.gpu-native-physical-staging-payload-copy.v1`. It measures host preparation
and describes a discriminator; decode TPS is never a diagnostic pass/fail gate.
No hardware result is implied by the portable tests.

Build from the reviewed, clean branch on Linux with the canonical command:

```sh
cd rust-engine
cargo build --release --features tokenizer
```

After code review and explicit authorization for the authoritative L4 inference
run, invoke the combined standalone and one-arm real-inference diagnostic:

```sh
./target/release/micro-expert-router diagnose-gpu-native-physical-staging-payload-copy \
  --config /home/randyap8/slice11-qwen3-coder-gpu-native.toml \
  --expected-adapter-name 'NVIDIA L4' \
  --warmup-iterations 3 \
  --iterations 20 \
  --report-out /home/randyap8/staging-payload-copy-attribution-v1.json
```

To run only the bounded RAM/WGPU experiment, without loading a model, add
`--standalone-only`. This mode also accepts another exact adapter name for
non-authoritative portable GPU investigation. Full real inference requires
Linux, NVIDIA L4/Vulkan, the existing frozen config path/hash, and 2048 MiB
managed expert VRAM. It reuses the existing physical-install workload runner:
the fixed Rust-addition prompt, greedy, 128 output tokens, one warmup, three
measured runs, cache reset `keep`, and the frozen disabled predictor contract.
It selects only `ProductionNoZeroFillTreatment`; it does not run a comparison
inference arm or change inference configuration.

The standalone `--iterations` range is 1–1000; `--warmup-iterations` is 1–100.
They do not change inference warmup or measured counts. Fewer than ten measured
pairs per width produces `no-clear-discriminator`. Widths are always 1, 2, 4, 8.

## JSON contents and timing definitions

- `build_provenance` and `executable_identity` identify the binary. `tree_sha`
  is null because the existing embedded build provenance does not supply one.
  The real arm additionally carries existing model, config, artifacts, runtime
  adapter, and executable provenance.
- `geometry` is payload offset 4, payload 2,654,208 B, stride 2,654,212 B,
  tail 0. `standalone.source_payload_sha256` identifies the deterministic
  versioned source pattern, independent of model data.
- `standalone.ram` and `standalone.wgpu_staging` each contain four width cases,
  including every measured batch, checked operations/bytes, exclusive copy
  time, host batch wall time, both throughputs, epoch/coverage time, observed
  job concurrency, failures and accounting errors. WGPU acquisition, view
  drop/scheduling, and submit/drain are separately reported.
- Host batch wall includes Rayon dispatch/join and each job's work. WGPU
  submit/drain and RAM verification are outside wall and copy timers. Decimal
  GB/s is `bytes / nanoseconds`; it is host-copy throughput, not DMA bandwidth.
  Width 1 is inline; wider jobs use MER's existing shared Rayon pool. Actual
  concurrency is reported even when the pool cannot exercise the full width.
- The same validated complete-overwrite helper measures both destination
  types. RAM destinations are preallocated and reused; full payload hashes
  are checked outside timing after every RAM phase. WGPU views are never read.
  The isolated queue submits and waits after every WGPU batch, including
  warmups/failures. At most N writes are pending for width N, bounded at
  21,233,696 B for width 8. These submits/polls never enter ordinary inference.
- `real_inference.warmup_attribution` and
  `real_inference.measured_production_attribution` separate the phases and
  report install counts, epoch/payload/staged bytes, validation, epoch write,
  payload copy, broad preparation, queue staging, checked residual, copy GB/s,
  copy share, residual share and accounting validity. Full observer and
  workload evidence remains in `real_inference.production_arm`.
- Production `physical_slot_validation_us` includes the existing pre-view
  checked-writer work plus the shared destination coverage check. The broad
  historical `physical_slot_prepare_us` still measures original checked-writer
  preparation plus outer view fill. Residual is the checked difference between
  broad prepare and the three subphases. Overflow, underflow or missing
  observation counts fail accounting; they cannot silently disappear.
- Production retains integer microseconds per install. Sub-microsecond epoch
  writes can truncate to zero, and residual includes quantization and timer
  overhead. Standalone retains nanoseconds and reports fractional microseconds.
  A zero epoch total does not establish zero epoch-write cost.

`complete` means the requested diagnostic phases completed and their accounting
is valid. It is not performance qualification. Failures produce an incomplete
report and nonzero exit. Standalone-only reports have `real_inference: null`.

The final `diagnostic_classification` follows documented conservative rules:

1. `staging-view-memory-slower-than-ram`: WGPU/RAM summed-copy throughput ratio
   and paired p90 are both at most 0.90 at three or more widths.
2. `concurrency-amortization-dominant`: all aggregate and paired p10/p90 copy
   ratios are within 0.90–1.10, and both destinations gain at least 25% in
   aggregate wall throughput from width 1 to width 8, with observed concurrency.
3. `host-memcpy-bandwidth-dominant`: copy ratios satisfy the same similarity
   requirement and valid real production evidence attributes at least 80% of
   broad preparation to payload copy.
4. Otherwise, `no-clear-discriminator`.

All classifications require at least ten paired samples at every width and no
standalone failures. Raw batches, ratios and production attribution remain
authoritative. These descriptive labels do not prove a WGPU memory type,
transfer causality, or a production optimization.

## Production isolation

Normal staging remains `stage_q4_expert_residency_inner::<false, true, true>`.
It calls the ordinary no-zero helper, with no new timers. The measured
specialization alone calls the observed equivalent. Both share the checked
coverage contract, validate before the first write, and write the same epoch
and payload bytes. Production source work, concurrency, slot policy,
publication, queue ordering/submission count, and recovery are unchanged.

Portable focused checks use `cargo test --features tokenizer payload_copy`.
When libtest needs a larger thread stack, set `RUST_MIN_STACK=8388608` for tests
only. The runtime does not require that environment variable.
