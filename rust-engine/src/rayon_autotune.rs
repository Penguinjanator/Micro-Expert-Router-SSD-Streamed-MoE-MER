//! CPU/Rayon thread-count autotuning.
//!
//! Rayon can build its global pool only once per process, so the tuner runs
//! each candidate in a child process with `RAYON_NUM_THREADS=<candidate>`.
//! The parent process never touches Rayon during probing; after it selects a
//! winner it initialises the global pool exactly once for the real run.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROFILE_VERSION: u32 = 1;
pub const DEFAULT_PROBE_TOKENS: u64 = 256;
pub const SLOW_REGIME_COMPUTE_US: u64 = 80_000;
pub const SELECTION_RULE: &str = "prefer candidates with compute_p95_us below slow threshold; then lowest compute_p50_us; tie-break highest sustained_tps; ignore failed/invalid probes";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuAutotuneKey {
    pub machine_fingerprint: String,
    pub model_fingerprint: String,
    pub data_dir_fingerprint: String,
    pub dtype: String,
    pub cache_slots: usize,
    pub backend: String,
}

impl CpuAutotuneKey {
    pub fn profile_path(&self) -> PathBuf {
        default_profile_dir().join(format!("{}.json", self.stable_id()))
    }

    pub fn stable_id(&self) -> String {
        stable_hash_hex(&format!(
            "v{}|machine={}|model={}|data={}|dtype={}|cache={}|backend={}",
            PROFILE_VERSION,
            self.machine_fingerprint,
            self.model_fingerprint,
            self.data_dir_fingerprint,
            self.dtype,
            self.cache_slots,
            self.backend
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuAutotuneProfile {
    pub version: u32,
    pub selected_rayon_threads: usize,
    pub candidate_results: Vec<CpuAutotuneProbeResult>,
    pub logical_cores: usize,
    pub cpu_model: Option<String>,
    pub machine_fingerprint: String,
    pub model_fingerprint: String,
    pub data_dir_fingerprint: String,
    pub dtype: String,
    pub cache_slots: usize,
    pub backend: String,
    pub selection_rule: String,
    pub created_at: String,
    pub created_at_unix_secs: u64,
}

impl CpuAutotuneProfile {
    pub fn new(
        key: &CpuAutotuneKey,
        logical_cores: usize,
        selected_rayon_threads: usize,
        candidate_results: Vec<CpuAutotuneProbeResult>,
    ) -> Self {
        let created_at_unix_secs = unix_now_secs();
        Self {
            version: PROFILE_VERSION,
            selected_rayon_threads,
            candidate_results,
            logical_cores,
            cpu_model: cpu_model(),
            machine_fingerprint: key.machine_fingerprint.clone(),
            model_fingerprint: key.model_fingerprint.clone(),
            data_dir_fingerprint: key.data_dir_fingerprint.clone(),
            dtype: key.dtype.clone(),
            cache_slots: key.cache_slots,
            backend: key.backend.clone(),
            selection_rule: SELECTION_RULE.to_string(),
            created_at: format!("unix:{created_at_unix_secs}"),
            created_at_unix_secs,
        }
    }

    pub fn matches_key(&self, key: &CpuAutotuneKey) -> bool {
        self.version == PROFILE_VERSION
            && self.machine_fingerprint == key.machine_fingerprint
            && self.model_fingerprint == key.model_fingerprint
            && self.data_dir_fingerprint == key.data_dir_fingerprint
            && self.dtype == key.dtype
            && self.cache_slots == key.cache_slots
            && self.backend == key.backend
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let body = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&body)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuAutotuneProbeResult {
    pub candidate_threads: usize,
    pub sustained_tps: Option<f64>,
    pub compute_p50_us: Option<u64>,
    pub compute_p95_us: Option<u64>,
    pub hit_rate_pct: Option<f64>,
    pub tokens: u64,
    pub cache_slots: usize,
    pub dtype: String,
    pub backend: Option<String>,
    pub quant_path: Option<String>,
    pub success: bool,
    pub error: Option<String>,
}

impl CpuAutotuneProbeResult {
    pub fn failure(
        candidate_threads: usize,
        tokens: u64,
        cache_slots: usize,
        dtype: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            candidate_threads,
            sustained_tps: None,
            compute_p50_us: None,
            compute_p95_us: None,
            hit_rate_pct: None,
            tokens,
            cache_slots,
            dtype: dtype.into(),
            backend: None,
            quant_path: None,
            success: false,
            error: Some(error.into()),
        }
    }

    fn valid_for_selection(&self) -> bool {
        self.success
            && self.compute_p50_us.is_some()
            && self
                .sustained_tps
                .is_some_and(|v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Clone, Debug)]
pub struct CpuAutotuneRun {
    pub profile: CpuAutotuneProfile,
    pub profile_path: PathBuf,
}

/// Candidate window around the existing smart fallback, plus the full visible
/// core count. For a 32-vCPU VM this returns 24..=32, covering the previously
/// observed fast band without assuming it is universal.
pub fn candidate_thread_counts(logical: usize) -> Vec<usize> {
    let logical = logical.max(1);
    let fallback = crate::parallel::default_compute_threads(logical);
    let radius = match logical {
        0..=8 => 2,
        9..=15 => 3,
        16..=31 => 4,
        _ => 6,
    };
    let lo = fallback.saturating_sub(radius).max(1);
    let hi = fallback.saturating_add(radius).min(logical);
    let mut set = BTreeSet::new();
    set.extend(lo..=hi);
    set.insert(fallback.min(logical).max(1));
    set.insert(logical);
    set.into_iter().collect()
}

pub fn select_best_candidate(
    results: &[CpuAutotuneProbeResult],
) -> Option<&CpuAutotuneProbeResult> {
    let mut valid: Vec<&CpuAutotuneProbeResult> =
        results.iter().filter(|r| r.valid_for_selection()).collect();
    if valid.is_empty() {
        return None;
    }
    if valid
        .iter()
        .any(|r| r.compute_p95_us.unwrap_or(u64::MAX) < SLOW_REGIME_COMPUTE_US)
    {
        valid.retain(|r| r.compute_p95_us.unwrap_or(u64::MAX) < SLOW_REGIME_COMPUTE_US);
    }
    valid.into_iter().min_by(|a, b| {
        let p50_cmp = a.compute_p50_us.cmp(&b.compute_p50_us);
        if p50_cmp != std::cmp::Ordering::Equal {
            return p50_cmp;
        }
        let a_tps = a.sustained_tps.unwrap_or(0.0);
        let b_tps = b.sustained_tps.unwrap_or(0.0);
        b_tps
            .partial_cmp(&a_tps)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub fn load_matching_profile(key: &CpuAutotuneKey) -> Option<(PathBuf, CpuAutotuneProfile)> {
    let path = key.profile_path();
    let profile = CpuAutotuneProfile::load(&path).ok()?;
    profile.matches_key(key).then_some((path, profile))
}

pub fn run_cpu_autotune(
    key: &CpuAutotuneKey,
    logical: usize,
    candidates: &[usize],
    probe_tokens: u64,
    cache_slots: usize,
    dtype: &str,
) -> Result<CpuAutotuneRun, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    let probe_tokens = probe_tokens.max(1);
    for &candidate in candidates {
        results.push(run_probe_child(candidate, probe_tokens, cache_slots, dtype));
    }
    let selected = select_best_candidate(&results).ok_or_else(|| {
        let failures = results
            .iter()
            .filter_map(|r| r.error.as_deref())
            .take(5)
            .collect::<Vec<_>>()
            .join("; ");
        if failures.is_empty() {
            "CPU/Rayon autotune produced no valid child probe results".to_string()
        } else {
            format!("CPU/Rayon autotune produced no valid child probe results: {failures}")
        }
    })?;
    let profile = CpuAutotuneProfile::new(key, logical, selected.candidate_threads, results);
    let profile_path = key.profile_path();
    profile.save(&profile_path)?;
    Ok(CpuAutotuneRun {
        profile,
        profile_path,
    })
}

fn run_probe_child(
    candidate: usize,
    probe_tokens: u64,
    cache_slots: usize,
    dtype: &str,
) -> CpuAutotuneProbeResult {
    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(e) => {
            return CpuAutotuneProbeResult::failure(
                candidate,
                probe_tokens,
                cache_slots,
                dtype,
                format!("failed to resolve current executable: {e}"),
            );
        }
    };
    let mut cmd = Command::new(exe);
    cmd.args(child_probe_args(probe_tokens, candidate))
        .env("RAYON_NUM_THREADS", candidate.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            return CpuAutotuneProbeResult::failure(
                candidate,
                probe_tokens,
                cache_slots,
                dtype,
                format!("failed to spawn child probe: {e}"),
            );
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    match parse_probe_stdout(&stdout) {
        Ok(mut result) => {
            result.candidate_threads = candidate;
            if !output.status.success() && result.success {
                result.success = false;
                result.error = Some(format!("child exited with status {}", output.status));
            }
            result
        }
        Err(e) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut msg = format!("failed to parse child probe JSON: {e}");
            if !output.status.success() {
                msg.push_str(&format!("; status={}", output.status));
            }
            let trimmed = stderr.trim();
            if !trimmed.is_empty() {
                msg.push_str("; stderr=");
                msg.push_str(&trimmed.chars().take(600).collect::<String>());
            }
            CpuAutotuneProbeResult::failure(candidate, probe_tokens, cache_slots, dtype, msg)
        }
    }
}

fn parse_probe_stdout(stdout: &str) -> Result<CpuAutotuneProbeResult, serde_json::Error> {
    let trimmed = stdout.trim();
    if let Some(line) = trimmed
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with('{'))
    {
        serde_json::from_str(line)
    } else {
        serde_json::from_str(trimmed)
    }
}

fn child_probe_args(probe_tokens: u64, candidate: usize) -> Vec<OsString> {
    let mut out = Vec::new();
    let mut args = std::env::args_os().skip(1).peekable();
    while let Some(arg) = args.next() {
        if arg == OsStr::new("--autotune-rayon") {
            continue;
        }
        if arg == OsStr::new("--autotune-tokens") {
            let _ = args.next();
            continue;
        }
        if os_arg_has_prefix(&arg, "--autotune-tokens=") {
            continue;
        }
        if arg == OsStr::new("--autotune-candidates") {
            let _ = args.next();
            continue;
        }
        if os_arg_has_prefix(&arg, "--autotune-candidates=") {
            continue;
        }
        if arg == OsStr::new("--tokens") {
            let _ = args.next();
            continue;
        }
        if os_arg_has_prefix(&arg, "--tokens=") {
            continue;
        }
        if arg == OsStr::new("--rayon-autotune-probe") {
            continue;
        }
        if arg == OsStr::new("--rayon-autotune-candidate") {
            let _ = args.next();
            continue;
        }
        if os_arg_has_prefix(&arg, "--rayon-autotune-candidate=") {
            continue;
        }
        out.push(arg);
    }
    out.push(OsString::from("--tokens"));
    out.push(OsString::from(probe_tokens.to_string()));
    out.push(OsString::from("--rayon-autotune-probe"));
    out.push(OsString::from("--rayon-autotune-candidate"));
    out.push(OsString::from(candidate.to_string()));
    out
}

fn os_arg_has_prefix(arg: &OsStr, prefix: &str) -> bool {
    arg.to_str().is_some_and(|s| s.starts_with(prefix))
}

pub fn default_profile_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("mer").join("autotune");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("mer")
            .join("autotune");
    }
    std::env::temp_dir().join("mer").join("autotune")
}

pub fn make_machine_fingerprint(logical: usize) -> String {
    let cpu = cpu_model().unwrap_or_else(|| "unknown-cpu".to_string());
    format!(
        "{}-{}-logical{}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        logical.max(1),
        cpu
    )
}

pub fn make_data_dir_fingerprint(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut parts = vec![canonical.display().to_string()];
    if let Ok(md) = std::fs::metadata(&canonical) {
        parts.push(format!("mtime={:?}", md.modified().ok()));
    }
    stable_hash_hex(&parts.join("|"))
}

#[allow(clippy::too_many_arguments)]
pub fn make_model_fingerprint(
    data_dir: &Path,
    num_experts: u32,
    top_k: usize,
    d_model: usize,
    d_ff: usize,
    expert_size: usize,
    num_layers: u32,
    dtype: &str,
    packed_blob: Option<&Path>,
    packed_manifest: Option<&Path>,
) -> String {
    let raw = format!(
        "data={}|experts={}|top_k={}|d_model={}|d_ff={}|expert_size={}|layers={}|dtype={}|packed_blob={}|packed_manifest={}",
        make_data_dir_fingerprint(data_dir),
        num_experts,
        top_k,
        d_model,
        d_ff,
        expert_size,
        num_layers,
        dtype,
        packed_blob
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        packed_manifest
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    stable_hash_hex(&raw)
}

pub fn cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let body = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                let model = rest.split_once(':')?.1.trim();
                if !model.is_empty() {
                    return Some(model.to_string());
                }
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        let model = String::from_utf8_lossy(&out.stdout).trim().to_string();
        (!model.is_empty()).then_some(model)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

pub fn stable_hash_hex(input: &str) -> String {
    // FNV-1a 64-bit: small, deterministic, and good enough for cache-file
    // names where collisions are inconvenient but not security-sensitive.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(candidate: usize, p50: u64, tps: f64) -> CpuAutotuneProbeResult {
        ok_with_p95(candidate, p50, p50 + 1000, tps)
    }

    fn ok_with_p95(candidate: usize, p50: u64, p95: u64, tps: f64) -> CpuAutotuneProbeResult {
        CpuAutotuneProbeResult {
            candidate_threads: candidate,
            sustained_tps: Some(tps),
            compute_p50_us: Some(p50),
            compute_p95_us: Some(p95),
            hit_rate_pct: Some(95.0),
            tokens: 64,
            cache_slots: 124,
            dtype: "q4_0".to_string(),
            backend: Some("cpu".to_string()),
            quant_path: Some("qmatmul-q4_0".to_string()),
            success: true,
            error: None,
        }
    }

    #[test]
    fn candidates_for_32_vcpus_cover_observed_fast_band() {
        let candidates = candidate_thread_counts(32);
        for expected in [24, 26, 27, 28, 29, 30, 31, 32] {
            assert!(
                candidates.contains(&expected),
                "missing candidate {expected}: {candidates:?}"
            );
        }
    }

    #[test]
    fn selection_prefers_stable_p95_over_slightly_lower_p50() {
        let results = vec![
            ok_with_p95(31, 55_711, 99_775, 12.0),
            ok_with_p95(24, 56_063, 59_007, 11.0),
        ];
        let selected = select_best_candidate(&results).unwrap();
        assert_eq!(selected.candidate_threads, 24);
    }

    #[test]
    fn stable_selection_prefers_lowest_p50_then_highest_tps() {
        let results = vec![
            ok(28, 55_000, 12.0),
            ok(29, 54_000, 11.0),
            ok(30, 54_000, 13.0),
        ];
        let selected = select_best_candidate(&results).unwrap();
        assert_eq!(selected.candidate_threads, 30);
    }

    #[test]
    fn selection_ignores_failed_invalid_candidates() {
        let mut failed = ok_with_p95(24, 40_000, 41_000, 99.0);
        failed.success = false;
        let mut invalid = ok(26, 39_000, f64::NAN);
        invalid.sustained_tps = Some(f64::NAN);
        let results = vec![failed, invalid, ok(30, 56_000, 12.0)];
        let selected = select_best_candidate(&results).unwrap();
        assert_eq!(selected.candidate_threads, 30);
    }

    #[test]
    fn selection_falls_back_to_p50_and_tps_when_all_candidates_have_slow_p95() {
        let results = vec![
            ok_with_p95(24, 56_063, 90_000, 15.0),
            ok_with_p95(31, 55_711, 99_775, 9.0),
            ok_with_p95(30, 55_711, 95_000, 11.0),
        ];
        let selected = select_best_candidate(&results).unwrap();
        assert_eq!(selected.candidate_threads, 30);
    }

    #[test]
    fn profile_round_trips_json() {
        let dir =
            std::env::temp_dir().join(format!("mer-autotune-profile-test-{}", unix_now_secs()));
        let key = CpuAutotuneKey {
            machine_fingerprint: "machine".to_string(),
            model_fingerprint: "model".to_string(),
            data_dir_fingerprint: "data".to_string(),
            dtype: "q4_0".to_string(),
            cache_slots: 124,
            backend: "cpu:rayon".to_string(),
        };
        let profile = CpuAutotuneProfile::new(&key, 32, 30, vec![ok(30, 55_000, 13.0)]);
        let path = dir.join("profile.json");
        profile.save(&path).unwrap();
        let loaded = CpuAutotuneProfile::load(&path).unwrap();
        assert!(loaded.matches_key(&key));
        assert_eq!(loaded.selected_rayon_threads, 30);
        assert_eq!(loaded.candidate_results.len(), 1);
        assert_eq!(loaded.selection_rule, SELECTION_RULE);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn probe_stdout_parser_uses_last_json_line_after_logs() {
        let stdout = "2026-07-08T00:00:00Z INFO child starting\n\
                      {\"candidate_threads\":2,\"sustained_tps\":10.0,\"compute_p50_us\":5,\"compute_p95_us\":7,\"hit_rate_pct\":100.0,\"tokens\":2,\"cache_slots\":2,\"dtype\":\"f32\",\"backend\":\"cpu\",\"quant_path\":\"f32-candle\",\"success\":true,\"error\":null}\n";
        let parsed = parse_probe_stdout(stdout).unwrap();
        assert_eq!(parsed.candidate_threads, 2);
        assert!(parsed.success);
    }
}
