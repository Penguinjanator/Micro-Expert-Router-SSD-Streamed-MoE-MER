use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::numa::{format_cpulist, EffectiveCpuAffinity};

pub const AUTOTUNE_PROBE_ENV: &str = "MER_RAYON_AUTOTUNE_PROBE";
pub const DEFAULT_AUTOTUNE_TOKENS: u64 = 2_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuAutotuneKey {
    pub machine_fingerprint: String,
    pub model_fingerprint: String,
    pub backend_fingerprint: String,
}

impl CpuAutotuneKey {
    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.machine_fingerprint, self.model_fingerprint, self.backend_fingerprint
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuAutotuneMachine {
    pub os: String,
    pub arch: String,
    pub cpu_features: Vec<&'static str>,
    pub available_logical_cores: usize,
    pub effective_cpu_mask: Option<Vec<usize>>,
}

impl CpuAutotuneMachine {
    pub fn from_effective_affinity(affinity: &EffectiveCpuAffinity) -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_features: detected_cpu_features(),
            available_logical_cores: affinity.logical_cores,
            effective_cpu_mask: affinity.cpus.clone(),
        }
    }

    pub fn fingerprint(&self) -> String {
        let mask = self
            .effective_cpu_mask
            .as_deref()
            .map(format_cpulist)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unavailable".to_string());
        let features = if self.cpu_features.is_empty() {
            "none".to_string()
        } else {
            self.cpu_features.join("+")
        };
        format!(
            "os={};arch={};features={};logical={};affinity={}",
            self.os, self.arch, features, self.available_logical_cores, mask
        )
    }
}

pub fn machine_fingerprint_from_affinity(affinity: &EffectiveCpuAffinity) -> String {
    CpuAutotuneMachine::from_effective_affinity(affinity).fingerprint()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detected_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if std::is_x86_feature_detected!("avx2") {
        features.push("avx2");
    }
    if std::is_x86_feature_detected!("avx512f") {
        features.push("avx512f");
    }
    if std::is_x86_feature_detected!("fma") {
        features.push("fma");
    }
    if std::is_x86_feature_detected!("amx-tile") {
        features.push("amx-tile");
    }
    features
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detected_cpu_features() -> Vec<&'static str> {
    Vec::new()
}

pub fn default_thread_candidates(effective_logical_cores: usize) -> Vec<usize> {
    let n = effective_logical_cores.max(1);
    let mut candidates = BTreeSet::new();
    candidates.insert(1);
    candidates.insert(n);
    candidates.insert(crate::parallel::default_compute_threads(n));

    for divisor in [2usize, 3, 4] {
        let lower = (n / divisor).max(1);
        let upper = n.div_ceil(divisor).max(1);
        candidates.insert(lower);
        candidates.insert(upper);
    }

    for delta in [1usize, 2, 4, 8] {
        if n > delta {
            candidates.insert(n - delta);
        }
    }

    let default = crate::parallel::default_compute_threads(n);
    for delta in [1usize, 2] {
        if default > delta {
            candidates.insert(default - delta);
        }
        let plus = default.saturating_add(delta);
        if plus <= n {
            candidates.insert(plus);
        }
    }

    candidates.into_iter().filter(|&c| c <= n).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayonAutotuneProbeResult {
    pub threads: usize,
    pub valid: bool,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub sustained_tps: f64,
}

pub fn select_best_probe(
    probes: &[RayonAutotuneProbeResult],
    slow_p95_threshold_ms: f64,
) -> Option<RayonAutotuneProbeResult> {
    let mut valid: Vec<RayonAutotuneProbeResult> =
        probes.iter().copied().filter(|p| p.valid).collect();
    if valid.is_empty() {
        return None;
    }
    let under_threshold: Vec<RayonAutotuneProbeResult> = valid
        .iter()
        .copied()
        .filter(|p| p.p95_ms <= slow_p95_threshold_ms)
        .collect();
    if !under_threshold.is_empty() {
        valid = under_threshold;
    }
    valid.sort_by(|a, b| {
        a.p50_ms
            .total_cmp(&b.p50_ms)
            .then_with(|| b.sustained_tps.total_cmp(&a.sustained_tps))
            .then_with(|| a.threads.cmp(&b.threads))
    });
    valid.into_iter().next()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgressWatchdogConfig {
    pub timeout: Option<Duration>,
}

impl ProgressWatchdogConfig {
    pub const fn disabled() -> Self {
        Self { timeout: None }
    }

    pub fn enabled(&self) -> bool {
        self.timeout.is_some()
    }
}

pub fn normalize_progress_timeout_secs(
    cli: Option<u64>,
    config: Option<u64>,
) -> ProgressWatchdogConfig {
    match cli.or(config) {
        Some(0) | None => ProgressWatchdogConfig::disabled(),
        Some(secs) => ProgressWatchdogConfig {
            timeout: Some(Duration::from_secs(secs)),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonAutotuneProfile {
    pub threads: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub sustained_tps: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RayonAutotuneProfileStore {
    profiles: BTreeMap<String, RayonAutotuneProfile>,
}

pub fn default_profile_path(data_dir: &Path) -> PathBuf {
    data_dir.join(".mer-rayon-autotune.json")
}

pub fn load_profile(path: &Path, key: &CpuAutotuneKey) -> Option<RayonAutotuneProfile> {
    let body = std::fs::read_to_string(path).ok()?;
    let store: RayonAutotuneProfileStore = serde_json::from_str(&body).ok()?;
    store.profiles.get(&key.cache_key()).cloned()
}

pub fn save_profile(
    path: &Path,
    key: &CpuAutotuneKey,
    profile: RayonAutotuneProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = match std::fs::read_to_string(path) {
        Ok(body) => serde_json::from_str::<RayonAutotuneProfileStore>(&body).unwrap_or_default(),
        Err(_) => RayonAutotuneProfileStore::default(),
    };
    store.profiles.insert(key.cache_key(), profile);
    let body = serde_json::to_string_pretty(&store)?;
    std::fs::write(path, body)?;
    Ok(())
}

pub fn percentile_ms(sorted_us: &[u64], q: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted_us.len() - 1) as f64 * q).round() as usize;
    sorted_us[idx] as f64 / 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa::EffectiveCpuAffinity;

    #[test]
    fn fingerprint_differs_for_different_effective_masks() {
        let a = EffectiveCpuAffinity {
            cpus: Some((0..25).collect()),
            logical_cores: 25,
            display: "0-24".to_string(),
        };
        let b = EffectiveCpuAffinity {
            cpus: Some((0..16).collect()),
            logical_cores: 16,
            display: "0-15".to_string(),
        };
        assert_ne!(
            machine_fingerprint_from_affinity(&a),
            machine_fingerprint_from_affinity(&b)
        );
    }

    #[test]
    fn fingerprint_separates_unknown_affinity_from_known_mask() {
        let unknown = EffectiveCpuAffinity {
            cpus: None,
            logical_cores: 25,
            display: "unavailable".to_string(),
        };
        let masked = EffectiveCpuAffinity {
            cpus: Some((0..25).collect()),
            logical_cores: 25,
            display: "0-24".to_string(),
        };
        assert_ne!(
            machine_fingerprint_from_affinity(&unknown),
            machine_fingerprint_from_affinity(&masked)
        );
    }

    #[test]
    fn candidate_generation_uses_effective_logical_cores() {
        let candidates = default_thread_candidates(25);
        assert!(candidates.contains(&25));
        assert!(candidates.contains(&crate::parallel::default_compute_threads(25)));
        assert!(candidates.iter().all(|&c| (1..=25).contains(&c)));
    }

    #[test]
    fn p95_aware_selection_prefers_non_slow_candidates_when_possible() {
        let probes = [
            RayonAutotuneProbeResult {
                threads: 8,
                valid: true,
                p50_ms: 8.0,
                p95_ms: 200.0,
                sustained_tps: 125.0,
            },
            RayonAutotuneProbeResult {
                threads: 12,
                valid: true,
                p50_ms: 9.0,
                p95_ms: 20.0,
                sustained_tps: 111.0,
            },
        ];
        assert_eq!(select_best_probe(&probes, 100.0).unwrap().threads, 12);
    }

    #[test]
    fn progress_timeout_zero_disables_watchdog() {
        assert!(!normalize_progress_timeout_secs(Some(0), Some(300)).enabled());
        assert!(!normalize_progress_timeout_secs(None, Some(0)).enabled());
    }

    #[test]
    fn positive_progress_timeout_enables_watchdog_config() {
        let cfg = normalize_progress_timeout_secs(Some(300), None);
        assert!(cfg.enabled());
        assert_eq!(cfg.timeout, Some(Duration::from_secs(300)));
    }
}
