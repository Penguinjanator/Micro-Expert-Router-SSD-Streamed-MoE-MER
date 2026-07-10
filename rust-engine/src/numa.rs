//! NUMA / CPU-affinity helpers — gist Phase 4.
//!
//! On modest single-socket boxes the PCIe-bus distance between the NVMe
//! root complex and an arbitrary CPU core barely matters, but on
//! dual-socket / multi-die parts (Threadripper, EPYC, Intel SPR with
//! sub-NUMA-clustering on) **a stray Infinity-Fabric / UPI hop per
//! expert read** can dominate the per-token tail latency the engine
//! advertises. This module wires the bare minimum we need to keep that
//! hop out of the inference critical path:
//!
//! * `--cpu-mask <CPULIST>` and `[performance].cpu_mask` can pin **the
//!   whole process** before Tokio/Rayon startup. `MER_PIN_CORES=N`
//!   remains as a lower-precedence compatibility fallback. Best-effort
//!   and Linux-only: anywhere else this is a logged no-op so dev machines
//!   (macOS, Windows) still boot.
//! * [`pin_current_thread_to_core`] is the lower-level primitive for
//!   future per-thread pinning (e.g. the io_uring completion thread).
//!
//! Why pin to "node 0" and not to whichever node the NVMe sits behind?
//! Because doing the latter properly requires either `libhwloc` or
//! walking `/sys/bus/pci/devices/.../local_cpulist`, which in turn
//! requires the user to tell us *which* PCIe device backs the data
//! drive — a config-and-discovery rabbithole that adds more failure
//! modes than it removes. Node 0 is the right answer on every
//! single-socket part the engine has been benchmarked on and a
//! reasonable default elsewhere; a deeper refactor (one io_uring ring
//! per node, per-node buffer pools) is the next step and is called
//! out in the README's *Known limitations* section.

use std::env;

#[cfg(target_os = "linux")]
use std::{fs, path::Path};

#[cfg(target_os = "linux")]
const MAX_CPULIST_CPUS: usize = libc::CPU_SETSIZE as usize;
#[cfg(not(target_os = "linux"))]
const MAX_CPULIST_CPUS: usize = 4096;

/// Environment variable that, if set to a positive integer `N`, pins
/// the process to the first `N` CPUs of NUMA node 0 at startup.
pub const MER_PIN_CORES_ENV: &str = "MER_PIN_CORES";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuMaskSource {
    Cli,
    Config,
    LegacyMerPinCores,
}

impl CpuMaskSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Config => "config",
            Self::LegacyMerPinCores => "MER_PIN_CORES",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuMaskRequest {
    pub source: CpuMaskSource,
    pub cpus: Vec<usize>,
    pub display: String,
}

/// Effective CPU-affinity context observed after startup placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveCpuAffinity {
    pub cpus: Option<Vec<usize>>,
    pub logical_cores: usize,
    pub display: String,
}

/// Resolve CPU placement precedence without applying it.
///
/// No mask is selected by default. Operators can opt in with `--cpu-mask`,
/// `[performance].cpu_mask`, or the legacy `MER_PIN_CORES=N` fallback, in
/// that order.
pub fn resolve_cpu_mask_request(
    cli_mask: Option<&str>,
    config_mask: Option<&str>,
    legacy_mer_pin_cores: Option<&str>,
) -> Result<Option<CpuMaskRequest>, String> {
    if let Some(mask) = cli_mask.and_then(nonempty) {
        return parse_explicit_cpu_mask(mask, CpuMaskSource::Cli).map(Some);
    }
    if let Some(mask) = config_mask.and_then(nonempty) {
        return parse_explicit_cpu_mask(mask, CpuMaskSource::Config).map(Some);
    }
    if let Some(raw) = legacy_mer_pin_cores.and_then(nonempty) {
        let n = raw
            .parse::<usize>()
            .map_err(|_| format!("{MER_PIN_CORES_ENV}={raw:?} must be a positive integer"))?;
        if n == 0 {
            return Err(format!("{MER_PIN_CORES_ENV}={raw:?} must be > 0"));
        }
        let cpus = first_n_node0_or_online_cpus(n);
        if cpus.is_empty() {
            return Err(format!(
                "{MER_PIN_CORES_ENV}={raw:?} resolved to an empty CPU set"
            ));
        }
        let display = format_cpulist(&cpus);
        return Ok(Some(CpuMaskRequest {
            source: CpuMaskSource::LegacyMerPinCores,
            cpus,
            display,
        }));
    }
    Ok(None)
}

fn nonempty(s: &str) -> Option<&str> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub fn parse_explicit_cpu_mask(
    mask: &str,
    source: CpuMaskSource,
) -> Result<CpuMaskRequest, String> {
    let cpus = normalize_cpus(parse_cpulist(mask)?);
    if cpus.is_empty() {
        return Err("cpu mask must contain at least one CPU".to_string());
    }
    let display = format_cpulist(&cpus);
    Ok(CpuMaskRequest {
        source,
        cpus,
        display,
    })
}

/// Outcome of an `apply_mer_pin_cores_env` call. Public so the caller
/// can log a single human-readable line at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinResult {
    /// `MER_PIN_CORES` was unset or empty — no pinning was attempted.
    NotRequested,
    /// `MER_PIN_CORES` was set but invalid (non-numeric or `<= 0`).
    BadValue(String),
    /// Linux only: pinned to the listed CPUs (already de-duplicated
    /// and clamped to what node 0 actually exposes).
    Pinned { cpus: Vec<usize> },
    /// Pinning was requested but the kernel / OS does not support the
    /// primitive (non-Linux, or `sched_setaffinity` returned an error).
    /// Carries a human-readable reason.
    Unsupported(String),
}

impl PinResult {
    pub fn as_log_line(&self) -> String {
        match self {
            PinResult::NotRequested => format!("{MER_PIN_CORES_ENV} unset, no NUMA pinning"),
            PinResult::BadValue(s) => format!("{MER_PIN_CORES_ENV}=\"{s}\" invalid, ignored"),
            PinResult::Pinned { cpus } => format!("pinned process to CPUs {:?}", cpus),
            PinResult::Unsupported(why) => {
                format!("NUMA pinning unsupported on this platform: {why}")
            }
        }
    }
}

/// Read `MER_PIN_CORES` and, on Linux, apply it via `sched_setaffinity(2)`.
///
/// This is a best-effort call: bad values are ignored, missing
/// `/sys/devices/system/node/node0/cpulist` falls back to the first
/// `N` logical CPUs of the system, and any `sched_setaffinity` error
/// is reported as [`PinResult::Unsupported`] rather than aborting
/// startup.
pub fn apply_mer_pin_cores_env() -> PinResult {
    let raw = env::var(MER_PIN_CORES_ENV).ok();
    apply_mer_pin_cores_value(raw.as_deref())
}

fn apply_mer_pin_cores_value(raw: Option<&str>) -> PinResult {
    let raw = match raw {
        Some(s) if !s.trim().is_empty() => s,
        _ => return PinResult::NotRequested,
    };
    let n: i64 = match raw.trim().parse() {
        Ok(v) => v,
        Err(_) => return PinResult::BadValue(raw.to_string()),
    };
    if n <= 0 {
        return PinResult::BadValue(raw.to_string());
    }
    let n = n as usize;
    pin_first_n_to_node0(n)
}

pub fn apply_cpu_mask_request(request: Option<&CpuMaskRequest>) -> PinResult {
    let Some(request) = request else {
        return PinResult::NotRequested;
    };
    apply_cpu_mask(&request.cpus)
}

#[cfg(target_os = "linux")]
pub fn apply_cpu_mask(cpus: &[usize]) -> PinResult {
    let cpus = normalize_cpus(cpus.to_vec());
    if cpus.is_empty() {
        return PinResult::BadValue("empty CPU mask".to_string());
    }
    match set_affinity(&cpus) {
        Ok(()) => PinResult::Pinned { cpus },
        Err(e) => PinResult::Unsupported(format!("sched_setaffinity: {e}")),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn apply_cpu_mask(cpus: &[usize]) -> PinResult {
    let _ = cpus;
    PinResult::Unsupported("sched_setaffinity(2) is Linux-only".into())
}

/// Pin the calling process to the first `n` CPUs of NUMA node 0.
/// On non-Linux this is a logged no-op.
#[cfg(target_os = "linux")]
pub fn pin_first_n_to_node0(n: usize) -> PinResult {
    let cpus = first_n_node0_or_online_cpus(n);
    if cpus.is_empty() {
        return PinResult::Unsupported(
            "no CPUs reported for NUMA node 0 and /proc/cpuinfo empty".into(),
        );
    }
    match set_affinity(&cpus) {
        Ok(()) => PinResult::Pinned { cpus },
        Err(e) => PinResult::Unsupported(format!("sched_setaffinity: {e}")),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_first_n_to_node0(_n: usize) -> PinResult {
    PinResult::Unsupported("sched_setaffinity(2) is Linux-only".into())
}

/// Pin the **current thread** to a single CPU. Returns `Ok(())` on
/// success, `Err(reason)` otherwise. Linux only — `Err` everywhere else.
#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_core(core: usize) -> Result<(), String> {
    set_affinity_thread(&[core])
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread_to_core(_core: usize) -> Result<(), String> {
    Err("sched_setaffinity(2) is Linux-only".into())
}

/// Read the CPU list of NUMA node 0 from sysfs.
///
/// Format is the standard kernel cpulist syntax: `0-3,8-11` etc.
#[cfg(target_os = "linux")]
fn node0_cpus() -> std::io::Result<Vec<usize>> {
    let path = Path::new("/sys/devices/system/node/node0/cpulist");
    let s = fs::read_to_string(path)?;
    parse_cpulist(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(target_os = "linux")]
fn num_cpus_online() -> usize {
    // `sysconf(_SC_NPROCESSORS_ONLN)` — well-supported and avoids the
    // `num_cpus` crate dep.
    // SAFETY: `libc::sysconf` is an FFI call into the C library that
    // takes a single integer constant and returns an integer. It has
    // no memory-safety preconditions and is thread-safe per POSIX.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n > 0 {
        n as usize
    } else {
        1
    }
}

#[cfg(target_os = "linux")]
fn first_n_node0_or_online_cpus(n: usize) -> Vec<usize> {
    let mut cpus = node0_cpus().unwrap_or_else(|_| {
        // Fall back to all online CPUs in ascending order if sysfs is
        // unavailable (containers, exotic kernels). The "first N"
        // ordering still matches the user's intent.
        let max = num_cpus_online();
        (0..max).collect()
    });
    cpus = normalize_cpus(cpus);
    cpus.truncate(n);
    cpus
}

#[cfg(not(target_os = "linux"))]
fn first_n_node0_or_online_cpus(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Parse a kernel cpulist string (e.g. `"0-3,8,10-11"`).
pub fn parse_cpulist(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("bad cpulist range start: {a:?}"))?;
            let b: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("bad cpulist range end: {b:?}"))?;
            if b < a {
                return Err(format!("descending cpulist range: {a}-{b}"));
            }
            let len = b
                .checked_sub(a)
                .and_then(|delta| delta.checked_add(1))
                .ok_or_else(|| format!("cpulist range is too large: {a}-{b}"))?;
            if len > MAX_CPULIST_CPUS {
                return Err(format!(
                    "cpulist range {a}-{b} expands to {len} CPUs, exceeding supported maximum {MAX_CPULIST_CPUS}"
                ));
            }
            if b >= MAX_CPULIST_CPUS {
                return Err(format!(
                    "cpu {b} exceeds supported maximum CPU id {}",
                    MAX_CPULIST_CPUS - 1
                ));
            }
            for c in a..=b {
                out.push(c);
            }
        } else {
            let c: usize = part
                .parse()
                .map_err(|_| format!("bad cpulist cpu: {part:?}"))?;
            if c >= MAX_CPULIST_CPUS {
                return Err(format!(
                    "cpu {c} exceeds supported maximum CPU id {}",
                    MAX_CPULIST_CPUS - 1
                ));
            }
            out.push(c);
        }
    }
    Ok(out)
}

pub fn normalize_cpus(mut cpus: Vec<usize>) -> Vec<usize> {
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

pub fn format_cpulist(cpus: &[usize]) -> String {
    let cpus = normalize_cpus(cpus.to_vec());
    if cpus.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &cpu in cpus.iter().skip(1) {
        if cpu == prev + 1 {
            prev = cpu;
            continue;
        }
        push_cpulist_range(&mut parts, start, prev);
        start = cpu;
        prev = cpu;
    }
    push_cpulist_range(&mut parts, start, prev);
    parts.join(",")
}

fn push_cpulist_range(parts: &mut Vec<String>, start: usize, end: usize) {
    if start == end {
        parts.push(start.to_string());
    } else {
        parts.push(format!("{start}-{end}"));
    }
}

pub fn effective_cpu_affinity() -> EffectiveCpuAffinity {
    #[cfg(target_os = "linux")]
    {
        match get_affinity() {
            Ok(cpus) if !cpus.is_empty() => {
                let cpus = normalize_cpus(cpus);
                let display = format_cpulist(&cpus);
                let logical_cores = cpus.len().max(1);
                EffectiveCpuAffinity {
                    cpus: Some(cpus),
                    logical_cores,
                    display,
                }
            }
            Ok(_) | Err(_) => effective_cpu_affinity_unavailable(),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        effective_cpu_affinity_unavailable()
    }
}

fn effective_cpu_affinity_unavailable() -> EffectiveCpuAffinity {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    EffectiveCpuAffinity {
        cpus: None,
        logical_cores,
        display: "unavailable".to_string(),
    }
}

#[cfg(target_os = "linux")]
fn set_affinity(cpus: &[usize]) -> Result<(), String> {
    set_affinity_pid(0, cpus)
}

#[cfg(target_os = "linux")]
fn set_affinity_thread(cpus: &[usize]) -> Result<(), String> {
    // pid==0 means "current task" — which is the current thread when
    // called from a non-leader thread. For pinning the whole process
    // we use the same syscall from the main thread before spawning.
    set_affinity_pid(0, cpus)
}

#[cfg(target_os = "linux")]
fn set_affinity_pid(pid: libc::pid_t, cpus: &[usize]) -> Result<(), String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            // Guard against CPU_SETSIZE overflow; libc::CPU_SET is a macro
            // that does no bounds-checking on some libc versions.
            if c >= libc::CPU_SETSIZE as usize {
                return Err(format!("cpu {c} exceeds CPU_SETSIZE"));
            }
            libc::CPU_SET(c, &mut set);
        }
        let rc = libc::sched_setaffinity(pid, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            return Err(format!("{e}"));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn get_affinity() -> Result<Vec<usize>, String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            return Err(format!("{e}"));
        }
        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                cpus.push(cpu);
            }
        }
        Ok(cpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpulist_simple() {
        assert_eq!(parse_cpulist("0").unwrap(), vec![0]);
        assert_eq!(parse_cpulist("0,2,4").unwrap(), vec![0, 2, 4]);
        assert_eq!(parse_cpulist("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0-1,4-5").unwrap(), vec![0, 1, 4, 5]);
        assert_eq!(parse_cpulist("  ").unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn formats_cpulist_as_ranges() {
        assert_eq!(format_cpulist(&[0, 1, 2, 4, 7, 8]), "0-2,4,7-8");
        assert_eq!(format_cpulist(&[4, 2, 2, 3]), "2-4");
    }

    #[test]
    fn no_cpu_mask_means_no_startup_placement_request() {
        let request = resolve_cpu_mask_request(None, None, None).unwrap();
        assert_eq!(request, None);
    }

    #[test]
    fn cli_cpu_mask_overrides_config() {
        let request = resolve_cpu_mask_request(Some("0-1"), Some("4-5"), None)
            .unwrap()
            .expect("mask");
        assert_eq!(request.source, CpuMaskSource::Cli);
        assert_eq!(request.display, "0-1");
        assert_eq!(request.cpus, vec![0, 1]);
    }

    #[test]
    fn config_cpu_mask_overrides_legacy_mer_pin_cores() {
        let request = resolve_cpu_mask_request(None, Some("2-3"), Some("8"))
            .unwrap()
            .expect("mask");
        assert_eq!(request.source, CpuMaskSource::Config);
        assert_eq!(request.display, "2-3");
        assert_eq!(request.cpus, vec![2, 3]);
    }

    #[test]
    fn parses_cpulist_with_whitespace() {
        assert_eq!(
            parse_cpulist("0-1, 4 , 6-7\n").unwrap(),
            vec![0, 1, 4, 6, 7]
        );
    }

    #[test]
    fn rejects_descending_range() {
        assert!(parse_cpulist("4-1").is_err());
    }

    #[test]
    fn rejects_oversized_range_before_expansion() {
        let err = parse_cpulist("0-999999999").unwrap_err();
        assert!(err.contains("exceeding supported maximum"));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_cpulist("nope").is_err());
        assert!(parse_cpulist("0-x").is_err());
    }

    #[test]
    fn pin_result_log_line_is_descriptive() {
        assert!(PinResult::NotRequested
            .as_log_line()
            .contains(MER_PIN_CORES_ENV));
        assert!(PinResult::BadValue("xyz".into())
            .as_log_line()
            .contains("xyz"));
        assert!(PinResult::Pinned { cpus: vec![0, 1] }
            .as_log_line()
            .contains("[0, 1]"));
        assert!(PinResult::Unsupported("nope".into())
            .as_log_line()
            .contains("nope"));
    }

    #[test]
    fn apply_with_unset_env_is_not_requested() {
        assert_eq!(apply_mer_pin_cores_value(None), PinResult::NotRequested);
        assert_eq!(apply_mer_pin_cores_value(Some("")), PinResult::NotRequested);
    }

    #[test]
    fn apply_with_bad_env_reports_bad_value() {
        let r = apply_mer_pin_cores_value(Some("abc"));
        assert!(matches!(r, PinResult::BadValue(_)));
    }
}
