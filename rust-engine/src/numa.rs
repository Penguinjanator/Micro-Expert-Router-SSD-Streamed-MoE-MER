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
//! * [`apply_startup_cpu_mask`] honours an explicit kernel cpulist such
//!   as `0-24` at startup and calls `sched_setaffinity(2)` before
//!   tokio/Rayon worker threads are created. If no explicit mask is
//!   supplied, [`apply_mer_pin_cores_env`] keeps the legacy
//!   `MER_PIN_CORES=N` behaviour and pins **the whole process** to the
//!   first `N` CPUs of NUMA node 0. Best-effort and Linux-only:
//!   anywhere else this is a logged no-op so dev machines (macOS,
//!   Windows) still boot.
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

/// Environment variable that, if set to a positive integer `N`, pins
/// the process to the first `N` CPUs of NUMA node 0 at startup.
pub const MER_PIN_CORES_ENV: &str = "MER_PIN_CORES";

/// Outcome of an `apply_mer_pin_cores_env` call. Public so the caller
/// can log a single human-readable line at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinResult {
    /// No explicit mask or legacy `MER_PIN_CORES` value was provided.
    NotRequested,
    /// A requested CPU mask or legacy `MER_PIN_CORES` value was invalid.
    BadValue(String),
    /// Linux only: pinned to the listed CPUs (already de-duplicated).
    Pinned { cpus: Vec<usize> },
    /// Pinning was requested but the kernel / OS does not support the
    /// primitive (non-Linux, or `sched_setaffinity` returned an error).
    /// Carries a human-readable reason.
    Unsupported(String),
}

impl PinResult {
    pub fn as_log_line(&self) -> String {
        match self {
            PinResult::NotRequested => "no CPU affinity pinning requested".to_string(),
            PinResult::BadValue(s) => format!("CPU affinity value \"{s}\" invalid, ignored"),
            PinResult::Pinned { cpus } => {
                format!("pinned process to CPUs {:?}", cpus)
            }
            PinResult::Unsupported(why) => {
                format!("CPU affinity pinning unsupported on this platform: {why}")
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
    let raw = match env::var(MER_PIN_CORES_ENV) {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return PinResult::NotRequested,
    };
    let n: i64 = match raw.trim().parse() {
        Ok(v) => v,
        Err(_) => return PinResult::BadValue(raw),
    };
    if n <= 0 {
        return PinResult::BadValue(raw);
    }
    let n = n as usize;
    pin_first_n_to_node0(n)
}

/// Apply an explicit kernel cpulist such as `"0-24"` or `"0-7,16-23"`.
///
/// When `requested` is `None`, this falls back to the legacy
/// [`MER_PIN_CORES_ENV`] behaviour. The explicit mask path is the
/// first-class production placement knob; it runs at process startup so
/// Tokio and Rayon workers inherit the selected affinity.
pub fn apply_startup_cpu_mask(requested: Option<&str>) -> PinResult {
    let raw = match requested {
        Some(s) if !s.trim().is_empty() => s.trim(),
        Some(s) => return PinResult::BadValue(s.to_string()),
        None => return apply_mer_pin_cores_env(),
    };
    let mut cpus = match parse_cpulist(raw) {
        Ok(cpus) => cpus,
        Err(_) => return PinResult::BadValue(raw.to_string()),
    };
    cpus.sort_unstable();
    cpus.dedup();
    if cpus.is_empty() {
        return PinResult::BadValue(raw.to_string());
    }
    pin_cpu_set(&cpus)
}

/// Pin the calling process to the first `n` CPUs of NUMA node 0.
/// On non-Linux this is a logged no-op.
#[cfg(target_os = "linux")]
pub fn pin_first_n_to_node0(n: usize) -> PinResult {
    let mut cpus = node0_cpus().unwrap_or_else(|_| {
        // Fall back to all online CPUs in ascending order if sysfs is
        // unavailable (containers, exotic kernels). The "first N"
        // ordering still matches the user's intent.
        let max = num_cpus_online();
        (0..max).collect()
    });
    cpus.sort_unstable();
    cpus.dedup();
    cpus.truncate(n);
    if cpus.is_empty() {
        return PinResult::Unsupported(
            "no CPUs reported for NUMA node 0 and /proc/cpuinfo empty".into(),
        );
    }
    pin_cpu_set(&cpus)
}

#[cfg(not(target_os = "linux"))]
pub fn pin_first_n_to_node0(_n: usize) -> PinResult {
    PinResult::Unsupported("sched_setaffinity(2) is Linux-only".into())
}

#[cfg(target_os = "linux")]
pub fn pin_cpu_set(cpus: &[usize]) -> PinResult {
    match set_affinity(cpus) {
        Ok(()) => PinResult::Pinned {
            cpus: cpus.to_vec(),
        },
        Err(e) => PinResult::Unsupported(format!("sched_setaffinity: {e}")),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_cpu_set(_cpus: &[usize]) -> PinResult {
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
    if n > 0 { n as usize } else { 1 }
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
            for c in a..=b {
                out.push(c);
            }
        } else {
            let c: usize = part
                .parse()
                .map_err(|_| format!("bad cpulist cpu: {part:?}"))?;
            out.push(c);
        }
    }
    Ok(out)
}

pub fn format_cpulist(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    let mut cpus = cpus.to_vec();
    cpus.sort_unstable();
    cpus.dedup();
    let mut ranges = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &cpu in cpus.iter().skip(1) {
        if cpu == prev + 1 {
            prev = cpu;
            continue;
        }
        ranges.push(format_cpu_range(start, prev));
        start = cpu;
        prev = cpu;
    }
    ranges.push(format_cpu_range(start, prev));
    ranges.join(",")
}

fn format_cpu_range(start: usize, end: usize) -> String {
    if start == end {
        start.to_string()
    } else {
        format!("{start}-{end}")
    }
}

#[cfg(target_os = "linux")]
pub fn current_affinity() -> Result<Vec<usize>, String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if rc != 0 {
            return Err(std::io::Error::last_os_error().to_string());
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

#[cfg(not(target_os = "linux"))]
pub fn current_affinity() -> Result<Vec<usize>, String> {
    Err("sched_getaffinity(2) is Linux-only".into())
}

pub fn current_affinity_cpulist() -> Option<String> {
    current_affinity().ok().map(|cpus| format_cpulist(&cpus))
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
    fn parses_cpulist_with_whitespace() {
        assert_eq!(parse_cpulist("0-1, 4 , 6-7\n").unwrap(), vec![0, 1, 4, 6, 7]);
    }

    #[test]
    fn formats_cpulist_compactly() {
        assert_eq!(format_cpulist(&[0, 1, 2, 4, 8, 7, 7]), "0-2,4,7-8");
        assert_eq!(format_cpulist(&[]), "");
    }

    #[test]
    fn rejects_descending_range() {
        assert!(parse_cpulist("4-1").is_err());
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
            .contains("no CPU affinity pinning requested"));
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
        // SAFETY: tests in this module are single-threaded relative to
        // this env var. We deliberately remove it before calling.
        // SAFETY (Rust 2024 set/remove_var are unsafe due to multi-thread races): single-thread tests.
        unsafe { env::remove_var(MER_PIN_CORES_ENV); }
        assert_eq!(apply_mer_pin_cores_env(), PinResult::NotRequested);
    }

    #[test]
    fn apply_with_bad_env_reports_bad_value() {
        unsafe { env::set_var(MER_PIN_CORES_ENV, "abc"); }
        let r = apply_mer_pin_cores_env();
        unsafe { env::remove_var(MER_PIN_CORES_ENV); }
        assert!(matches!(r, PinResult::BadValue(_)));
    }
}
