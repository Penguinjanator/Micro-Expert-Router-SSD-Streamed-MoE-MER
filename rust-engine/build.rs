use std::path::Path;
use std::process::Command;

fn git_output(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .expect("CARGO_MANIFEST_DIR is set by Cargo");
    let repo = manifest_dir.parent().unwrap_or(&manifest_dir);

    let sha = git_output(repo, &["rev-parse", "HEAD"])
        .filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
    let dirty = git_output(repo, &["status", "--porcelain", "--untracked-files=no"])
        .map(|status| !status.is_empty());

    println!(
        "cargo:rustc-env=MER_BUILD_GIT_SHA={}",
        sha.as_deref().unwrap_or("unavailable")
    );
    println!(
        "cargo:rustc-env=MER_BUILD_GIT_DIRTY={}",
        dirty
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    );

    // A commit or staged/unstaged tracked-source change must invalidate the
    // embedded provenance. These paths work for the normal checkout used by
    // this repository; Cargo still has explicit unavailable semantics if Git
    // metadata cannot be read.
    println!(
        "cargo:rerun-if-changed={}",
        repo.join(".git/HEAD").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo.join(".git/index").display()
    );
    // Once any `rerun-if-changed` directive is emitted Cargo stops using its
    // broad package-directory fallback. Watch every tracked path so a dirty
    // tracked file cannot reuse provenance captured by an earlier clean build.
    if let Some(tracked) = git_output(repo, &["ls-files"]) {
        for path in tracked.lines().filter(|path| !path.is_empty()) {
            println!("cargo:rerun-if-changed={}", repo.join(path).display());
        }
    }
}
