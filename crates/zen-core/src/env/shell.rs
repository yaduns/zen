use crate::env::ExecutionError;
use crate::env::PathBuf;
use std::path::Path;

/// Pure-data output of resolve_shell, describing which shell to use and with what arguments and transport to execute commands.
pub struct ShellConfig {
    /// Resolved shell executable path
    pub shell: PathBuf,
    /// Fixed arguments passed to the shell (unix defaults to ["-c"]; legacy WSL bash uses ["-s"])
    pub args: Vec<String>,
    /// Argv: command appended to the end of args; Stdin: command written through the stdin pipe
    pub transport: ShellTransport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellTransport {
    /// Pass the command via the `-c` argument (default).
    Argv,
    /// Pass the command through the stdin pipe (legacy WSL bash requires `-s`).
    Stdin,
}

/// Resolve the available shell configuration platform-independently.
///
/// Behavior (refer to pi-229 `getShellConfig` + `getBashShellConfig`):
/// - When `custom_shell` is provided: use it if it exists; legacy WSL path (`is_legacy_wsl_bash_path`)
///   uses `["-s"]` + `Stdin`, otherwise `["-c"]` + `Argv`; if it does not exist, return `ShellUnavailable`.
/// - unix: if `/bin/bash` exists → use it; otherwise `find_bash_on_path()` finds `bash`; then fall back to `sh -c` (Argv).
/// - windows: candidates `%ProgramFiles%\Git\bin\bash.exe`, `%ProgramFiles(x86)%\Git\bin\bash.exe`,
///   then `find_bash_on_path()` finds `bash.exe`; if neither is found, return `ShellUnavailable`.
///
/// Difference from pi: PATH search uses `std::env::split_paths` iteration + existence checks,
/// no longer spawns `which`/`where` subprocesses (removes Node specialization).
pub fn resolve_shell(custom_shell: Option<&Path>) -> Result<ShellConfig, ExecutionError> {
    // 1) When custom_shell is provided: use it if it exists, otherwise ShellUnavailable
    if let Some(shell) = custom_shell {
        if shell.is_file() {
            return Ok(bash_shell_config(shell.to_path_buf()));
        }
        return Err(ExecutionError::ShellUnavailable(format!(
            "custom shell path not found: {}",
            shell.display()
        )));
    }

    #[cfg(windows)]
    {
        // 2a) Two Git Bash candidates; use the first hit (matches pi's candidates loop)
        for env_key in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(pf) = std::env::var_os(env_key) {
                let candidate = PathBuf::from(pf).join("Git\\bin\\bash.exe");
                if candidate.is_file() {
                    return Ok(bash_shell_config(candidate));
                }
            }
        }
        // 2b) Find bash.exe in PATH
        if let Some(bash) = find_bash_on_path() {
            return Ok(bash_shell_config(bash));
        }
        // 2c) Neither found → ShellUnavailable
        return Err(ExecutionError::ShellUnavailable(
            "no bash shell found".into(),
        ));
    }

    #[cfg(not(windows))]
    {
        // 3a) Use /bin/bash if it exists
        if Path::new("/bin/bash").is_file() {
            return Ok(bash_shell_config(PathBuf::from("/bin/bash")));
        }
        // 3b) Find bash in PATH
        if let Some(bash) = find_bash_on_path() {
            return Ok(bash_shell_config(bash));
        }
        // 3c) Fall back to sh -c (Argv) — same as pi: unix never fails
        Ok(ShellConfig {
            shell: PathBuf::from("sh"),
            args: vec!["-c".into()],
            transport: ShellTransport::Argv,
        })
    }
}

fn bash_shell_config(shell: PathBuf) -> ShellConfig {
    if is_legacy_wsl_bash_path(&shell) {
        // ← here
        ShellConfig {
            shell,
            args: vec!["-s".into()],
            transport: ShellTransport::Stdin,
        }
    } else {
        ShellConfig {
            shell,
            args: vec!["-c".into()],
            transport: ShellTransport::Argv,
        }
    }
}

/// Find bash in PATH (unix looks for `bash`, windows looks for `bash.exe`), returning the first existing one; otherwise `None`.
fn find_bash_on_path() -> Option<PathBuf> {
    // Platform-specific executable file name
    let exe_name = if cfg!(windows) { "bash.exe" } else { "bash" };
    // Return None directly when PATH is unset (`?` short-circuit)
    let path_var = std::env::var_os("PATH")?;
    // Join each PATH entry with the candidate path, taking the first existing file
    // is_file() existence check
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(exe_name))
        .find(|candidate| candidate.is_file())
}

/// Determine whether a path is legacy WSL bash (`bash.exe` under `System32`/`Sysnative`),
/// Case-insensitive (matches pi `isLegacyWslBashPath`); when matched, requires `-s` + stdin transport.
fn is_legacy_wsl_bash_path(path: &Path) -> bool {
    // Normalize `/` to `\` and lowercase, so paths mixing case/separators also match
    let normalized = path.to_string_lossy().replace('/', "\\").to_lowercase();

    // Strip suffixes segment by segment, anchoring the regex's three fixed parts
    let Some(rest) = normalized.strip_suffix("\\bash.exe") else {
        return false;
    };
    let Some(rest) = rest
        .strip_suffix("\\system32")
        .or_else(|| rest.strip_suffix("\\sysnative"))
    else {
        return false;
    };
    let Some(rest) = rest.strip_suffix("\\windows") else {
        return false;
    };

    // What remains must be a single-letter drive + colon (corresponds to regex ^[a-z]:)
    matches!(rest.as_bytes(), [c, b':'] if c.is_ascii_lowercase())
}

/// Kill the child process tree rooted at `pid` (best-effort, no errors thrown; aligns with pi `killProcessTree`).
/// unix: send SIGKILL to process group `-pid` (precondition: child spawned with `process_group(0)`),
/// on failure, fall back to the single `pid`.
#[cfg(unix)]
pub(crate) fn kill_process_tree(pid: u32) {
    // kill the process group first (-pid); on failure (e.g. group does not exist) fall back to the single process
    let group_killed = unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) } == 0;
    if !group_killed {
        // best effort: ignore the return value, the process may have already exited
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }
}

/// Kill the child process tree rooted at `pid` (best-effort, no errors thrown; aligns with pi `killProcessTree`).
/// windows: spawn `taskkill /T /F /PID <pid>`; the spawn result is ignored.
#[cfg(windows)]
pub(crate) fn kill_process_tree(pid: u32) {
    use std::os::windows::process::CommandExt;
    let _ = std::process::Command::new("taskkill")
        .arg("/F")
        .arg("/T")
        .arg("/PID")
        .arg(pid.to_string())
        // CREATE_NO_WINDOW: no black console window, matching pi's windowsHide
        .creation_flags(0x08000000)
        // stdio: "ignore" → set all to null to avoid occupying pipes
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    // ── is_legacy_wsl_bash_path (pure string normalization logic) ──────────────

    #[test]
    fn test_is_legacy_wsl_bash_path_true_cases() {
        for p in [
            r"C:\Windows\System32\bash.exe",
            r"C:\windows\system32\bash.exe",
            r"c:\WINDOWS\SYSNATIVE\BASH.EXE",
            r"C:/Windows/System32/bash.exe", // normalize forward slashes to backslashes
            r"D:\Windows\Sysnative\bash.exe",
        ] {
            assert!(is_legacy_wsl_bash_path(Path::new(p)), "expected true: {p}");
        }
    }

    #[test]
    fn test_is_legacy_wsl_bash_path_false_cases() {
        for p in [
            r"C:\Windows\System32\WSL\bash.exe", // there is a suffix after \system32
            r"C:\Program Files\Git\bin\bash.exe",
            "/bin/bash", // does not end with \bash.exe
            r"C:\Windows\System32\bash", // no .exe
            r"CC:\Windows\System32\bash.exe", // drive letter is longer than one character
            r"\Windows\System32\bash.exe", // no drive letter
            "",
        ] {
            assert!(!is_legacy_wsl_bash_path(Path::new(p)), "expected false: {p}");
        }
    }

    // ── bash_shell_config (WSL detected → -s/Stdin, otherwise -c/Argv) ──────

    #[test]
    fn test_bash_shell_config_legacy_wsl_uses_stdin() {
        let shell = PathBuf::from(r"C:\Windows\System32\bash.exe");
        let cfg = bash_shell_config(shell.clone());
        assert_eq!(cfg.shell, shell);
        assert_eq!(cfg.args, vec!["-s".to_string()]);
        assert_eq!(cfg.transport, ShellTransport::Stdin);
    }

    #[test]
    fn test_bash_shell_config_normal_uses_argv() {
        let shell = PathBuf::from("/bin/bash");
        let cfg = bash_shell_config(shell.clone());
        assert_eq!(cfg.shell, shell);
        assert_eq!(cfg.args, vec!["-c".to_string()]);
        assert_eq!(cfg.transport, ShellTransport::Argv);
    }

    // ── resolve_shell ─────────────────────────────────────────────

    #[test]
    fn test_resolve_shell_custom_shell_not_found() {
        let res = resolve_shell(Some(Path::new("/nonexistent/shell")));
        assert!(matches!(res, Err(ExecutionError::ShellUnavailable(_))));
    }

    #[test]
    fn test_resolve_shell_custom_shell_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shell = dir.path().join("myshell");
        std::fs::write(&shell, "#!/bin/sh\n").expect("write fake shell");
        let cfg = resolve_shell(Some(&shell)).expect("resolve shell");
        assert_eq!(cfg.shell, shell);
        assert_eq!(cfg.args, vec!["-c".to_string()]);
        assert_eq!(cfg.transport, ShellTransport::Argv);
    }

    #[cfg(unix)]
    #[test]
    fn test_resolve_shell_unix_bin_bash_preferred() {
        // on this machine (macOS/Linux) /bin/bash usually exists → should hit directly;
        // if absent, fall back to PATH lookup or sh; should still succeed (unix never fails)
        let cfg = resolve_shell(None).expect("resolve shell");
        if Path::new("/bin/bash").is_file() {
            assert_eq!(cfg.shell, PathBuf::from("/bin/bash"));
        }
        assert!(matches!(cfg.transport, ShellTransport::Argv));
    }

    // ── find_bash_on_path (reads the process PATH; kept in a single test to avoid parallel races) ─

    #[test]
    fn test_find_bash_on_path_scenarios() {
        let old_path = std::env::var_os("PATH");
        let set_path = |p: Option<&Path>| match p {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        };
        let restore = || match &old_path {
            Some(p) => std::env::set_var("PATH", p),
            None => std::env::remove_var("PATH"),
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let exe = if cfg!(windows) { "bash.exe" } else { "bash" };
        let bash_path = dir.path().join(exe);
        std::fs::write(&bash_path, "#!/bin/sh\n").expect("write fake bash");

        // case 1: PATH points to a directory containing bash → found
        set_path(Some(dir.path()));
        assert_eq!(find_bash_on_path(), Some(bash_path.clone()));
        restore();

        // case 2: PATH points to an empty directory → None
        let empty = tempfile::tempdir().expect("tempdir");
        set_path(Some(empty.path()));
        assert!(find_bash_on_path().is_none());
        restore();

        // case 3: PATH is not set → None
        set_path(None);
        assert!(find_bash_on_path().is_none());
        restore();
    }
}
