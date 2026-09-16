// REQ-APP-08 (pi-361): data migration at startup
// See reference/pi-original/packages/coding-agent/src/migrations.ts
// Implementation notes in tasks/T-AP-08-7-implemented.md
// v1 keeps 3 of 5 sub-migrations (session relocation / commands→prompts / key-name upgrade); reviving auth→auth.json belongs to REQ-V2-02,
// reviving tools→bin and the extension-deprecation warning belongs to REQ-V2-01 (see "Ecosystem reuse verification conclusions" in the task doc).

use crate::config::{get_agent_dir, CONFIG_DIR_NAME};
use crate::keybindings::migrate_keybindings_config;
use crate::session::jsonl::encode_cwd;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// One-shot data migration run at zen startup (called by main.rs after mode determination and before session selection) — brings legacy
/// sessions stray in the root directory, old `commands/` directories, and old key-name keybinding configs into zen's current layout, so existing users after upgrading
/// don't have their data silently dropped. Each migration tolerates failure independently (missing directories / bad files are skipped and logged via tracing),
/// and never interrupts startup (returns () with no error — v1 has no consumer needing migration results; progress info goes to tracing/stderr:
/// in print mode stdout is taken over by structured output, so println is forbidden). The `cwd` parameter locates the project-level `<cwd>/.zen` directory.
/// Consumer: `main.rs` (wired up in this task).
pub fn run_migrations(cwd: &Path) {
    let Some(agent_dir) = get_agent_dir().ok() else {
        // Extreme environments where home is unavailable: skip all migrations (matches pi sub-migrations returning silently when reading the directory fails)
        warn!("[migrations] agent dir unavailable, skipping startup migrations");
        return;
    };
    run_migrations_with_agent_dir(cwd, &agent_dir);
}

/// Actual body of `run_migrations` — `agent_dir` is injected explicitly (test seam: unit tests pass a temp directory directly, avoiding env vars).
/// Migration order matches pi `runMigrations` (session relocation → commands→prompts (global + project) → key-name upgrade);
/// sub-migrations are independent; any failure doesn't affect the others.
fn run_migrations_with_agent_dir(cwd: &Path, agent_dir: &Path) {
    let moved = migrate_sessions_from_agent_root(agent_dir);
    if moved > 0 {
        info!(moved = %moved, "[migrations] relocated stray session files under agent root");
    }

    migrate_commands_to_prompts(agent_dir, "Global");
    let project_dir = cwd.join(CONFIG_DIR_NAME);
    migrate_commands_to_prompts(&project_dir, "Project");

    migrate_keybindings_config_file(agent_dir);
}

/// Migrate sessions left stray in the root directory by the v0.30.0 layout bug — move `.jsonl` files directly under `<agent_dir>`
/// (which should be in `sessions/<encoded-cwd>/` subdirectories) into the correct subdirectory based on the cwd in the first-line session header, so the session selector
/// can find them again. The first line uses a lenient check (`type == "session"` with a non-empty `cwd` field), and **must not** reuse
/// the strict version==3 validation of `jsonl::parse_header_line` — the migration targets are precisely old-version files; directory encoding reuses
/// `session::jsonl::encode_cwd` (same source as SessionManager's current write layout). When a file with the same name already exists at the target,
/// skip it (keep the original file); .jsonl files whose root is unreadable, whose first line is not JSON, or that have no cwd are all silently skipped.
/// Returns the number of files moved successfully (for test assertions and tracing logs).
fn migrate_sessions_from_agent_root(agent_dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(agent_dir) else {
        return 0; // Root directory missing/unreadable → nothing to migrate (pi readdirSync catch)
    };
    let mut moved = 0;
    for entry in entries.flatten() {
        let file = entry.path();
        let Some(ext) = file.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext != "jsonl" || !file.is_file() {
            continue;
        }
        // Read only the first line (avoid pi readFileSync reading large session files in full)
        let first_line = match read_first_line(&file) {
            Some(line) => line,
            None => continue,
        };
        let line = first_line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(header) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // Lenient check: type == "session" and cwd non-empty (pi `!header.cwd` truthiness semantics)
        if header.get("type").and_then(|v| v.as_str()) != Some("session") {
            continue;
        }
        let cwd = match header.get("cwd").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };
        let correct_dir = agent_dir.join("sessions").join(encode_cwd(cwd));
        let Some(file_name) = file.file_name() else {
            continue;
        };
        let new_path = correct_dir.join(file_name);
        if new_path.exists() {
            continue; // Target already exists → keep the original file (same semantics as pi)
        }
        if std::fs::create_dir_all(&correct_dir).is_err() {
            continue;
        }
        match std::fs::rename(&file, &new_path) {
            Ok(()) => moved += 1,
            Err(err) => warn!(
                error = %err,
                path = %file.display(),
                "[migrations] failed to relocate stray session file"
            ),
        }
    }
    moved
}

/// Reads the file's first line (including trailing newline); returns None on read failure.
fn read_first_line(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    Some(line)
}

/// Renames old `commands/` directories (global `<agent_dir>/commands` and project `<cwd>/.zen/commands`) to
/// `prompts/` — the prompt template loader only reads the prompts directory (`prompt_templates.rs` in T-AP-03-4),
/// so without migration user templates silently stop working. Rename only when commands exists and prompts does **not**
/// (on name conflict prompts wins and commands is left untouched); symlinks work too (`fs::rename` moves the link itself).
/// `label` ("Global"/"Project") is only used in tracing log text (matching pi's `Migrated {label} commands/ → prompts/`).
/// Returns whether the rename was performed.
fn migrate_commands_to_prompts(base_dir: &Path, label: &str) -> bool {
    let commands_dir = base_dir.join("commands");
    let prompts_dir = base_dir.join("prompts");
    if !commands_dir.exists() || prompts_dir.exists() {
        return false;
    }
    match std::fs::rename(&commands_dir, &prompts_dir) {
        Ok(()) => {
            info!(label, "[migrations] migrated {label} commands/ → prompts/");
            true
        }
        Err(err) => {
            // Matches pi's yellow warning (chalk.yellow → tracing warn/stderr)
            warn!(
                error = %err,
                label,
                "[migrations] could not migrate {label} commands/ to prompts/"
            );
            false
        }
    }
}

/// Upgrades legacy pi keybinding config key names to namespaced canonical names — keys like `cursorUp` in `<agent_dir>/keybindings.json`
/// would be silently dropped by `load_user_keybindings` in zen as unknown actions; this function
/// rewrites them to `tui.editor.cursorUp` so users' old configs keep working. File IO shell: read JSON → delegate to
/// the `keybindings::migrate_keybindings_config` pure function → if changed, atomically write back pretty + trailing newline
/// (matching pi `JSON.stringify(config, null, 2) + "\n"`; temp file + rename follows
/// the `config/settings.rs::write_atomic` precedent); missing file / bad JSON / non-object is silently skipped.
/// Returns whether the file was written back.
fn migrate_keybindings_config_file(agent_dir: &Path) -> bool {
    let config_path = agent_dir.join("keybindings.json");
    if !config_path.exists() {
        return false;
    }
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(_) => return false,
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false; // Bad JSON silently skipped (pi catch semantics)
    };
    let Some(migrated) = migrate_keybindings_config(&parsed) else {
        return false; // No legacy key names, no rewrite
    };
    let rewritten = match serde_json::to_string_pretty(&migrated) {
        Ok(s) => format!("{s}\n"),
        Err(_) => return false,
    };
    // Atomic write: temp file in same directory + rename replace (process crash won't corrupt existing file)
    let parent = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut tmp = match tempfile::NamedTempFile::new_in(&parent) {
        Ok(tmp) => tmp,
        Err(_) => return false,
    };
    if tmp.write_all(rewritten.as_bytes()).is_err() {
        return false;
    }
    match tmp.persist(&config_path) {
        Ok(_) => {
            info!(
                path = %config_path.display(),
                "[migrations] migrated legacy keybinding names in keybindings.json"
            );
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    fn session_header_json(cwd: &str) -> String {
        format!(r#"{{"type":"session","cwd":"{cwd}","version":3,"id":"s1"}}"#)
    }

    // ── migrate_sessions_from_agent_root ─────────────────────────

    #[test]
    fn moves_stray_session_file_into_cwd_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(
            &agent.join("stray.jsonl"),
            &session_header_json("/home/user/project"),
        );
        assert_eq!(migrate_sessions_from_agent_root(agent), 1);
        let correct_dir = agent
            .join("sessions")
            .join(encode_cwd("/home/user/project"));
        assert!(correct_dir.join("stray.jsonl").exists());
        assert!(!agent.join("stray.jsonl").exists(), "original file should be moved away");
    }

    #[test]
    fn moves_multiple_files_into_separate_cwd_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(&agent.join("a.jsonl"), &session_header_json("/p/one"));
        write(&agent.join("b.jsonl"), &session_header_json("/p/two"));
        assert_eq!(migrate_sessions_from_agent_root(agent), 2);
        assert!(agent
            .join("sessions")
            .join(encode_cwd("/p/one"))
            .join("a.jsonl")
            .exists());
        assert!(agent
            .join("sessions")
            .join(encode_cwd("/p/two"))
            .join("b.jsonl")
            .exists());
    }

    #[test]
    fn skips_non_session_and_invalid_files() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(&agent.join("a.jsonl"), r#"{"type":"entry","cwd":"/x"}"#); // wrong type
        write(&agent.join("b.jsonl"), r#"{"type":"session"}"#); // no cwd
        write(&agent.join("c.jsonl"), "not json"); // bad JSON
        write(&agent.join("d.jsonl"), ""); // empty first line
        write(&agent.join("e.txt"), &session_header_json("/x")); // non-jsonl extension
        write(&agent.join("f.jsonl"), &session_header_json("")); // empty cwd
        assert_eq!(migrate_sessions_from_agent_root(agent), 0);
    }

    #[test]
    fn skips_when_target_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(&agent.join("stray.jsonl"), &session_header_json("/p"));
        let target_dir = agent.join("sessions").join(encode_cwd("/p"));
        std::fs::create_dir_all(&target_dir).unwrap();
        write(&target_dir.join("stray.jsonl"), "existing");
        assert_eq!(migrate_sessions_from_agent_root(agent), 0);
        assert!(agent.join("stray.jsonl").exists(), "original file should be preserved");
    }

    #[test]
    fn missing_agent_dir_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            migrate_sessions_from_agent_root(&dir.path().join("absent")),
            0
        );
    }

    // ── migrate_commands_to_prompts ──────────────────────────────

    #[test]
    fn renames_commands_to_prompts_when_prompts_absent() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        std::fs::create_dir_all(base.join("commands")).unwrap();
        assert!(migrate_commands_to_prompts(base, "Global"));
        assert!(base.join("prompts").exists());
        assert!(!base.join("commands").exists());
    }

    #[test]
    fn keeps_commands_when_prompts_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        std::fs::create_dir_all(base.join("commands")).unwrap();
        std::fs::create_dir_all(base.join("prompts")).unwrap();
        assert!(!migrate_commands_to_prompts(base, "Project"));
        assert!(base.join("commands").exists());
        assert!(base.join("prompts").exists());
    }

    #[test]
    fn no_commands_dir_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!migrate_commands_to_prompts(dir.path(), "Global"));
    }

    // ── migrate_keybindings_config_file ──────────────────────────

    #[test]
    fn rewrites_file_with_canonical_names_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(
            &agent.join("keybindings.json"),
            r#"{ "cursorUp": "ctrl+k" }"#,
        );
        assert!(migrate_keybindings_config_file(agent));
        let content = std::fs::read_to_string(agent.join("keybindings.json")).unwrap();
        let obj = serde_json::from_str::<serde_json::Value>(&content)
            .unwrap()
            .as_object()
            .unwrap()
            .clone();
        assert!(obj.contains_key("tui.editor.cursorUp"));
        assert!(!obj.contains_key("cursorUp"), "Old key name should be rewritten");
        // Idempotent: second call no longer rewrites
        assert!(!migrate_keybindings_config_file(agent));
    }

    #[test]
    fn no_legacy_names_no_write() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        write(
            &agent.join("keybindings.json"),
            r#"{ "app.interrupt": "esc" }"#,
        );
        assert!(!migrate_keybindings_config_file(agent));
        let content = std::fs::read_to_string(agent.join("keybindings.json")).unwrap();
        assert!(content.contains("app.interrupt"));
    }

    #[test]
    fn missing_file_or_bad_json_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        assert!(!migrate_keybindings_config_file(agent));
        write(&agent.join("keybindings.json"), "{not json");
        assert!(!migrate_keybindings_config_file(agent));
    }

    // ── run_migrations (via explicit agent_dir test seam) ──────────────────

    #[test]
    fn run_migrations_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        let cwd = dir.path().join("project");
        // Three kinds of legacy data: stray root-directory sessions / global and project commands / old key-name configs
        write(
            &agent.join("stray.jsonl"),
            &session_header_json("/work/proj"),
        );
        std::fs::create_dir_all(agent.join("commands")).unwrap();
        write(
            &agent.join("keybindings.json"),
            r#"{ "cursorUp": "ctrl+k" }"#,
        );
        std::fs::create_dir_all(cwd.join(".zen").join("commands")).unwrap();

        run_migrations_with_agent_dir(&cwd, agent);

        // Relocate sessions to sessions/<encoded>/
        let correct_dir = agent.join("sessions").join(encode_cwd("/work/proj"));
        assert!(correct_dir.join("stray.jsonl").exists());
        // Global + project commands → prompts
        assert!(agent.join("prompts").exists());
        assert!(!agent.join("commands").exists());
        assert!(cwd.join(".zen").join("prompts").exists());
        assert!(!cwd.join(".zen").join("commands").exists());
        // Key-name upgrade
        let kb = std::fs::read_to_string(agent.join("keybindings.json")).unwrap();
        assert!(kb.contains("tui.editor.cursorUp"));
    }
}
