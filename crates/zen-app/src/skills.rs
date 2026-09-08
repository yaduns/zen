//! Skill loading assembly layer: multi-directory assembly + source provenance + dedup collisions + metadata validation.
//!
//! REQ-APP-03 (pi-309): skill loading (directory discovery + frontmatter).
//! Application-layer assembly completion form: in-directory discovery / single-file frontmatter / formatting are already covered by zen-core
//! (T-AG-05-2 `load_skills`/`build_skill`, T-AP-03-5 frontmatter); this module supplies the remaining pi-309
//! capabilities — default directories (`<agentDir>/skills` + `<cwd>/.zen/skills`) + explicit path assembly,
//! source provenance (lightweight `SkillSource`, full SourceInfo belongs to T-AP-05-4), name/realpath dedup and
//! collision diagnostics, validateName/validateDescription length and character warning validation.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use zen_core::paths::{canonicalize_path, resolve_path, PathInputOptions};
use zen_core::skills::load::{load_skill_from_file, load_skills as core_load_skills};
use zen_core::skills::skill::Skill;

use crate::config::CONFIG_DIR_NAME;

/// Maximum skill name length (pi `MAX_NAME_LENGTH`).
const MAX_NAME_LENGTH: usize = 64;
/// Maximum skill description length (pi `MAX_DESCRIPTION_LENGTH`).
const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// Lightweight provenance tag for skill source — resolution of the scope semantics of pi `createSkillSourceInfo`.
///
/// v1 has no extension system; pi's `source` is always `"local"` and `origin` is always `"top-level"`, so the only real semantics left is the source level
/// (user/project/temporary path); `baseDir` is derived by the consumer from the parent directory of `skill.file_path`, not a field.
/// The full `SourceInfo` resource model (scope/origin/baseDir) is spec-assigned to **T-AP-05-4** (pi-319),
/// Then T-AP-05-3 resource loader will map `SkillSource → SourceInfo` uniformly. Pure in-memory assembly product, not serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillSource {
    /// Skills under the default directory `<agentDir>/skills` (pi scope `"user"`).
    User,
    /// Skills under the default directory `<cwd>/.zen/skills` (pi scope `"project"`).
    Project,
    /// Skills loaded from explicit paths (CLI `--skill` / settings.skills extension directories) (pi scope `"temporary"`).
    Path,
}

/// A loaded skill + its source — the zen assembly form of pi `Skill` (including sourceInfo/baseDir):
/// zen-core `Skill` (pure domain data, finalized in T-AG-05-2) + application-layer source provenance;
/// `baseDir` is derived by the consumer from the parent directory of `skill.file_path`, not a field.
#[derive(Debug)]
pub struct LoadedSkill {
    /// zen-core loaded skill (name/description/content/file_path/disable_model_invocation).
    pub skill: Skill,
    /// Skill source (default directory user/project / explicit path).
    pub source: SkillSource,
}

/// Payload for same-name skill collision diagnostics — the skill-specific form of pi `ResourceCollision`, with `resourceType` fixed to `"skill"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCollision {
    /// The colliding skill name.
    pub name: String,
    /// Path of the first loader (name first occurrence kept).
    pub winner_path: String,
    /// Path of the later loader (discarded).
    pub loser_path: String,
}

/// Non-fatal skill loading diagnostic — pi `ResourceDiagnostic`'s `type` string discriminant resolved into a Rust enum.
#[derive(Debug)]
pub enum SkillLoadDiagnostic {
    /// Validation/path warning (the skill may still be loaded; pi type `"warning"`).
    Warning {
        /// Human-readable message.
        message: String,
        /// Associated path.
        path: String,
    },
    /// Same-name collision (later loader is discarded; pi type `"collision"`).
    Collision(SkillCollision),
}

/// Return aggregate of `load_skills` — "warnings don't abort" semantics: some skill failures don't affect the rest, diagnostics and skills may both be non-empty.
#[derive(Debug, Default)]
pub struct LoadSkillsResult {
    /// Successfully loaded and deduplicated skills (name first-occurrence order).
    pub skills: Vec<LoadedSkill>,
    /// Load warnings and collision diagnostics.
    pub diagnostics: Vec<SkillLoadDiagnostic>,
}

/// Assembly input for `load_skills` (pi `LoadSkillsOptions`). Pure input parameters, no behavior.
pub struct LoadSkillsOptions {
    /// Session working directory (base for the project skills directory `<cwd>/.zen/skills`).
    pub cwd: PathBuf,
    /// Global agent config directory (base for `<agentDir>/skills`; produced by T-AP-02-1 `get_agent_dir`).
    pub agent_dir: PathBuf,
    /// Explicit skill paths (file or directory); CLI `--skill` + settings.skills are merged and passed in by the caller (T-AP-05-3/assembly chain).
    pub skill_paths: Vec<PathBuf>,
    /// When true, loads the default directories `<agentDir>/skills` (User) + `<cwd>/.zen/skills` (Project).
    pub include_defaults: bool,
}

/// Assembles the skill list from all configuration locations — system prompt assembly (T-AP-03-1's `BuildSystemPromptOptions.skills`),
/// the resource loader (T-AP-05-3) and slash commands (T-AP-03-2) share this source: it resolves "default directories + explicit paths" in one pass
/// into a deduplicated skill set, sparing each consumer from repeating directory discovery and collision handling.
///
/// Behavior (aligned with pi loadSkills): when `include_defaults`, first load `<agentDir>/skills` (source=User) and
/// `<cwd>/.zen/skills` (source=Project), then process `skill_paths` in order (file or directory, source=Path;
/// non-existent paths record a Warning diagnostic and continue, non-`.md` files record a Warning); directory discovery is delegated to zen-core
/// `load_skills` (SKILL.md only); a single `.md` file goes through zen-core `load_skill_from_file`.
/// Deduplication: same name keeps the first occurrence, later ones record a Collision diagnostic (`SkillCollision`); the same file appearing again via symlink
/// is detected via `paths::canonicalize_path` and silently skipped (aligned with pi realPathSet).
/// Loaded skills additionally run `validate_skill_metadata` (name length/character rules, description length; warnings do not abort).
///
/// # Parameters
/// - `options`: assembly input (cwd/agent_dir/skill_paths/include_defaults).
///
/// # Returns
/// Deduplicated `LoadedSkill` list + non-fatal diagnostics.
pub async fn load_skills(options: &LoadSkillsOptions) -> LoadSkillsResult {
    let mut result = LoadSkillsResult::default();
    let mut seen_real_paths: HashSet<PathBuf> = HashSet::new();

    if options.include_defaults {
        // Default directory ①: <agentDir>/skills (User)
        let user_dir = options.agent_dir.join("skills");
        let (loaded, diags) = load_skills_from_dir(&user_dir, SkillSource::User).await;
        result.diagnostics.extend(diags);
        add_skills(loaded, &mut result, &mut seen_real_paths);

        // Default directory ②: <cwd>/.zen/skills (Project)
        let project_dir = options.cwd.join(CONFIG_DIR_NAME).join("skills");
        let (loaded, diags) = load_skills_from_dir(&project_dir, SkillSource::Project).await;
        result.diagnostics.extend(diags);
        add_skills(loaded, &mut result, &mut seen_real_paths);
    }

    // Explicit paths (source is always Path, aligned with pi includeDefaults=true semantics)
    for raw_path in &options.skill_paths {
        let resolved = resolve_path(
            &raw_path.to_string_lossy(),
            &options.cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        );
        let resolved = match resolved {
            Ok(p) => p,
            Err(_) => continue, // Path resolution failure (no home directory, etc.) is silently skipped
        };

        // Existence check: missing → Warning diagnostic, continue
        let metadata = match tokio::fs::metadata(&resolved).await {
            Ok(m) => m,
            Err(_) => {
                result.diagnostics.push(SkillLoadDiagnostic::Warning {
                    message: "skill path does not exist".to_string(),
                    path: resolved.to_string_lossy().into_owned(),
                });
                continue;
            }
        };

        if metadata.is_dir() {
            let (loaded, diags) = load_skills_from_dir(&resolved, SkillSource::Path).await;
            result.diagnostics.extend(diags);
            add_skills(loaded, &mut result, &mut seen_real_paths);
        } else if metadata.is_file() && resolved.extension().is_some_and(|e| e == "md") {
            // Single .md file skill (aligned with pi loadSkillFromFile branch)
            let (skill, diags) = load_skill_from_file(&resolved).await;
            for d in diags {
                result.diagnostics.push(SkillLoadDiagnostic::Warning {
                    message: d.message,
                    path: d.path,
                });
            }
            if let Some(skill) = skill {
                add_skills(
                    vec![LoadedSkill {
                        skill,
                        source: SkillSource::Path,
                    }],
                    &mut result,
                    &mut seen_real_paths,
                );
            }
        } else {
            result.diagnostics.push(SkillLoadDiagnostic::Warning {
                message: "skill path is not a markdown file".to_string(),
                path: resolved.to_string_lossy().into_owned(),
            });
        }
    }

    // Metadata validation warnings (name/description length and character rules; warnings do not abort)
    for loaded in &result.skills {
        result
            .diagnostics
            .extend(validate_skill_metadata(&loaded.skill));
    }

    result
}

/// Loads skills from a single directory and attaches source — delegates to zen-core `load_skills` (directory traversal + SKILL.md loading,
/// T-AG-05-2), converting core diagnostics (`SkillDiagnostic`) into application-level `SkillLoadDiagnostic::Warning`.
/// Only maps "directory → skill with source"; no deduplication (handled centrally by the caller's `add_skills`).
async fn load_skills_from_dir(
    dir: &Path,
    source: SkillSource,
) -> (Vec<LoadedSkill>, Vec<SkillLoadDiagnostic>) {
    let core = core_load_skills(&[dir.to_path_buf()]).await;
    let skills = core
        .skills
        .into_iter()
        .map(|skill| LoadedSkill { skill, source })
        .collect();
    let diagnostics = core
        .diagnostics
        .into_iter()
        .map(|d| SkillLoadDiagnostic::Warning {
            message: d.message,
            path: d.path,
        })
        .collect();
    (skills, diagnostics)
}

/// Merges a batch of sourced skills into the result — two-level deduplication (aligned with pi addSkills): ① the same real file via symlink
/// appearing again → silently skipped; ② different files with the same name → Collision diagnostic, later one discarded. First occurrence of a name wins.
fn add_skills(
    loaded: Vec<LoadedSkill>,
    result: &mut LoadSkillsResult,
    seen_real_paths: &mut HashSet<PathBuf>,
) {
    for loaded in loaded {
        // ① realpath deduplication (pi realPathSet): on canonicalize failure, fall back to the original path
        let real_path = canonicalize_path(Path::new(&loaded.skill.file_path));
        if seen_real_paths.contains(&real_path) {
            continue;
        }
        // ② name deduplication (pi skillMap): second occurrence of a name → collision, later one discarded
        if let Some(existing) = result
            .skills
            .iter()
            .find(|s| s.skill.name == loaded.skill.name)
        {
            result
                .diagnostics
                .push(SkillLoadDiagnostic::Collision(SkillCollision {
                    name: loaded.skill.name.clone(),
                    winner_path: existing.skill.file_path.clone(),
                    loser_path: loaded.skill.file_path.clone(),
                }));
            continue;
        }
        seen_real_paths.insert(real_path);
        result.skills.push(loaded);
    }
}

/// Validates the name and description metadata of a single loaded skill — at assembly time, covering what zen-core `build_skill` does not
/// pi validation surface (name ≤64/`[a-z0-9-]`/no leading, trailing, or consecutive hyphens; description ≤1024) produces warning diagnostics,
/// while the skill itself is still loaded (aligned with the non-aborting warning semantics of pi validateName/validateDescription).
fn validate_skill_metadata(skill: &Skill) -> Vec<SkillLoadDiagnostic> {
    let mut diagnostics = Vec::new();
    let name_chars = skill.name.chars().count();
    if name_chars > MAX_NAME_LENGTH {
        diagnostics.push(SkillLoadDiagnostic::Warning {
            message: format!("name exceeds {MAX_NAME_LENGTH} characters ({name_chars})"),
            path: skill.file_path.clone(),
        });
    }
    if !skill
        .name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        diagnostics.push(SkillLoadDiagnostic::Warning {
            message: "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_string(),
            path: skill.file_path.clone(),
        });
    }
    if skill.name.starts_with('-') || skill.name.ends_with('-') {
        diagnostics.push(SkillLoadDiagnostic::Warning {
            message: "name must not start or end with a hyphen".to_string(),
            path: skill.file_path.clone(),
        });
    }
    if skill.name.contains("--") {
        diagnostics.push(SkillLoadDiagnostic::Warning {
            message: "name must not contain consecutive hyphens".to_string(),
            path: skill.file_path.clone(),
        });
    }
    let desc_chars = skill.description.chars().count();
    if desc_chars > MAX_DESCRIPTION_LENGTH {
        diagnostics.push(SkillLoadDiagnostic::Warning {
            message: format!(
                "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({desc_chars})"
            ),
            path: skill.file_path.clone(),
        });
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_core::skills::skill::Skill;

    fn make_skill(name: &str, description: &str, file_path: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: description.to_string(),
            content: "body".to_string(),
            file_path: file_path.to_string(),
            disable_model_invocation: false,
        }
    }

    // ─── validate_skill_metadata ─────────────────────────────────

    #[test]
    fn test_validate_skill_metadata_valid_no_diagnostics() {
        let skill = make_skill("my-skill", "does foo", "/s/my-skill/SKILL.md");
        assert!(validate_skill_metadata(&skill).is_empty());
    }

    #[test]
    fn test_validate_skill_metadata_name_too_long() {
        let name = "a".repeat(MAX_NAME_LENGTH + 1);
        let skill = make_skill(&name, "d", "/s/x/SKILL.md");
        let diags = validate_skill_metadata(&skill);
        assert_eq!(diags.len(), 1);
        assert!(
            matches!(&diags[0], SkillLoadDiagnostic::Warning { message, .. }
            if message.contains("exceeds"))
        );
    }

    #[test]
    fn test_validate_skill_metadata_name_invalid_chars() {
        let skill = make_skill("Bad_Name", "d", "/s/x/SKILL.md");
        let diags = validate_skill_metadata(&skill);
        assert_eq!(diags.len(), 1);
        assert!(
            matches!(&diags[0], SkillLoadDiagnostic::Warning { message, .. }
            if message.contains("invalid characters"))
        );
    }

    #[test]
    fn test_validate_skill_metadata_name_hyphen_edges() {
        let skill = make_skill("-foo", "d", "/s/x/SKILL.md");
        let diags = validate_skill_metadata(&skill);
        assert_eq!(diags.len(), 1);
        assert!(
            matches!(&diags[0], SkillLoadDiagnostic::Warning { message, .. }
            if message.contains("start or end"))
        );
    }

    #[test]
    fn test_validate_skill_metadata_name_consecutive_hyphens() {
        let skill = make_skill("a--b", "d", "/s/x/SKILL.md");
        let diags = validate_skill_metadata(&skill);
        assert_eq!(diags.len(), 1);
        assert!(
            matches!(&diags[0], SkillLoadDiagnostic::Warning { message, .. }
            if message.contains("consecutive hyphens"))
        );
    }

    #[test]
    fn test_validate_skill_metadata_description_too_long() {
        let desc = "d".repeat(MAX_DESCRIPTION_LENGTH + 1);
        let skill = make_skill("ok", &desc, "/s/x/SKILL.md");
        let diags = validate_skill_metadata(&skill);
        assert_eq!(diags.len(), 1);
        assert!(
            matches!(&diags[0], SkillLoadDiagnostic::Warning { message, .. }
            if message.contains("description exceeds"))
        );
    }

    // ─── add_skills (deduplication + collisions) ────────────────

    #[test]
    fn test_add_skills_name_collision_keeps_first() {
        let mut result = LoadSkillsResult::default();
        let mut seen = HashSet::new();
        let first = make_skill("foo", "d1", "/a/SKILL.md");
        let second = make_skill("foo", "d2", "/b/SKILL.md");
        add_skills(
            vec![
                LoadedSkill {
                    skill: first,
                    source: SkillSource::User,
                },
                LoadedSkill {
                    skill: second,
                    source: SkillSource::Path,
                },
            ],
            &mut result,
            &mut seen,
        );
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].source, SkillSource::User); // first occurrence (User) wins
        assert_eq!(result.diagnostics.len(), 1);
        match &result.diagnostics[0] {
            SkillLoadDiagnostic::Collision(c) => {
                assert_eq!(c.name, "foo");
                assert_eq!(c.winner_path, "/a/SKILL.md");
                assert_eq!(c.loser_path, "/b/SKILL.md");
            }
            _ => panic!("expected collision"),
        }
    }

    #[test]
    fn test_add_skills_distinct_names_both_kept() {
        let mut result = LoadSkillsResult::default();
        let mut seen = HashSet::new();
        add_skills(
            vec![
                LoadedSkill {
                    skill: make_skill("a", "d", "/a/SKILL.md"),
                    source: SkillSource::User,
                },
                LoadedSkill {
                    skill: make_skill("b", "d", "/b/SKILL.md"),
                    source: SkillSource::Project,
                },
            ],
            &mut result,
            &mut seen,
        );
        assert_eq!(result.skills.len(), 2);
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn test_add_skills_same_real_path_deduped_silently() {
        // Same file path added twice (simulating symlink duplication) → second one silently skipped
        let mut result = LoadSkillsResult::default();
        let mut seen = HashSet::new();
        let skill = make_skill("foo", "d", "/a/SKILL.md");
        add_skills(
            vec![
                LoadedSkill {
                    skill,
                    source: SkillSource::User,
                },
                LoadedSkill {
                    skill: make_skill("foo", "d", "/a/SKILL.md"),
                    source: SkillSource::User,
                },
            ],
            &mut result,
            &mut seen,
        );
        assert_eq!(result.skills.len(), 1);
        assert!(result.diagnostics.is_empty());
    }

    // ─── load_skills integration (tempdir + real files) ─────────

    #[tokio::test]
    async fn test_load_skills_defaults_user_and_project() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("skills")).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME).join("skills")).unwrap();
        std::fs::write(
            agent_dir.join("skills").join("SKILL.md"),
            "---\nname: user-skill\ndescription: from user\n---\nbody",
        )
        .unwrap();
        std::fs::write(
            cwd.join(CONFIG_DIR_NAME).join("skills").join("SKILL.md"),
            "---\nname: proj-skill\ndescription: from project\n---\nbody",
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            include_defaults: true,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 2);
        assert!(result.diagnostics.is_empty());

        let user = result
            .skills
            .iter()
            .find(|s| s.skill.name == "user-skill")
            .unwrap();
        assert_eq!(user.source, SkillSource::User);
        let proj = result
            .skills
            .iter()
            .find(|s| s.skill.name == "proj-skill")
            .unwrap();
        assert_eq!(proj.source, SkillSource::Project);
    }

    #[tokio::test]
    async fn test_load_skills_explicit_dir_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("SKILL.md"),
            "---\nname: extra-skill\ndescription: from extra\n---\nbody",
        )
        .unwrap();
        // Single .md file skill
        let single_file = tmp.path().join("single.md");
        std::fs::write(
            &single_file,
            "---\nname: single-skill\ndescription: single\n---\nbody",
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            skill_paths: vec![extra_dir, single_file],
            include_defaults: false,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 2);
        assert!(result.skills.iter().all(|s| s.source == SkillSource::Path));
        assert!(result.skills.iter().any(|s| s.skill.name == "extra-skill"));
        assert!(result.skills.iter().any(|s| s.skill.name == "single-skill"));
    }

    #[tokio::test]
    async fn test_load_skills_nonexistent_and_non_md_warnings() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let non_md = tmp.path().join("note.txt");
        std::fs::write(&non_md, "hello").unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            skill_paths: vec![tmp.path().join("missing"), non_md],
            include_defaults: false,
        };
        let result = load_skills(&opts).await;
        assert!(result.skills.is_empty());
        assert_eq!(result.diagnostics.len(), 2);
        assert!(result.diagnostics.iter().any(|d| matches!(d,
            SkillLoadDiagnostic::Warning { message, .. } if message.contains("does not exist"))));
        assert!(result.diagnostics.iter().any(|d| matches!(d,
            SkillLoadDiagnostic::Warning { message, .. } if message.contains("not a markdown file"))));
    }

    #[tokio::test]
    async fn test_load_skills_collision_default_wins_over_explicit() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("skills")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        // Same-name skill: default directory (User) loaded first, explicit path later → User wins
        std::fs::write(
            agent_dir.join("skills").join("SKILL.md"),
            "---\nname: clash\ndescription: from user\n---\nbody",
        )
        .unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("SKILL.md"),
            "---\nname: clash\ndescription: from extra\n---\nbody",
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: vec![extra_dir],
            include_defaults: true,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].source, SkillSource::User);
        assert_eq!(result.diagnostics.len(), 1);
        assert!(matches!(
            &result.diagnostics[0],
            SkillLoadDiagnostic::Collision(_)
        ));
    }

    #[tokio::test]
    async fn test_load_skills_validation_warning_for_long_name() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let long_name = "a".repeat(MAX_NAME_LENGTH + 1);
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("SKILL.md"),
            format!("---\nname: {long_name}\ndescription: d\n---\nbody"),
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            skill_paths: vec![extra_dir],
            include_defaults: false,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 1); // Validation warnings do not abort loading
        assert_eq!(result.diagnostics.len(), 1);
        assert!(matches!(&result.diagnostics[0],
            SkillLoadDiagnostic::Warning { message, .. } if message.contains("exceeds")));
    }

    #[tokio::test]
    async fn test_load_skills_comment_frontmatter_via_core_chain() {
        // Comment-line frontmatter (T-AP-03-5 fix semantics) loads normally through the zen-core path
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("SKILL.md"),
            "---\n# just a comment\nname: commented\ndescription: d\n---\nbody",
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            skill_paths: vec![extra_dir],
            include_defaults: false,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].skill.name, "commented");
    }

    #[tokio::test]
    async fn test_load_skills_crlf_normalized_via_core_chain() {
        // CRLF frontmatter (T-AP-03-5 fix semantics) loads normally through the zen-core path
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("SKILL.md"),
            "---\r\nname: crlf-skill\r\ndescription: d\r\n---\r\nbody",
        )
        .unwrap();

        let opts = LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            skill_paths: vec![extra_dir],
            include_defaults: false,
        };
        let result = load_skills(&opts).await;
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].skill.name, "crlf-skill");
    }
}
