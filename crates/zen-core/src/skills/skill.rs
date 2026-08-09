/// Describes a loaded SKILL.md skill. Pure data with no loading or execution logic.
pub struct Skill {
    /// Stable skill name ([a-z0-9-], length ≤64, defaults to the parent directory name)
    pub name: String,
    /// Model-visible description (length ≤1024; if missing, discard the skill and record a diagnostic)
    pub description: String,
    /// Body content after the frontmatter
    pub content: String,
    /// Absolute path to the skill file
    pub file_path: String,
    /// When true, exclude it from the system prompt's skill block; it can still be invoked explicitly
    pub disable_model_invocation: bool,
}
