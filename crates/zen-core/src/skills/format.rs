use crate::skills::skill::Skill;

/// Format the skill list into an agentskills.io-compatible XML `<available_skills>` block.
/// Filter out `disable_model_invocation == true`; if empty after filtering, return an empty string.
/// Each visible skill emits `<name>`/`<description>`/`<location>` (location is file_path); field values are escaped via `escape_xml`.
pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();
    if visible.is_empty() {
        return String::new();
    }

    let mut lines = vec![
           "The following skills provide specialized instructions for specific tasks.".to_string(),
           "Read the full skill file when the task matches its description.".to_string(),
           "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the
 path) and use that absolute path in tool commands.".to_string(),
           String::new(),
           "<available_skills>".to_string(),
       ];
    for skill in &visible {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".to_string());
    }
    lines.push("</available_skills>".to_string());
    lines.join("\n")
}

/// XML-escape the five characters `& < > " '` (`&amp; &lt; &gt; &quot; &apos;`).
///
/// Single-pass scan: all escapes are single-byte ASCII, so consecutive safe spans between
/// the two special chars are `push_str`'d as a whole; split points naturally fall on UTF-8 char boundaries, so multi-byte chars are never broken.
/// Single allocation, and since `&` in the escaped output is not re-processed, order-independent.
fn escape_xml(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut run_start = 0;
    for (i, &b) in value.as_bytes().iter().enumerate() {
        let replacement = match b {
            b'&' => "&amp;",
            b'<' => "&lt;",
            b'>' => "&gt;",
            b'"' => "&quot;",
            b'\'' => "&apos;",
            _ => continue,
        };
        out.push_str(&value[run_start..i]);
        out.push_str(replacement);
        run_start = i + 1;
    }
    out.push_str(&value[run_start..]);
    out
}

/// Formats the explicit invocation prompt for a single skill.
/// Generates `<skill name="..." location="...">\nReferences are relative to <parent directory>.\n\n<content>\n</skill>`,
/// When `additional_instructions` is non-empty, appends it after the block separated by a blank line.
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let file_path = skill.file_path.trim_end_matches('/');
    let parent_dir = file_path
        .rfind('/')
        .filter(|&i| i > 0) // Corresponds to TS dirnameEnvPath: slashIndex <= 0 → "/"
        .map_or("/", |i| &file_path[..i]);
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name, skill.file_path, parent_dir, skill.content
    );
    match additional_instructions {
        Some(inst) if !inst.is_empty() => format!("{block}\n\n{inst}"),
        _ => block,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Skill fields constructed per design (name/description/content/file_path/disable_model_invocation) assumption; adjust if inconsistent with the actual definition.
    fn make_skill(name: &str, description: &str, file_path: &str, disable: bool) -> Skill {
        Skill {
            name: name.to_string(),
            description: description.to_string(),
            content: "skill content".to_string(),
            file_path: file_path.to_string(),
            disable_model_invocation: disable,
        }
    }

    // ─── escape_xml ───────────────────────────────────────────────
    #[test]
    fn test_escape_xml_empty() {
        assert_eq!(escape_xml(""), "");
    }

    #[test]
    fn test_escape_xml_no_special_chars() {
        assert_eq!(escape_xml("plain text 123"), "plain text 123");
    }

    #[test]
    fn test_escape_xml_each_special_char() {
        assert_eq!(escape_xml("&"), "&amp;");
        assert_eq!(escape_xml("<"), "&lt;");
        assert_eq!(escape_xml(">"), "&gt;");
        assert_eq!(escape_xml("\""), "&quot;");
        assert_eq!(escape_xml("'"), "&apos;");
    }

    #[test]
    fn test_escape_xml_combined() {
        assert_eq!(
            escape_xml("a&b<c>d\"e'f"),
            "a&amp;b&lt;c&gt;d&quot;e&apos;f"
        );
    }

    #[test]
    fn test_escape_xml_adjacent_special_chars() {
        assert_eq!(escape_xml("&&"), "&amp;&amp;");
        assert_eq!(escape_xml("<>"), "&lt;&gt;");
    }

    #[test]
    fn test_escape_xml_unicode_preserved() {
        assert_eq!(escape_xml("Chinese & test"), "Chinese &amp; test");
    }

    // ─── format_skills_for_system_prompt ───────────────────────────
    #[test]
    fn test_format_skills_for_system_prompt_empty() {
        assert_eq!(format_skills_for_system_prompt(&[]), "");
    }

    #[test]
    fn test_format_skills_for_system_prompt_all_disabled() {
        let skills = vec![make_skill("a", "da", "/skills/a/SKILL.md", true)];
        assert_eq!(format_skills_for_system_prompt(&skills), "");
    }

    #[test]
    fn test_format_skills_for_system_prompt_single_skill() {
        let skills = vec![make_skill("foo", "does foo", "/skills/foo/SKILL.md", false)];
        let out = format_skills_for_system_prompt(&skills);
        assert!(out.contains("The following skills provide specialized instructions"));
        assert!(out.contains("<available_skills>"));
        assert!(out.contains("  <skill>"));
        assert!(out.contains("    <name>foo</name>"));
        assert!(out.contains("    <description>does foo</description>"));
        assert!(out.contains("    <location>/skills/foo/SKILL.md</location>"));
        assert!(out.contains("</available_skills>"));
    }

    #[test]
    fn test_format_skills_for_system_prompt_disabled_filtered() {
        let skills = vec![
            make_skill("visible", "dv", "/s/visible/SKILL.md", false),
            make_skill("hidden", "dh", "/s/hidden/SKILL.md", true),
        ];
        let out = format_skills_for_system_prompt(&skills);
        assert!(out.contains("<name>visible</name>"));
        assert!(!out.contains("hidden"));
    }

    #[test]
    fn test_format_skills_for_system_prompt_escapes_fields() {
        let skills = vec![make_skill("a&b", "d<e", "/s/x\"y/SKILL.md", false)];
        let out = format_skills_for_system_prompt(&skills);
        assert!(out.contains("<name>a&amp;b</name>"));
        assert!(out.contains("<description>d&lt;e</description>"));
        assert!(out.contains("<location>/s/x&quot;y/SKILL.md</location>"));
    }

    #[test]
    fn test_format_skills_for_system_prompt_multiple_skills_order() {
        let skills = vec![
            make_skill("first", "d1", "/s/first/SKILL.md", false),
            make_skill("second", "d2", "/s/second/SKILL.md", false),
        ];
        let out = format_skills_for_system_prompt(&skills);
        let first = out.find("<name>first</name>").unwrap();
        let second = out.find("<name>second</name>").unwrap();
        assert!(first < second);
    }

    // ─── format_skill_invocation ───────────────────────────────────
    #[test]
    fn test_format_skill_invocation_basic_block() {
        let skill = make_skill("foo", "d", "/skills/foo/SKILL.md", false);
        let out = format_skill_invocation(&skill, None);
        assert!(out.contains("<skill name=\"foo\" location=\"/skills/foo/SKILL.md\">"));
        assert!(out.contains("References are relative to /skills/foo."));
        assert!(out.contains("skill content"));
        assert!(out.ends_with("</skill>"));
    }

    #[test]
    fn test_format_skill_invocation_parent_dir_top_level() {
        let skill = make_skill("foo", "d", "/SKILL.md", false);
        let out = format_skill_invocation(&skill, None);
        assert!(out.contains("References are relative to /."));
    }

    #[test]
    fn test_format_skill_invocation_parent_dir_no_slash() {
        let skill = make_skill("foo", "d", "SKILL.md", false);
        let out = format_skill_invocation(&skill, None);
        assert!(out.contains("References are relative to /."));
    }

    #[test]
    fn test_format_skill_invocation_with_instructions() {
        let skill = make_skill("foo", "d", "/skills/foo/SKILL.md", false);
        let out = format_skill_invocation(&skill, Some("extra instructions"));
        assert!(out.ends_with("</skill>\n\nextra instructions"));
    }

    #[test]
    fn test_format_skill_invocation_empty_instructions_ignored() {
        let skill = make_skill("foo", "d", "/skills/foo/SKILL.md", false);
        let out_none = format_skill_invocation(&skill, None);
        let out_empty = format_skill_invocation(&skill, Some(""));
        assert_eq!(out_none, out_empty);
        assert!(out_empty.ends_with("</skill>"));
    }
}
