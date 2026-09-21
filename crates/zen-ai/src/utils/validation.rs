//! Runtime validation and type coercion for tool-argument JSON Schema.
//!
//! REQ-AI-03 (pi-110): validation engine uses the `jsonschema` crate; type coercion is self-implemented
//! (aligned with pi `coercePrimitiveByType`/`coerceWithJsonSchema`—the jsonschema crate
//! only validates and does not convert).
//! See reference/pi-original/packages/ai/src/utils/validation.ts

use crate::tool::{Tool, ToolRegistry};
use crate::types::ToolCall;
use jsonschema::JSONSchema;

/// Tool-argument validation error; failure surface for `validate_tool_arguments`/`validate_tool_call`.
///
/// Three variants cover "tool not found in registry", "arguments failed Schema validation", and "tool-argument Schema
/// itself is invalid (jsonschema compilation failed)" three failure classes; Display messages align with pi's error contract
/// （`Validation failed for tool "X":\n  - path: msg\n\nReceived arguments:\n{json}`），
/// and are the text source the agent loop uses to feed bad arguments back to the LLM for retry.
#[derive(Debug, thiserror::Error)]
pub enum ToolArgumentError {
    /// Tool not found by name in the registry (`validate_tool_call` path)
    #[error("Tool \"{tool_name}\" not found")]
    ToolNotFound {
        /// Tool name passed by the caller
        tool_name: String,
    },
    /// Arguments failed Schema validation; includes per-field error lines and argument echo (after prepare, before coerce)
    #[error(
        "Validation failed for tool \"{tool_name}\":\n{errors}\n\nReceived arguments:\n{received}"
    )]
    ValidationFailed {
        /// Tool name (used for message location)
        tool_name: String,
        /// Preformatted error lines (`  - path: message`, separated by \n)
        errors: String,
        /// Echoed argument JSON (`serde_json::to_string_pretty`, i.e., the validation input itself)
        received: String,
    },
    /// Tool-argument Schema itself is invalid; jsonschema compilation failed
    #[error("Tool \"{tool_name}\" parameter schema is invalid: {message}")]
    SchemaCompile {
        /// Tool name (used for message location)
        tool_name: String,
        /// jsonschema compilation error text
        message: String,
    },
}

/// Last gate before tool execution: the agent loop, after `prepare_arguments` preprocessing and before actually executing
/// the tool, calls this function to finalize the arguments (possibly transformed by per-tool hooks) into a Schema-valid
/// form—"the LLM occasionally returns `"42"` instead of `42`" is converted back to number here; unconvertible cases produce per-field
/// validation errors, so the model retries next round instead of bad arguments being passed to the tool.
///
/// Flow (aligned with pi `validateToolArguments`; input corresponds to pi `preparedToolCall.arguments`):
/// First compile `tool.parameters()` (throws `SchemaCompile` on failure), then use `coerce_with_json_schema`
/// to perform deep type coercion, and finally validate the result with jsonschema; on failure returns
/// `ToolArgumentError::ValidationFailed` (errors are line-by-line `  - path: message`; the echo is
/// **this input `args`**—i.e., the arguments after prepare and before coerce, corresponding to pi
/// `JSON.stringify(preparedToolCall.arguments)`）。
///
/// # Parameters
/// - `tool`: the tool found by name (`Tool::parameters()` is the validation Schema source,
///   `tool.name()` is used in error messages).
/// - `args`: the arguments to validate—the agent loop passes the output of `prepare_arguments`;
///   The manual execution path can pass the raw `tool_call.arguments`.
///
/// # Returns
/// `Ok(Value)` is the validated (possibly type-coerced) argument object; the caller should use the return value
/// to replace `args` and continue the flow (aligned with pi `args: validatedArgs`); for `Err`, see `ToolArgumentError`.
///
/// # Bounds
/// - Schema itself is invalid (compilation failure) → `ToolArgumentError::SchemaCompile`.
/// - When the argument is not an object (e.g. top level is a string), no object-level coercion is performed; type checking is left directly to jsonschema to report an error.
pub fn validate_tool_arguments(
    tool: &dyn Tool,
    args: &serde_json::Value,
) -> Result<serde_json::Value, ToolArgumentError> {
    let schema = tool.parameters();
    let validator =
        compile_validator(schema).map_err(|message| ToolArgumentError::SchemaCompile {
            tool_name: tool.name().to_string(),
            message,
        })?;

    let coerced = coerce_with_json_schema(args, schema);

    if let Err(errors) = validator.validate(&coerced) {
        let mut lines = Vec::new();
        for error in errors {
            lines.push(format!("  - {}: {}", format_validation_path(&error), error));
        }
        let received = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
        return Err(ToolArgumentError::ValidationFailed {
            tool_name: tool.name().to_string(),
            errors: lines.join("\n"),
            received,
        });
    }

    Ok(coerced)
}

/// Entry point of the manual tool execution loop (aligned with pi README public API `validateToolCall`): the caller holds
/// the entire tool registry and one LLM tool call; this function looks up by `tool_call.name` and delegates to
/// `validate_tool_arguments`—the two failure classes, "tool not found by name" and "invalid arguments", converge into
/// the same error type, saving the caller from looking up and assembling error messages itself.
///
/// What is passed in is the **raw arguments** of `tool_call.arguments`, without calling `prepare_arguments` (aligned with pi
/// `validateToolCall` passing directly without going through `prepareToolCallArguments`—prepare is
/// a pipeline exclusive to the agent loop; the manual execution path validates using raw arguments).
///
/// Current zen production code has no manual execution loop (the agent loop goes directly through `validate_tool_arguments`),
/// this function is retained for alignment with the pi public API, for future manual execution paths.
///
/// # Returns
/// `Ok(Value)` is the validated (possibly coerced) arguments; `Err(ToolArgumentError::ToolNotFound)`
/// means no such tool name exists in the registry; for `Err(ValidationFailed)`, see `validate_tool_arguments`.
pub fn validate_tool_call(
    tools: &ToolRegistry,
    tool_call: &ToolCall,
) -> Result<serde_json::Value, ToolArgumentError> {
    let tool =
        tools
            .get_by_name(&tool_call.name)
            .ok_or_else(|| ToolArgumentError::ToolNotFound {
                tool_name: tool_call.name.clone(),
            })?;
    validate_tool_arguments(tool.as_ref(), &tool_call.arguments)
}

/// Recursive engine for deep type coercion: `validate_tool_arguments` calls it before jsonschema validation,
/// converting LLM stringified arguments layer by layer according to the Schema into real types such as number/boolean/integer—
/// "the Schema wants a number, so here `\"42\"` is converted back to `42`."
///
/// Processing order aligned with pi `coerceWithJsonSchema`: allOf → anyOf/oneOf (per-member coercion + member
/// sub-validator validation, first passing member wins, all failing returns the original value) → by `type` (string or array), perform
/// primitive type coercion (pi's guard is `schemaTypes.length > 1 && a matching member exists`; in the single-type case
/// the value is unchanged—coerce is a no-op for already-matching values) → object's properties/additionalProperties
/// recursion → array's items (single Schema or tuple item-by-item) recursion.
///
/// # Returns
/// Returns a new (possibly coerced) `Value`; coercion does not modify the input (Rust value semantics).
pub(crate) fn coerce_with_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let mut next = value.clone();

    // allOf: coerce each member schema in sequence
    if let Some(all_of) = schema.get("allOf").and_then(|v| v.as_array()) {
        for member in all_of {
            next = coerce_with_json_schema(&next, member);
        }
    }

    // anyOf / oneOf: per-member coercion + sub-validator validation, first passing member wins
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        next = coerce_with_union_schema(&next, any_of);
    }
    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        next = coerce_with_union_schema(&next, one_of);
    }

    let schema_types = get_schema_types(schema);
    let matches_union_member =
        schema_types.len() > 1 && schema_types.iter().any(|t| matches_json_type(&next, t));
    if !schema_types.is_empty() && !matches_union_member {
        for type_name in &schema_types {
            let candidate = coerce_primitive_by_type(&next, type_name);
            if candidate != next {
                next = candidate;
                break;
            }
        }
    }

    if schema_types.contains(&"object") && next.is_object() {
        next = apply_schema_object_coercion(next, schema);
    }
    if schema_types.contains(&"array") && next.is_array() {
        next = apply_schema_array_coercion(next, schema);
    }

    next
}

/// Single-type coercion primitive: when `coerce_with_json_schema` handles the `type` keyword, for each type branch
/// it calls this function to coerce null/string/boolean to the target type—"null→0/false/\"\", `\"42\"`→42,
/// `true`→1、`1`→true、`""`/`0`/`false`→null」。
///
/// when no coercion is possible, return as-is (no partial coercion): `number` accepts only strings that are non-empty after trim and parse as finite numbers
/// (`\"42abc\"` is not coerced); `integer` uses f64 parsing + no fractional part (aligned with pi
/// `Number.isInteger` floating-point semantics)—`\"42.0\"`/`\"42.00\"` coerce successfully (→ 42),
/// `\"42.5\"` is not coerced (returns the original value).
/// Mapping table aligns item by item with pi `coercePrimitiveByType`'s 6 branches (number/integer/boolean/
/// string/null/default）。
pub(crate) fn coerce_primitive_by_type(
    value: &serde_json::Value,
    type_name: &str,
) -> serde_json::Value {
    match type_name {
        "number" | "integer" => {
            if value.is_null() {
                return serde_json::json!(0);
            }
            if let Some(s) = value.as_str() {
                if !s.trim().is_empty() {
                    if let Some(parsed) = parse_numeric_string(s) {
                        // number: coerce if finite; integer: only when the parsed result is an integer (aligned with
                        // pi `Number.isInteger(Number(s))`—"42.0" is an integer in the JS domain).
                        if type_name == "number" || parsed.fract() == 0.0 {
                            return number_from_f64(parsed);
                        }
                    }
                }
            }
            if let Some(b) = value.as_bool() {
                return serde_json::json!(if b { 1 } else { 0 });
            }
            value.clone()
        }
        "boolean" => {
            if value.is_null() {
                return serde_json::json!(false);
            }
            if let Some(s) = value.as_str() {
                if s == "true" {
                    return serde_json::json!(true);
                }
                if s == "false" {
                    return serde_json::json!(false);
                }
            }
            if let Some(f) = value.as_number().and_then(|n| n.as_f64()) {
                if f == 1.0 {
                    return serde_json::json!(true);
                }
                if f == 0.0 {
                    return serde_json::json!(false);
                }
            }
            value.clone()
        }
        "string" => {
            if value.is_null() {
                return serde_json::json!("");
            }
            if let Some(n) = value.as_number() {
                return serde_json::json!(number_to_string(n));
            }
            if let Some(b) = value.as_bool() {
                return serde_json::json!(b.to_string());
            }
            value.clone()
        }
        "null" => {
            if value == "" || value == 0 || value == false {
                return serde_json::Value::Null;
            }
            value.clone()
        }
        _ => value.clone(),
    }
}

/// anyOf/oneOf branch handling: for each member, "clone for trial → coerce by member Schema → validate with member sub-validator",
/// first passing member wins; if all fail (including sub-validator compilation failure), return the original value (aligned with pi `coerceWithUnionSchema`,
/// compilation failure corresponds to `getSubSchemaValidator`'s try/catch swallowing errors).
fn coerce_with_union_schema(
    value: &serde_json::Value,
    members: &[serde_json::Value],
) -> serde_json::Value {
    for member in members {
        let candidate = coerce_with_json_schema(value, member);
        if let Ok(validator) = compile_validator(member) {
            if validator.is_valid(&candidate) {
                return candidate;
            }
        }
    }
    value.clone()
}

/// object recursion: keys present in schema.properties are coerced by their property Schema; additionalProperties
/// when it is in object form, keys **not defined in properties** are coerced by additionalProperties
/// (aligned with pi `applySchemaObjectCoercion`, definedKeys exemption-set semantics).
fn apply_schema_object_coercion(
    mut value: serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let properties = schema.get("properties").and_then(|v| v.as_object());
    let defined_keys: std::collections::HashSet<&str> = properties
        .map(|p| p.keys().map(String::as_str).collect())
        .unwrap_or_default();

    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            if let Some(property_value) = value.get(key) {
                value[key] = coerce_with_json_schema(property_value, property_schema);
            }
        }
    }

    if let Some(additional) = schema.get("additionalProperties").filter(|v| v.is_object()) {
        if let Some(obj) = value.as_object_mut() {
            for (key, property_value) in obj.iter_mut() {
                if defined_keys.contains(key.as_str()) {
                    continue;
                }
                *property_value = coerce_with_json_schema(property_value, additional);
            }
        }
    }

    value
}

/// array recursion: when items is in tuple-array form, coerce position by position (out-of-bounds positions unchanged); when items is in object form
/// coerce element by element (aligned with pi `applySchemaArrayCoercion`).
fn apply_schema_array_coercion(
    mut value: serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let Some(items) = schema.get("items") else {
        return value;
    };

    if let Some(tuple) = items.as_array() {
        if let Some(arr) = value.as_array_mut() {
            for (index, item_schema) in tuple.iter().enumerate() {
                if index >= arr.len() {
                    break;
                }
                let coerced = coerce_with_json_schema(&arr[index], item_schema);
                arr[index] = coerced;
            }
        }
    } else if items.is_object() {
        if let Some(arr) = value.as_array_mut() {
            for item in arr.iter_mut() {
                let coerced = coerce_with_json_schema(item, items);
                *item = coerced;
            }
        }
    }

    value
}

/// Reads the schema's `type` keyword (string or array of strings); returns an empty list for any other form.
fn get_schema_types(schema: &serde_json::Value) -> Vec<&str> {
    match schema.get("type") {
        Some(serde_json::Value::String(t)) => vec![t.as_str()],
        Some(serde_json::Value::Array(types)) => types.iter().filter_map(|t| t.as_str()).collect(),
        _ => Vec::new(),
    }
}

/// Determines whether a value already matches a JSON Schema primitive type (aligned with pi `matchesJsonType`).
///
/// `integer`'s criteria match the jsonschema side: accepts i64/u64 representations and **f64 without a fractional part**—
/// tested jsonschema 0.18.3 accepts `42.0`/`1e3` for `{\"type\":\"integer\"}` (fract==0 semantics),
/// so using only `is_i64() || is_u64()` would be inconsistent with the validation engine.
fn matches_json_type(value: &serde_json::Value, type_name: &str) -> bool {
    match type_name {
        "number" => value.is_number(),
        "integer" => value.as_number().is_some_and(|n| {
            n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0)
        }),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

/// Compiles a JSON Schema into a jsonschema validator; returns error text on compilation failure.
///
/// Compilation failure on the main validation path must be explicitly propagated as `SchemaCompile` (must not be swallowed); for union sub-validation's
/// trial path, use `compile_validator(member).ok()` to swallow errors (aligned with pi `getSubSchemaValidator`).
fn compile_validator(schema: &serde_json::Value) -> Result<JSONSchema, String> {
    JSONSchema::options()
        .compile(schema)
        .map_err(|e| e.to_string())
}

/// jsonschema error → pi-style path text: strip leading `/` from `instance_path`, replace `/` with `.`
/// (empty → `\"root\"`); for `required` keyword errors, append the missing property name after the base path
/// (if the base path is empty, take only the property name; aligned with pi `formatValidationPath`).
fn format_validation_path(error: &jsonschema::ValidationError) -> String {
    if let jsonschema::error::ValidationErrorKind::Required { property } = &error.kind {
        if let Some(property) = property.as_str() {
            let base_path = format_instance_path(&error.instance_path);
            return if base_path.is_empty() {
                property.to_string()
            } else {
                format!("{}.{}", base_path, property)
            };
        }
    }
    let path = format_instance_path(&error.instance_path);
    if path.is_empty() {
        "root".to_string()
    } else {
        path
    }
}

/// `instance_path` (JSONPointer text such as `/a/b` or an empty string) → `a.b` or an empty string.
fn format_instance_path(path: &jsonschema::paths::JSONPointer) -> String {
    let s = path.to_string();
    let stripped = s.strip_prefix('/').unwrap_or(&s);
    stripped.replace('/', ".")
}

/// Parses a decimal numeric string as a finite f64 (aligned with pi `Number(s)`'s finiteness check; trim semantics
/// handled explicitly—Rust's f64 `FromStr` does not guarantee trimming, while JS `Number` trims).
/// Intentional divergence: JS `Number` also accepts hexadecimal/octal/binary literals (`\"0x10\"`→16),
/// Rust f64 parsing does not accept them—LLMs produce decimal, so this does not matter.
fn parse_numeric_string(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parsed: f64 = trimmed.parse().ok()?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

/// f64 → serde_json Number: when there is no fractional part and it is within integer range, normalize to i64/u64 representation,
/// so the serialized form aligns with pi (`JSON.stringify(42)` = `"42"` rather than `"42.0"`), and is
/// friendly to integer deserialization such as `serde_json::from_value::<i32>` (f64 representation would fail deserialization).
fn number_from_f64(f: f64) -> serde_json::Value {
    // i64 range [-2^63, 2^63): use strictly less than for the upper bound (2^63 in f64 is exactly equal to i64::MAX as f64,
    // must be excluded to avoid `f as i64` still passing the round-trip check after saturation)
    const I64_MIN: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCL: f64 = 9_223_372_036_854_775_808.0;
    if f.fract() == 0.0 && (I64_MIN..I64_MAX_EXCL).contains(&f) {
        let i = f as i64;
        if i as f64 == f {
            return serde_json::json!(i);
        }
    }
    const U64_MAX_EXCL: f64 = 18_446_744_073_709_551_616.0;
    if f.fract() == 0.0 && (0.0..U64_MAX_EXCL).contains(&f) {
        let u = f as u64;
        if u as f64 == f {
            return serde_json::json!(u);
        }
    }
    serde_json::Number::from_f64(f)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// serde_json Number → text: use integer text for integer types (JS `String(42)` = `"42"`),
/// use ryu representation for floats (`String(42.5)` = `"42.5"`).
fn number_to_string(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolError, ToolExecutionResult, ToolUpdateCallback};
    use crate::types::{ContentBlock, TextContent, ToolResultMessage, Usage};
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn num(v: serde_json::Value) -> serde_json::Value {
        v
    }

    // ─── coerce_primitive_by_type ────────────────────────────────────────

    #[test]
    fn number_coercion_maps_pi_branches() {
        // null → 0
        assert_eq!(
            coerce_primitive_by_type(&serde_json::Value::Null, "number"),
            num(serde_json::json!(0))
        );
        // "42" → 42 (integer representation)
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42"), "number"),
            num(serde_json::json!(42))
        );
        // " 42 " → 42（trim）
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(" 42 "), "number"),
            num(serde_json::json!(42))
        );
        // "42.5" → 42.5
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42.5"), "number"),
            num(serde_json::json!(42.5))
        );
        // "42abc" is not parseable → keep as-is
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42abc"), "number"),
            num(serde_json::json!("42abc"))
        );
        // empty string (empty after trim) → keep as-is
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("  "), "number"),
            num(serde_json::json!("  "))
        );
        // true/false → 1/0
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(true), "number"),
            num(serde_json::json!(1))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(false), "number"),
            num(serde_json::json!(0))
        );
        // already a number → keep as-is
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(7), "number"),
            num(serde_json::json!(7))
        );
    }

    #[test]
    fn integer_coercion_accepts_integral_strings_only() {
        assert_eq!(
            coerce_primitive_by_type(&serde_json::Value::Null, "integer"),
            num(serde_json::json!(0))
        );
        // "42" → 42
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42"), "integer"),
            num(serde_json::json!(42))
        );
        // "42.0" → 42 (JS Number("42.0")=42 is an integer)
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42.0"), "integer"),
            num(serde_json::json!(42))
        );
        // "42.00" → 42
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42.00"), "integer"),
            num(serde_json::json!(42))
        );
        // "42.5" has a fractional part → do not convert
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42.5"), "integer"),
            num(serde_json::json!("42.5"))
        );
        // "42abc" is not parseable → do not convert
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("42abc"), "integer"),
            num(serde_json::json!("42abc"))
        );
        // true/false → 1/0
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(true), "integer"),
            num(serde_json::json!(1))
        );
        // already an integer → keep as-is
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(7), "integer"),
            num(serde_json::json!(7))
        );
    }

    #[test]
    fn boolean_coercion_maps_pi_branches() {
        assert_eq!(
            coerce_primitive_by_type(&serde_json::Value::Null, "boolean"),
            num(serde_json::json!(false))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("true"), "boolean"),
            num(serde_json::json!(true))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("false"), "boolean"),
            num(serde_json::json!(false))
        );
        // string other than true/false → do not convert
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("yes"), "boolean"),
            num(serde_json::json!("yes"))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(1), "boolean"),
            num(serde_json::json!(true))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(0), "boolean"),
            num(serde_json::json!(false))
        );
        // 1.0 (f64 representation) → true
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(1.0), "boolean"),
            num(serde_json::json!(true))
        );
        // 2 → do not convert
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(2), "boolean"),
            num(serde_json::json!(2))
        );
    }

    #[test]
    fn string_coercion_maps_pi_branches() {
        assert_eq!(
            coerce_primitive_by_type(&serde_json::Value::Null, "string"),
            num(serde_json::json!(""))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(42), "string"),
            num(serde_json::json!("42"))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(42.5), "string"),
            num(serde_json::json!("42.5"))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(true), "string"),
            num(serde_json::json!("true"))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("x"), "string"),
            num(serde_json::json!("x"))
        );
    }

    #[test]
    fn null_coercion_maps_pi_branches() {
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(""), "null"),
            serde_json::Value::Null
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(0), "null"),
            serde_json::Value::Null
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(false), "null"),
            serde_json::Value::Null
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("x"), "null"),
            num(serde_json::json!("x"))
        );
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!(1), "null"),
            num(serde_json::json!(1))
        );
    }

    #[test]
    fn unknown_type_returns_value_unchanged() {
        assert_eq!(
            coerce_primitive_by_type(&serde_json::json!("x"), "object"),
            num(serde_json::json!("x"))
        );
    }

    // ─── coerce_with_json_schema ─────────────────────────────────────────

    #[test]
    fn object_properties_recursed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "timeout": { "type": "number" },
                "name": { "type": "string" }
            }
        });
        let args = serde_json::json!({ "timeout": "42", "name": "ls" });
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!({ "timeout": 42, "name": "ls" })
        );
    }

    #[test]
    fn object_nested_properties_recursed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": { "count": { "type": "integer" } }
                }
            }
        });
        let args = serde_json::json!({ "inner": { "count": "3" } });
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!({ "inner": { "count": 3 } })
        );
    }

    #[test]
    fn additional_properties_recursed_for_undefined_keys() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "known": { "type": "string" } },
            "additionalProperties": { "type": "number" }
        });
        let args = serde_json::json!({ "known": "a", "extra": "5" });
        // convert known fields according to properties (a string that is already a string is left unchanged); extra fields according to additionalProperties
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!({ "known": "a", "extra": 5 })
        );
    }

    #[test]
    fn array_items_single_schema_recursed() {
        let schema = serde_json::json!({
            "type": "array",
            "items": { "type": "number" }
        });
        let args = serde_json::json!(["1", "2.5", 3]);
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!([1, 2.5, 3])
        );
    }

    #[test]
    fn array_tuple_items_recursed_by_position() {
        let schema = serde_json::json!({
            "type": "array",
            "items": [
                { "type": "number" },
                { "type": "string" }
            ]
        });
        let args = serde_json::json!(["1", 2]);
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!([1, "2"])
        );
    }

    #[test]
    fn union_any_of_first_member_that_validates_wins() {
        let schema = serde_json::json!({
            "anyOf": [
                { "type": "number", "minimum": 10 },
                { "type": "string" }
            ]
        });
        // "5": try as number first → 5 fails minimum:10 → switch to string member → "5" passes as-is
        let args = serde_json::json!("5");
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!("5")
        );
        // "42": try as number → 42 passes minimum → 42
        let args = serde_json::json!("42");
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!(42)
        );
    }

    #[test]
    fn all_of_chain_applies_in_order() {
        let schema = serde_json::json!({
            "allOf": [
                { "type": "number" },
                { "minimum": 10 }
            ]
        });
        let args = serde_json::json!("5");
        // member 1 converts "5" to 5; member 2 has no type and does not convert further
        assert_eq!(
            coerce_with_json_schema(&args, &schema),
            serde_json::json!(5)
        );
    }

    #[test]
    fn union_type_array_tries_types_in_order() {
        // type: ["string", "number"] and value true matches neither member → try in order
        // the string branch converts true to "true" (first changer takes effect)
        let schema = serde_json::json!({ "type": ["string", "number"] });
        assert_eq!(
            coerce_with_json_schema(&serde_json::json!(true), &schema),
            serde_json::json!("true")
        );
        // already matched member → no-op
        assert_eq!(
            coerce_with_json_schema(&serde_json::json!("x"), &schema),
            serde_json::json!("x")
        );
        assert_eq!(
            coerce_with_json_schema(&serde_json::json!(42), &schema),
            serde_json::json!(42)
        );
    }

    #[test]
    fn schema_without_type_returns_value_unchanged() {
        let schema = serde_json::json!({ "description": "loose" });
        let args = serde_json::json!({ "a": "1" });
        assert_eq!(coerce_with_json_schema(&args, &schema), args);
    }

    // ─── matches_json_type / get_schema_types ────────────────────────────

    #[test]
    fn integer_match_is_consistent_with_jsonschema() {
        // jsonschema 0.18.3 accepts f64 without fractional part as integer (42.0/1e3),
        // matches_json_type must use the same criteria (only is_i64/is_u64 would misjudge 42.0)
        assert!(matches_json_type(&serde_json::json!(42), "integer"));
        assert!(matches_json_type(&serde_json::json!(42.0), "integer"));
        assert!(matches_json_type(&serde_json::json!(1e3), "integer"));
        assert!(!matches_json_type(&serde_json::json!(42.5), "integer"));
        assert!(!matches_json_type(&serde_json::json!("42"), "integer"));
    }

    #[test]
    fn schema_types_reads_string_and_array() {
        assert_eq!(
            get_schema_types(&serde_json::json!({ "type": "object" })),
            vec!["object"]
        );
        assert_eq!(
            get_schema_types(&serde_json::json!({ "type": ["string", "null"] })),
            vec!["string", "null"]
        );
        assert!(get_schema_types(&serde_json::json!({})).is_empty());
    }

    // ─── format_validation_path ──────────────────────────────────────────

    fn first_validation_path(schema: &serde_json::Value, instance: &serde_json::Value) -> String {
        // ValidationError borrows instance, so it must be consumed in the same scope before returning an owned String
        let validator = JSONSchema::options()
            .compile(schema)
            .expect("schema compiles");
        let err = validator
            .validate(instance)
            .expect_err("instance must fail")
            .next()
            .expect("at least one error");
        format_validation_path(&err)
    }

    #[test]
    fn validation_path_root_required_uses_property_name() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        });
        assert_eq!(
            first_validation_path(&schema, &serde_json::json!({})),
            "command"
        );
    }

    #[test]
    fn validation_path_nested_required_appends_property() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": { "x": { "type": "number" } },
                    "required": ["x"]
                }
            }
        });
        assert_eq!(
            first_validation_path(&schema, &serde_json::json!({ "nested": {} })),
            "nested.x"
        );
    }

    #[test]
    fn validation_path_type_error_uses_dotted_path() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "command": { "type": "string" } }
        });
        assert_eq!(
            first_validation_path(&schema, &serde_json::json!({ "command": 42 })),
            "command"
        );
    }

    #[test]
    fn validation_path_root_non_required_returns_root() {
        let schema = serde_json::json!({ "type": "number" });
        assert_eq!(
            first_validation_path(&schema, &serde_json::json!("x")),
            "root"
        );
    }

    // ─── validate_tool_arguments / validate_tool_call ────────────────────

    /// Minimal Tool implementation for tests: the validation path only consumes `name()`/`parameters()`.
    struct TestTool {
        name: String,
        params: serde_json::Value,
    }

    impl std::fmt::Debug for TestTool {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TestTool")
                .field("name", &self.name)
                .finish()
        }
    }

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn parameters(&self) -> &serde_json::Value {
            &self.params
        }
        fn label(&self) -> &str {
            "Test"
        }
        async fn execute(
            &self,
            tool_call_id: &str,
            _params: &serde_json::Value,
            _cancel: &CancellationToken,
            _on_update: ToolUpdateCallback,
        ) -> Result<ToolExecutionResult, ToolError> {
            Ok(ToolExecutionResult {
                message: ToolResultMessage {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name: self.name.clone(),
                    content: vec![ContentBlock::Text(TextContent {
                        text: "ok".into(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: Some(Usage::default()),
                    is_error: false,
                    added_tool_names: None,
                    timestamp: 0,
                },
                terminate: false,
            })
        }
    }

    fn test_tool(name: &str, params: serde_json::Value) -> TestTool {
        TestTool {
            name: name.to_string(),
            params,
        }
    }

    fn test_tool_call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments: args,
            thought_signature: None,
        }
    }

    #[test]
    fn validate_ok_returns_coerced_args() {
        let tool = test_tool(
            "bash",
            serde_json::json!({
                "type": "object",
                "properties": { "timeout": { "type": "number" } },
                "required": ["timeout"]
            }),
        );
        let args = serde_json::json!({ "timeout": "42" });
        let validated = validate_tool_arguments(&tool, &args).expect("validates after coercion");
        assert_eq!(validated, serde_json::json!({ "timeout": 42 }));
    }

    #[test]
    fn validate_ok_passes_through_valid_args() {
        let tool = test_tool("bash", serde_json::json!({ "type": "object" }));
        let args = serde_json::json!({ "command": "ls" });
        let validated = validate_tool_arguments(&tool, &args).expect("validates");
        assert_eq!(validated, args);
    }

    #[test]
    fn validate_failure_reports_paths_and_received_echo() {
        let tool = test_tool(
            "bash",
            serde_json::json!({
                "type": "object",
                "properties": { "timeout": { "type": "number", "minimum": 100 } },
                "required": ["timeout"]
            }),
        );
        // "5" is coerced to 5 but still fails minimum:100 → validation fails
        let args = serde_json::json!({ "timeout": "5" });
        let err = validate_tool_arguments(&tool, &args).expect_err("must fail");
        match &err {
            ToolArgumentError::ValidationFailed {
                tool_name,
                errors,
                received,
            } => {
                assert_eq!(tool_name, "bash");
                assert!(errors.contains("  - timeout:"), "errors: {errors}");
                // echoes the original input before coercion (the "5" string), not the coerced 5
                assert!(
                    received.contains("\"timeout\": \"5\""),
                    "received: {received}"
                );
                assert!(
                    err.to_string()
                        .contains("Validation failed for tool \"bash\":"),
                    "{}",
                    err
                );
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn validate_failure_missing_required() {
        let tool = test_tool(
            "bash",
            serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        );
        let err = validate_tool_arguments(&tool, &serde_json::json!({})).expect_err("must fail");
        assert!(err.to_string().contains("  - command:"), "{}", err);
    }

    #[test]
    fn validate_schema_compile_failure_is_reported() {
        // `{"type": "bogus"}` cannot compile into a valid Schema → SchemaCompile
        let tool = test_tool("broken", serde_json::json!({ "type": "bogus" }));
        let err = validate_tool_arguments(&tool, &serde_json::json!({})).expect_err("must fail");
        match &err {
            ToolArgumentError::SchemaCompile { tool_name, message } => {
                assert_eq!(tool_name, "broken");
                assert!(!message.is_empty());
            }
            other => panic!("expected SchemaCompile, got {other:?}"),
        }
    }

    #[test]
    fn validate_tool_call_delegates_and_reports_not_found() {
        let tool = test_tool("bash", serde_json::json!({ "type": "object" }));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(tool));

        let ok = validate_tool_call(
            &registry,
            &test_tool_call("bash", serde_json::json!({ "a": 1 })),
        )
        .expect("found and validated");
        assert_eq!(ok, serde_json::json!({ "a": 1 }));

        let err = validate_tool_call(&registry, &test_tool_call("nope", serde_json::json!({})))
            .expect_err("not found");
        assert_eq!(err.to_string(), "Tool \"nope\" not found");
        assert!(matches!(err, ToolArgumentError::ToolNotFound { .. }));
    }
}
