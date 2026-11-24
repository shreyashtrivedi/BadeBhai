"""File classification and tree-sitter language specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import PurePosixPath

CODE_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".tf": "terraform",
    ".lua": "lua",
    ".ex": "elixir",
}
DOC_EXT = {
    ".md": "markdown",
    ".mdx": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".adoc": "asciidoc",
}
SPECIAL_NAMES = {
    "dockerfile": ("code", "dockerfile"),
    "makefile": ("code", "make"),
    "procfile": ("code", "text"),
    "readme": ("doc", "text"),
    "changelog": ("doc", "text"),
    "history": ("doc", "text"),
    "authors": ("doc", "text"),
    "contributing": ("doc", "text"),
    "license": ("doc", "text"),
}


def classify(path: str) -> tuple[str, str | None]:
    """Return (kind, language) for a repo-relative path. kind is 'code' or 'doc'."""
    p = PurePosixPath(path)
    ext = p.suffix.lower()
    if ext in DOC_EXT:
        return "doc", DOC_EXT[ext]
    if ext in CODE_EXT:
        return "code", CODE_EXT[ext]
    special = SPECIAL_NAMES.get(p.stem.lower()) or SPECIAL_NAMES.get(p.name.lower())
    if special:
        return special
    return "code", None


def is_test_path(path: str) -> bool:
    p = path.lower()
    name = PurePosixPath(p).name
    return (
        p.startswith(("tests/", "test/", "testing/"))
        or "/tests/" in p
        or "/test/" in p
        or "/__tests__/" in p
        or name.startswith("test_")
        or name.endswith(
            ("_test.py", "_test.go", ".spec.ts", ".test.ts", ".spec.js", ".test.js", "test.java")
        )
    )


@dataclass(frozen=True)
class TSSpec:
    """How to find definitions in one tree-sitter grammar."""

    defs: dict[str, str]  # node type -> symbol type
    containers: frozenset[str]  # def types whose body may be split into member chunks
    wrappers: dict[str, str] = field(default_factory=dict)  # wrapper node type -> field holding the def
    body_field: str = "body"


_JS_DEFS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "lexical_declaration": "variable",
}
_TS_DEFS = {
    **_JS_DEFS,
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "abstract_class_declaration": "class",
    "module": "module",
    "internal_module": "module",
}

TS_SPECS: dict[str, TSSpec] = {
    "python": TSSpec(
        defs={"function_definition": "function", "class_definition": "class"},
        containers=frozenset({"class_definition"}),
        wrappers={"decorated_definition": "definition"},
    ),
    "javascript": TSSpec(
        defs=_JS_DEFS,
        containers=frozenset({"class_declaration"}),
        wrappers={"export_statement": "declaration"},
    ),
    "typescript": TSSpec(
        defs=_TS_DEFS,
        containers=frozenset({"class_declaration", "abstract_class_declaration", "interface_declaration"}),
        wrappers={"export_statement": "declaration"},
    ),
    "go": TSSpec(
        defs={"function_declaration": "function", "method_declaration": "method", "type_declaration": "type"},
        containers=frozenset(),
    ),
    "java": TSSpec(
        defs={
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "class",
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        },
        containers=frozenset(
            {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
        ),
    ),
    "rust": TSSpec(
        defs={
            "function_item": "function",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
            "mod_item": "module",
            "macro_definition": "macro",
            "type_item": "type",
            "const_item": "const",
            "static_item": "static",
        },
        containers=frozenset({"impl_item", "trait_item", "mod_item"}),
    ),
}
TS_SPECS["tsx"] = TS_SPECS["typescript"]


@cache
def get_parser(language: str):
    """Return a tree-sitter Parser for `language`, or None if no grammar is installed."""
    try:
        import tree_sitter as ts

        if language == "python":
            import tree_sitter_python as mod

            lang = ts.Language(mod.language())
        elif language == "javascript":
            import tree_sitter_javascript as mod

            lang = ts.Language(mod.language())
        elif language in ("typescript", "tsx"):
            import tree_sitter_typescript as mod

            lang = ts.Language(mod.language_tsx() if language == "tsx" else mod.language_typescript())
        elif language == "go":
            import tree_sitter_go as mod

            lang = ts.Language(mod.language())
        elif language == "java":
            import tree_sitter_java as mod

            lang = ts.Language(mod.language())
        elif language == "rust":
            import tree_sitter_rust as mod

            lang = ts.Language(mod.language())
        else:
            return None
        return ts.Parser(lang)
    except Exception:  # missing grammar or ABI mismatch -> sliding-window fallback
        return None
