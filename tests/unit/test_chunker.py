import itertools

from repomind.ingestion.chunker import Chunker, expand_identifiers, split_identifier
from repomind.ingestion.languages import classify, is_test_path

PY = (
    '''import os

CONST = 1


class Big:
    """Docstring."""

    attr = 2

'''
    + "\n".join(
        f"    def method_{i}(self):\n        x = {i}\n        y = x + 1\n        return y\n"
        for i in range(30)
    )
    + """

@decorator
def top_level(a, b):
    return a + b
"""
)


def _lines_match(chunk, text):
    lines = text.split("\n")
    assert chunk.content == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_python_function_and_class_chunks():
    text = "def a():\n    return 1\n\n\nclass B:\n    def m(self):\n        pass\n"
    chunks = Chunker().chunk("x.py", text, "code", "python")
    assert [(c.symbol, c.symbol_type, c.start_line, c.end_line) for c in chunks] == [
        ("a", "function", 1, 2),
        ("B", "class", 5, 7),
    ]
    for c in chunks:
        _lines_match(c, text)


def test_large_class_is_split_into_methods_with_qualified_names():
    chunker = Chunker(max_lines=40, window=20, overlap=5)
    chunks = chunker.chunk("big.py", PY, "code", "python")
    symbols = [c.symbol for c in chunks]
    assert "Big.method_0" in symbols and "Big.method_29" in symbols
    assert "top_level" in symbols
    deco = next(c for c in chunks if c.symbol == "top_level")
    assert deco.content.startswith("@decorator")  # decorator included in the function chunk
    for c in chunks:
        _lines_match(c, PY)
        assert c.end_line - c.start_line + 1 <= 40


def test_huge_function_falls_back_to_windows_with_symbol():
    body = "\n".join(f"    v{i} = {i}" for i in range(200))
    text = f"def huge():\n{body}\n"
    chunks = Chunker(max_lines=50, window=40, overlap=10).chunk("h.py", text, "code", "python")
    assert len(chunks) > 1
    assert all(c.symbol == "huge" for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 201
    for a, b in itertools.pairwise(chunks):
        assert b.start_line <= a.end_line  # overlap


def test_javascript_and_go_and_rust():
    js = "export function handler(req) {\n  return auth(req);\n}\n\nclass Foo {\n  bar() { return 1; }\n}\n"
    js_chunks = Chunker().chunk("a.js", js, "code", "javascript")
    assert [c.symbol for c in js_chunks] == ["handler", "Foo"]

    go = 'package main\n\n// Login checks credentials.\nfunc Login(u string) bool {\n\treturn u != ""\n}\n'
    go_chunks = Chunker().chunk("a.go", go, "code", "go")
    login = next(c for c in go_chunks if c.symbol == "Login")
    assert login.content.startswith("// Login checks")  # doc comment attached

    rs = "struct S;\n\nimpl Display for S {\n    fn fmt(&self) {}\n}\n"
    rs_chunks = Chunker().chunk("a.rs", rs, "code", "rust")
    assert "Display for S" in [c.symbol for c in rs_chunks]


def test_unsupported_language_uses_sliding_windows():
    text = "\n".join(f"line {i}" for i in range(1, 121))
    chunks = Chunker(max_lines=80, window=50, overlap=10).chunk("a.kt", text, "code", "kotlin")
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 50), (41, 90), (81, 120)]


def test_markdown_sections_use_heading_path_and_skip_code_fences():
    md = "# Title\n\nIntro text here.\nMore intro.\n\n## Install\n\n```bash\n# not a heading\npip install x\n```\nDone.\n\n## Usage\n\nCall it.\nTwice.\nThrice.\n"
    chunks = Chunker().chunk("README.md", md, "doc", "markdown")
    titles = [c.symbol for c in chunks]
    assert titles == ["Title", "Title > Install", "Title > Usage"]
    assert "# not a heading" in chunks[1].content
    for c in chunks:
        _lines_match(c, md)


def test_rst_sections():
    rst = (
        "Quickstart\n==========\n\nSome text.\nMore.\nEven more.\n\nAuth\n----\n\nUse auth.\nReally.\nYes.\n"
    )
    chunks = Chunker().chunk("docs/quick.rst", rst, "doc", "rst")
    assert [c.symbol for c in chunks] == ["Quickstart", "Quickstart > Auth"]


def test_blank_file_produces_no_chunks():
    assert Chunker().chunk("e.py", "\n\n  \n", "code", "python") == []


def test_identifier_expansion():
    assert split_identifier("HTTPBasicAuth") == ["http", "basic", "auth"]
    assert split_identifier("max_retries") == ["max", "retries"]
    assert set(expand_identifiers("x = HTTPDigestAuth(max_retries)").split()) >= {"digest", "retries"}


def test_classify_and_test_paths():
    assert classify("src/a.py") == ("code", "python")
    assert classify("docs/guide.md") == ("doc", "markdown")
    assert classify("README") == ("doc", "text")
    assert classify("Dockerfile") == ("code", "dockerfile")
    assert is_test_path("tests/test_auth.py") and is_test_path("pkg/foo_test.go")
    assert not is_test_path("src/auth.py")
