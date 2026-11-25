from pathlib import Path

from badebhai.ingestion.filters import list_candidate_files, read_text_file
from badebhai.ingestion.secrets import REDACTED, contains_secret, redact


def test_redacts_known_token_formats_and_preserves_lines():
    text = (
        "aws = 'AKIAABCDEFGHIJKLMNOP'\n"
        "gh = 'ghp_" + "a" * 36 + "'\n"
        "anthropic = 'sk-ant-api03-" + "x" * 30 + "'\n"
        "normal_line = 42\n"
    )
    out, n = redact(text)
    assert n == 3
    assert out.count("\n") == text.count("\n")
    assert "AKIA" not in out and "ghp_" not in out and "sk-ant" not in out
    assert "normal_line = 42" in out


def test_generic_assignment_redacted_only_when_random_looking():
    real = 'API_KEY = "Zx81kPq0LmN3vB7wQe2R"\n'
    placeholder = 'API_KEY = "your-api-key-here"\n'
    assert REDACTED in redact(real)[0]
    assert redact(placeholder)[1] == 0


def test_private_key_block_redacted_line_by_line():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nAbCdEf\n-----END RSA PRIVATE KEY-----\nok\n"
    out, n = redact(pem)
    assert n == 2
    assert out.split("\n")[1] == REDACTED and out.split("\n")[2] == REDACTED
    assert out.count("\n") == pem.count("\n")


def test_contains_secret():
    assert contains_secret("my token is ghp_" + "b" * 36)
    assert not contains_secret("I work on the payments module")


def test_candidate_files_respect_gitignore_and_skip_binaries(tiny_repo: Path):
    files = list_candidate_files(tiny_repo)
    assert "app/auth.py" in files and "README.md" in files
    assert "logo.png" not in files
    assert not any(f.startswith("secret_dir/") for f in files)


def test_read_text_file_caps_size_and_rejects_binary(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 1000)
    assert read_text_file(big, max_bytes=100) is None
    binf = tmp_path / "b.dat"
    binf.write_bytes(b"abc\0def")
    assert read_text_file(binf, max_bytes=10_000) is None
    ok = tmp_path / "ok.py"
    ok.write_bytes(b"print(1)\n")
    assert read_text_file(ok, max_bytes=10_000)[1] == "print(1)\n"
