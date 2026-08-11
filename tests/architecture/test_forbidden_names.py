from pathlib import Path

def test_source_has_no_legacy_model_dependency():
    text = "\n".join(p.read_text(errors="ignore") for p in Path("src").rglob("*.py"))
    assert "mcpire_pdo" not in text.lower()
    assert "triperturb" not in text.lower()
