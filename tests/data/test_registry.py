from drug_screen.data.registry import validate_registry

def test_registry_is_valid():
    assert validate_registry() == []
