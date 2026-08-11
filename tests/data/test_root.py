from drug_screen.data.root import data_root

def test_default_data_root():
    assert data_root().name == "data"
    assert data_root().is_dir()
