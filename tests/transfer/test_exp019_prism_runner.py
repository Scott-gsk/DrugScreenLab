import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('run_exp019_prism',ROOT/'scripts/run_exp019_prism.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_cell_line_expression_is_broadcast_to_all_drug_records():
    cells=np.array([[1.,2.],[8.,9.]])
    got=module.broadcast_cell_rows(cells,['A','B'],['A','A','B','A','B'])
    assert np.array_equal(got,np.array([[1.,2.],[1.,2.],[8.,9.],[1.,2.],[8.,9.]]))

def test_broadcast_rejects_unknown_or_duplicate_cell_line_identity():
    with pytest.raises(ValueError,match='unique'):
        module.broadcast_cell_rows(np.ones((2,1)),['A','A'],['A'])
    with pytest.raises(ValueError,match='unknown'):
        module.broadcast_cell_rows(np.ones((1,1)),['A'],['B'])
