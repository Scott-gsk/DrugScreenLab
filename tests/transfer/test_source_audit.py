"""The source audit must remain separate from model-performance decisions."""
from drug_screen.transfer.chemistry import standardize_smiles


def test_stereochemical_identity_loss_is_detectable():
    specified = standardize_smiles('N[C@@H](C)C(=O)O')
    unspecified = standardize_smiles('NC(C)C(=O)O')
    assert specified.inchikey != unspecified.inchikey
    assert specified.inchikey.split('-')[0] == unspecified.inchikey.split('-')[0]
