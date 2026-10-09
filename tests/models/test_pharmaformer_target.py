import torch
from torch import nn

from drug_screen.models.pharmaformer_target import configure_target_variant


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(6, 1))

    def forward(self, value):
        return self.output(value)


class TinyOfficialSurface(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.gene_fc1 = nn.Linear(3, 4)
        self.feature_extractor.gene_fc2 = nn.Linear(4, 4)
        self.feature_extractor.smiles_fc = nn.Linear(2, 2)
        self.feature_dim = 2
        self.seq_len = 3
        self.transformer = TinyTransformer()

    def forward(self, gene, smiles):
        e = self.feature_extractor
        values = torch.cat((torch.relu(e.gene_fc2(torch.relu(e.gene_fc1(gene)))),
                            torch.relu(e.smiles_fc(smiles))), dim=1)
        return self.transformer(values.reshape(len(values), self.seq_len, self.feature_dim))


def test_zero_initialized_adapter_preserves_official_output_and_freezes_backbone():
    torch.manual_seed(3)
    backbone = TinyOfficialSurface()
    gene, smiles = torch.randn(2, 3), torch.randn(2, 2)
    expected = backbone(gene, smiles)

    adapted = configure_target_variant(backbone, "source_adapter", adapter_rank=2)

    torch.testing.assert_close(adapted(gene, smiles), expected, rtol=0, atol=0)
    assert all(not p.requires_grad for p in backbone.feature_extractor.parameters())
    assert all(p.requires_grad for p in backbone.transformer.output.parameters())
    assert all(p.requires_grad for p in adapted.adapter.parameters())


def test_source_head_updates_only_official_output_head():
    backbone = TinyOfficialSurface()
    selected = configure_target_variant(backbone, "source_head")

    assert selected is backbone
    assert all(not p.requires_grad for p in backbone.feature_extractor.parameters())
    assert all(p.requires_grad for p in backbone.transformer.output.parameters())
