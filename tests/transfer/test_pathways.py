import numpy as np
import pytest

from drug_screen.transfer.pathways import (
    gene_to_pathways,
    parse_gmt,
    require_distinct_resources,
    score_pathways,
    score_progeny,
)


def test_gmt_mapping_preserves_versioned_resource_and_gene_membership():
    resource = parse_gmt("P1\tdescription\tTP53\tEGFR\nP2\tdescription\tEGFR\tKRAS", resource_id="msigdb", version="2024.1")
    assert resource.metadata["version"] == "2024.1"
    assert gene_to_pathways(resource)["EGFR"] == ("P1", "P2")
    with pytest.raises(ValueError, match="distinct"):
        require_distinct_resources(resource, resource)


def test_pathway_scores_are_sample_local_and_respect_coverage():
    resource = parse_gmt("P\tdescription\tA\tB\tC", resource_id="gmt", version="1")
    scored = score_pathways(np.array([[1.0, 3.0], [2.0, np.nan]]), ["A", "B"], resource, min_coverage=0.5)
    np.testing.assert_allclose(scored.values[0], [2.0])
    assert np.isnan(scored.values[1, 0])  # only one of three genes is observed
    np.testing.assert_allclose(scored.coverage[:, 0], [2 / 3, 1 / 3])


def test_signed_progeny_requires_an_explicit_weight_resource():
    with pytest.raises(NotImplementedError, match="weight resource"):
        score_progeny(np.array([[1.0]]), ["A"])


def test_signed_progeny_uses_explicit_weights_and_per_sample_coverage():
    scored = score_progeny(
        np.array([[4.0, 3.0], [4.0, np.nan]]), ["B", "A"],
        weights={"signal": {"A": 2.0, "B": -1.0}}, version="v1", min_coverage=1.0,
    )
    np.testing.assert_allclose(scored.values[0], [2.0])
    assert np.isnan(scored.values[1, 0])
    np.testing.assert_allclose(scored.coverage[:, 0], [1.0, 0.5])
    assert scored.resource_metadata == {"resource_id": "PROGENy", "version": "v1", "method": "signed_linear"}
