import numpy as np
import pytest
import torch

from drug_screen.models.exp045_gene_axis import (
    Exp045GeneAxisSlice,
    build_exp045_gene_axis_slice,
    gene_axis_sha256,
    slice_exp045_gene_fc1_weight,
    slice_exp045_transform_vectors,
)


_SOURCE = ("A", "B", "C", "D")
_TARGET = ("A", "C")


def _plan():
    source = ["A", "B", "C", "D"]
    target = ["A", "C", "D"]
    return build_exp045_gene_axis_slice(
        source,
        target,
        expected_source_sha256=gene_axis_sha256(source),
        expected_target_sha256=gene_axis_sha256(target),
    )


def _manual_plan(
    *,
    source_genes=_SOURCE,
    target_genes=_TARGET,
    positions=(0, 2),
    source_sha256=None,
    target_sha256=None,
):
    """Construct the public dataclass directly, including intentionally bad plans."""
    return Exp045GeneAxisSlice(
        source_genes=source_genes,
        target_genes=target_genes,
        positions=positions,
        source_sha256=source_sha256 if source_sha256 is not None else gene_axis_sha256(_SOURCE),
        target_sha256=target_sha256 if target_sha256 is not None else gene_axis_sha256(_TARGET),
    )


def test_gene_axis_slice_selects_exact_source_columns_without_mutation():
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = weight.clone()

    sliced = slice_exp045_gene_fc1_weight(weight, _plan())

    torch.testing.assert_close(sliced, original[:, [0, 2, 3]], rtol=0, atol=0)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert sliced.shape == (3, 3)


def test_transform_vectors_use_the_same_frozen_column_map():
    transform = {
        "mean": np.array([10.0, 20.0, 30.0, 40.0]),
        "scale": np.array([1.0, 2.0, 3.0, 4.0]),
        "min": np.array([-1.0, -2.0, -3.0, -4.0]),
        "range": np.array([5.0, 6.0, 7.0, 8.0]),
    }

    sliced = slice_exp045_transform_vectors(transform, _plan())

    np.testing.assert_array_equal(sliced["mean"], [10.0, 30.0, 40.0])
    np.testing.assert_array_equal(sliced["scale"], [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(sliced["min"], [-1.0, -3.0, -4.0])
    np.testing.assert_array_equal(sliced["range"], [5.0, 7.0, 8.0])


def test_axis_slice_fails_closed_on_wrong_hash_missing_or_reordered_genes():
    source = ["A", "B", "C"]
    target = ["A", "C"]
    with pytest.raises(ValueError, match="source gene axis checksum"):
        build_exp045_gene_axis_slice(
            source,
            target,
            expected_source_sha256="0" * 64,
            expected_target_sha256=gene_axis_sha256(target),
        )
    with pytest.raises(ValueError, match="absent from source"):
        build_exp045_gene_axis_slice(
            source,
            ["A", "D"],
            expected_source_sha256=gene_axis_sha256(source),
            expected_target_sha256=gene_axis_sha256(["A", "D"]),
        )
    with pytest.raises(ValueError, match="preserve the source"):
        build_exp045_gene_axis_slice(
            source,
            ["C", "A"],
            expected_source_sha256=gene_axis_sha256(source),
            expected_target_sha256=gene_axis_sha256(["C", "A"]),
        )


def test_weight_and_transform_must_match_the_frozen_source_axis():
    with pytest.raises(ValueError, match="input width"):
        slice_exp045_gene_fc1_weight(torch.zeros(2, 3), _plan())
    with pytest.raises(ValueError, match="does not match"):
        slice_exp045_transform_vectors({"mean": np.zeros(2)}, _plan())


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ((-1, 2), "within the source"),
        ((0, 0), "strictly increasing"),
        ((2, 0), "strictly increasing"),
        ((0, 4), "within the source"),
        ((1, 2), "do not map"),
        ((0,), "length must equal"),
        ((True, 2), "strict integers"),
        ((np.int64(0), 2), "strict integers"),
    ],
)
def test_public_plan_constructor_rejects_invalid_positions(positions, message):
    with pytest.raises(ValueError, match=message):
        _manual_plan(positions=positions)


@pytest.mark.parametrize(
    ("field", "axis", "message"),
    [
        ("source_genes", ["A", "B", "C", "D"], "must be a tuple"),
        ("target_genes", ["A", "C"], "must be a tuple"),
        ("source_genes", ("A", " B", "C", "D"), "normalized"),
        ("target_genes", ("A", " C"), "normalized"),
        ("source_genes", (), "non-empty"),
        ("target_genes", (), "non-empty"),
        ("source_genes", ("A", "A", "C", "D"), "duplicate"),
        ("target_genes", ("A", "A"), "duplicate"),
    ],
)
def test_public_plan_constructor_rejects_noncanonical_axes(field, axis, message):
    with pytest.raises(ValueError, match=message):
        _manual_plan(**{field: axis})


def test_public_plan_constructor_rejects_mutable_positions():
    with pytest.raises(ValueError, match="immutable tuple"):
        _manual_plan(positions=[0, 2])


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_sha256", "source gene axis checksum"),
        ("target_sha256", "target gene axis checksum"),
    ],
)
def test_public_plan_constructor_recomputes_and_checks_axis_hashes(field, message):
    with pytest.raises(ValueError, match=message):
        _manual_plan(**{field: "0" * 64})
