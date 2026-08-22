from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.build_exp008_target_pathway_features import build_feature_contract


def _write_fixture_h5ad(path: Path) -> None:
    import anndata as ad

    obs = pd.DataFrame(
        {
            "pert_id": ["BRD-A", "BRD-B", "BRD-A", "BRD-C"],
            "pert_idx": [2, 0, 2, 1],
            "split_cold_drug_1": ["train", "test", "train", "test"],
            "split_cold_cell_1": ["train", "train", "test", "test"],
        }
    )
    var = pd.DataFrame(
        {
            "gene_id": ["101", "102", "103"] + [str(value) for value in range(104, 1079)],
            "gene_symbol": ["A", "B", "C"] + [f"G{value}" for value in range(104, 1079)],
        }
    )
    ad.AnnData(X=np.zeros((4, 978), dtype=np.float32), obs=obs, var=var).write_h5ad(path)


def _write_fixture_crosswalk(tmp_path: Path) -> Path:
    crosswalk = tmp_path / "crosswalk.tsv"
    crosswalk.write_text(
        "accession\tentrez_id\tmapping_status\tambiguous\n"
        "P00001\t101\tmapped\tfalse\n"
        "P00002\t102\tmapped\tfalse\n",
        encoding="utf-8",
    )
    return crosswalk


def _write_fixture_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    chembl = tmp_path / "chembl.sqlite"
    import sqlite3

    with sqlite3.connect(chembl) as connection:
        connection.executescript(
            """
            CREATE TABLE drug_identity (pert_id TEXT PRIMARY KEY, chembl_id TEXT);
            CREATE TABLE direct_human_single_protein (
                chembl_id TEXT, target_accession TEXT, direct_interaction INTEGER,
                organism TEXT, target_type TEXT);
            CREATE TABLE accession_to_entrez (accession TEXT PRIMARY KEY, entrez_id TEXT);
            INSERT INTO drug_identity VALUES ('BRD-A', 'CHEMBL1');
            INSERT INTO direct_human_single_protein VALUES
              ('CHEMBL1', 'P00001', 1, 'Homo sapiens', 'SINGLE PROTEIN');
            INSERT INTO accession_to_entrez VALUES ('P00001', '101');
            """
        )
    string_path = tmp_path / "9606.protein.links.v12.0.txt"
    string_path.write_text("protein1 protein2 combined_score\n9606.ENSP1 9606.ENSP2 900\n", encoding="utf-8")
    (tmp_path / "9606.protein.physical.links.v12.0.txt").write_text(
        "protein1 protein2 combined_score\n9606.ENSP1 9606.ENSP2 900\n", encoding="utf-8"
    )
    aliases_path = tmp_path / "9606.protein.aliases.v12.0.txt"
    aliases_path.write_text(
        "#string_protein_id\talias\tsource\n"
        "9606.ENSP1\t101\tEnsembl_EntrezGene\n"
        "9606.ENSP2\t102\tEnsembl_EntrezGene\n",
        encoding="utf-8",
    )
    gmt_path = tmp_path / "go_bp_human.gmt"
    gmt_path.write_text("GO_TERM\tGO:0000001\t101\t102\n", encoding="utf-8")
    return chembl, string_path, gmt_path


def test_build_preserves_unique_sdst_drug_index_and_frozen_978_axis(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "sdst.h5ad"
    _write_fixture_h5ad(h5ad_path)
    chembl, string_path, gmt_path = _write_fixture_sources(tmp_path)

    built = build_feature_contract(
        h5ad_path=h5ad_path,
        chembl_path=chembl,
        string_path=string_path,
        go_bp_gmt_path=gmt_path,
        output_dir=tmp_path / "out",
        crosswalk_path=_write_fixture_crosswalk(tmp_path),
    )

    matrix = np.load(tmp_path / "out" / "target_pathway_features.npy")
    contract = json.loads((tmp_path / "out" / "MECHANISM_CONTRACT.json").read_text(encoding="utf-8"))
    assert matrix.shape[0] == 3
    assert contract["drug_index"]["ordered_pert_idx"] == [0, 1, 2]
    assert contract["gene_axis"]["entrez_ids"][:3] == ["101", "102", "103"]
    assert len(contract["gene_axis"]["entrez_ids"]) == 978
    # 该 3 药物 fixture 的完整特征覆盖率为 1/3，按真实 30% 门槛可 READY；
    # 它仅验证构建语义，不能代表正式 SDST 训练放行。
    assert built["status"] == "READY_FOR_TRAINING"
    assert matrix[2, 0] == 1.0
    assert matrix[2, 978 + 1] == 1.0
    assert contract["string"]["status"] == "OK"
    assert contract["go_bp"]["summary_nonzero_drugs"] == 1


def test_conflicting_pert_idx_is_rejected_before_deduplication(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "sdst.h5ad"
    _write_fixture_h5ad(h5ad_path)
    import anndata as ad

    sdst = ad.read_h5ad(h5ad_path)
    # AnnData stores this fixture column as categorical; widen it before
    # injecting a new identity so the test reaches the builder's conflict check.
    sdst.obs["pert_id"] = sdst.obs["pert_id"].astype(object)
    sdst.obs.iloc[2, sdst.obs.columns.get_loc("pert_id")] = "BRD-CONFLICT"
    sdst.write_h5ad(h5ad_path)
    chembl, string_path, gmt_path = _write_fixture_sources(tmp_path)

    import pytest
    with pytest.raises(ValueError, match="multiple pert_id values share"):
        build_feature_contract(
            h5ad_path=h5ad_path,
            chembl_path=chembl,
            string_path=string_path,
            go_bp_gmt_path=gmt_path,
            output_dir=tmp_path / "out",
            crosswalk_path=_write_fixture_crosswalk(tmp_path),
        )


def test_unmapped_drug_is_retained_as_all_zero_vector(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "sdst.h5ad"
    _write_fixture_h5ad(h5ad_path)
    chembl, string_path, gmt_path = _write_fixture_sources(tmp_path)

    build_feature_contract(
        h5ad_path=h5ad_path,
        chembl_path=chembl,
        string_path=string_path,
        go_bp_gmt_path=gmt_path,
        output_dir=tmp_path / "out",
        crosswalk_path=_write_fixture_crosswalk(tmp_path),
    )

    matrix = np.load(tmp_path / "out" / "target_pathway_features.npy")
    assert np.array_equal(matrix[0], np.zeros(matrix.shape[1], dtype=matrix.dtype))
    assert np.array_equal(matrix[1], np.zeros(matrix.shape[1], dtype=matrix.dtype))


def test_string_aliases_without_qualified_neighbor_are_blocked(tmp_path: Path) -> None:
    from scripts.data.build_exp008_target_pathway_features import _string_neighbors

    string_path = tmp_path / "9606.protein.links.v12.0.txt"
    (tmp_path / "9606.protein.physical.links.v12.0.txt").write_text(
        "protein1 protein2 combined_score\n9606.ENSP1 9606.ENSP2 600\n", encoding="utf-8"
    )
    (tmp_path / "9606.protein.aliases.v12.0.txt").write_text(
        "#string_protein_id\talias\tsource\n"
        "9606.ENSP1\t101\tEnsembl_EntrezGene\n"
        "9606.ENSP2\t102\tEnsembl_EntrezGene\n", encoding="utf-8"
    )
    neighbors, audit = _string_neighbors(string_path, {"101"}, {"101", "102"})
    assert neighbors == {"101": set()}
    assert audit["status"] == "NOT_USED_DATA_BLOCKED"


def test_missing_go_bp_is_audited_data_blocked(tmp_path: Path) -> None:
    from scripts.data.build_exp008_target_pathway_features import _go_bp_summary

    _, terms, audit = _go_bp_summary(tmp_path / "missing.gmt", {"BRD-A": {"101"}}, {"101"})
    assert terms == []
    assert audit["status"] == "DATA_BLOCKED"
    assert "unavailable or unreadable" in audit["reason"]


def test_unique_crosswalk_excludes_ambiguous_accessions(tmp_path: Path) -> None:
    from scripts.data.build_exp008_target_pathway_features import _load_accession_crosswalk

    crosswalk = tmp_path / "crosswalk.tsv"
    crosswalk.write_text(
        "accession\tentrez_id\tmapping_status\tambiguous\n"
        "P00001\t101\tmapped\tfalse\n"
        "P00002\t102\tmapped\ttrue\n"
        "P00003\t103\tmissing\tfalse\n",
        encoding="utf-8",
    )
    mapped, audit = _load_accession_crosswalk(crosswalk)
    assert mapped == {"P00001": "101"}
    assert audit["ambiguous_rows"] == 1
    assert audit["excluded_accessions"] == ["P00002", "P00003"]


def test_non_singleton_mapped_accessions_are_excluded_from_audit(tmp_path: Path) -> None:
    from scripts.data.build_exp008_target_pathway_features import _load_accession_crosswalk

    crosswalk = tmp_path / "crosswalk.tsv"
    crosswalk.write_text(
        "accession\tentrez_id\tmapping_status\tambiguous\n"
        "P00001\t101\tmapped\tfalse\n"
        "P00001\t102\tmapped\tfalse\n",
        encoding="utf-8",
    )

    mapped, audit = _load_accession_crosswalk(crosswalk)

    assert mapped == {}
    assert audit["excluded_accessions"] == ["P00001"]


def test_repeated_build_is_deterministic_and_does_not_depend_on_efficacy_paths(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "sdst.h5ad"
    _write_fixture_h5ad(h5ad_path)
    chembl, string_path, gmt_path = _write_fixture_sources(tmp_path)

    first = build_feature_contract(
        h5ad_path=h5ad_path,
        chembl_path=chembl,
        string_path=string_path,
        go_bp_gmt_path=gmt_path,
        output_dir=tmp_path / "first",
        crosswalk_path=_write_fixture_crosswalk(tmp_path),
    )
    second = build_feature_contract(
        h5ad_path=h5ad_path,
        chembl_path=chembl,
        string_path=string_path,
        go_bp_gmt_path=gmt_path,
        output_dir=tmp_path / "second",
        crosswalk_path=_write_fixture_crosswalk(tmp_path),
    )

    assert first["output_sha256"] == second["output_sha256"]
    assert first["forbidden_data"]["efficacy_data_read"] is False
    import ast

    source = Path(build_feature_contract.__code__.co_filename).read_text(encoding="utf-8")
    imports = [
        node.names[0].name.lower()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import) and node.names
    ]
    assert all("prism" not in module for module in imports)
