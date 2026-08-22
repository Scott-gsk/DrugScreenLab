"""Shared LINCS landmark-gene and exact-context contracts.

This module contains only response-blind identifiers needed by context/PDO
readiness code. Historical EXP-007 Oracle evaluation code is intentionally not
part of this contract.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable

import pandas as pd


CRC_EXACT_CONTEXTS = (
    "CL34",
    "HCT116",
    "HT29",
    "LOVO",
    "RKO",
    "SNUC4",
    "SNUC5",
    "SW480",
    "SW620",
    "SW948",
)
ORDERED_GENE_IDS_SHA256 = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"


def ordered_landmark_gene_ids(
    gene_info: pd.DataFrame,
    *,
    gctx_row_ids: Iterable[str] | None = None,
) -> list[str]:
    """Return the 978 landmark gene IDs in source GCTX order when supplied."""
    flags = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_is_lm"].astype(str)))
    if gctx_row_ids is None:
        ordered = [gene_id for gene_id, flag in flags.items() if flag == "1"]
    else:
        ordered = [str(gene_id) for gene_id in gctx_row_ids if flags.get(str(gene_id)) == "1"]
    if len(ordered) != 978:
        raise ValueError(f"expected 978 landmark genes, found {len(ordered)}")
    return ordered


def gene_order_digest(gene_ids: Iterable[str]) -> str:
    digest = sha256()
    for gene_id in gene_ids:
        digest.update(str(gene_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
