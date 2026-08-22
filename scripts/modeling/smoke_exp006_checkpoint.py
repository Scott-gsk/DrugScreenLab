"""CPU smoke: B inherits official XPert champion plus uninitialized adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

from drug_screen.foundation.exp006_transfer import build_xpert_genetic_transfer_model
from drug_screen.foundation.xpert_extension import load_xpert_checkpoint
from models.model_XPert import XPertNet


def main() -> int:
    source = Path("data/external/xpert_source")
    config = yaml.safe_load((source / "configs" / "config_l1000_foundation_bounded.yaml").read_text())
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(source / "processed_data" / "PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(
        source / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy"
    )
    args = SimpleNamespace(
        mode="train",
        dataset="l1000_sdst",
        drug_feat="unimol",
        device="cpu",
        pretrained_mode="global",
        include_cell_idx=True,
        wo_HG=False,
        wo_atom=False,
        wo_atom_HG=False,
        wo_unimol=False,
        wo_ppi=False,
        use_gene_pos_emed=False,
        output_attention=False,
        output_cls_embed=False,
    )
    logger = logging.getLogger("exp006.smoke")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    model = build_xpert_genetic_transfer_model(XPertNet)(args, config, "cpu", logger)
    model.init_weights()
    audit = load_xpert_checkpoint(
        model,
        source / "saved_model" / "l1000_sdst_warm_split.pth",
        map_location="cpu",
    )
    adapter_keys = [key for key in model.state_dict() if key.startswith("genetic_adapter.")]
    payload = {
        "official_checkpoint_loaded": bool(model.official_checkpoint_loaded),
        "missing_official": audit["missing_official"],
        "missing_extension": audit["missing_extension"],
        "adapter_parameter_count": int(len(adapter_keys)),
        "adapter_keys": adapter_keys,
    }
    print(json.dumps(payload, sort_keys=True))
    if audit["missing_official"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
