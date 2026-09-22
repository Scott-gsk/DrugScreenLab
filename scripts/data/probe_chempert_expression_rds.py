#!/usr/bin/env python3
"""Read-only structural probe of the ChemPert expression RDS (S4 object)."""

from __future__ import annotations

import rdata
from rdata.parser._parser import RObjectType

RDS = "data/raw/resource_acquisition_20260916/improve_chempert/chempert_gene_expression_profiles.rds"


def sym_name(tag) -> str | None:
    if tag is None:
        return None
    try:
        v = tag.value
        while hasattr(v, "value"):
            v = v.value
        if isinstance(v, bytes):
            return v.decode()
        return str(v)
    except Exception:
        return "?"


def iter_pairlist(node):
    """Yield (tag_name, car) for each node of an R pairlist."""
    while node is not None and node.info.type == RObjectType.LIST:
        car, cdr = node.value[0], node.value[1]
        yield sym_name(node.tag), car
        node = cdr


def walk(obj, label: str, depth: int, max_depth: int) -> None:
    pad = "  " * depth
    t = obj.info.type.name
    desc = ""
    if t == "REAL":
        try:
            arr = obj.value
            desc = f" len={len(arr)} first={arr[:3]}"
        except Exception:
            pass
    elif t == "INT":
        try:
            arr = obj.value
            desc = f" len={len(arr)} first={list(arr[:3])}"
        except Exception:
            pass
    elif t == "CHAR":
        desc = f" value={obj.value!r}"
    print(f"{pad}{label}: {t}{desc}")
    if depth >= max_depth:
        return
    if t == "LIST":
        for tag, car in iter_pairlist(obj):
            walk(car, f"[{tag}]", depth + 1, max_depth)
    elif obj.attributes is not None:
        for tag, car in iter_pairlist(obj.attributes):
            walk(car, f"@{tag}", depth + 1, max_depth)


def main() -> None:
    parsed = rdata.parser.parse_file(RDS)
    top = parsed.object
    print(f"top: {top.info.type.name}")
    attrs = top.attributes
    print(f"attrs.tag of S4: {sym_name(attrs.tag) if attrs else None}")
    print("--- structure (depth<=6) ---")
    walk(top, "root", 0, 6)
    print("--- hunting REAL/INT arrays with Dim ---")
    found = []

    def hunt(obj, path, depth=0):
        if depth > 10:
            return
        if obj.attributes is not None:
            for tag, car in iter_pairlist(obj.attributes):
                if tag == "Dim":
                    try:
                        dim = [int(x) for x in car.value]
                    except Exception:
                        dim = None
                    found.append((path, obj.info.type.name, dim))
                    print(f"ARRAY {path}: {obj.info.type.name} dim={dim}")
        if obj.info.type == RObjectType.LIST:
            for tag, car in iter_pairlist(obj):
                hunt(car, f"{path}[{tag}]", depth + 1)
        elif obj.attributes is not None:
            for tag, car in iter_pairlist(obj.attributes):
                hunt(car, f"{path}@{tag}", depth + 1)

    hunt(top, "root")
    print("done")


if __name__ == "__main__":
    main()
