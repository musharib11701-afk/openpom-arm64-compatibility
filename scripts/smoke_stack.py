#!/usr/bin/env python3
"""Smoke-test the pinned OpenPOM dependency stack on native Linux ARM64."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise RuntimeError(f"Expected native ARM64, got {machine!r}")

    import numpy as np
    import torch
    import dgl
    import deepchem as dc
    from dgllife.utils import CanonicalAtomFeaturizer, smiles_to_bigraph
    from rdkit import Chem, rdBase

    # Confirm the core numeric backend executes on CPU.
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    product = a @ a
    if product.shape != (2, 2) or not torch.isfinite(product).all().item():
        raise RuntimeError("PyTorch CPU tensor operation failed")

    # Exercise DGL's native graph operations with the PyTorch backend.
    graph = dgl.graph(
        (torch.tensor([0, 1], dtype=torch.int64), torch.tensor([1, 2], dtype=torch.int64)),
        num_nodes=3,
    )
    graph = dgl.add_self_loop(graph)
    if graph.num_nodes() != 3 or graph.num_edges() != 5:
        raise RuntimeError(
            f"Unexpected DGL graph dimensions: nodes={graph.num_nodes()}, edges={graph.num_edges()}"
        )

    # Parse ethanol with RDKit and construct a DGL-LifeSci molecular graph.
    molecule = Chem.MolFromSmiles("CCO")
    if molecule is None or molecule.GetNumAtoms() != 3:
        raise RuntimeError("RDKit failed to parse ethanol SMILES CCO")

    molecular_graph = smiles_to_bigraph(
        "CCO",
        add_self_loop=True,
        node_featurizer=CanonicalAtomFeaturizer(),
    )
    if molecular_graph.num_nodes() != 3:
        raise RuntimeError("DGL-LifeSci molecular graph has an unexpected node count")
    if "h" not in molecular_graph.ndata:
        raise RuntimeError("DGL-LifeSci did not produce the expected atom feature field 'h'")
    if molecular_graph.ndata["h"].shape[0] != 3:
        raise RuntimeError("DGL-LifeSci atom feature row count is incorrect")

    # Exercise DeepChem's RDKit-backed fingerprint featurizer.
    fingerprint = dc.feat.CircularFingerprint(size=32, radius=2)
    features = fingerprint.featurize(["CCO"])
    if features.shape[0] != 1 or features.shape[1] != 32:
        raise RuntimeError(f"Unexpected DeepChem fingerprint shape: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("DeepChem fingerprint contains non-finite values")

    report = {
        "status": "PASS",
        "architecture": machine,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "dgl": dgl.__version__,
        "rdkit_distribution": distribution_version("rdkit"),
        "rdkit_runtime": rdBase.rdkitVersion,
        "dgllife": distribution_version("dgllife"),
        "deepchem": dc.__version__,
        "numpy": np.__version__,
        "checks": [
            "PyTorch CPU matrix multiplication",
            "DGL graph construction and self-loops",
            "RDKit ethanol SMILES parsing",
            "DGL-LifeSci molecular graph and atom features",
            "DeepChem circular fingerprint featurization",
        ],
    }
    print(json.dumps(report, indent=2))

    expected = {
        "torch": "2.4.0",
        "dgl": "2.4.0",
        "rdkit": "2026.3.6",
        "dgllife": "0.3.2",
        "deepchem": "2.8.1.dev20260821122258",
        "numpy": "1.26.4",
    }
    actual = {
        "torch": torch.__version__.split("+")[0],
        "dgl": dgl.__version__,
        "rdkit": distribution_version("rdkit"),
        "dgllife": distribution_version("dgllife"),
        "deepchem": dc.__version__,
        "numpy": np.__version__,
    }
    mismatches = {key: {"expected": value, "actual": actual[key]} for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError("Version mismatch: " + json.dumps(mismatches, sort_keys=True))

    print("\nALL ARM64 DEPENDENCY SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # Keep a clear failure reason in the Actions log.
        print(f"ARM64 SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
