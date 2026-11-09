#!/usr/bin/env python3
"""Run the official 10-checkpoint OpenPOM ensemble on ARM64 CPU."""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import deepchem as dc
import numpy as np
import psutil
import torch

UPSTREAM_URL = "https://github.com/BioMachineLearning/openpom.git"
UPSTREAM_COMMIT = "5434c4b7badebca8e7d180aa34d2f787402121ee"
ROOT = Path(os.environ.get("OPENPOM_SOURCE_DIR", "/tmp/openpom-upstream"))
MODEL_ROOT = ROOT / "models" / "ensemble_models"
EXPECTED_MODELS = 10
EXPECTED_TASKS = 138
SMILES = ["CCO", "c1ccccc1", "COc1cc(C=O)ccc1O"]


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def extract_tasks(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASKS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return value
    raise RuntimeError("Could not safely extract TASKS list from upstream predict_odors.py")


def ensure_upstream() -> None:
    # Avoid downloading every LFS object during clone/checkout. Only the ten
    # checkpoint2.pt files are fetched explicitly below.
    if not ROOT.exists():
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "clone", UPSTREAM_URL, str(ROOT)], env=clone_env)

    run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=ROOT)
    run(["git", "lfs", "install", "--local", "--skip-smudge"], cwd=ROOT)
    run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=ROOT)
    run(
        [
            "git",
            "lfs",
            "pull",
            "--include=models/ensemble_models/experiments_*/checkpoint2.pt",
        ],
        cwd=ROOT,
    )


def main() -> None:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise RuntimeError(f"Expected native ARM64, got {platform.machine()}")
    if torch.cuda.is_available():
        raise RuntimeError("This test must run with CPU-only PyTorch")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may already have initialized its inter-op pool.
        pass

    ensure_upstream()
    source_file = ROOT / "predict_odors.py"
    tasks = extract_tasks(source_file)
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} odor tasks, got {len(tasks)}")

    # Import the pinned upstream source directly; this avoids invoking setup.py
    # or allowing it to change the already-validated environment.
    sys.path.insert(0, str(ROOT))
    from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
    from openpom.models.mpnn_pom import MPNNPOMModel

    featurized = GraphFeaturizer().featurize(SMILES)
    if len(featurized) != len(SMILES):
        raise RuntimeError("Featurizer did not return one graph per SMILES")

    dataset = dc.data.NumpyDataset(featurized)
    predictions: list[np.ndarray] = []
    checkpoint_report: list[dict[str, object]] = []
    process = psutil.Process()
    start_all = time.perf_counter()

    for index in range(1, EXPECTED_MODELS + 1):
        checkpoint = MODEL_ROOT / f"experiments_{index}" / "checkpoint2.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Missing or unresolved Git LFS checkpoint: {checkpoint}"
            )

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = MPNNPOMModel(
            n_tasks=len(tasks),
            batch_size=128,
            class_imbalance_ratio=None,
            loss_aggr_type="sum",
            learning_rate=0.001,
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type="sum",
            mode="classification",
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type="set2set",
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation="relu",
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name="adam",
            log_frequency=32,
            device_name="cpu",
            model_dir=str(checkpoint.parent),
        )
        model.restore(str(checkpoint))
        model.model.eval()
        parameter_device = next(model.model.parameters()).device.type
        if parameter_device != "cpu":
            raise RuntimeError(f"Model {index} loaded onto {parameter_device}, not CPU")

        started = time.perf_counter()
        pred = np.asarray(model.predict(dataset))
        elapsed = time.perf_counter() - started
        expected_shape = (len(SMILES), len(tasks))
        if pred.shape != expected_shape:
            raise RuntimeError(
                f"Model {index} returned shape {pred.shape}; expected {expected_shape}"
            )
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Model {index} produced NaN or infinite values")

        predictions.append(pred)
        checkpoint_report.append(
            {
                "model": index,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest,
                "inference_seconds": round(elapsed, 4),
            }
        )
        print(f"Model {index}/{EXPECTED_MODELS}: PASS; {elapsed:.3f}s", flush=True)
        del model
        gc.collect()

    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    if ensemble.shape != (len(SMILES), EXPECTED_TASKS) or not np.isfinite(ensemble).all():
        raise RuntimeError("Ensemble output failed shape/finite-value validation")

    top_results = []
    for row, smiles in enumerate(SMILES):
        top_indices = np.argsort(ensemble[row])[::-1][:10]
        top_results.append(
            {
                "smiles": smiles,
                "top_10": [
                    {"odor": tasks[int(i)], "score": float(ensemble[row, i])}
                    for i in top_indices
                ],
            }
        )

    import dgl
    import rdkit

    report = {
        "status": "PASS",
        "upstream_repository": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "architecture": platform.machine(),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dgl": dgl.__version__,
        "rdkit": rdkit.__version__,
        "deepchem": getattr(dc, "__version__", "unknown"),
        "models_loaded": len(checkpoint_report),
        "tasks": len(tasks),
        "smiles_tested": len(SMILES),
        "total_inference_seconds": round(time.perf_counter() - start_all, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "current_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "checkpoints": checkpoint_report,
        "predictions": top_results,
    }
    print(json.dumps(report, indent=2), flush=True)
    Path("openpom-arm64-inference-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("ALL 10 OFFICIAL OPENPOM CHECKPOINTS PASSED CPU INFERENCE", flush=True)


if __name__ == "__main__":
    main()
