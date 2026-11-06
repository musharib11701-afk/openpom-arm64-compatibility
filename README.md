# OpenPOM ARM64 compatibility harness

This repository is an isolated, public-safe compatibility test. It contains only a GitHub Actions workflow, pinned package requirements, and a small smoke-test script. It does **not** contain OmniAgent/Nexus source code, credentials, `.env` files, OpenPOM model weights, or private data.


> **Runtime support note:** Python 3.10 reached end of life on October 1, 2026. This workflow deliberately reproduces the existing OpenPOM environment for compatibility testing only. Do not use Python 3.10 as a new production runtime. We must test a supported Python version before considering permanent hosting.

## Purpose

Run the existing OpenPOM dependency versions on a native GitHub-hosted Linux ARM64 runner and check whether the stack can install and execute basic CPU operations:

- Python 3.10
- PyTorch 2.4.0 (CPU)
- DGL 2.4.0 (built from the official `v2.4.0` source tag)
- RDKit 2026.3.6
- DGL-LifeSci 0.3.2
- DeepChem 2.8.1.dev20260821122258
- NumPy 1.26.4

The workflow uses `ubuntu-24.04-arm`. GitHub documents this public-repository standard runner as a native ARM64 VM with 4 vCPUs and 16 GB RAM. Standard hosted runner usage is free for public repositories. This is a temporary CI machine, **not** a continuously running API host.

## Run

1. Create an empty **public** GitHub repository named `openpom-arm64-compatibility`.
2. Extract this archive.
3. From PowerShell, enter the extracted `openpom-arm64-compatibility` folder, initialize Git, and push the files to that new repository. Do not place this folder inside the OmniAgent project.
4. Open the repository's **Actions** tab and select **ARM64 OpenPOM stack compatibility**. The first push to `main` also starts the workflow automatically. Later runs can be started with **Run workflow**.

The workflow has read-only repository permissions, does not use repository secrets, and does not publish or deploy anything.

## What a pass means

A successful run establishes that the pinned dependencies import and perform basic PyTorch CPU operations, DGL graph operations, RDKit molecule parsing, DGL-LifeSci graph featurization, and a DeepChem fingerprint operation on ARM64.

It does **not** yet prove that the full OpenPOM trained ensemble or its checkpoints run on ARM64. That is a separate follow-up test after the dependency build passes and an official, permitted source for the model weights is verified. Do not upload model weights to this repository.

## Safety and scope

- Never add API keys, tokens, passwords, `.env` files, personal information, or proprietary application files.
- Never upload `.pt`, `.pth`, `.ckpt`, or other model-weight files.
- Do not connect this repository to Vercel or any production deployment.
- No cloud server is provisioned by this test.
