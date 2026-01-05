# Installation

This page shows how to download and install the packaged version of SimWorld. The package includes the executable file of SimWorld server and the Python client library.

## Before you begin
The following device requirements should be fulfilled before installing SimWorld:

| Category | Minimum | Recommended |
|---|---|---|
| OS | Windows / Linux | Windows / Linux |
| GPU | ≥ 6 GB VRAM (dedicated GPU recommended) | ≥ 8 GB VRAM (dedicated GPU strongly recommended) |
| Memory (RAM) | 32 GB | 32 GB+ |
| Disk Space | 50 GB (Base version) | 200 GB+ (Additional environments / 100+ maps) |

| We use RTX 4090 with 24 GB VRAM for demo video recording, and hardware raytracing is enabled for enhanced visual fidelity. |

**Note:**
1. Python: Python 3.10 or later is required.
2. Network: A stable internet connection and two available TCP ports (9000 and 9001 by default) are required.

## Installation
### SimWorld Python Client Library
SimWorld Python Client contains code for SimWorld's agent and environment layer. Download the Python library from GitHub:

[SimWorld Python Client Library](https://github.com/SimWorld-AI/SimWorld.git)

```bash
git clone https://github.com/SimWorld-AI/SimWorld.git
cd SimWorld

# install simworld
conda create -n simworld python=3.10
conda activate simworld
pip install -e .
```


### Unreal Engine Backend
Download the SimWorld Unreal Engine backend executable from AWS S3, choose the version according to your OS and the edition you want to use, the usage of the additional environments version can be shown in the [Additional Environments](../getting_started/additional_environments) page.

| Platform | Package | Scenes/Maps Included | Download | Notes |
| --- | --- | --- | --- | --- |
| Windows | Base | Foundation procedural generated city scenes | [Download (Base)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Win64-v0_1_0-Foundation.zip) | Full agent features; smaller download. |
| Windows | Additional Environment | 100+ maps (includes Base) | [Download (100+ Maps)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Win64-v0_1_0-100Maps.zip) | Full agent features; larger download. |
| Linux | Base | Foundation procedural generated city scenes | [Download (Base)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-Foundation.zip) | Full agent features; smaller download. |
| Linux | Additional Environment | 100+ maps (includes Base) | [Download (100+ Maps)](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-100Maps.zip) | Full agent features; larger download. |

**Note:**
If you only need core functionality for development or testing, use **Base**. If you want richer demonstrations and more scenes, use the **Environments Pack (100+ Maps)**.
