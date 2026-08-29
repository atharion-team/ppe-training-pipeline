# PPE Training Pipeline

This is a training pipeline to support the development of a vision system that turns a construction site camera feed into a stream of **PPE compliance events**: a finetuned YOLO detector finds workers and safety gear, a multi-object tracker holds each worker's identity, and an association + compliance layer emits one deduplicated event per violation.

For the design, diagrams, and open decisions, run the docs site (`make serve`, then <http://localhost:8000>).

> **Early stage.** The environment and docs are set up; the pipeline itself is not built yet. See the docs for intended design and [open decisions](docs/decisions/README.md).

## Prerequisites

- **Python 3.11** specifically. The pinned PyTorch wheel (`torch==2.5.1+cu121`) is built for 3.11; newer Python versions may have no matching wheel
- **Windows** (the only platform tested so far)
- **Git** to clone the repo, and required by the docs revision-date and author plugins
- **Docker Desktop** to serve or build the docs
- **GNU Make**, used by the `make` targets, not installed on Windows by default. Install it (`winget install GnuWin32.Make` or `choco install make`) or run the underlying `docker compose` commands directly (see [Docs](#docs))
- Optional: an **NVIDIA GPU with a CUDA driver** for training. CPU works but is far slower

## Setup

First clone the repository. Then, on Windows (PowerShell):

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script with an execution-policy error, run this once, then activate again: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins a **CUDA 12.1** PyTorch build (`torch==2.5.1+cu121`), change accordingly.

> **The `cu121` build matches my driver. Do not install it blindly.** Your machine may expose a different CUDA version, check yours and install the matching wheel instead.

Check which CUDA version your driver exposes:

```powershell
nvidia-smi
```

Then install the matching PyTorch wheels (adjust `cu121` to match, common values are `cu118`, `cu121`, `cu124`, `cu126`, `cu128`):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

No NVIDIA GPU? Install the CPU build instead:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Use the official selector at <https://pytorch.org/get-started/locally/> if you are unsure which build to pick.

### 3. Verify the installation

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expect `CUDA available: True` and your GPU name. Use `--device 0` in any script to target the first GPU, or `--device cpu` to force CPU regardless of what is available.

## Docs

```powershell
make serve   # live docs at http://localhost:8000
make docs    # build the static site into site/
make help    # list targets
```

No `make` installed? Run the same commands through Docker directly:

```powershell
docker compose -f docker/docker-compose.yml up               # serve at http://localhost:8000
docker compose -f docker/docker-compose.yml run --rm docs build   # build into site/
```
