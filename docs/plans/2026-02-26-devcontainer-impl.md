# DevContainer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a DevContainer that provides a full GPU training environment with persistent AI agent auth (Claude Code, OpenCode) and ML credentials (W&B, HuggingFace).

**Architecture:** Single `.devcontainer/` directory with Dockerfile (based on NGC PyTorch, extending existing `docker/Dockerfile.ngc.vllm`), `devcontainer.json` (GPU runtime, bind mounts for credentials), and `post-create.sh` (editable install + permission fixes).

**Tech Stack:** Docker, VS Code Dev Containers, NVIDIA Container Toolkit, NGC PyTorch 24.05, Node.js 22 LTS

---

### Task 1: Create .devcontainer directory

**Files:**
- Create: `.devcontainer/` (directory)

**Step 1: Create the directory**

```bash
mkdir -p .devcontainer
```

**Step 2: Verify**

```bash
ls -la .devcontainer/
```

Expected: Empty directory exists.

---

### Task 2: Create Dockerfile

**Files:**
- Create: `.devcontainer/Dockerfile`
- Reference: `docker/Dockerfile.ngc.vllm` (reuse dependency logic)

**Step 1: Write the Dockerfile**

```dockerfile
# DevContainer: Full GPU training environment for simpleRL-reason
# Based on docker/Dockerfile.ngc.vllm with dev tools + AI coding agents

FROM nvcr.io/nvidia/pytorch:24.05-py3

# ============================================================
# Layer 1: Clean NGC fork packages, install vanilla PyTorch 2.4.0
# ============================================================
RUN pip3 uninstall -y \
    pytorch-quantization \
    pytorch-triton \
    torch \
    torch-tensorrt \
    torchvision \
    xgboost \
    transformer_engine \
    flash_attn \
    apex \
    megatron-core

RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# Layer 2: Python dependencies
# ============================================================
RUN pip3 install --no-cache-dir \
    "torch==2.4.0" \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    numpy \
    pybind11 \
    tensordict \
    "transformers>=4.51.0" \
    peft \
    liger-kernel \
    word2number \
    "math-verify[antlr4_11_0]>=0.6.0" \
    wandb \
    py-spy

# ============================================================
# Layer 3: vLLM + flash-attn
# ============================================================
RUN pip3 install --no-cache-dir vllm==0.6.3
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2

# ============================================================
# Layer 4: NVIDIA Apex + Transformer Engine (slow, cached)
# ============================================================
RUN MAX_JOBS=4 pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex

# Transformer Engine requires older flash-attn, then we restore
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir --no-build-isolation flash-attn==2.5.8
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir \
    git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# Restore flash-attn to target version
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2

# ============================================================
# Layer 5: System dev tools
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    screen \
    vim \
    curl \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 LTS (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 6: AI coding agents
# ============================================================
RUN npm install -g @anthropic-ai/claude-code

# OpenCode
RUN curl -fsSL https://get.opencode.ai | bash || true

# ============================================================
# Layer 7: Non-root user with sudo
# ============================================================
RUN groupadd -g 1000 vscode \
    && useradd -m -u 1000 -g 1000 -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode

# Set working directory
WORKDIR /workspace

USER vscode
```

**Step 2: Verify Dockerfile syntax**

```bash
docker build --check -f .devcontainer/Dockerfile .
```

If `--check` not available, just verify file exists and has correct content:

```bash
head -5 .devcontainer/Dockerfile
```

Expected: First lines show the FROM instruction.

---

### Task 3: Create devcontainer.json

**Files:**
- Create: `.devcontainer/devcontainer.json`

**Step 1: Write devcontainer.json**

```jsonc
{
  "name": "simpleRL-reason Training Env",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },

  // GPU support — works for both local and remote SSH
  "runArgs": [
    "--gpus", "all",
    "--ipc=host",
    "--ulimit", "memlock=-1",
    "--ulimit", "stack=67108864"
  ],

  // Credential persistence via bind mounts
  "mounts": [
    "source=${localEnv:HOME}/.claude,target=/home/vscode/.claude,type=bind,consistency=cached",
    "source=${localEnv:HOME}/.config/opencode,target=/home/vscode/.config/opencode,type=bind,consistency=cached",
    "source=${localEnv:HOME}/.netrc,target=/home/vscode/.netrc,type=bind,consistency=cached",
    "source=${localEnv:HOME}/.cache/huggingface,target=/home/vscode/.cache/huggingface,type=bind,consistency=cached"
  ],

  "remoteUser": "vscode",
  "containerUser": "vscode",
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached",

  "postCreateCommand": "bash .devcontainer/post-create.sh",

  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3",
        "terminal.integrated.defaultProfile.linux": "bash"
      }
    }
  }
}
```

**Step 2: Verify JSON syntax**

```bash
python3 -c "import json; json.load(open('.devcontainer/devcontainer.json'))"
```

Note: This will fail because of JSONC comments. That's fine — VS Code Dev Containers supports JSONC natively. To validate, just check the file exists:

```bash
cat .devcontainer/devcontainer.json | head -5
```

---

### Task 4: Create post-create.sh

**Files:**
- Create: `.devcontainer/post-create.sh`

**Step 1: Write the script**

```bash
#!/bin/bash
set -e

echo "=== simpleRL-reason: Post-create setup ==="

# Install project in editable mode
echo "Installing verl in editable mode..."
pip install -e /workspace

# Ensure Claude Code config directory and terms file exist
if [ ! -f "$HOME/.claude/.claude.json" ]; then
    mkdir -p "$HOME/.claude"
    echo '{}' > "$HOME/.claude/.claude.json"
    echo "Created Claude Code config stub."
fi

# Fix ownership of bind-mounted directories if created by root
for dir in "$HOME/.claude" "$HOME/.config/opencode" "$HOME/.cache/huggingface"; do
    if [ -d "$dir" ]; then
        sudo find "$dir" -user root -exec chown "$(id -u):$(id -g)" {} + 2>/dev/null || true
    fi
done

# Fix .netrc ownership if needed
if [ -f "$HOME/.netrc" ]; then
    sudo chown "$(id -u):$(id -g)" "$HOME/.netrc" 2>/dev/null || true
    chmod 600 "$HOME/.netrc"
fi

echo "=== Setup complete ==="
echo "Available tools:"
command -v claude && echo "  - Claude Code: $(claude --version 2>/dev/null || echo 'installed')"
command -v opencode && echo "  - OpenCode: $(opencode --version 2>/dev/null || echo 'installed')"
echo "  - Python: $(python3 --version)"
echo "  - PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected (expected if no NVIDIA Container Toolkit)"
```

**Step 2: Make executable**

```bash
chmod +x .devcontainer/post-create.sh
```

**Step 3: Verify**

```bash
ls -la .devcontainer/post-create.sh
```

Expected: File is executable (`-rwxr-xr-x`).

---

### Task 5: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Check if .devcontainer is already in .gitignore**

```bash
grep -n "devcontainer" .gitignore
```

Expected: No matches (we want `.devcontainer/` to be tracked in git).

**Step 2: No changes needed**

The `.devcontainer/` directory should be committed to git so all team members get the same dev environment. No `.gitignore` changes required.

---

### Task 6: Commit

**Step 1: Stage and commit**

```bash
git add .devcontainer/ docs/plans/2026-02-26-devcontainer-design.md docs/plans/2026-02-26-devcontainer-impl.md
git commit -m "feat: add devcontainer with GPU training env and agent auth persistence"
```

---

### Task 7: Verify (manual, post-build)

These steps are for manual verification after building:

**Step 1: Ensure host directories exist**

```bash
mkdir -p ~/.claude ~/.config/opencode ~/.cache/huggingface
touch ~/.netrc
```

**Step 2: Build and open in VS Code**

Open the project in VS Code, press `Ctrl+Shift+P` > "Dev Containers: Reopen in Container".

**Step 3: Verify inside container**

```bash
# Check GPU access
nvidia-smi

# Check agents
claude --version
opencode --version

# Check credentials are mounted
ls -la ~/.claude/
ls -la ~/.config/opencode/
cat ~/.netrc

# Check project is installed
python3 -c "import verl; print(verl.__version__)"

# Check training can start (dry run)
bash experiments/scripts/train.sh --config grpo_qwen3_1.7b --dry_run
```
