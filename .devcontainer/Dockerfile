# DevContainer: Full GPU training environment for simpleRL-reason
# Based on docker/Dockerfile.ngc.vllm with dev tools + AI coding agents
#
# Build time: ~30-60 min (first build), fast rebuilds via Docker cache
# Image size: ~25-30 GB (NGC base is ~20 GB)

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

# Transformer Engine requires older flash-attn for build
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir --no-build-isolation flash-attn==2.5.8
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir \
    git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# Restore flash-attn to target version after TE build
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

# Node.js 22 LTS (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 6: AI coding agents
# ============================================================
# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# OpenCode
RUN curl -fsSL https://get.opencode.ai | bash || true

# ============================================================
# Layer 7: Non-root user with sudo
# ============================================================
RUN groupadd -g 1000 vscode \
    && useradd -m -u 1000 -g 1000 -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode

WORKDIR /workspace

USER vscode
