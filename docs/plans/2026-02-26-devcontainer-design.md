# DevContainer Design for simpleRL-reason

**Date:** 2026-02-26
**Status:** Approved

## Summary

Create a DevContainer configuration that provides a complete GPU-enabled training
environment with persistent AI coding agent authentication (Claude Code, OpenCode)
and ML platform credentials (W&B, HuggingFace).

## Requirements

| Dimension | Requirement |
|-----------|-------------|
| Runtime environment | Local GPU + Remote SSH (both supported) |
| Base image | NGC PyTorch (`nvcr.io/nvidia/pytorch:24.05-py3`) |
| Environment scope | Full training env (flash-attn, vllm, apex, Transformer Engine) |
| Agent persistence | Claude Code (`~/.claude`) + OpenCode (`~/.config/opencode`) |
| Extra credentials | W&B (`~/.netrc`) + HuggingFace (`~/.cache/huggingface`) |

## Approach

**Single Dockerfile + devcontainer.json** (Option A, selected over docker-compose
and pre-built image approaches for simplicity and compatibility).

## File Structure

```
.devcontainer/
├── devcontainer.json       # Main config: GPU runtime, mounts, extensions
├── Dockerfile              # Full training env based on NGC PyTorch
└── post-create.sh          # Post-creation init: pip install -e ., permissions
```

## Dockerfile Layers

1. **Base:** `nvcr.io/nvidia/pytorch:24.05-py3`
2. **Clean NGC fork + install PyTorch 2.4.0** (reuse Dockerfile.ngc.vllm logic)
3. **pip dependencies:** accelerate, datasets, hydra-core, tensordict, transformers, etc.
4. **vllm 0.6.3 + flash-attn 2.7.0.post2**
5. **Apex + Transformer Engine v1.7** (compiled from source, slowest layer, cached)
6. **Dev tools:** git, tmux, screen, vim, curl, jq, Node.js 22 LTS
7. **AI Agents:** Claude Code (npm), OpenCode
8. **User setup:** non-root `vscode` user with sudo

## devcontainer.json Key Config

### GPU Support

```jsonc
"runArgs": [
  "--gpus", "all",
  "--ipc=host",
  "--ulimit", "memlock=-1",
  "--ulimit", "stack=67108864"
]
```

### Credential Persistence (Bind Mounts)

```jsonc
"mounts": [
  "source=${localEnv:HOME}/.claude,target=/home/vscode/.claude,type=bind,consistency=cached",
  "source=${localEnv:HOME}/.config/opencode,target=/home/vscode/.config/opencode,type=bind,consistency=cached",
  "source=${localEnv:HOME}/.netrc,target=/home/vscode/.netrc,type=bind,consistency=cached",
  "source=${localEnv:HOME}/.cache/huggingface,target=/home/vscode/.cache/huggingface,type=bind,consistency=cached"
]
```

### Post-Create Script

- `pip install -e .` (editable install of verl)
- Create Claude Code terms file if missing
- Fix ownership of bind-mounted directories (Docker may create as root)

## Local vs Remote SSH Compatibility

| Aspect | Local | Remote SSH |
|--------|-------|------------|
| `--gpus all` | Needs local NVIDIA Container Toolkit | Needs remote NVIDIA Container Toolkit |
| Bind mounts | `${localEnv:HOME}` = local home | `${localEnv:HOME}` = remote home |
| OAuth browser redirect | Direct browser popup | Requires prior login on remote or port forwarding |

## Security

- Bind mounts keep tokens on host, never in image layers
- `.gitignore` already ignores `.env`
- Mount targets are hidden dirs under user home, not in project tree
- First-time users must `mkdir -p` host dirs before container build

## Prerequisites for Users

```bash
# On host machine (local or remote), before first use:
mkdir -p ~/.claude ~/.config/opencode ~/.cache/huggingface
# Also ensure ~/.netrc exists if using W&B
touch ~/.netrc
```
