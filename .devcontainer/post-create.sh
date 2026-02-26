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
command -v claude && echo "  - Claude Code: $(claude --version 2>/dev/null || echo 'installed')" || true
command -v opencode && echo "  - OpenCode: $(opencode --version 2>/dev/null || echo 'installed')" || true
echo "  - Python: $(python3 --version)"
echo "  - PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected (expected if no NVIDIA Container Toolkit)"
