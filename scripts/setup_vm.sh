#!/bin/bash
# Setup script for fresh Ubuntu VM to run training experiments
# Assumes CUDA drivers are already installed

set -e

echo "=== Updating package lists ==="
sudo apt-get update

echo "=== Installing system dependencies ==="
# Build essentials and git
sudo apt-get install -y build-essential git curl

# Audio libraries for librosa
sudo apt-get install -y libsndfile1 ffmpeg

# Optional: useful for debugging audio issues
sudo apt-get install -y sox libsox-dev

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

echo "=== Verifying installation ==="
uv --version

echo "=== Setup complete ==="
echo "Run 'source ~/.bashrc' or start a new shell to use uv"
echo "Then: uv sync --group cuda"
