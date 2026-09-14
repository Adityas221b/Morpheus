#!/bin/bash
# Server setup script for VLM jailbreak experiments
# Run on 80GB VRAM server (172.24.16.132)

set -e

echo "=== VLM Jailbreak Server Setup ==="

# Check GPU
echo "Checking GPU..."
nvidia-smi || { echo "ERROR: nvidia-smi not found. Is this a GPU server?"; exit 1; }

# Check Python
echo "Checking Python..."
python3 --version || { echo "ERROR: python3 not found"; exit 1; }

# Check CUDA
echo "Checking CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || {
    echo "PyTorch not found. Installing..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

# Clone repo (if not already present)
if [ ! -d "AutoDAN-LLL" ]; then
    echo "Cloning Morpheus repo..."
    git clone https://github.com/Adityas221b/Morpheus.git AutoDAN-LLL
fi

cd AutoDAN-LLL

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install bitsandbytes accelerate transformers sentencepiece protobuf pillow pyyaml numpy

# Install Ollama (for attacker model)
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama and pull attacker model
echo "Starting Ollama and pulling attacker model..."
ollama serve &
sleep 3
ollama pull qwen3:1.7b

# Verify setup
echo ""
echo "=== Setup Complete ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else '')
"

echo ""
echo "Ready to run experiments!"
echo "Example: python3 run_vlm_hf.py --model qwen3-vl-4b --quick"
