#!/bin/bash
# =============================================================================
# VLM Jailbreak Setup + Launch — paste this into your SSH session
# Server: 80GB A100 (172.24.16.132)
# =============================================================================
set -e

echo "=========================================="
echo "  VLM Jailbreak Server Setup"
echo "=========================================="

# 1. Check GPU
echo "[1/7] Checking GPU..."
nvidia-smi

# 2. Check Python + PyTorch
echo "[2/7] Checking Python..."
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 3. Clone repo
echo "[3/7] Setting up Morpheus repo..."
if [ ! -d "$HOME/Morpheus" ]; then
    git clone https://github.com/Adityas221b/Morpheus.git $HOME/Morpheus
fi
cd $HOME/Morpheus

# 4. Install dependencies
echo "[4/7] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
pip install -r requirements.txt
pip install bitsandbytes accelerate transformers sentencepiece protobuf pillow pyyaml numpy

# 5. Install Ollama + attacker model
echo "[5/7] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama serve &>/dev/null &
sleep 3
ollama pull qwen3:1.7b

# 6. Verify everything
echo "[6/7] Verification..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import transformers, accelerate
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
"

# 7. Pre-download the smallest model (qwen3-vl-4b) to test
echo "[7/7] Pre-downloading qwen3-vl-4b..."
python3 -c "
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
print('Downloading processor...')
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', trust_remote_code=True)
print('Downloading model...')
Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', torch_dtype='auto', trust_remote_code=True)
print('Done!')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Next: run the experiment script"
echo "=========================================="
