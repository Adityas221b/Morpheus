#!/bin/bash
# =============================================================================
# Download all VLM models via huggingface-cli
# Run this FIRST, then run experiments separately
# =============================================================================
set -e

export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /tmp/hf_cache

echo "=========================================="
echo "  Downloading VLM Models to /tmp/hf_cache"
echo "=========================================="
echo ""

# Phase 1: Small models
echo ">>> Phase 1: Small models <<<"

echo "[1/12] LLaVA 1.6 7B..."
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf 2>&1 | tail -2
echo "  Done."

echo "[2/12] Kimi-VL 3B..."
huggingface-cli download moonshotai/Kimi-VL-A3B-Instruct 2>&1 | tail -2
echo "  Done."

echo "[3/12] Qwen3-VL 4B..."
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct 2>&1 | tail -2
echo "  Done."

# Phase 2: Medium models
echo ""
echo ">>> Phase 2: Medium models <<<"

echo "[4/12] GLM-4V 9B..."
huggingface-cli download THUDM/GLM-4V-9B 2>&1 | tail -2
echo "  Done."

echo "[5/12] Qwen3-VL 8B..."
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct 2>&1 | tail -2
echo "  Done."

echo "[6/12] Llama 3.2 Vision 11B..."
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct 2>&1 | tail -2
echo "  Done."

# Phase 3: Large models (INT4)
echo ""
echo ">>> Phase 3: Large models (INT4) <<<"

echo "[7/12] Gemma3 27B..."
huggingface-cli download google/gemma-3-27b-it 2>&1 | tail -2
echo "  Done."

echo "[8/12] Qwen3-VL 32B..."
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct 2>&1 | tail -2
echo "  Done."

# Phase 4: XL models (INT4)
echo ""
echo ">>> Phase 4: XL models (INT4) <<<"

echo "[9/12] Qwen3-VL 72B..."
huggingface-cli download Qwen/Qwen3-VL-72B-Instruct 2>&1 | tail -2
echo "  Done."

echo "[10/12] InternVL 2.5 78B..."
huggingface-cli download OpenGVLab/InternVL2.5-78B-Instruct 2>&1 | tail -2
echo "  Done."

echo "[11/12] Llama 3.2 Vision 90B..."
huggingface-cli download meta-llama/Llama-3.2-90B-Vision-Instruct 2>&1 | tail -2
echo "  Done."

echo "[12/12] Molmo 72B..."
huggingface-cli download allenai/Molmo-72B-Instruct 2>&1 | tail -2
echo "  Done."

echo ""
echo "=========================================="
echo "  All downloads complete!"
echo "=========================================="
du -sh /tmp/hf_cache/hub/models--*/ 2>/dev/null || du -sh /tmp/hf_cache/ 2>/dev/null
