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

download_model() {
    local name="$1"
    local repo="$2"
    echo ">>> Downloading: $name ($repo)"
    huggingface-cli download "$repo" --resume-download
    if [ $? -eq 0 ]; then
        echo "  [OK] $name downloaded"
    else
        echo "  [FAIL] $name failed"
    fi
    echo ""
}

# Phase 1: Small models
echo ">>> Phase 1: Small models <<<"
download_model "LLaVA 1.6 7B" "llava-hf/llava-v1.6-mistral-7b-hf"
download_model "Kimi-VL 3B" "moonshotai/Kimi-VL-A3B-Instruct"
download_model "Qwen3-VL 4B" "Qwen/Qwen3-VL-4B-Instruct"

# Phase 2: Medium models
echo ">>> Phase 2: Medium models <<<"
download_model "GLM-4V 9B" "THUDM/GLM-4V-9B"
download_model "Qwen3-VL 8B" "Qwen/Qwen3-VL-8B-Instruct"
download_model "Llama 3.2 Vision 11B" "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Phase 3: Large models (INT4)
echo ">>> Phase 3: Large models (INT4) <<<"
download_model "Gemma3 27B" "google/gemma-3-27b-it"
download_model "Qwen3-VL 32B" "Qwen/Qwen3-VL-32B-Instruct"

# Phase 4: XL models (INT4)
echo ">>> Phase 4: XL models (INT4) <<<"
download_model "Qwen3-VL 72B" "Qwen/Qwen3-VL-72B-Instruct"
download_model "InternVL 2.5 78B" "OpenGVLab/InternVL2.5-78B-Instruct"
download_model "Llama 3.2 Vision 90B" "meta-llama/Llama-3.2-90B-Vision-Instruct"
download_model "Molmo 72B" "allenai/Molmo-72B-Instruct"

echo ""
echo "=========================================="
echo "  All downloads complete!"
echo "=========================================="
du -sh /tmp/hf_cache/hub/models--*/ 2>/dev/null || du -sh /tmp/hf_cache/ 2>/dev/null
