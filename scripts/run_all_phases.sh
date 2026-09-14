#!/bin/bash
# =============================================================================
# VLM Jailbreak Experiment Launcher
# Run from: $HOME/Morpheus
# Starts with LOW safety models, progresses to HIGH safety
# =============================================================================
set -e

cd "$(dirname "$0")/.."
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L | head -1)"

ALGORITHM="${1:-ucb1}"
RUNS="${2:-1}"
PROMPTS="data/prompts/advbench_nsfw_50.json"

echo ""
echo "=========================================="
echo "  VLM Jailbreak Experiments"
echo "  Algorithm: $ALGORITHM | Runs: $RUNS"
echo "=========================================="

# Phase 1: LOW safety models (fast, high expected ASR)
echo ""
echo ">>> PHASE 1: Low safety models <<<"
echo "  LLaVA 1.6 7B → Kimi-VL 3B → Qwen3-VL 4B"

python3 run_vlm_hf.py --model llava-16-7b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase1_llava.log
python3 run_vlm_hf.py --model kimi-vl-3b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase1_kimi.log
python3 run_vlm_hf.py --model qwen3-vl-4b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase1_qwen4b.log

echo ""
echo ">>> PHASE 1 COMPLETE <<<"
echo "Results so far:"
python3 -c "
import json, glob
for f in sorted(glob.glob('logs/run_summary_vlm_*_run*.json')):
    r = json.load(open(f))
    print(f\"  {r['target']:<25} ASR={r['asr']:.1f}%  ({r['successes']}/{r['total']})\")
"

# Phase 2: MEDIUM safety models
echo ""
echo ">>> PHASE 2: Medium safety models <<<"
echo "  GLM-4V 9B → Qwen3-VL 8B → Llama 3.2 11B"

python3 run_vlm_hf.py --model glm4v-9b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase2_glm4v.log
python3 run_vlm_hf.py --model qwen3-vl-8b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase2_qwen8b.log
python3 run_vlm_hf.py --model llama32-vision-11b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase2_llama11b.log

echo ""
echo ">>> PHASE 2 COMPLETE <<<"

# Phase 3: HIGH safety models (INT4 quantized)
echo ""
echo ">>> PHASE 3: High safety models (INT4) <<<"
echo "  Gemma3 27B → Qwen3-VL 32B"

python3 run_vlm_hf.py --model gemma3-27b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase3_gemma27b.log
python3 run_vlm_hf.py --model qwen3-vl-32b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase3_qwen32b.log

echo ""
echo ">>> PHASE 3 COMPLETE <<<"

# Phase 4: VERY HIGH safety models (large, INT4)
echo ""
echo ">>> PHASE 4: Very high safety models (INT4) <<<"
echo "  Qwen3-VL 72B → InternVL 2.5 78B → Llama 3.2 90B → Molmo 72B"

python3 run_vlm_hf.py --model qwen3-vl-72b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase4_qwen72b.log
python3 run_vlm_hf.py --model internvl25-78b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase4_internvl.log
python3 run_vlm_hf.py --model llama32-vision-90b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase4_llama90b.log
python3 run_vlm_hf.py --model molmo-72b --algorithm "$ALGORITHM" --runs "$RUNS" --prompts "$PROMPTS" 2>&1 | tee logs/phase4_molmo.log

echo ""
echo ">>> ALL PHASES COMPLETE <<<"

# Final summary
echo ""
echo "=========================================="
echo "  FINAL RESULTS"
echo "=========================================="
python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('logs/run_summary_vlm_*_run*.json')):
    r = json.load(open(f))
    results.append(r)

# Sort by safety level (model size as proxy)
print(f\"{'Model':<30} {'HF ID':<50} {'ASR':<8} {'Success':<10}\")
print('-' * 100)
for r in sorted(results, key=lambda x: x.get('hf_id', '')):
    print(f\"{r['target']:<30} {r.get('hf_id',''):<50} {r['asr']:.1f}%   {r['successes']}/{r['total']}\")
"
echo ""
echo "Results saved to: logs/vlm_hf_analysis.json"
