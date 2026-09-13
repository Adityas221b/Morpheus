"""Minimal test: 1 prompt, uncensored attacker, qwen3:8b target"""
import asyncio
import json
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.agents.attacker import Attacker

async def test():
    print("[1] Creating orchestrator...", flush=True)
    t0 = time.time()
    orch = Orchestrator(run_id="test_uncensored_v2")
    print(f"[1] Orchestrator created in {time.time()-t0:.1f}s", flush=True)

    # Patch foundation config
    foundation_cfg = orch.model_config.get("foundation", {})
    foundation_cfg["model"] = "richardyoung/qwen2.5-7b-instruct-abliterated:Q4_K_M"
    foundation_cfg["think"] = False
    orch.model_config["foundation"] = foundation_cfg
    orch.model_config["targets"]["qwen3_8b_target"]["model"] = "qwen3:8b"

    # Recreate attacker with patched config
    print("[2] Creating attacker...", flush=True)
    orch.attacker = Attacker(
        config=foundation_cfg,
        ollama_url=orch.ollama_url,
        timeout=orch.model_timeout,
        prompts=orch.prompts.get("attacker", {}),
        think=False,
    )
    print(f"[2] Attacker model: {orch.attacker.model}", flush=True)

    # Load just 1 prompt
    with open("data/prompts/advbench_20.json") as f:
        prompts = json.load(f)[:1]
    print(f"[3] Loaded {len(prompts)} prompt(s)", flush=True)

    # Patch settings
    orch.settings["bandit"]["enabled"] = True
    orch.settings["bandit"]["algorithm"] = "ucb1"

    print("[4] Starting attack loop...", flush=True)
    t1 = time.time()
    try:
        await orch.run(
            requests=prompts,
            target_names=["qwen3_8b_target"],
            epochs=1,
            requests_name="test_uncensored",
        )
        print(f"[4] Attack loop completed in {time.time()-t1:.1f}s", flush=True)
    except Exception as e:
        print(f"[ERR] Attack loop failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[5] Total time: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    asyncio.run(test())
