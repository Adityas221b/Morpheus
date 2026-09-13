import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.agents.attacker import Attacker

async def test():
    orch = Orchestrator(run_id="test_uncensored")
    foundation_cfg = orch.model_config.get("foundation", {})
    foundation_cfg["model"] = "richardyoung/qwen2.5-7b-instruct-abliterated:Q4_K_M"
    foundation_cfg["think"] = False
    orch.model_config["foundation"] = foundation_cfg
    orch.model_config["targets"]["qwen3_8b_target"]["model"] = "qwen3:8b"
    orch.attacker = Attacker(
        config=foundation_cfg,
        ollama_url=orch.ollama_url,
        timeout=orch.model_timeout,
        prompts=orch.prompts.get("attacker", {}),
        think=False,
    )
    with open("data/prompts/advbench_20.json") as f:
        prompts = json.load(f)[:2]
    await orch.run(requests=prompts, target_names=["qwen3_8b_target"], epochs=1, requests_name="test")
    print("DONE - check logs/ for summary")

asyncio.run(test())
