import asyncio
import sys
import io
import time
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.orchestrator import Orchestrator, load_requests

prompts = load_requests("data/prompts/advbench_2.json")
orchestrator = Orchestrator(run_id="speed_test")
orchestrator.settings["bandit"]["algorithm"] = "ucb1"
orchestrator.settings["training"]["epochs"] = 1
orchestrator.settings["training"]["max_attempts"] = 3
orchestrator.settings["training"]["timeout_seconds"] = 180
orchestrator.concurrency = 8

start = time.time()
asyncio.run(orchestrator.run(
    requests=prompts,
    target_names=["qwen3_8b_target"],
    epochs=1,
    requests_name="advbench_2",
))
elapsed = time.time() - start
print(f"\n=== SPEED TEST: {len(prompts)} prompts in {elapsed:.0f}s ({elapsed/len(prompts):.0f}s per prompt) ===")
