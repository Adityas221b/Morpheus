"""ICML 2026 Experiment Launcher

Sets Ollama env vars, verifies GPU config, then runs all 30 experiments:
  5 bandit algorithms x 2 target models x 3 runs = 30 experiments

Usage:
    python run_experiments.bat     (double-click or from terminal)
    python run_experiments.py      (from terminal)
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ALGORITHMS = ["ucb1", "thompson", "exp3", "eps_greedy", "ucb_v"]
TARGET_MODELS = ["qwen3:1.7b", "qwen3:8b"]
RUNS_PER_CONFIG = 3
PROMPTS_FILE = "data/prompts/advbench_20.json"

# Ollama env vars — these must be set BEFORE Ollama server starts
OLLAMA_ENV = {
    "OLLAMA_NUM_PARALLEL": "2",
    "OLLAMA_MAX_LOADED_MODELS": "2",
    "OLLAMA_HOST": "127.0.0.1:11434",
}


def ensure_ollama_server():
    """Kill any existing Ollama, restart with correct env vars."""
    print("[SETUP] Restarting Ollama with OLLAMA_NUM_PARALLEL=8 ...")
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True)
    time.sleep(3)

    env = os.environ.copy()
    env.update(OLLAMA_ENV)

    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    # Verify
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "qwen3" in result.stdout:
        print(f"[SETUP] Ollama running. Models:\n{result.stdout}")
        return True
    else:
        print(f"[ERR] Ollama not ready: {result.stdout} {result.stderr}")
        return False


def run_experiment(algorithm: str, target_model: str, run_number: int, total: list) -> dict:
    """Run a single experiment via run_loop_v1.py."""
    idx = total[0]
    total[0] += 1
    model_safe = target_model.replace(":", "_")
    run_id = f"icml_{algorithm}_{model_safe}_run{run_number}"

    print(f"\n{'='*60}")
    print(f"[{idx}/30] {algorithm} | {target_model} | run {run_number}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update(OLLAMA_ENV)

    cmd = [
        sys.executable, "run_icml_experiment.py",
        "--algorithm", algorithm,
        "--model", target_model,
        "--runs", "1",
        "--run-number", str(run_number),
        "--prompts", PROMPTS_FILE,
    ]

    start = time.time()
    result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"[{idx}/30] {status} | {elapsed:.0f}s | {algorithm} | {target_model} | run {run_number}")
    return {"algorithm": algorithm, "model": target_model, "run": run_number, "elapsed": elapsed, "rc": result.returncode}


def main():
    print("=" * 60)
    print("ICML 2026 Scaling Experiments")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Models:     {TARGET_MODELS}")
    print(f"Runs:       {RUNS_PER_CONFIG} per config")
    print(f"Total:      {len(ALGORITHMS) * len(TARGET_MODELS) * RUNS_PER_CONFIG} experiments")
    print("=" * 60)

    if not ensure_ollama_server():
        sys.exit(1)

    # Verify prompts exist
    if not Path(PROMPTS_FILE).exists():
        print(f"[ERR] Prompts file not found: {PROMPTS_FILE}")
        sys.exit(1)

    counter = [1]
    results = []
    total_start = time.time()

    for algorithm in ALGORITHMS:
        for target_model in TARGET_MODELS:
            for run_number in range(1, RUNS_PER_CONFIG + 1):
                r = run_experiment(algorithm, target_model, run_number, counter)
                results.append(r)

                # Save intermediate results
                with open("logs/icml_results_live.json", "w") as f:
                    json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL DONE | {total_elapsed/3600:.1f} hours | {len(results)} experiments")
    print(f"{'='*60}")

    successes = [r for r in results if r["rc"] == 0]
    failures = [r for r in results if r["rc"] != 0]
    print(f"  Passed: {len(successes)}")
    print(f"  Failed: {len(failures)}")
    if failures:
        for f in failures:
            print(f"    {f['algorithm']} | {f['model']} | run {f['run']}")

    # Aggregate analysis
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_icml_experiment import generate_analysis
        # Load all run summaries
        all_summaries = []
        for r in results:
            model_safe = r["model"].replace(":", "_")
            summary_path = Path(f"logs/run_summary_icml_{r['algorithm']}_{model_safe}_run{r['run']}.json")
            if summary_path.exists():
                with open(summary_path) as f:
                    all_summaries.append(json.load(f))

        if all_summaries:
            # Build analysis-compatible format
            analysis_input = []
            for s in all_summaries:
                analysis_input.append({
                    "algorithm": s.get("run_id", "").split("_")[1] if "run_id" in s else "unknown",
                    "target_model": s.get("targets", [""])[0] if s.get("targets") else "",
                    "run_number": 1,
                    "success": True,
                    "summary": s,
                })
            analysis = generate_analysis(analysis_input)
            with open("logs/icml_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to logs/icml_analysis.json")

            if analysis["summary_table"]:
                print(f"\n{'Algorithm':<15} {'Model':<20} {'Mean ASR':<10} {'Std':<8}")
                print("-" * 53)
                for row in analysis["summary_table"]:
                    print(f"{row['algorithm']:<15} {row['target_model']:<20} {row['mean_asr']:<10.1f} {row['std_asr']:<8.1f}")
    except Exception as e:
        print(f"[WARN] Analysis generation failed: {e}")


if __name__ == "__main__":
    main()
