"""ICML 2026 Experiment Runner

Runs the full scaling experiment matrix:
- 5 bandit algorithms: UCB1, Thompson Sampling, EXP3, epsilon-Greedy, UCB-V
- 2 target models: qwen3:1.7b, qwen3:8b
- 3 runs per configuration (for variance estimation)
- All 20 prompts from AdvBench

Usage:
    python run_icml_experiment.py                    # Full experiment (5 algorithms x 2 models x 3 runs)
    python run_icml_experiment.py --quick            # Quick test (2 algorithms x 1 model x 1 run)
    python run_icml_experiment.py --algorithm thompson --runs 5  # Single algorithm, 5 runs
"""

import asyncio
import json
import time
import argparse
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

from src.orchestrator import Orchestrator, load_requests


# Experiment configurations
BANDIT_ALGORITHMS = ["ucb1", "thompson", "exp3", "eps_greedy", "ucb_v"]
TARGET_MODELS = ["qwen3:1.7b", "qwen3:8b"]
DEFAULT_RUNS = 3
PROMPTS_FILE = "data/prompts/advbench_20.json"


def create_experiment_config(algorithm: str, target_model: str, run_id: str, attacker_model: str = None) -> dict:
    """Create a settings override dict for a specific experiment."""
    return {
        "bandit": {
            "enabled": True,
            "algorithm": algorithm,
            "exploration_weight": 1.0,
            "temperature": 0.0,
        },
        "genetic_algorithm": {
            "enabled": True,
            "population_size": 10,
            "crossover_rate": 0.7,
            "mutation_rate": 0.3,
            "elite_count": 2,
            "n_offspring": 3,
            "offspring_per_epoch": 2,
        },
        "multipass": {
            "enabled": False,
            "k": 3,
        },
        "soc": {
            "enabled": False,
            "max_turns": 3,
        },
        "metacognition": {
            "enabled": False,
        },
    }


def patch_settings(settings: dict, overrides: dict) -> dict:
    """Deep-merge overrides into settings."""
    for key, value in overrides.items():
        if key in settings and isinstance(settings[key], dict) and isinstance(value, dict):
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


async def run_single_experiment(
    algorithm: str,
    target_model: str,
    run_number: int,
    prompts: list[dict],
    total_runs: int,
    attacker_model: str = None,
) -> dict:
    """Run a single experiment configuration."""
    attacker_label = attacker_model or "default"
    run_id = f"icml_{algorithm}_{target_model.replace(':', '_')}_run{run_number}"
    print(f"\n{'='*60}")
    print(f"Experiment: {algorithm} | {target_model} | attacker={attacker_label} | Run {run_number}/{total_runs}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")

    # Create orchestrator
    orchestrator = Orchestrator(run_id=run_id)

    # Patch settings with experiment-specific config
    overrides = create_experiment_config(algorithm, target_model, run_id)
    orchestrator.settings = patch_settings(orchestrator.settings, overrides)

    # Override target model
    orchestrator.model_config["targets"]["qwen3_8b_target"]["model"] = target_model

    # Override attacker (foundation) model if specified
    if attacker_model:
        orchestrator.model_config["foundation"]["model"] = attacker_model

    # Recreate attacker with updated foundation config (model + think flag)
    from src.agents.attacker import Attacker
    foundation_cfg = orchestrator.model_config.get("foundation", {})
    orchestrator.attacker = Attacker(
        config=foundation_cfg,
        ollama_url=orchestrator.ollama_url,
        timeout=orchestrator.model_timeout,
        prompts=orchestrator.prompts.get("attacker", {}),
        think=foundation_cfg.get("think", True),
    )

    # Run the experiment
    start_time = time.time()
    try:
        await orchestrator.run(
            requests=prompts,
            target_names=["qwen3_8b_target"],
            epochs=1,
            requests_name="advbench_20",
        )
        elapsed = time.time() - start_time
        success = True
    except Exception as e:
        print(f"  [ERR] Experiment failed: {e}")
        elapsed = time.time() - start_time
        success = False

    # Read the run summary
    summary_path = Path(f"logs/run_summary_{run_id}.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"run_id": run_id, "error": "No summary generated"}

    return {
        "algorithm": algorithm,
        "target_model": target_model,
        "run_number": run_number,
        "run_id": run_id,
        "success": success,
        "elapsed_sec": round(elapsed, 1),
        "summary": summary,
    }


async def run_experiment_matrix(
    algorithms: list[str],
    target_models: list[str],
    runs_per_config: int,
    prompts: list[dict],
    attacker_model: str = None,
    runs_list: list[int] = None,
):
    """Run the full experiment matrix."""
    results = []
    if runs_list is None:
        runs_list = list(range(1, runs_per_config + 1))
    total_configs = len(algorithms) * len(target_models)
    total_experiments = total_configs * len(runs_list)

    print(f"\n{'#'*60}")
    print(f"ICML 2026 Scaling Experiments")
    print(f"Algorithms: {algorithms}")
    print(f"Target models: {target_models}")
    print(f"Attacker model: {attacker_model or 'config default'}")
    print(f"Runs per config: {runs_per_config}")
    print(f"Total experiments: {total_experiments}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'#'*60}")

    experiment_num = 0
    for algorithm in algorithms:
        for target_model in target_models:
            for run_number in runs_list:
                experiment_num += 1
                print(f"\n[Experiment {experiment_num}/{total_experiments}]")

                result = await run_single_experiment(
                    algorithm=algorithm,
                    target_model=target_model,
                    run_number=run_number,
                    prompts=prompts,
                    total_runs=runs_per_config,
                    attacker_model=attacker_model,
                )
                results.append(result)

                # Save intermediate results
                results_path = Path("logs/icml_experiment_results.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

    return results


def generate_analysis(results: list[dict]) -> dict:
    """Generate aggregate analysis from experiment results."""
    analysis = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_experiments": len(results),
        "by_algorithm": {},
        "by_model": {},
        "summary_table": [],
    }

    # Group by algorithm
    for result in results:
        algo = result["algorithm"]
        model = result["target_model"]
        run = result["run_number"]

        if algo not in analysis["by_algorithm"]:
            analysis["by_algorithm"][algo] = []
        analysis["by_algorithm"][algo].append(result)

        if model not in analysis["by_model"]:
            analysis["by_model"][model] = []
        analysis["by_model"][model].append(result)

    # Compute ASR per algorithm-model pair
    for algo in analysis["by_algorithm"]:
        algo_results = analysis["by_algorithm"][algo]
        for model in set(r["target_model"] for r in algo_results):
            model_runs = [r for r in algo_results if r["target_model"] == model]
            asrs = []
            for run in model_runs:
                if run["success"] and "stats" in run.get("summary", {}):
                    asr = run["summary"]["stats"].get("overall_asr", 0)
                    asrs.append(asr)

            if asrs:
                mean_asr = sum(asrs) / len(asrs)
                std_asr = (sum((x - mean_asr) ** 2 for x in asrs) / len(asrs)) ** 0.5
                analysis["summary_table"].append({
                    "algorithm": algo,
                    "target_model": model,
                    "mean_asr": round(mean_asr, 2),
                    "std_asr": round(std_asr, 2),
                    "n_runs": len(asrs),
                    "asrs": [round(a, 2) for a in asrs],
                })

    # Sort summary table
    analysis["summary_table"].sort(key=lambda x: x["mean_asr"], reverse=True)

    return analysis


def main():
    parser = argparse.ArgumentParser(description="ICML 2026 Experiment Runner")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 algorithms, 1 model, 1 run)")
    parser.add_argument("--algorithm", type=str, help="Single algorithm to test")
    parser.add_argument("--model", type=str, help="Single target model to test")
    parser.add_argument("--attacker", type=str, help="Attacker (foundation) model to use (overrides config)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per configuration")
    parser.add_argument("--run-number", type=int, default=None, help="Override run number (for batch launcher)")
    parser.add_argument("--prompts", type=str, default=PROMPTS_FILE, help="Path to prompts JSON file")
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # Determine experiment configurations
    if args.quick:
        algorithms = ["ucb1", "thompson"]
        target_models = ["qwen3:1.7b"]
        runs = 1
    elif args.algorithm:
        algorithms = [args.algorithm]
        target_models = [args.model] if args.model else TARGET_MODELS
        runs = args.runs
    else:
        algorithms = BANDIT_ALGORITHMS
        target_models = TARGET_MODELS
        runs = args.runs

    # If --run-number is specified, run only that specific run
    if args.run_number is not None:
        runs_list = [args.run_number]
    else:
        runs_list = list(range(1, runs + 1))

    # Run experiments
    results = asyncio.run(run_experiment_matrix(
        algorithms=algorithms,
        target_models=target_models,
        runs_per_config=runs,
        prompts=prompts,
        attacker_model=args.attacker,
        runs_list=runs_list,
    ))

    # Generate analysis
    analysis = generate_analysis(results)

    # Save analysis
    analysis_path = Path("logs/icml_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} {'Target Model':<20} {'Mean ASR':<10} {'Std ASR':<10} {'Runs':<5}")
    print(f"{'-'*60}")
    for row in analysis["summary_table"]:
        print(f"{row['algorithm']:<15} {row['target_model']:<20} {row['mean_asr']:<10.2f} {row['std_asr']:<10.2f} {row['n_runs']:<5}")

    print(f"\nResults saved to: logs/icml_experiment_results.json")
    print(f"Analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()
