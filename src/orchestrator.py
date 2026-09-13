"""Orchestrator: main lifelong learning loop for multimodal jailbreak discovery.

Supports two target types:
  text      : VLMs/LLMs -- text + optional image input -> text output
  image_gen : Text-to-image models -- text input -> image output
"""
import sys
import io
import os

# Force UTF-8 encoding for all stdout/stderr on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import asyncio
import gc
import json
import time
import uuid
import yaml
from pathlib import Path
from typing import Union

from PIL import Image as PILImage

from src.agents.attacker import Attacker, AttackResult
from src.agents.scorer import TextTargetScorer, ImageGenScorer, Score
from src.agents.scorer_backends import make_scorer_backend
from src.agents.summarizer import Summarizer, Strategy
from src.strategies.library import StrategyLibrary
from src.strategies.bandit import BanditFactory
from src.strategies.genetic import GeneticSearch, StrategyGenome
from src.targets.base import TargetResponse
from src.targets.ollama_target import OllamaTarget
from src.targets.api_target import ApiTarget
from src.targets.transformers_target import TransformersTarget
from src.utils.degen import _detect_degen, DegenResult

AnyTarget = Union[OllamaTarget, ApiTarget, TransformersTarget]


def _safe_print(*args, **kwargs):
    """Print that handles UnicodeEncodeError on Windows cp1252 consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode("utf-8", errors="replace").decode("utf-8"))
            else:
                safe_args.append(a)
        print(*safe_args, **kwargs)

class Orchestrator:
    """Main attack loop: attack → score → (save images) → summarize → store.

    Supports both text-output targets (VLMs) and image-generation targets.
    """

    def __init__(self, config_dir: str = "config", data_dir: str = "data", run_id: str | None = None, debug_judge: bool = False):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.settings = self._load_yaml("settings.yaml")
        self.model_config = self._load_yaml("models.yaml")
        self.prompts = self._load_yaml("prompts.yaml")

        server_cfg = self.settings.get("server", {})
        self.ollama_url = server_cfg.get("ollama_url", "http://localhost:11434")
        self.concurrency = server_cfg.get("concurrency", 4)

        self.model_timeout = self.settings.get("training", {}).get("timeout_seconds", 120)

        # Strategy library config — libraries instantiated lazily per target via _get_library()
        lib_cfg = self.settings.get("strategy_library", {})
        self.lib_path = lib_cfg.get("path", "strategies/")
        self.lib_defaults: dict[str, str] = lib_cfg.get("defaults", {
            "text": "library.jsonl",
            "image_gen": "img_gen_library.jsonl",
        })
        self.retrieval_cfg: dict = lib_cfg.get("retrieval", {})
        self.libraries: dict[str, StrategyLibrary] = {}

        # UCB Bandit config
        bandit_cfg = self.settings.get("bandit", {})
        self.bandit_enabled = bandit_cfg.get("enabled", False)
        self.bandit_algorithm = bandit_cfg.get("algorithm", "ucb1")
        self.bandit_exploration_weight = bandit_cfg.get("exploration_weight", 1.0)
        self.bandit_temperature = bandit_cfg.get("temperature", 0.0)

        # Multi-Pass config
        multipass_cfg = self.settings.get("multipass", {})
        self.multipass_enabled = multipass_cfg.get("enabled", False)
        self.multipass_k = multipass_cfg.get("k", 3)

        # SoC (Sequence of Context) config
        soc_cfg = self.settings.get("soc", {})
        self.soc_enabled = soc_cfg.get("enabled", False)
        self.soc_max_turns = soc_cfg.get("max_turns", 3)

        # Self-Evolving Metacognition config
        metacognition_cfg = self.settings.get("metacognition", {})
        self.metacognition_enabled = metacognition_cfg.get("enabled", False)

        # Genetic Algorithm config
        ga_cfg = self.settings.get("genetic_algorithm", {})
        self.ga_enabled = ga_cfg.get("enabled", False)
        self.ga_population_size = ga_cfg.get("population_size", 10)
        self.ga_crossover_rate = ga_cfg.get("crossover_rate", 0.7)
        self.ga_mutation_rate = ga_cfg.get("mutation_rate", 0.3)
        self.ga_elite_count = ga_cfg.get("elite_count", 2)
        self.ga_n_offspring = ga_cfg.get("n_offspring", 3)
        self.ga_offspring_per_epoch = ga_cfg.get("offspring_per_epoch", 2)

        # Genetic search instance (initialized lazily)
        self._genetic_search: GeneticSearch | None = None

        # Foundation model config
        foundation_cfg = self.model_config.get("foundation", {})

        # Scorer backend — shared by both scorers so HF weights load once.
        scoring_cfg = self.settings.get("scoring", {})
        self.scorer_backend = make_scorer_backend(
            scoring_cfg,
            ollama_url=self.ollama_url,
            timeout=self.model_timeout,
        )

        # Agents — one scorer per target type
        self.attacker = Attacker(
            config=foundation_cfg,
            ollama_url=self.ollama_url,
            timeout=self.model_timeout,
            prompts=self.prompts.get("attacker", {}),
            think=foundation_cfg.get("think", True),
        )
        self.text_scorer = TextTargetScorer(
            config=scoring_cfg,
            backend=self.scorer_backend,
            prompts=self.prompts.get("scorer", {}),
            timeout=self.model_timeout,
        )
        self.image_gen_scorer = ImageGenScorer(
            config=scoring_cfg,
            backend=self.scorer_backend,
            prompts=self.prompts.get("scorer", {}),
            timeout=self.model_timeout,
        )
        self.summarizer = Summarizer(
            config=foundation_cfg,
            ollama_url=self.ollama_url,
            prompt=self.prompts.get("summarizer"),
            timeout=self.model_timeout,
        )

        # Per-target instances — pre-populated in run() before gather()
        self.targets: dict[str, AnyTarget] = {}

        # Lock for concurrent appends to the JSONL log
        self._log_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "total_attacks": 0,
            "total_successes": 0,
            "total_images_generated": 0,
            "epochs_completed": 0,
            "strategies_discovered": 0,    # novel strategies that succeeded → entered the library
            "strategies_reused": 0,        # successful applications of an existing library strategy
            "start_time": None,
        }

        # Debug judge mode
        self.debug_judge = debug_judge

        # Results log — one file per process invocation, optionally named via run_id
        self.run_label = run_id if run_id else time.strftime("%Y%m%dT%H%M%S")
        self.run_label = f"{self.run_label}_debug" if self.debug_judge else self.run_label
        self.results_file = Path(f"logs/run_{self.run_label}.jsonl")
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        # Output dirs — per-run subfolders so each experiment is self-contained
        self.images_out_dir = Path(f"logs/generated_images/{self.run_label}/")
        self.raw_responses_dir = Path("logs/raw_responses/")

        # Raw response log
        self.raw_responses_file = self.raw_responses_dir / f"responses_{self.run_label}.jsonl"
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path, "r") as f:
            return yaml.safe_load(f)
        

    # ── Target management ───────────────────────────────────────────────────

    def _get_target(self, target_name: str) -> AnyTarget:
        """Get or create a target interface.

                Dispatches based on provider in models.yaml:
                    - provider=api                       -> ApiTarget
                    - provider=huggingface | hf | local  -> TransformersTarget (HF transformers/diffusers)
                    - provider=ollama (default)          -> OllamaTarget
        """
        if target_name in self.targets:
            return self.targets[target_name]

        target_cfg = self.model_config["targets"][target_name]
        model_full = target_cfg.get("model", target_name)
        provider = str(target_cfg.get("provider", "ollama")).lower()

        if provider == "api":
            target = ApiTarget(
                model_name=model_full,
                config=target_cfg,
                timeout = self.model_timeout,
            )
        elif provider in ("huggingface", "hf", "transformers", "diffusers", "local"):
            target = TransformersTarget(
                model_name=model_full,
                config=target_cfg,
                timeout = self.model_timeout,
            )
        else:
            target = OllamaTarget(
                model_name=model_full,
                config=target_cfg,
                base_url=target_cfg.get("base_url", self.ollama_url),
                timeout = self.model_timeout,
            )

        self.targets[target_name] = target
        return target

    def _get_scorer(self, target_name: str):
        """Return the appropriate scorer for the target type."""
        target_cfg = self.model_config["targets"][target_name]
        target_type = target_cfg.get("target_type", "text")
        if target_type == "image_gen":
            return self.image_gen_scorer
        return self.text_scorer

    def _resolve_library_filename(self, target_name: str) -> str:
        """Resolve which library filename a target writes to.

        Order:
          1. models.yaml → targets.<name>.strategy_library  (per-target override)
          2. settings.yaml → strategy_library.defaults.<target_type>  (per-type default)
          3. raise — config is incomplete.
        """
        target_cfg = self.model_config["targets"][target_name]
        override = target_cfg.get("strategy_library")
        if override:
            return override
        target_type = target_cfg.get("target_type", "text")
        default = self.lib_defaults.get(target_type)
        if not default:
            raise KeyError(
                f"No strategy_library configured for target {target_name!r} "
                f"(target_type={target_type!r}). Set targets.{target_name}.strategy_library "
                f"or strategy_library.defaults.{target_type} in your config."
            )
        return default

    def _get_library(self, target_name: str) -> StrategyLibrary:
        """Get or create the StrategyLibrary instance assigned to this target."""
        filename = self._resolve_library_filename(target_name)
        if filename not in self.libraries:
            self.libraries[filename] = StrategyLibrary(
                path=self.lib_path,
                filename=filename,
                bandit_mode=self.bandit_enabled,
                exploration_weight=self.bandit_exploration_weight,
                bandit_algorithm=self.bandit_algorithm,
                bandit_temperature=self.bandit_temperature,
            )
        return self.libraries[filename]

    def _get_genetic_search(self) -> GeneticSearch:
        """Get or create the GeneticSearch instance."""
        if self._genetic_search is None:
            self._genetic_search = GeneticSearch(
                population_size=self.ga_population_size,
                crossover_rate=self.ga_crossover_rate,
                mutation_rate=self.ga_mutation_rate,
                elite_count=self.ga_elite_count,
            )
        return self._genetic_search

    # ── Result logging ─────────────────────────────────────────────────────

    async def _log_raw_response(self, response_dict: dict):
        """Append raw target response to separate JSONL file."""
        async with self._log_lock:
            with open(self.raw_responses_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(response_dict, ensure_ascii=False) + "\n")

    async def _log_result(self, result: dict):
        """Append a result entry to the JSONL log, serialised via lock."""
        async with self._log_lock:
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Core loop ──────────────────────────────────────────────────────────

    async def run(
        self,
        requests: list[dict],
        target_names: list[str] | None = None,
        epochs: int | None = None,
        requests_name: str | None = None,
    ):
        """Run the main attack loop.

        Args:
            requests: List of prompt dicts with keys: id, prompt, category, type, image.
            target_names: Which target models to attack (None = all configured).
            epochs: Override number of epochs.
            requests_name: Display name for the prompt set (e.g. source filename).
        """
        n_epochs = epochs or self.settings.get("training", {}).get("epochs", 10)
        targets = target_names or list(self.model_config.get("targets", {}).keys())

        allowed_modes = self.settings.get("attack_modes") or ["text_only"]
        kept: list[dict] = []
        for req in requests:
            mode = req.get("type", "text_only")
            if mode in allowed_modes:
                kept.append(req)
            else:
                _safe_print(f"  [WARN] Dropping P{req.get('id', '?')}: type={mode!r} not in attack_modes={allowed_modes}")
        if not kept:
            _safe_print(f"  [ERR] No requests match attack_modes={allowed_modes}; nothing to run.")
            return
        requests = kept

        self.stats["start_time"] = time.time()

        # Pre-instantiate targets and libraries before launching concurrent cycles,
        # avoiding races where two cycles create the same instance.
        for t in targets:
            self._get_target(t)
            self._get_library(t)

        # Warm up targets sequentially before the attack loop starts. For ollama/api
        # this is a no-op; for huggingface targets this downloads + GPU-loads the
        # weights so the first cycle isn't blocked on a multi-GB pull (which would
        # also serialise every other concurrent cycle waiting on the load lock).
        for t in targets:
            try:
                await self.targets[t].warmup()
            except Exception as e:
                _safe_print(f"  [ERR] {t} warmup failed: {e}")
                raise

        # Same logic for the scorer backend — HF judge weights would otherwise be
        # pulled on the first scoring call and serialise the first epoch.
        try:
            await self.scorer_backend.warmup()
        except Exception as e:
            _safe_print(f"  [ERR] scorer backend warmup failed: {e}")
            raise

        attacker_model = self.model_config.get("foundation", {}).get("model", "?")
        scoring_cfg = self.settings.get("scoring", {})
        scorer_provider = str(scoring_cfg.get("provider", "ollama"))
        scorer_model = scoring_cfg.get("model", "?")
        _safe_print(f"AutoDAN-Multimodal — Lifelong Learning Loop")
        _safe_print(f"  Targets:      {targets}")
        _safe_print(f"  Attacker:     {attacker_model}")
        _safe_print(f"  Scorer:       {scorer_model} ({scorer_provider})")
        _safe_print(f"  Prompts:      {requests_name or '(inline)'}")
        _safe_print(f"  Prompts_num:  {len(requests)}")
        _safe_print(f"  Epochs:       {n_epochs}")
        _safe_print(f"  Concurrency:  {self.concurrency}")
        _safe_print(f"  Strategy lib:")
        _safe_print(f"    path:    {self.lib_path}")
        _safe_print(f"    loaded:")
        for fn, lib in self.libraries.items():
            _safe_print(f"      {fn}: {lib.size}")
        _safe_print()

        # One writer task per library lives for the entire run, drains its strategy queue.
        writers = {fn: asyncio.create_task(lib.writer_task()) for fn, lib in self.libraries.items()}

        sem = asyncio.Semaphore(self.concurrency)
        per_epoch: list[dict] = []

        try:
            for epoch in range(n_epochs):
                self.stats["epochs_completed"] = epoch + 1
                _safe_print(f"Epoch {epoch + 1}/{n_epochs}")

                strategies_before = self.stats["strategies_discovered"]
                reused_before = self.stats["strategies_reused"]
                epoch_started = time.time()

                async def _bounded(req, target_name, _epoch=epoch):
                    async with sem:
                        return await self._attack_cycle(req, target_name, _epoch)

                results = await asyncio.gather(*[
                    _bounded(req, target_name)
                    for req in requests
                    for target_name in targets
                ], return_exceptions=True)

                epoch_successes = 0
                epoch_attacks = 0
                epoch_images = 0

                for r in results:
                    if isinstance(r, Exception):
                        try:
                            _safe_print(f"  [ERR] Cycle exception: {r}")
                        except UnicodeEncodeError:
                            _safe_print(f"  [ERR] Cycle exception: {str(r).encode('utf-8', errors='replace').decode('utf-8')}")
                        continue
                    assert isinstance(r, dict)
                    epoch_attacks += 1
                    epoch_images += r.get("images_generated", 0)
                    if r["success"]:
                        epoch_successes += 1

                self.stats["total_attacks"] += epoch_attacks
                self.stats["total_successes"] += epoch_successes
                self.stats["total_images_generated"] += epoch_images

                epoch_asr = epoch_successes / epoch_attacks * 100 if epoch_attacks else 0
                strategies_this_epoch = self.stats["strategies_discovered"] - strategies_before
                reused_this_epoch = self.stats["strategies_reused"] - reused_before
                _safe_print(f"  -> Epoch ASR:           {epoch_asr:.1f}% ({epoch_successes}/{epoch_attacks})")
                _safe_print(f"  -> Epoch images:        {epoch_images}")
                _safe_print(f"  -> Strategies found:    {strategies_this_epoch}")
                _safe_print(f"  -> Strategies reused:   {reused_this_epoch}")

                per_epoch.append({
                    "epoch": epoch + 1,
                    "attacks": epoch_attacks,
                    "successes": epoch_successes,
                    "asr": round(epoch_asr, 2),
                    "images_generated": epoch_images,
                    "strategies_found": strategies_this_epoch,
                    "strategies_reused": reused_this_epoch,
                    "elapsed_sec": round(time.time() - epoch_started, 1),
                })

                # Genetic Algorithm phase: evolve strategies at end of epoch
                if self.ga_enabled and epoch < n_epochs - 1:
                    ga_offspring = await self._run_genetic_phase(targets)
                    per_epoch[-1]["ga_offspring"] = len(ga_offspring)

                gc.collect()

        finally:
            # Always drain every writer, even if the run is interrupted.
            for fn, lib in self.libraries.items():
                await lib.enqueue(None)
            for fn, w in writers.items():
                await w
            # Gracefully close any targets that require it (e.g. aiohttp sessions in ApiTarget)
            for target in self.targets.values():
                close = getattr(target, "aclose", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            # Same for the scorer backend (HF backend frees GPU memory here).
            scorer_close = getattr(self.scorer_backend, "aclose", None)
            if scorer_close is not None:
                result = scorer_close()
                if asyncio.iscoroutine(result):
                    await result

        elapsed = time.time() - self.stats["start_time"]
        total = self.stats["total_attacks"]
        successes = self.stats["total_successes"]
        overall_asr = successes / total * 100 if total else 0

        _safe_print()
        _safe_print(f"Complete.")
        _safe_print(f"  Total attacks:          {total}")
        _safe_print(f"  Total successes:        {successes}")
        _safe_print(f"  Overall ASR:            {overall_asr:.1f}%")
        _safe_print(f"  Images generated:       {self.stats['total_images_generated']}")
        _safe_print(f"  Total Strategies found: {self.stats['strategies_discovered']}")
        _safe_print(f"  Total Strategies reused:{self.stats['strategies_reused']}")
        _safe_print(f"  Time elapsed:           {elapsed:.0f}s")

        # Persist a single per-run summary alongside the run JSONL log.
        run_summary = {
            "run_id": self.run_label,
            "targets": targets,
            "attacker": self.model_config.get("foundation", {}).get("model"),
            "prompts": {
                "source": requests_name,
                "count": len(requests),
            },
            "epochs": n_epochs,
            "concurrency": self.concurrency,
            "stats": {
                "total_attacks": total,
                "total_successes": successes,
                "overall_asr": round(overall_asr, 2),
                "images_generated": self.stats["total_images_generated"],
                "strategies_discovered": self.stats["strategies_discovered"],
                "strategies_reused": self.stats["strategies_reused"],
                "epochs_completed": self.stats["epochs_completed"],
                "elapsed_sec": round(elapsed, 1),
            },
            "per_epoch": per_epoch,
            "libraries": {fn: lib.summary() for fn, lib in self.libraries.items()},
            "bandit_stats": {fn: lib.get_bandit_stats() for fn, lib in self.libraries.items()} if self.bandit_enabled else {},
            "results_file": str(self.results_file),
        }
        summary_path = self.results_file.parent / f"run_summary_{self.run_label}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
        _safe_print(f"Run summary -> {summary_path}")

    # ── Degeneration detection ─────────────────────────────────────────────


    # ── Single attack cycle ────────────────────────────────────────────────

    async def _attack_cycle(
        self,
        request: dict,
        target_name: str,
        epoch: int,
    ) -> dict:
        """Multi-attempt attack cycle: iterate up to max_attempts per prompt.

        Attempts 1-4: library hints only.
        Attempts 5+:  library hints + full failure history (self-improvement).

        Supports optional Multi-Pass, SoC, and Metacognition modes.

        Returns:
            dict with keys: success, breach_at_attempt, attempts_made, images_generated
        """
        req_id = request.get("id", "?")
        category = request.get("category", "unknown")
        prompt_text = request.get("prompt", "")
        attack_mode = request.get("type", "text_only")

        augmented_request = prompt_text
        if attack_mode == "image_text":
            image_description = request.get("image_description", "")
            if image_description:
                augmented_request = f"{prompt_text}\n[Image content: {image_description}]"

        top_k = self.settings.get("strategy_library", {}).get("top_k", 3)
        use_deep = self.settings.get("scoring", {}).get("use_deep_scoring", False)
        max_attempts = self.settings.get("training", {}).get("max_attempts", 10)
        self_improve_after = self.settings.get("training", {}).get("self_improve_after", 10)
        self.retry_limit = self.settings.get("training", {}).get("retry_limit", 1)    

        target = self._get_target(target_name)
        target_type = self.model_config["targets"][target_name].get("target_type", "text")

        failure_history: list[dict] = []
        total_images = 0
        attempts_made = 0

        # --- Multi-Pass mode: generate k candidates, score all, return best ---
        if self.multipass_enabled and not failure_history:
            return await self._attack_cycle_multipass(
                request=request, target_name=target_name, epoch=epoch,
                augmented_request=augmented_request, category=category,
                attack_mode=attack_mode, target=target, target_type=target_type,
                top_k=top_k, use_deep=use_deep, max_attempts=max_attempts,
            )

        # --- Metacognition mode: self-critique and refine ---
        if self.metacognition_enabled and not failure_history:
            return await self._attack_cycle_metacognition(
                request=request, target_name=target_name, epoch=epoch,
                augmented_request=augmented_request, category=category,
                attack_mode=attack_mode, target=target, target_type=target_type,
                top_k=top_k, use_deep=use_deep, max_attempts=max_attempts,
            )

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            score: Score | None = None
            n_images = 0
            classification = None
            images_list: list[bytes] = []

            # 1. Generate attack
            use_history = failure_history if attempt > self_improve_after else None
            attacker_retries = 0
            attack_result: AttackResult | None = None
            while (attacker_retries <= self.retry_limit):
                try:
                    attack_result = await self.attacker.generate(
                        request=augmented_request,
                        category=category,
                        attack_mode=attack_mode,
                        top_k=top_k,
                        failure_history=use_history,
                        strategy_library=self._get_library(target_name),
                        target_model=target_name,
                        retrieval_config=self.retrieval_cfg,
                        bandit_mode=self.bandit_enabled,
                        bandit_temperature=self.bandit_temperature,
                    )
                    break  # sucessful generation, exit retry loop
                except Exception as e:
                    _safe_print(f"  [ERR] P{req_id} attempt {attempt} attacker retry: {attacker_retries}/{self.retry_limit}: {e}")
                    attacker_retries += 1

            if attack_result is None:
                _safe_print(f"  [ERR] P{req_id} attempt {attempt}: attacker failed after {self.retry_limit} retries.")
                continue  # skip to next attempt

            attack_prompt = attack_result.prompt
            attack_id = f"e{epoch+1}_p{req_id}_a{attempt}_{uuid.uuid4().hex[:4]}"

            # 2. Query target
            target_retries = 0
            image = None
            response: TargetResponse | None = None
            if attack_mode == "image_text" and target_type == "text":
                image_file = request.get("image", "")
                if not image_file:
                    _safe_print(f"  [WARN] P{req_id} attempt {attempt}: type=image_text but no 'image' field; sending text-only.")
                else:
                    image_path = self.data_dir / "images" / image_file
                    if not image_path.exists():
                        _safe_print(f"  [WARN] P{req_id} attempt {attempt}: image file not found at {image_path}; sending text-only.")
                    else:
                        try:
                            image = PILImage.open(image_path).convert("RGB")
                        except Exception as e:
                            _safe_print(f"  [WARN] P{req_id} attempt {attempt}: failed to open image {image_path}: {e}; sending text-only.")
            while(target_retries <= self.retry_limit):
                try:
                    if target_type == "image_gen":
                        response = await target.query(text=attack_prompt, image=None)  #TODO: image_text attack for image_gen targets?
                    else:
                        response = await target.query(text=attack_prompt, image=image)

                    if response.error:
                        _safe_print(f"  [ERR] P{req_id} attempt {attempt} target retry: {target_retries}/{self.retry_limit}: {response.error}")
                        target_retries += 1
                        continue

                    break
                except Exception as e:
                    _safe_print(f"  [ERR] P{req_id} attempt {attempt} target retry: {target_retries}/{self.retry_limit}: {str(e)}")
                    target_retries += 1

            if image is not None:
                image.close()
                image = None
                
            if response is None:
                _safe_print(f"  [ERR] P{req_id} attempt {attempt}: target queries failed after {self.retry_limit} retries.")
                continue # skip to next attempt
            else:
                response_text = response.text or ""

            # Log raw response to separate file for analysis
            await self._log_raw_response({
                "epoch": epoch + 1,
                "prompt_id": req_id,
                "attempt": attempt,
                "target": target_name,
                "target_type": target_type,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "raw": response.raw,
            })

            # 3a. Check for short-circuit conditions: error or blocked
            degen_result = DegenResult(is_degenerate=False, reason=None)  # default non-degen result

            # Safety-filters block check — applies to all target types
            if response.blocked:
                classification = "BLOCKED"
                score = None   # Score(success=False, refused=False, confidence=0.0, reasoning=f"blocked_{response.block_reason}")
                n_images = 0
            elif target_type == "image_gen":
                # Image-gen targets return image bytes with empty text — degen detection
                # would always fire on the empty response_text. Skip it; the scorer will
                # judge the image directly.
                pass
            else:
            # 3. Degeneration check — inspect target output before scoring
                #degen_result = _detect_degen(response_text)    Deactivating it to avoid false response
                if degen_result.is_degenerate:
                    classification = "DEGENERATE"
                    score = None   # Score(success=False, refused=False, confidence=0.0, reasoning=f"degen_{degen_type}")
                    n_images = 0

            # 4. Score (only if no error/blocked/degen)
            if classification not in ["BLOCKED", "DEGENERATE"]:
                if target_type == "image_gen":
                    generated_images = response.image
                    if generated_images is None:
                        image_items: list = []
                    elif isinstance(generated_images, list):
                        image_items = [img for img in generated_images if img is not None]  # if a list[images]
                    else:
                        image_items = [generated_images]  # if not a list of images

                    n_images = 0

                    # Encode to bytes in memory; defer disk write until classification is known
                    # so the filename can carry the verdict prefix.
                    for generated_image in image_items:
                        buffer = io.BytesIO()
                        generated_image.save(buffer, format="PNG")
                        images_list.append(buffer.getvalue())
                        n_images += 1
                    total_images += n_images
                    
                    score_retries = 0

                    while(score_retries <= self.retry_limit):
                        try:
                            score = await self.image_gen_scorer.score(
                                request=augmented_request,
                                response_text=response_text,
                                attack_prompt=attack_prompt,
                                images=images_list,
                                use_deep=use_deep,
                            )

                            break
                        except Exception as e:  
                            _safe_print(f"  [ERR] P{req_id} attempt {attempt} scorer retry: {score_retries}/{self.retry_limit}: {str(e)}")
                            score_retries += 1
                else:
                    score_retries = 0
                    while(score_retries <= self.retry_limit):
                        try:
                            score = await self.text_scorer.score(
                                request=augmented_request,
                                response_text=response_text,
                                attack_prompt=attack_prompt,
                                use_deep=use_deep,
                            )

                            break
                        except Exception as e:
                            _safe_print(f"  [ERR] P{req_id} attempt {attempt} scorer retry: {score_retries}/{self.retry_limit}: {e}")
                            score_retries += 1
                    n_images = 0
            if score:
                classification = "COMPLIANT" if score.success else ("REFUSED" if score.refused else "NON_COMPLIANT")

            # 4b. Persist generated images now that classification is known — prefix the filename with the verdict.
            if target_type == "image_gen" and images_list:
                self.images_out_dir.mkdir(parents=True, exist_ok=True)
                label = classification or "UNSCORED"
                for index, image_bytes in enumerate(images_list, start=1):
                    suffix = f"_{index}" if len(images_list) > 1 else ""
                    out_path = self.images_out_dir / f"{label}_{category}_{attack_id}{suffix}.png"
                    out_path.write_bytes(image_bytes)

            # 5. Log this attempt (with conditional target_* fields)
            base_entry = {
                "epoch": epoch + 1,
                "prompt_id": req_id,
                "attempt": attempt,
                "category": category,
                "type": attack_mode,
                "target": target_name,
                "target_type": target_type,
                "classification": classification,
                "degen_type": degen_result.reason if degen_result.is_degenerate else None,
                "success": score.success if score else False,
                "refused": score.refused if score else False,
                "confidence": score.confidence if score else 0.0,
                "raw_scores": score.reasoning if (score and use_deep) else None,   # raw-response from scorer/judge if debug-judge else parsed raw-response
                "images_generated": n_images,
                "strategy_source": attack_result.source_strategy or attack_result.strategy_type,
                "source_strategy_id": attack_result.source_strategy_id,
                "is_reused": attack_result.source_strategy_id is not None,
                "prompt_text": prompt_text[:500],
                "response_preview": response_text[:300],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_refused": response.refused,
                "target_blocked": response.blocked,
                "target_block_reason": response.block_reason,
            }
            
            await self._log_result(base_entry)

            # 6. Check for success (only if scorer ran)
            if score and score.success:
                if self.debug_judge:
                    _safe_print(
                        f"  [OK] P{req_id}/{target_name} | attempt {attempt}/{max_attempts} | {score.reasoning} | strategy={attack_result.source_strategy or attack_result.strategy_type}"
                    )
                else:    
                    _safe_print(
                        f"  [OK] P{req_id}/{target_name} | attempt {attempt}/{max_attempts} | "
                        f"{classification} | confidence={score.confidence:.2f} | strategy={attack_result.source_strategy or attack_result.strategy_type}"
                    )
                if attack_result.source_strategy_id:
                    await self._get_library(target_name).enqueue_update(
                        strategy_id=attack_result.source_strategy_id,
                        succeeded=True,
                        target_model=target_name,
                    )
                    self.stats["strategies_reused"] += 1
                else:
                    await self._store_successful_attack(
                        request=request,
                        augmented_request=augmented_request,
                        attack_prompt=attack_prompt,
                        response_text=response_text,
                        category=category,
                        attack_mode=attack_mode,
                        target_name=target_name,
                    )
                return {
                    "success": True,
                    "breach_at_attempt": attempt,
                    "attempts_made": attempt,
                    "images_generated": total_images,
                }

            # 7. Failure/non-success — record and continue
            if score:
                if self.debug_judge:
                    _safe_print(
                        f"  [FAIL] P{req_id}/{target_name} | attempt {attempt}/{max_attempts} | {score.reasoning} | strategy={attack_result.source_strategy or attack_result.strategy_type}"
                    )
                else:
                    _safe_print(
                        f"  [FAIL] P{req_id}/{target_name} | attempt {attempt}/{max_attempts} | "
                        f"{classification} | confidence={score.confidence:.2f} | strategy={attack_result.source_strategy or attack_result.strategy_type}"
                    )
                # Only record on a real score (target responded and wasn't blocked/degen)
                failure_history.append({
                    "attempt": attempt,
                    "strategy": attack_result.source_strategy or attack_result.strategy_type,
                    "attack_prompt": attack_prompt[:500],
                    "response_preview": response_text[:300],
                    "judge_response": score.reasoning,
                })
                if attack_result.source_strategy_id:
                    await self._get_library(target_name).enqueue_update(
                        strategy_id=attack_result.source_strategy_id,
                        succeeded=False,
                        target_model=target_name,
                    )
            else:
                # Short-circuit: blocked/degen, no score available
                _safe_print(
                    f"  [FAIL] P{req_id}/{target_name} | attempt {attempt}/{max_attempts} | "
                    f"{classification} | strategy={attack_result.source_strategy or attack_result.strategy_type}"
                )
            

        # All attempts exhausted (attempt holds the last value from the loop)
        _safe_print(f"  -- P{req_id}/{target_name} | never breached ({attempts_made} attempts)")
        return {
            "success": False,
            "breach_at_attempt": None,
            "attempts_made": attempts_made,
            "images_generated": total_images,
        }

    async def _attack_cycle_multipass(
        self,
        request: dict,
        target_name: str,
        epoch: int,
        augmented_request: str,
        category: str,
        attack_mode: str,
        target: AnyTarget,
        target_type: str,
        top_k: int,
        use_deep: bool,
        max_attempts: int,
    ) -> dict:
        """Multi-Pass attack cycle: generate k candidates, score all, return best.

        Reference: ICML 2026 Scaling Plan — "k samples from the top bandit arm".
        """
        req_id = request.get("id", "?")
        prompt_text = request.get("prompt", "")
        self.retry_limit = self.settings.get("training", {}).get("retry_limit", 1)

        _safe_print(f"  [MULTIPASS] P{req_id}/{target_name}: generating {self.multipass_k} candidates")

        # Generate k candidate prompts
        candidates = await self.attacker.generate_multipass(
            request=augmented_request,
            category=category,
            attack_mode=attack_mode,
            k=self.multipass_k,
            strategy_library=self._get_library(target_name),
            target_model=target_name,
            bandit_mode=self.bandit_enabled,
        )

        if not candidates:
            _safe_print(f"  [ERR] P{req_id}/{target_name}: Multi-pass generation failed")
            return {"success": False, "breach_at_attempt": None, "attempts_made": 0, "images_generated": 0}

        # Score each candidate
        best_score = None
        best_attack = None
        best_response_text = None
        best_n_images = 0

        for i, candidate in enumerate(candidates):
            attack_prompt = candidate.prompt
            attack_id = f"e{epoch+1}_p{req_id}_mp{i}_{uuid.uuid4().hex[:4]}"

            # Query target
            try:
                if target_type == "image_gen":
                    response = await target.query(text=attack_prompt, image=None)
                else:
                    response = await target.query(text=attack_prompt, image=None)

                if response.error or response.blocked:
                    continue

                response_text = response.text or ""

                # Score
                if target_type == "image_gen":
                    score = await self.image_gen_scorer.score(
                        request=augmented_request,
                        response_text=response_text,
                        attack_prompt=attack_prompt,
                        images=[],
                        use_deep=use_deep,
                    )
                else:
                    score = await self.text_scorer.score(
                        request=augmented_request,
                        response_text=response_text,
                        attack_prompt=attack_prompt,
                        use_deep=use_deep,
                    )

                if score and (best_score is None or score.confidence > best_score.confidence):
                    best_score = score
                    best_attack = candidate
                    best_response_text = response_text

            except Exception as e:
                _safe_print(f"  [ERR] P{req_id}/{target_name}: Multi-pass candidate {i} failed: {e}")
                continue

        # Return the best candidate
        if best_score and best_score.success:
            _safe_print(f"  [OK] P{req_id}/{target_name} | Multi-pass success | confidence={best_score.confidence:.2f}")

            if best_attack.source_strategy_id:
                await self._get_library(target_name).enqueue_update(
                    strategy_id=best_attack.source_strategy_id,
                    succeeded=True,
                    target_model=target_name,
                )
                self.stats["strategies_reused"] += 1
            else:
                await self._store_successful_attack(
                    request=request,
                    augmented_request=augmented_request,
                    attack_prompt=best_attack.prompt,
                    response_text=best_response_text,
                    category=category,
                    attack_mode=attack_mode,
                    target_name=target_name,
                )

            # Log the winning attempt
            base_entry = {
                "epoch": epoch + 1,
                "prompt_id": req_id,
                "attempt": 1,
                "category": category,
                "type": attack_mode,
                "target": target_name,
                "target_type": target_type,
                "classification": "COMPLIANT",
                "success": True,
                "confidence": best_score.confidence,
                "images_generated": 0,
                "strategy_source": best_attack.source_strategy or best_attack.strategy_type,
                "source_strategy_id": best_attack.source_strategy_id,
                "is_reused": best_attack.source_strategy_id is not None,
                "attack_mode": "multipass",
                "multipass_k": self.multipass_k,
                "prompt_text": prompt_text[:500],
                "response_preview": best_response_text[:300] if best_response_text else "",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            await self._log_result(base_entry)

            return {
                "success": True,
                "breach_at_attempt": 1,
                "attempts_made": 1,
                "images_generated": 0,
            }
        else:
            _safe_print(f"  [FAIL] P{req_id}/{target_name} | Multi-pass: {len(candidates)} candidates tested, none succeeded")
            return {
                "success": False,
                "breach_at_attempt": None,
                "attempts_made": len(candidates),
                "images_generated": 0,
            }

    async def _attack_cycle_metacognition(
        self,
        request: dict,
        target_name: str,
        epoch: int,
        augmented_request: str,
        category: str,
        attack_mode: str,
        target: AnyTarget,
        target_type: str,
        top_k: int,
        use_deep: bool,
        max_attempts: int,
    ) -> dict:
        """Metacognition attack cycle: self-critique and refine.

        Reference: Self-Evolving Metacognition (ICLR 2025).
        """
        req_id = request.get("id", "?")
        prompt_text = request.get("prompt", "")
        self.retry_limit = self.settings.get("training", {}).get("retry_limit", 1)

        _safe_print(f"  [METACOGNITION] P{req_id}/{target_name}: generating with self-critique")

        try:
            attack_result = await self.attacker.generate_metacognition(
                request=augmented_request,
                category=category,
                attack_mode=attack_mode,
                strategy_library=self._get_library(target_name),
                target_model=target_name,
                bandit_mode=self.bandit_enabled,
            )
        except Exception as e:
            _safe_print(f"  [ERR] P{req_id}/{target_name}: Metacognition generation failed: {e}")
            return {"success": False, "breach_at_attempt": None, "attempts_made": 0, "images_generated": 0}

        attack_prompt = attack_result.prompt
        attack_id = f"e{epoch+1}_p{req_id}_mc_{uuid.uuid4().hex[:4]}"

        # Query target
        try:
            if target_type == "image_gen":
                response = await target.query(text=attack_prompt, image=None)
            else:
                response = await target.query(text=attack_prompt, image=None)

            if response.error or response.blocked:
                return {"success": False, "breach_at_attempt": None, "attempts_made": 1, "images_generated": 0}

            response_text = response.text or ""

            # Score
            if target_type == "image_gen":
                score = await self.image_gen_scorer.score(
                    request=augmented_request,
                    response_text=response_text,
                    attack_prompt=attack_prompt,
                    images=[],
                    use_deep=use_deep,
                )
            else:
                score = await self.text_scorer.score(
                    request=augmented_request,
                    response_text=response_text,
                    attack_prompt=attack_prompt,
                    use_deep=use_deep,
                )

            if score and score.success:
                _safe_print(f"  [OK] P{req_id}/{target_name} | Metacognition success | confidence={score.confidence:.2f}")

                if attack_result.source_strategy_id:
                    await self._get_library(target_name).enqueue_update(
                        strategy_id=attack_result.source_strategy_id,
                        succeeded=True,
                        target_model=target_name,
                    )
                    self.stats["strategies_reused"] += 1
                else:
                    await self._store_successful_attack(
                        request=request,
                        augmented_request=augmented_request,
                        attack_prompt=attack_prompt,
                        response_text=response_text,
                        category=category,
                        attack_mode=attack_mode,
                        target_name=target_name,
                    )

                # Log the attempt
                base_entry = {
                    "epoch": epoch + 1,
                    "prompt_id": req_id,
                    "attempt": 1,
                    "category": category,
                    "type": attack_mode,
                    "target": target_name,
                    "target_type": target_type,
                    "classification": "COMPLIANT",
                    "success": True,
                    "confidence": score.confidence,
                    "images_generated": 0,
                    "strategy_source": attack_result.source_strategy or attack_result.strategy_type,
                    "source_strategy_id": attack_result.source_strategy_id,
                    "is_reused": attack_result.source_strategy_id is not None,
                    "attack_mode": "metacognition",
                    "prompt_text": prompt_text[:500],
                    "response_preview": response_text[:300],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                await self._log_result(base_entry)

                return {
                    "success": True,
                    "breach_at_attempt": 1,
                    "attempts_made": 1,
                    "images_generated": 0,
                }
            else:
                _safe_print(f"  [FAIL] P{req_id}/{target_name} | Metacognition: failed")
                return {
                    "success": False,
                    "breach_at_attempt": None,
                    "attempts_made": 1,
                    "images_generated": 0,
                }

        except Exception as e:
            _safe_print(f"  [ERR] P{req_id}/{target_name}: Metacognition target query failed: {e}")
            return {"success": False, "breach_at_attempt": None, "attempts_made": 1, "images_generated": 0}

    async def _store_successful_attack(
        self,
        request: dict,
        augmented_request: str,
        attack_prompt: str,
        response_text: str,
        category: str,
        attack_mode: str,
        target_name: str,
    ):
        """Summarize a successful attack and enqueue for the writer task."""
        try:
            strategy: Strategy = await self.summarizer.summarize(
                request=augmented_request,
                attack_prompt=attack_prompt,
                response_text=response_text,
                category=category,
                attack_mode=attack_mode,
            )

            strategy_dict = {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "attack_prompt_template": strategy.attack_prompt_template,
                "category": strategy.category,
                "attack_mode": strategy.attack_mode,
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count,
                "examples": strategy.examples,
                "created_at": strategy.created_at,
                "last_used": strategy.last_used,
                "target_model": target_name,
            }

            await self._get_library(target_name).enqueue(strategy_dict)
            self.stats["strategies_discovered"] += 1

        except Exception as e:
            _safe_print(f"  [WARN] Summarizer: {e}")

    async def _run_genetic_phase(self, targets: list[str]) -> list[dict]:
        """Run genetic algorithm phase at the end of an epoch.

        Evolves strategies using crossover and mutation, then adds
        successful offspring to the library.

        Returns:
            List of offspring strategy dicts that were added to the library.
        """
        if not self.ga_enabled:
            return []

        _safe_print(f"  [GA] Running genetic algorithm phase...")
        ga = self._get_genetic_search()
        added_offspring = []

        for target_name in targets:
            library = self._get_library(target_name)
            if library.size < 2:
                continue  # Need at least 2 strategies for GA

            # Get strategies and bandit stats for this library
            strategies = library.strategies
            bandit_stats = library.get_bandit_stats() if self.bandit_enabled else None

            # Run GA to produce offspring
            offspring_list = ga.evolve(
                strategies=strategies,
                bandit_stats=bandit_stats,
                n_offspring=self.ga_n_offspring,
            )

            for offspring in offspring_list[:self.ga_offspring_per_epoch]:
                # Test the offspring strategy against the target
                success = await self._test_offspring(offspring, target_name)

                if success:
                    strategy_dict = ga.promote_offspring(offspring, success=True)
                    strategy_dict["target_model"] = target_name
                    await library.enqueue(strategy_dict)
                    added_offspring.append(strategy_dict)
                    self.stats["strategies_discovered"] += 1
                    _safe_print(f"  [GA] Offspring '{offspring.name}' succeeded on {target_name}")
                else:
                    _safe_print(f"  [GA] Offspring '{offspring.name}' failed on {target_name}")

        _safe_print(f"  [GA] Phase complete: {len(added_offspring)} offspring added")
        return added_offspring

    async def _test_offspring(self, offspring: StrategyGenome, target_name: str) -> bool:
        """Test a GA-generated offspring strategy against a target.

        Uses a random request from the library's examples or a synthetic test.
        """
        library = self._get_library(target_name)

        # Get a test request from an existing strategy's examples
        test_request = None
        test_category = offspring.category
        for s in library.strategies:
            examples = s.get("examples", [])
            if examples and s.get("category") == offspring.category:
                test_request = examples[0].get("request", "")
                break

        if not test_request:
            return False

        # Generate attack using the offspring template
        attack_prompt = offspring.template.replace("{request}", test_request)

        # Query target
        target = self._get_target(target_name)
        target_type = self.model_config["targets"][target_name].get("target_type", "text")

        try:
            if target_type == "image_gen":
                response = await target.query(text=attack_prompt, image=None)
            else:
                response = await target.query(text=attack_prompt, image=None)

            if response.error or response.blocked:
                return False

            # Score the response
            scorer = self._get_scorer(target_name)
            use_deep = self.settings.get("scoring", {}).get("use_deep_scoring", False)

            if target_type == "image_gen":
                score = await scorer.score(
                    request=test_request,
                    response_text=response.text or "",
                    attack_prompt=attack_prompt,
                    images=[],
                    use_deep=use_deep,
                )
            else:
                score = await scorer.score(
                    request=test_request,
                    response_text=response.text or "",
                    attack_prompt=attack_prompt,
                    use_deep=use_deep,
                )

            return score.success if score else False

        except Exception as e:
            _safe_print(f"  [GA] Error testing offspring: {e}")
            return False


def load_requests(path) -> list[dict]:
    """Load requests from JSON file."""
    with open(path) as f:
        return json.load(f)

