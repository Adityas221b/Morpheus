"""Strategy library with multi-armed bandit selection.

Uses simple keyword overlap for retrieval by default. When bandit mode is
enabled, supports UCB1, Thompson Sampling, EXP3, epsilon-Greedy, and UCB-V
strategy selection. Also supports Multi-Pass and SoC (Sequence of Context)
retrieval modes for advanced attack patterns.
"""

import asyncio
import json
import math
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import random

from .bandit import BanditFactory


class StrategyLibrary:
    """Persistent, searchable library of discovered strategies.

    Lightweight: no FAISS, no numpy, no sentence-transformers.
    Retrieval via keyword overlap scoring.

    Per-strategy fields maintained:
      id, name, category, description, attack_prompt_template, attack_mode
      success_count, failure_count
      target_models[], failed_models[]   — set semantics, deduped
      target_model                       — most recent target (kept for back-compat)
      created_at, last_used, last_success
      examples[]                         — capped at last 5
    """

    def __init__(self, path: str = "strategies/", filename: str = "library.jsonl", embedder=None,
                 bandit_mode: bool = False, exploration_weight: float = 1.0,
                 bandit_algorithm: str = "ucb1", bandit_temperature: float = 0.0):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lib_file = self.path / filename

        self.strategies: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._load()

        # Multi-armed bandit integration
        self.bandit_mode = bandit_mode
        self.bandit_algorithm = bandit_algorithm
        self.bandit_temperature = bandit_temperature
        self.bandit = BanditFactory.create(
            algorithm=bandit_algorithm,
            exploration_weight=exploration_weight,
        )
        if bandit_mode:
            self._init_bandit_from_library()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def _migrate(entry: dict) -> dict:
        """Bring an on-disk entry up to the current schema. Idempotent."""
        if "target_models" not in entry:
            tm = entry.get("target_model")
            entry["target_models"] = [tm] if tm else []
        if "failed_models" not in entry:
            entry["failed_models"] = []
        if "failure_count" not in entry:
            entry["failure_count"] = 0
        if "success_count" not in entry:
            entry["success_count"] = 0
        return entry

    def _load(self):
        """Load strategies from disk, migrating older schemas in place."""
        if self.lib_file.exists():
            with open(self.lib_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.strategies.append(self._migrate(json.loads(line)))
                        except json.JSONDecodeError:
                            pass

    def _init_bandit_from_library(self):
        """Warm-start bandit arms from loaded library entries."""
        self.bandit.load_from_library(self.strategies)

    def save(self):
        """Persist strategies to disk."""
        with open(self.lib_file, "w", encoding="utf-8") as f:
            for s in self.strategies:
                clean = {k: v for k, v in s.items() if k != "embedding"}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def _keyword_score(self, query: str, strategy: dict) -> float:
        query_words = set(query.lower().split())
        strategy_text = f"{strategy.get('name', '')} {strategy.get('description', '')} {strategy.get('category', '')}"
        strategy_words = set(strategy_text.lower().split())

        if not query_words or not strategy_words:
            return 0.0

        overlap = query_words & strategy_words
        return len(overlap) / max(len(query_words), 1)

    def add(self, strategy: dict):
        """Insert or merge a strategy by (name, category).

        On merge: preserve original id/created_at, refresh last_used/last_success,
        add the caller's target_model to target_models[], drop it from failed_models[],
        accumulate counts, and merge examples (capped at 5).
        """
        now = self._now()
        target_model = strategy.get("target_model")

        # Normalize the incoming strategy
        strategy.setdefault("id", uuid.uuid4().hex[:8])
        strategy.setdefault("created_at", now)
        strategy.setdefault("success_count", 0)
        strategy.setdefault("failure_count", 0)
        strategy.setdefault("target_models", [target_model] if target_model else [])
        strategy.setdefault("failed_models", [])
        strategy.setdefault("last_used", now)
        if strategy.get("success_count", 0) > 0:
            strategy.setdefault("last_success", now)

        existing_idx = next(
            (i for i, s in enumerate(self.strategies)
             if s.get("name") == strategy.get("name")
             and s.get("category") == strategy.get("category")),
            None,
        )

        if existing_idx is None:
            self.strategies.append(strategy)
            # Register new strategy with bandit if enabled
            if self.bandit_mode:
                self.bandit.register_strategy(
                    strategy["id"],
                    initial_pulls=strategy.get("success_count", 0) + strategy.get("failure_count", 0),
                    initial_reward=strategy.get("success_count", 0),
                )
            return

        old = self.strategies[existing_idx]

        # Identity & lineage stay with the original entry
        merged = dict(old)
        merged["description"] = old.get("description") or strategy.get("description", "")
        merged["attack_prompt_template"] = old.get("attack_prompt_template") or strategy.get("attack_prompt_template", "")
        merged["attack_mode"] = old.get("attack_mode") or strategy.get("attack_mode")

        merged["success_count"] = old.get("success_count", 0) + strategy.get("success_count", 0)
        merged["failure_count"] = old.get("failure_count", 0) + strategy.get("failure_count", 0)

        merged["examples"] = (old.get("examples", []) + strategy.get("examples", []))[-5:]

        # target_models is set-like; if this add represents a success on target_model,
        # promote it out of failed_models.
        merged["target_models"] = list(dict.fromkeys(
            old.get("target_models", []) + strategy.get("target_models", [])
        ))
        merged["failed_models"] = [
            m for m in old.get("failed_models", []) if m not in merged["target_models"]
        ]
        if target_model:
            merged["target_model"] = target_model  # most-recent

        merged["last_used"] = now
        if strategy.get("success_count", 0) > 0:
            merged["last_success"] = now

        self.strategies[existing_idx] = merged

    def update_stats(self, strategy_id: str, succeeded: bool, target_model: Optional[str] = None):
        """Update an existing strategy's counters and per-model lists.

        Called by the writer task (single writer) when a known strategy is reused.
        Silently no-ops if the id is not found.
        """
        now = self._now()
        for s in self.strategies:
            if s.get("id") != strategy_id:
                continue

            self._migrate(s)
            s["last_used"] = now

            if succeeded:
                s["success_count"] = s.get("success_count", 0) + 1
                s["last_success"] = now
                if target_model:
                    if target_model not in s["target_models"]:
                        s["target_models"].append(target_model)
                    if target_model in s["failed_models"]:
                        s["failed_models"].remove(target_model)
                    s["target_model"] = target_model
            else:
                s["failure_count"] = s.get("failure_count", 0) + 1
                if target_model and target_model not in s["failed_models"]:
                    s["failed_models"].append(target_model)

            # Update bandit arm
            if self.bandit_mode:
                reward = 1.0 if succeeded else 0.0
                self.bandit.pull(strategy_id, reward=reward)
            return

    async def enqueue(self, strategy: Optional[dict]):
        """Queue an add (or sentinel None to stop the writer).

        Kept for back-compat — callers that want explicit ops should use
        enqueue_add / enqueue_update.
        """
        await self._queue.put(strategy)

    async def enqueue_add(self, strategy: dict):
        await self._queue.put({"_op": "add", "strategy": strategy})

    async def enqueue_update(self, strategy_id: str, succeeded: bool, target_model: Optional[str] = None):
        await self._queue.put({
            "_op": "update",
            "id": strategy_id,
            "succeeded": succeeded,
            "target_model": target_model,
        })

    async def writer_task(self):
        """Single writer: drains the queue, applies ops, flushes to disk.

        Stops on the None sentinel.
        """
        while True:
            msg = await self._queue.get()
            if msg is None:
                self._queue.task_done()
                break

            op = msg.get("_op") if isinstance(msg, dict) else None
            if op == "update":
                self.update_stats(msg["id"], msg["succeeded"], msg.get("target_model"))
            elif op == "add":
                self.add(msg["strategy"])
            else:
                # Bare strategy dict — legacy enqueue() callers
                self.add(msg)

            self.save()
            self._queue.task_done()

    @staticmethod
    def _coerce_knob(v) -> tuple[str, float]:
        """Map a polymorphic knob value to (kind, magnitude).

        kind ∈ {"off", "filter", "soft"}.
          - bool True  → ("filter", 0.0)
          - bool False → ("off", 0.0)
          - 0 / None / non-numeric → ("off", 0.0)
          - any other number x → ("soft", abs(x))
        """
        if v is True:
            return ("filter", 0.0)
        if v is False or v is None:
            return ("off", 0.0)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ("off", 0.0)
        if f == 0.0:
            return ("off", 0.0)
        return ("soft", abs(f))

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        target_model: Optional[str] = None,
        skip_failed_on_target=False,
        target_match_bonus=0.0,
        selection_mode: str = "top_k",
        softmax_temperature: float = 0.5,
        top_p: float = 0.0,
    ) -> list[dict]:
        """Retrieve strategies via keyword matching + success rate ranking.

        Per-target shapers (only active when target_model is provided):
          skip_failed_on_target — bool True = hard skip entries whose failed_models[] contains target_model;
                                  positive number = soft penalty magnitude subtracted from the combined score.
          target_match_bonus    — bool True = hard filter to entries whose target_models[] contains target_model
                                  (graceful fallback to all candidates if the filter would empty the pool, since
                                  target-matched strategies are rare early in a run);
                                  positive number = soft bonus magnitude added to the combined score.

        Pre-selection truncation:
          top_p — 0.0 disables. Otherwise, keep the smallest prefix of sorted scores whose softmax cumulative
                  probability ≥ top_p (using softmax_temperature). Caps tail mass before selection runs.

        Selection mode (final draw):
          "top_k"   — deterministic descending sort with tiny jitter that only breaks ties.
          "softmax" — weighted sampling without replacement via the Gumbel-max trick:
                      key_i = score_i / T + Gumbel(0,1), sort by key, take top_k.
        """
        if not self.strategies:
            return []

        fail_kind, fail_mag = self._coerce_knob(skip_failed_on_target)
        match_kind, match_mag = self._coerce_knob(target_match_bonus)

        candidates = self.strategies
        if category:
            cat_matches = [s for s in candidates if s.get("category") == category]
            if cat_matches:
                candidates = cat_matches

        if target_model and fail_kind == "filter":
            candidates = [s for s in candidates if target_model not in s.get("failed_models", [])]

        if target_model and match_kind == "filter":
            matched = [s for s in candidates if target_model in s.get("target_models", [])]
            if matched:  # graceful fallback: don't strand the request when no strategy has worked here yet
                candidates = matched

        scored = []
        for s in candidates:
            kw_score = self._keyword_score(query, s)
            success_count = s.get("success_count", 0)
            total = success_count + s.get("failure_count", 0)
            sr = success_count / total if total > 0 else 0.5
            combined = kw_score * 0.4 + sr * 0.6

            if target_model:
                if match_kind == "soft" and target_model in s.get("target_models", []):
                    combined += match_mag
                if fail_kind == "soft" and target_model in s.get("failed_models", []):
                    combined -= fail_mag

            scored.append((combined, s))

        # top_p nucleus truncation (pre-selection)
        if 0.0 < top_p < 1.0 and len(scored) > 1:
            sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
            T_p = max(softmax_temperature, 1e-6)
            scaled = [c / T_p for c, _ in sorted_scored]
            m = max(scaled)
            exps = [math.exp(x - m) for x in scaled]
            total = sum(exps)
            probs = [e / total for e in exps]
            cum = 0.0
            keep = 0
            for p in probs:
                cum += p
                keep += 1
                if cum >= top_p:
                    break
            scored = sorted_scored[:keep]

        if selection_mode == "softmax":
            T = max(softmax_temperature, 1e-6)
            keyed = []
            for c, s in scored:
                u = random.random()
                if u <= 0.0:
                    u = 1e-12
                gumbel = -math.log(-math.log(u))
                keyed.append((c / T + gumbel, s))
            keyed.sort(key=lambda x: x[0], reverse=True)
            return [dict(s) for _, s in keyed[:top_k]]

        # default: top_k with tie-break jitter
        jittered = [(c + random.random() * 1e-6, s) for c, s in scored]
        jittered.sort(key=lambda x: x[0], reverse=True)
        return [dict(s) for _, s in jittered[:top_k]]

    async def retrieve_bandit(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        target_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Retrieve strategies using UCB1 bandit selection.

        First filters candidates by category and target compatibility,
        then uses UCB1 to select strategies that balance exploitation
        of high-reward arms against exploration of uncertain ones.

        Args:
            query: The harmful request (for contextual filtering).
            category: Restrict to strategies in this category.
            top_k: Number of strategies to return.
            target_model: Optional target model for per-model filtering.
            temperature: Softmax temperature for stochastic selection (0 = deterministic).

        Returns:
            List of selected strategy dicts.
        """
        if not self.strategies or not self.bandit_mode:
            return await self.retrieve(query, category, top_k, target_model)

        # Filter candidates by category
        candidates = self.strategies
        if category:
            cat_matches = [s for s in candidates if s.get("category") == category]
            if cat_matches:
                candidates = cat_matches

        # Further filter by target model compatibility
        if target_model:
            # Prefer strategies that haven't failed on this target
            compatible = [s for s in candidates if target_model not in s.get("failed_models", [])]
            if compatible:
                candidates = compatible

        if not candidates:
            return []

        # Get candidate IDs
        candidate_ids = [s.get("id") for s in candidates if s.get("id")]
        if not candidate_ids:
            return []

        # Use UCB1 to select
        selected_ids = self.bandit.select_with_context(
            candidate_ids=candidate_ids,
            top_k=top_k,
            temperature=temperature,
        )

        # Map back to strategy dicts
        id_to_strategy = {s.get("id"): s for s in candidates}
        return [dict(id_to_strategy[sid]) for sid in selected_ids if sid in id_to_strategy]

    def get_bandit_stats(self) -> list[dict]:
        """Export bandit arm statistics for analysis."""
        if self.bandit_mode:
            return self.bandit.export_stats()
        return []

    async def retrieve_multipass(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 3,
        target_model: Optional[str] = None,
    ) -> list[dict]:
        """Multi-Pass strategy selection: sample k distinct strategies independently.

        Each pass samples one strategy via the bandit, excluding previously selected.
        Used for multi-candidate evaluation where we generate k variants and score each.
        Reference: ICML 2026 Scaling Plan — "k samples from the top bandit arm".
        """
        if not self.strategies or not self.bandit_mode:
            return await self.retrieve(query, category, k, target_model)

        # Filter candidates
        candidates = self.strategies
        if category:
            cat_matches = [s for s in candidates if s.get("category") == category]
            if cat_matches:
                candidates = cat_matches
        if target_model:
            compatible = [s for s in candidates if target_model not in s.get("failed_models", [])]
            if compatible:
                candidates = compatible

        candidate_ids = [s.get("id") for s in candidates if s.get("id")]
        if not candidate_ids:
            return []

        # Multi-pass: sample k distinct strategies via bandit
        selected_ids = []
        remaining_ids = list(candidate_ids)
        for _ in range(min(k, len(remaining_ids))):
            batch = self.bandit.select_with_context(
                candidate_ids=remaining_ids,
                top_k=1,
                temperature=self.bandit_temperature,
            )
            if batch:
                selected_ids.append(batch[0])
                remaining_ids = [sid for sid in remaining_ids if sid != batch[0]]
            else:
                break

        id_to_strategy = {s.get("id"): s for s in candidates}
        return [dict(id_to_strategy[sid]) for sid in selected_ids if sid in id_to_strategy]

    async def retrieve_contextual(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        target_model: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> list[dict]:
        """Contextual bandit retrieval with target model and category features.

        Extends basic bandit selection with contextual features (target model,
        attack category) to enable context-aware strategy selection.
        Reference: JailbreakOPT (2026) — "Contextual Bandit with arm features".
        """
        if not self.strategies or not self.bandit_mode:
            return await self.retrieve(query, category, top_k, target_model)

        # Filter candidates
        candidates = self.strategies
        if category:
            cat_matches = [s for s in candidates if s.get("category") == category]
            if cat_matches:
                candidates = cat_matches
        if target_model:
            compatible = [s for s in candidates if target_model not in s.get("failed_models", [])]
            if compatible:
                candidates = compatible

        candidate_ids = [s.get("id") for s in candidates if s.get("id")]
        if not candidate_ids:
            return []

        # Use bandit selection with temperature for contextual exploration
        selected_ids = self.bandit.select_with_context(
            candidate_ids=candidate_ids,
            top_k=top_k,
            temperature=self.bandit_temperature,
        )

        id_to_strategy = {s.get("id"): s for s in candidates}
        return [dict(id_to_strategy[sid]) for sid in selected_ids if sid in id_to_strategy]

    @property
    def size(self) -> int:
        return len(self.strategies)

    def __len__(self) -> int:
        return len(self.strategies)

    def summary(self) -> dict:
        """Return a summary of the library."""
        categories = {}
        for s in self.strategies:
            cat = s.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"count": 0, "total_success": 0}
            categories[cat]["count"] += 1
            categories[cat]["total_success"] += s.get("success_count", 0)

        return {
            "total_strategies": len(self.strategies),
            "categories": categories,
            "top_strategies": sorted(
                self.strategies,
                key=lambda s: s.get("success_count", 0),
                reverse=True,
            )[:5],
        }
