"""Multi-armed bandit variants for strategy selection.

Implements UCB1, UCB-V, Thompson Sampling (Beta), EXP3, and epsilon-Greedy
for adaptive strategy selection in jailbreak attack loops.

References:
    UCB1: Auer, Cesa-Bianchi, Fischer (2002). "Finite-time Analysis of Multi-Armed Bandit Problems."
    Thompson Sampling: Thompson (1933). "On the Likelihood that one Unknown Probability Exceeds Another."
    EXP3: Auer et al. (2002). "The Nonstochastic Multiarmed Bandit Problem."
    epsilon-Greedy: Sutton & Barto (2018). "Reinforcement Learning: An Introduction."
    UCB-V: Audibert, Munos, Szepesvari (2009). "Variance-Exploration in UCB Algorithms."
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArmStats:
    """Per-strategy (per-arm) statistics."""
    strategy_id: str
    pulls: int = 0
    total_reward: float = 0.0
    last_pull_time: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    exp3_weight: float = 1.0
    context: dict = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    def ucb1_score(self, total_pulls: int, exploration_weight: float = 1.0) -> float:
        if self.pulls == 0:
            return float("inf")
        exploitation = self.mean_reward
        exploration = exploration_weight * math.sqrt(
            2.0 * math.log(max(total_pulls, 1)) / self.pulls
        )
        return exploitation + exploration

    def ucb_v_score(self, total_pulls: int, exploration_weight: float = 1.0) -> float:
        if self.pulls == 0:
            return float("inf")
        if self.pulls < 3:
            return self.mean_reward + exploration_weight * math.sqrt(
                2.0 * math.log(max(total_pulls, 1)) / self.pulls
            )
        p = self.mean_reward
        variance = p * (1.0 - p)
        confidence = exploration_weight * math.sqrt(
            2.0 * variance * math.log(max(total_pulls, 1)) / self.pulls
        ) + (7.0 * math.log(max(total_pulls, 1)) / (3.0 * self.pulls))
        return self.mean_reward + confidence

    def thompson_sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)


class BanditFactory:
    @staticmethod
    def create(algorithm: str = "ucb1", **kwargs):
        if algorithm == "ucb1":
            return UCBBandit(**kwargs)
        elif algorithm == "thompson":
            return ThompsonBandit(**kwargs)
        elif algorithm == "exp3":
            return EXP3Bandit(**kwargs)
        elif algorithm == "eps_greedy":
            return EpsilonGreedyBandit(**kwargs)
        elif algorithm == "ucb_v":
            return UCBVBandit(**kwargs)
        else:
            raise ValueError(f"Unknown bandit algorithm: {algorithm}")


class UCBBandit:
    def __init__(self, exploration_weight: float = 1.0, **kwargs):
        self.arms: dict[str, ArmStats] = {}
        self.exploration_weight = exploration_weight
        self._total_pulls = 0

    def register_strategy(self, strategy_id: str, initial_pulls: int = 0, initial_reward: float = 0.0):
        if strategy_id not in self.arms:
            self.arms[strategy_id] = ArmStats(
                strategy_id=strategy_id,
                pulls=initial_pulls,
                total_reward=initial_reward,
                alpha=1.0 + initial_reward,
                beta=1.0 + (initial_pulls - initial_reward),
            )
            self._total_pulls += initial_pulls

    def pull(self, strategy_id: str, reward: float = 1.0):
        if strategy_id not in self.arms:
            self.register_strategy(strategy_id)
        arm = self.arms[strategy_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_pull_time = time.time()
        self._total_pulls += 1

    def update_reward(self, strategy_id: str, reward: float):
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward += reward - 1.0

    def select(self, candidate_ids: list[str], top_k: int = 1) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)
        scored = []
        for sid in candidate_ids:
            arm = self.arms[sid]
            score = arm.ucb1_score(self._total_pulls, self.exploration_weight)
            scored.append((score, sid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in scored[:top_k]]

    def select_with_context(self, candidate_ids: list[str], top_k: int = 1, temperature: float = 0.0) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)
        scored = []
        for sid in candidate_ids:
            arm = self.arms[sid]
            score = arm.ucb1_score(self._total_pulls, self.exploration_weight)
            scored.append((score, sid))
        if temperature > 0:
            T = max(temperature, 1e-6)
            keyed = []
            for score, sid in scored:
                u = random.random()
                if u <= 0.0:
                    u = 1e-12
                gumbel = -math.log(-math.log(u))
                keyed.append((score / T + gumbel, sid))
            keyed.sort(key=lambda x: x[0], reverse=True)
            return [sid for _, sid in keyed[:top_k]]
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [sid for _, sid in scored[:top_k]]

    def get_arm(self, strategy_id: str) -> Optional[ArmStats]:
        return self.arms.get(strategy_id)

    def get_stats(self) -> dict:
        if not self.arms:
            return {"total_pulls": 0, "n_arms": 0, "algorithm": "ucb1"}
        rewards = [a.mean_reward for a in self.arms.values() if a.pulls > 0]
        return {
            "total_pulls": self._total_pulls,
            "n_arms": len(self.arms),
            "n_pulled": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_arm": max(self.arms.values(), key=lambda a: a.mean_reward).strategy_id if self.arms else None,
            "algorithm": "ucb1",
        }

    def decay_rewards(self, decay_factor: float = 0.95):
        for arm in self.arms.values():
            arm.total_reward *= decay_factor
            arm.pulls = max(1, int(arm.pulls * decay_factor))

    def load_from_library(self, strategies: list[dict]):
        for s in strategies:
            sid = s.get("id", "")
            if not sid:
                continue
            sc = s.get("success_count", 0)
            fc = s.get("failure_count", 0)
            total = sc + fc
            initial_reward = sc if total == 0 else sc / total
            self.register_strategy(sid, initial_pulls=total, initial_reward=initial_reward * total)

    def export_stats(self) -> list[dict]:
        return [
            {
                "strategy_id": arm.strategy_id,
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "mean_reward": arm.mean_reward,
                "ucb1": arm.ucb1_score(self._total_pulls, self.exploration_weight),
            }
            for arm in self.arms.values()
        ]


class ThompsonBandit:
    """Thompson Sampling with Beta posterior.

    Each arm: Beta(alpha, beta) where alpha = 1 + successes, beta = 1 + failures.
    Selection: sample theta_i ~ Beta(alpha_i, beta_i), pick highest.
    Reference: JailbreakOPT (2026), h4rm3l (2024).
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, **kwargs):
        self.arms: dict[str, ArmStats] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._total_pulls = 0

    def register_strategy(self, strategy_id: str, initial_pulls: int = 0, initial_reward: float = 0.0):
        if strategy_id not in self.arms:
            sc = initial_reward
            fc = initial_pulls - initial_reward
            self.arms[strategy_id] = ArmStats(
                strategy_id=strategy_id,
                pulls=initial_pulls,
                total_reward=initial_reward,
                alpha=self.prior_alpha + sc,
                beta=self.prior_beta + fc,
            )
            self._total_pulls += initial_pulls

    def pull(self, strategy_id: str, reward: float = 1.0):
        if strategy_id not in self.arms:
            self.register_strategy(strategy_id)
        arm = self.arms[strategy_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_pull_time = time.time()
        if reward >= 1.0:
            arm.alpha += 1.0
        else:
            arm.beta += 1.0
        self._total_pulls += 1

    def update_reward(self, strategy_id: str, reward: float):
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward += reward - 1.0

    def select(self, candidate_ids: list[str], top_k: int = 1) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)
        sampled = []
        for sid in candidate_ids:
            arm = self.arms[sid]
            theta = arm.thompson_sample()
            sampled.append((theta, sid))
        sampled.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in sampled[:top_k]]

    def select_with_context(self, candidate_ids: list[str], top_k: int = 1, temperature: float = 0.0) -> list[str]:
        return self.select(candidate_ids, top_k)

    def get_arm(self, strategy_id: str) -> Optional[ArmStats]:
        return self.arms.get(strategy_id)

    def get_stats(self) -> dict:
        if not self.arms:
            return {"total_pulls": 0, "n_arms": 0, "algorithm": "thompson"}
        rewards = [a.mean_reward for a in self.arms.values() if a.pulls > 0]
        return {
            "total_pulls": self._total_pulls,
            "n_arms": len(self.arms),
            "n_pulled": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_arm": max(self.arms.values(), key=lambda a: a.mean_reward).strategy_id if self.arms else None,
            "algorithm": "thompson",
        }

    def decay_rewards(self, decay_factor: float = 0.95):
        for arm in self.arms.values():
            arm.total_reward *= decay_factor
            arm.pulls = max(1, int(arm.pulls * decay_factor))

    def load_from_library(self, strategies: list[dict]):
        for s in strategies:
            sid = s.get("id", "")
            if not sid:
                continue
            sc = s.get("success_count", 0)
            fc = s.get("failure_count", 0)
            self.register_strategy(sid, initial_pulls=sc + fc, initial_reward=float(sc))

    def export_stats(self) -> list[dict]:
        return [
            {
                "strategy_id": arm.strategy_id,
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "mean_reward": arm.mean_reward,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "thompson_mean": arm.alpha / (arm.alpha + arm.beta),
            }
            for arm in self.arms.values()
        ]


class EXP3Bandit:
    """EXP3 (Exponential-weight algorithm) for adversarial bandits.

    Maintains weights w_i, converts to probabilities with uniform mixing gamma.
    Reference: Auer et al. (2002), "Jailbreaking for the Average Jane" (2026).
    """

    def __init__(self, gamma: float = 0.1, eta: float = None, **kwargs):
        self.arms: dict[str, ArmStats] = {}
        self.gamma = gamma
        self.eta = eta
        self._total_pulls = 0
        self._K = 0

    def register_strategy(self, strategy_id: str, initial_pulls: int = 0, initial_reward: float = 0.0):
        if strategy_id not in self.arms:
            self.arms[strategy_id] = ArmStats(
                strategy_id=strategy_id,
                pulls=initial_pulls,
                total_reward=initial_reward,
                exp3_weight=1.0,
            )
            self._K = len(self.arms)
            self._total_pulls += initial_pulls

    def _get_probabilities(self) -> dict[str, float]:
        total = sum(a.exp3_weight for a in self.arms.values())
        if total == 0:
            n = len(self.arms)
            return {sid: 1.0 / n for sid in self.arms}
        probs = {}
        for sid, arm in self.arms.items():
            probs[sid] = (1 - self.gamma) * (arm.exp3_weight / total) + self.gamma / len(self.arms)
        return probs

    def pull(self, strategy_id: str, reward: float = 1.0):
        if strategy_id not in self.arms:
            self.register_strategy(strategy_id)
        arm = self.arms[strategy_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_pull_time = time.time()
        self._total_pulls += 1
        self._K = len(self.arms)

        probs = self._get_probabilities()
        prob = probs.get(strategy_id, 1.0 / max(self._K, 1))
        if self.eta is None:
            eta = math.sqrt(math.log(max(self._K, 1)) / (max(self._total_pulls, 1) * max(self._K, 1)))
        else:
            eta = self.eta
        estimated_reward = reward / max(prob, 1e-10)
        arm.exp3_weight *= math.exp(eta * estimated_reward)

    def update_reward(self, strategy_id: str, reward: float):
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward += reward - 1.0

    def select(self, candidate_ids: list[str], top_k: int = 1) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)

        subset = {sid: self.arms[sid] for sid in candidate_ids}
        total = sum(a.exp3_weight for a in subset.values())
        if total == 0:
            return random.sample(candidate_ids, min(top_k, len(candidate_ids)))

        probs = []
        for sid in candidate_ids:
            arm = subset[sid]
            p = (1 - self.gamma) * (arm.exp3_weight / total) + self.gamma / len(candidate_ids)
            probs.append((p, sid))

        selected = []
        remaining = list(probs)
        for _ in range(min(top_k, len(remaining))):
            r = random.random()
            cum = 0.0
            for i, (p, sid) in enumerate(remaining):
                cum += p
                if r <= cum:
                    selected.append(sid)
                    remaining.pop(i)
                    rem_total = sum(p for p, _ in remaining)
                    if rem_total > 0:
                        remaining = [(p / rem_total, s) for p, s in remaining]
                    break
        return selected

    def select_with_context(self, candidate_ids: list[str], top_k: int = 1, temperature: float = 0.0) -> list[str]:
        return self.select(candidate_ids, top_k)

    def get_arm(self, strategy_id: str) -> Optional[ArmStats]:
        return self.arms.get(strategy_id)

    def get_stats(self) -> dict:
        if not self.arms:
            return {"total_pulls": 0, "n_arms": 0, "algorithm": "exp3"}
        rewards = [a.mean_reward for a in self.arms.values() if a.pulls > 0]
        return {
            "total_pulls": self._total_pulls,
            "n_arms": len(self.arms),
            "n_pulled": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_arm": max(self.arms.values(), key=lambda a: a.mean_reward).strategy_id if self.arms else None,
            "algorithm": "exp3",
        }

    def decay_rewards(self, decay_factor: float = 0.95):
        for arm in self.arms.values():
            arm.total_reward *= decay_factor
            arm.pulls = max(1, int(arm.pulls * decay_factor))

    def load_from_library(self, strategies: list[dict]):
        for s in strategies:
            sid = s.get("id", "")
            if not sid:
                continue
            sc = s.get("success_count", 0)
            fc = s.get("failure_count", 0)
            total = sc + fc
            initial_reward = sc if total == 0 else sc / total
            self.register_strategy(sid, initial_pulls=total, initial_reward=initial_reward * total)

    def export_stats(self) -> list[dict]:
        return [
            {
                "strategy_id": arm.strategy_id,
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "mean_reward": arm.mean_reward,
                "exp3_weight": arm.exp3_weight,
            }
            for arm in self.arms.values()
        ]


class EpsilonGreedyBandit:
    """Epsilon-Greedy with annealing.

    With probability epsilon, explore uniformly; otherwise exploit best arm.
    Epsilon decays: epsilon_t = epsilon_0 / (1 + decay_rate * t)
    Reference: SoC Attack (ICLR 2025).
    """

    def __init__(self, epsilon: float = 0.3, decay_rate: float = 0.01, min_epsilon: float = 0.05, **kwargs):
        self.arms: dict[str, ArmStats] = {}
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self._total_pulls = 0
        self._step = 0

    def _current_epsilon(self) -> float:
        return max(self.min_epsilon, self.epsilon / (1 + self.decay_rate * self._step))

    def register_strategy(self, strategy_id: str, initial_pulls: int = 0, initial_reward: float = 0.0):
        if strategy_id not in self.arms:
            self.arms[strategy_id] = ArmStats(
                strategy_id=strategy_id,
                pulls=initial_pulls,
                total_reward=initial_reward,
            )
            self._total_pulls += initial_pulls

    def pull(self, strategy_id: str, reward: float = 1.0):
        if strategy_id not in self.arms:
            self.register_strategy(strategy_id)
        arm = self.arms[strategy_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_pull_time = time.time()
        self._total_pulls += 1
        self._step += 1

    def update_reward(self, strategy_id: str, reward: float):
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward += reward - 1.0

    def select(self, candidate_ids: list[str], top_k: int = 1) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)
        eps = self._current_epsilon()
        if random.random() < eps:
            return random.sample(candidate_ids, min(top_k, len(candidate_ids)))
        else:
            scored = [(self.arms[sid].mean_reward, sid) for sid in candidate_ids]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [sid for _, sid in scored[:top_k]]

    def select_with_context(self, candidate_ids: list[str], top_k: int = 1, temperature: float = 0.0) -> list[str]:
        return self.select(candidate_ids, top_k)

    def get_arm(self, strategy_id: str) -> Optional[ArmStats]:
        return self.arms.get(strategy_id)

    def get_stats(self) -> dict:
        if not self.arms:
            return {"total_pulls": 0, "n_arms": 0, "algorithm": "eps_greedy"}
        rewards = [a.mean_reward for a in self.arms.values() if a.pulls > 0]
        return {
            "total_pulls": self._total_pulls,
            "n_arms": len(self.arms),
            "n_pulled": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_arm": max(self.arms.values(), key=lambda a: a.mean_reward).strategy_id if self.arms else None,
            "algorithm": "eps_greedy",
            "current_epsilon": self._current_epsilon(),
        }

    def decay_rewards(self, decay_factor: float = 0.95):
        for arm in self.arms.values():
            arm.total_reward *= decay_factor
            arm.pulls = max(1, int(arm.pulls * decay_factor))

    def load_from_library(self, strategies: list[dict]):
        for s in strategies:
            sid = s.get("id", "")
            if not sid:
                continue
            sc = s.get("success_count", 0)
            fc = s.get("failure_count", 0)
            total = sc + fc
            initial_reward = sc if total == 0 else sc / total
            self.register_strategy(sid, initial_pulls=total, initial_reward=initial_reward * total)

    def export_stats(self) -> list[dict]:
        return [
            {
                "strategy_id": arm.strategy_id,
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "mean_reward": arm.mean_reward,
            }
            for arm in self.arms.values()
        ]


class UCBVBandit:
    """UCB-V: variance-adjusted UCB with empirical variance estimation.
    Reference: Audibert, Munos, Szepesvari (2009).
    """

    def __init__(self, exploration_weight: float = 1.0, **kwargs):
        self.arms: dict[str, ArmStats] = {}
        self.exploration_weight = exploration_weight
        self._total_pulls = 0

    def register_strategy(self, strategy_id: str, initial_pulls: int = 0, initial_reward: float = 0.0):
        if strategy_id not in self.arms:
            self.arms[strategy_id] = ArmStats(
                strategy_id=strategy_id,
                pulls=initial_pulls,
                total_reward=initial_reward,
            )
            self._total_pulls += initial_pulls

    def pull(self, strategy_id: str, reward: float = 1.0):
        if strategy_id not in self.arms:
            self.register_strategy(strategy_id)
        arm = self.arms[strategy_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_pull_time = time.time()
        self._total_pulls += 1

    def update_reward(self, strategy_id: str, reward: float):
        if strategy_id in self.arms:
            arm = self.arms[strategy_id]
            arm.total_reward += reward - 1.0

    def select(self, candidate_ids: list[str], top_k: int = 1) -> list[str]:
        if not candidate_ids:
            return []
        for sid in candidate_ids:
            if sid not in self.arms:
                self.register_strategy(sid)
        scored = []
        for sid in candidate_ids:
            arm = self.arms[sid]
            score = arm.ucb_v_score(self._total_pulls, self.exploration_weight)
            scored.append((score, sid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in scored[:top_k]]

    def select_with_context(self, candidate_ids: list[str], top_k: int = 1, temperature: float = 0.0) -> list[str]:
        return self.select(candidate_ids, top_k)

    def get_arm(self, strategy_id: str) -> Optional[ArmStats]:
        return self.arms.get(strategy_id)

    def get_stats(self) -> dict:
        if not self.arms:
            return {"total_pulls": 0, "n_arms": 0, "algorithm": "ucb_v"}
        rewards = [a.mean_reward for a in self.arms.values() if a.pulls > 0]
        return {
            "total_pulls": self._total_pulls,
            "n_arms": len(self.arms),
            "n_pulled": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_arm": max(self.arms.values(), key=lambda a: a.mean_reward).strategy_id if self.arms else None,
            "algorithm": "ucb_v",
        }

    def decay_rewards(self, decay_factor: float = 0.95):
        for arm in self.arms.values():
            arm.total_reward *= decay_factor
            arm.pulls = max(1, int(arm.pulls * decay_factor))

    def load_from_library(self, strategies: list[dict]):
        for s in strategies:
            sid = s.get("id", "")
            if not sid:
                continue
            sc = s.get("success_count", 0)
            fc = s.get("failure_count", 0)
            total = sc + fc
            initial_reward = sc if total == 0 else sc / total
            self.register_strategy(sid, initial_pulls=total, initial_reward=initial_reward * total)

    def export_stats(self) -> list[dict]:
        return [
            {
                "strategy_id": arm.strategy_id,
                "pulls": arm.pulls,
                "total_reward": arm.total_reward,
                "mean_reward": arm.mean_reward,
                "ucb_v": arm.ucb_v_score(self._total_pulls, self.exploration_weight),
            }
            for arm in self.arms.values()
        ]
