"""
Bi-Phasic Replay Engine (TriBrain Section 4.4.3)

This module implements a two-phase replay strategy to stabilize iterative refinement loops:

1) Online micro-replay:
   - Small, targeted batches during active refinement.
   - Focuses on recent, high-surprise/high-error episodes from the current task family.

2) Offline consolidation ("sleep"):
   - Larger, interleaved batches spanning task families and time.
   - Designed to reduce interference and long-horizon drift via interleaving + diversity constraints.

Key robustness features in this implementation:
- Safe behavior on empty inputs (no ZeroDivisionError / quantile on empty list).
- Deterministic sampling option via injected RNG (reproducibility).
- Replay-bias mitigation via replay-count penalties and diversity-based sampling.
- Flexible EpisodeStore adapter: works with common method names or plain in-memory lists.

Expected episode fields (best-effort; only some are required):
- id (str) or episode_id (str) [optional but strongly recommended]
- task_family (str) [recommended]
- timestamp (float seconds) [recommended]
- iteration (int) [recommended]
- delta (float) and/or improvement (float) [optional, used for TD-error proxy]
- critic_scores (dict[str,float]) or fused_scores (dict[str,float]) [optional]

The engine never mutates episodes; it only selects them for replay.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class ReplayBatch:
    """Batch of episodes for replay."""
    episodes: List[Dict[str, Any]]
    phase: str  # "online" or "offline"
    interleaved: bool = False
    timestamp: float = 0.0


class EpisodeStore(Protocol):
    """
    Optional protocol for an EpisodeStore used by this replay engine.

    If your store implements any of these, the engine will use them:
      - get_recent_episodes(task_family: str, current_iter: int, max_age_iters: int, limit: int) -> list[dict]
      - get_recent(task_family: str, limit: int) -> list[dict]
      - get_by_family(task_family: str, limit: int) -> list[dict]
      - list_episodes() -> list[dict]

    Otherwise, the engine will try to treat the store as an iterable of episode dicts.
    """
    ...


def _canonical_episode_id(ep: Dict[str, Any]) -> str:
    """Best-effort stable identifier for an episode."""
    for k in ("id", "episode_id", "uuid"):
        v = ep.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: hash a canonical JSON projection (stable, but may be heavy if ep is huge)
    try:
        payload = json_dumps_canonical(ep)
    except Exception:
        payload = repr(ep)
    return "hash:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def json_dumps_canonical(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _get_float(ep: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = ep.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


class BiPhaseReplayEngine:
    """
    Two-phase replay system for stable continual learning.
    """

    def __init__(
        self,
        online_batch_size: int = 8,
        offline_batch_size: int = 64,
        interleave_ratio: float = 0.5,          # fraction sampled from "old" set in offline
        online_lr: float = 0.001,
        offline_lr: float = 0.01,
        max_replays_per_episode: int = 25,      # soft cap to prevent "rumination"
        replay_penalty_alpha: float = 0.05,     # how strongly replay counts downweight sampling
        diversity_key: str = "signature",       # key used to enforce diversity if present
        rng: Optional[random.Random] = None,
    ):
        if online_batch_size <= 0:
            raise ValueError("online_batch_size must be > 0")
        if offline_batch_size <= 0:
            raise ValueError("offline_batch_size must be > 0")
        if not (0.0 <= interleave_ratio <= 1.0):
            raise ValueError("interleave_ratio must be in [0,1]")

        self.online_batch_size = online_batch_size
        self.offline_batch_size = offline_batch_size
        self.interleave_ratio = interleave_ratio
        self.online_lr = online_lr
        self.offline_lr = offline_lr

        self.max_replays_per_episode = max_replays_per_episode
        self.replay_penalty_alpha = replay_penalty_alpha
        self.diversity_key = diversity_key

        self._rng = rng or random.Random()

        # Track which episodes have been replayed (by stable ID)
        self.replay_counts: Dict[str, int] = {}

    # -----------------------------
    # Public API
    # -----------------------------

    def sample_online_batch(
        self,
        episode_store: Any,
        current_task_family: str,
        current_iter: int,
    ) -> ReplayBatch:
        """
        Sample a small batch for online micro-replay.

        Focus on recent, high-surprise episodes from the current task.
        """
        recent = self._get_recent_episodes(
            episode_store,
            task_family=current_task_family,
            current_iter=current_iter,
            max_age_iters=10,
            limit=max(self.online_batch_size * 4, self.online_batch_size),
        )
        if not recent:
            return ReplayBatch(episodes=[], phase="online", interleaved=False, timestamp=time.time())

        prioritized = self._prioritize_for_sampling(recent)

        # Enforce replay cap + diversity
        batch = self._sample_diverse(prioritized, k=self.online_batch_size)

        self._mark_replayed(batch)
        return ReplayBatch(episodes=batch, phase="online", interleaved=False, timestamp=time.time())

    def sample_offline_batch(
        self,
        episode_store: Any,
        task_families: List[str],
    ) -> ReplayBatch:
        """
        Sample a large interleaved batch for offline consolidation.

        Mix episodes across task families and time to reduce interference.
        """
        if not task_families:
            return ReplayBatch(episodes=[], phase="offline", interleaved=False, timestamp=time.time())

        per_family = max(1, (self.offline_batch_size // len(task_families)) + 1)

        all_eps: List[Dict[str, Any]] = []
        for fam in task_families:
            all_eps.extend(self._get_episodes_by_family(episode_store, task_family=fam, limit=per_family * 4))

        if not all_eps:
            return ReplayBatch(episodes=[], phase="offline", interleaved=False, timestamp=time.time())

        # Split into old/new by timestamp percentile (robust to missing timestamps)
        ts = [float(ep.get("timestamp", 0.0) or 0.0) for ep in all_eps]
        ts_sorted = sorted(ts)
        # 30th percentile index
        idx = int(0.3 * (len(ts_sorted) - 1))
        old_threshold = ts_sorted[idx] if ts_sorted else 0.0

        old_eps = [ep for ep in all_eps if float(ep.get("timestamp", 0.0) or 0.0) < old_threshold]
        new_eps = [ep for ep in all_eps if float(ep.get("timestamp", 0.0) or 0.0) >= old_threshold]

        n_old = int(self.offline_batch_size * self.interleave_ratio)
        n_new = self.offline_batch_size - n_old

        # Prioritize within each pool
        old_pool = self._prioritize_for_sampling(old_eps)
        new_pool = self._prioritize_for_sampling(new_eps)

        sampled: List[Dict[str, Any]] = []
        sampled.extend(self._sample_diverse(old_pool, k=n_old))
        sampled.extend(self._sample_diverse(new_pool, k=n_new))

        # If one side was empty, top up from the other
        if len(sampled) < self.offline_batch_size:
            remaining = self.offline_batch_size - len(sampled)
            fallback_pool = new_pool if len(new_pool) > len(old_pool) else old_pool
            sampled.extend(self._sample_diverse(fallback_pool, k=remaining, already=sampled))

        # Shuffle to truly interleave
        self._rng.shuffle(sampled)

        # Truncate to batch size
        sampled = sampled[: self.offline_batch_size]

        self._mark_replayed(sampled)

        return ReplayBatch(episodes=sampled, phase="offline", interleaved=True, timestamp=time.time())

    def apply_replay(
        self,
        batch: ReplayBatch,
        update_fn: Callable[[Dict[str, Any], float], None],
    ) -> None:
        """
        Apply replay updates with phase-appropriate learning rate.

        update_fn(episode, lr) is user-provided (e.g., update meta-value model / critic reliabilities / schemas).
        """
        lr = self.online_lr if batch.phase == "online" else self.offline_lr
        for ep in batch.episodes:
            update_fn(ep, lr)

    def get_replay_stats(self) -> Dict[str, Any]:
        """Get replay statistics."""
        counts = list(self.replay_counts.values())
        return {
            "unique_episodes_replayed": len(self.replay_counts),
            "avg_replay_count": sum(counts) / len(counts) if counts else 0.0,
            "max_replay_count": max(counts) if counts else 0,
            "min_replay_count": min(counts) if counts else 0,
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _mark_replayed(self, episodes: Sequence[Dict[str, Any]]) -> None:
        for ep in episodes:
            ep_id = _canonical_episode_id(ep)
            self.replay_counts[ep_id] = self.replay_counts.get(ep_id, 0) + 1

    def _prioritize_for_sampling(self, episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize by TD-error proxy, with penalties for over-replayed items.
        """
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ep in episodes:
            ep_id = _canonical_episode_id(ep)
            replay_n = self.replay_counts.get(ep_id, 0)
            if replay_n >= self.max_replays_per_episode:
                continue
            td = self._td_error_proxy(ep)
            score = td - self.replay_penalty_alpha * replay_n
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored]

    def _td_error_proxy(self, ep: Dict[str, Any]) -> float:
        """
        Proxy for "surprise": abs(predicted - observed).

        Uses available fields; if missing, falls back to |delta| or 0.
        """
        delta = _get_float(ep, "delta", 0.0)
        improvement = _get_float(ep, "improvement", delta)
        # High error = unexpected outcomes
        return abs(delta - improvement)

    def _sample_diverse(
        self,
        pool: Sequence[Dict[str, Any]],
        k: int,
        already: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample up to k items from pool while enforcing diversity by diversity_key (if present).
        Deterministic given rng seed.
        """
        if k <= 0 or not pool:
            return []

        already = list(already) if already else []
        used_ids = {_canonical_episode_id(ep) for ep in already}
        used_div = {ep.get(self.diversity_key) for ep in already if ep.get(self.diversity_key) is not None}

        out: List[Dict[str, Any]] = []
        for ep in pool:
            if len(out) >= k:
                break
            ep_id = _canonical_episode_id(ep)
            if ep_id in used_ids:
                continue
            div = ep.get(self.diversity_key)
            if div is not None and div in used_div:
                continue
            out.append(ep)
            used_ids.add(ep_id)
            if div is not None:
                used_div.add(div)

        # If not enough, fill without diversity constraint
        if len(out) < k:
            remaining = [ep for ep in pool if _canonical_episode_id(ep) not in used_ids]
            # deterministic random fill
            self._rng.shuffle(remaining)
            out.extend(remaining[: (k - len(out))])

        return out[:k]

    def _get_recent_episodes(
        self,
        store: Any,
        task_family: str,
        current_iter: int,
        max_age_iters: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Best-effort retrieval of recent episodes."""
        # Common method names
        for name in ("get_recent_episodes", "recent_episodes", "fetch_recent_episodes"):
            fn = getattr(store, name, None)
            if callable(fn):
                try:
                    return list(fn(task_family=task_family, current_iter=current_iter, max_age_iters=max_age_iters, limit=limit))
                except TypeError:
                    # try positional
                    try:
                        return list(fn(task_family, current_iter, max_age_iters, limit))
                    except Exception:
                        pass

        for name in ("get_recent", "recent", "fetch_recent"):
            fn = getattr(store, name, None)
            if callable(fn):
                try:
                    eps = list(fn(task_family=task_family, limit=limit))
                except TypeError:
                    try:
                        eps = list(fn(task_family, limit))
                    except Exception:
                        eps = []
                # filter by iteration if present
                if eps:
                    min_iter = current_iter - max_age_iters
                    out = []
                    for ep in eps:
                        it = ep.get("iteration")
                        if it is None:
                            out.append(ep)
                        else:
                            try:
                                if int(it) >= min_iter:
                                    out.append(ep)
                            except Exception:
                                out.append(ep)
                    return out[:limit]
                return eps

        # If store has list_episodes
        fn = getattr(store, "list_episodes", None)
        if callable(fn):
            try:
                all_eps = list(fn())
            except Exception:
                all_eps = []
        else:
            # Treat store as iterable
            try:
                all_eps = list(store)
            except Exception:
                all_eps = []

        min_iter = current_iter - max_age_iters
        out: List[Dict[str, Any]] = []
        for ep in all_eps:
            if not isinstance(ep, dict):
                continue
            if ep.get("task_family") != task_family:
                continue
            it = ep.get("iteration")
            if it is None:
                out.append(ep)
            else:
                try:
                    if int(it) >= min_iter:
                        out.append(ep)
                except Exception:
                    out.append(ep)

        # sort by iteration/time descending
        out.sort(key=lambda e: (float(e.get("timestamp", 0.0) or 0.0), int(e.get("iteration", 0) or 0)), reverse=True)
        return out[:limit]

    def _get_episodes_by_family(
        self,
        store: Any,
        task_family: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Best-effort retrieval of episodes for a task family."""
        for name in ("get_by_family", "get_episodes_by_family", "fetch_by_family"):
            fn = getattr(store, name, None)
            if callable(fn):
                try:
                    return list(fn(task_family=task_family, limit=limit))
                except TypeError:
                    try:
                        return list(fn(task_family, limit))
                    except Exception:
                        pass

        # fallback to list_episodes / iterable
        fn = getattr(store, "list_episodes", None)
        if callable(fn):
            try:
                all_eps = list(fn())
            except Exception:
                all_eps = []
        else:
            try:
                all_eps = list(store)
            except Exception:
                all_eps = []

        out = [ep for ep in all_eps if isinstance(ep, dict) and ep.get("task_family") == task_family]
        out.sort(key=lambda e: float(e.get("timestamp", 0.0) or 0.0), reverse=True)
        return out[:limit]


class ConsolidationScheduler:
    """
    Schedules offline consolidation ("sleep") phases.

    Triggers offline replay when:
    - sufficient new episodes accumulated, OR
    - scheduled interval reached, OR
    - plateau detected (optional)
    """

    def __init__(
        self,
        episodes_per_consolidation: int = 100,
        consolidation_interval_s: float = 3600.0,
        force_on_plateau: bool = True,
    ):
        if episodes_per_consolidation <= 0:
            raise ValueError("episodes_per_consolidation must be > 0")
        if consolidation_interval_s <= 0:
            raise ValueError("consolidation_interval_s must be > 0")

        self.episodes_per_consolidation = episodes_per_consolidation
        self.consolidation_interval_s = consolidation_interval_s
        self.force_on_plateau = force_on_plateau

        self.episodes_since_last: int = 0
        self.last_consolidation_time: float = time.time()
        self.consolidation_count: int = 0

    def notify_episode(self, n: int = 1) -> None:
        """Call this after recording new episodes."""
        self.episodes_since_last += max(0, int(n))

    def should_run(self, current_time: Optional[float] = None, plateau_detected: bool = False) -> bool:
        """Return True if offline consolidation should run now."""
        now = time.time() if current_time is None else float(current_time)
        time_due = (now - self.last_consolidation_time) >= self.consolidation_interval_s
        count_due = self.episodes_since_last >= self.episodes_per_consolidation
        plateau_due = plateau_detected and self.force_on_plateau
        return bool(time_due or count_due or plateau_due)

    def mark_done(self, current_time: Optional[float] = None) -> None:
        """Record that consolidation completed."""
        now = time.time() if current_time is None else float(current_time)
        self.episodes_since_last = 0
        self.last_consolidation_time = now
        self.consolidation_count += 1

    def get_stats(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Get scheduler statistics."""
        now = time.time() if current_time is None else float(current_time)
        return {
            "consolidations_run": self.consolidation_count,
            "episodes_since_last": self.episodes_since_last,
            "time_since_last_s": max(0.0, now - self.last_consolidation_time),
            "interval_s": self.consolidation_interval_s,
            "episodes_per_consolidation": self.episodes_per_consolidation,
        }
