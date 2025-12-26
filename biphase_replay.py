"""
Bi-Phasic Replay Engine (Section 4.4.3).

Implements two-phase replay:
1. Online micro-replay: Small, targeted updates during active refinement
2. Offline consolidation: Large interleaved batches across tasks (sleep phase)

This prevents catastrophic forgetting and reduces interference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class ReplayBatch:
    """Batch of episodes for replay."""
    episodes: list[dict[str, Any]]
    phase: str  # "online" or "offline"
    interleaved: bool = False
    timestamp: float = 0.0


class BiPhaseReplayEngine:
    """
    Two-phase replay system for stable continual learning.
    
    Online phase: Fast micro-updates during active refinement
    Offline phase: Slower consolidation across task families
    """
    
    def __init__(
        self,
        online_batch_size: int = 8,
        offline_batch_size: int = 64,
        interleave_ratio: float = 0.5,  # Mix old and new episodes
        online_lr: float = 0.001,  # Small learning rate for online
        offline_lr: float = 0.01,  # Larger for offline
    ):
        self.online_batch_size = online_batch_size
        self.offline_batch_size = offline_batch_size
        self.interleave_ratio = interleave_ratio
        self.online_lr = online_lr
        self.offline_lr = offline_lr
        
        # Track which episodes have been replayed
        self.replay_counts: dict[str, int] = {}
        
    def sample_online_batch(
        self,
        episode_store: Any,  # EpisodeStore
        current_task_family: str,
        current_iter: int,
    ) -> ReplayBatch:
        """
        Sample a small batch for online micro-replay.
        
        Focus on recent, high-error episodes from the current task.
        """
        # Get recent episodes from current task
        recent_episodes = self._get_recent_episodes(
            episode_store,
            task_family=current_task_family,
            max_age_iters=10,
            limit=self.online_batch_size * 2,
        )
        
        # Prioritize by TD error (high-error episodes)
        prioritized = self._prioritize_by_td_error(recent_episodes)
        
        # Take top K
        batch_eps = prioritized[:self.online_batch_size]
        
        # Update replay counts
        for ep in batch_eps:
            ep_id = ep.get("id", "")
            self.replay_counts[ep_id] = self.replay_counts.get(ep_id, 0) + 1
            
        return ReplayBatch(
            episodes=batch_eps,
            phase="online",
            interleaved=False,
        )
        
    def sample_offline_batch(
        self,
        episode_store: Any,
        task_families: list[str],
    ) -> ReplayBatch:
        """
        Sample a large interleaved batch for offline consolidation.
        
        Mix episodes across task families and time to reduce interference.
        """
        # Get episodes from all task families
        all_episodes = []
        
        for family in task_families:
            family_eps = self._get_episodes_by_family(
                episode_store,
                task_family=family,
                limit=self.offline_batch_size // len(task_families) + 1,
            )
            all_episodes.extend(family_eps)
            
        # Interleave old and new
        old_threshold = np.quantile([ep.get("timestamp", 0) for ep in all_episodes], 0.3)
        old_episodes = [ep for ep in all_episodes if ep.get("timestamp", 0) < old_threshold]
        new_episodes = [ep for ep in all_episodes if ep.get("timestamp", 0) >= old_threshold]
        
        # Mix according to interleave_ratio
        n_old = int(self.offline_batch_size * self.interleave_ratio)
        n_new = self.offline_batch_size - n_old
        
        batch_eps = (
            random.sample(old_episodes, min(n_old, len(old_episodes))) +
            random.sample(new_episodes, min(n_new, len(new_episodes)))
        )
        
        # Shuffle to truly interleave
        random.shuffle(batch_eps)
        
        # Update replay counts
        for ep in batch_eps:
            ep_id = ep.get("id", "")
            self.replay_counts[ep_id] = self.replay_counts.get(ep_id, 0) + 1
            
        return ReplayBatch(
            episodes=batch_eps[:self.offline_batch_size],
            phase="offline",
            interleaved=True,
        )
        
    def apply_replay(
        self,
        batch: ReplayBatch,
        update_fn: Callable[[dict, float], None],
    ):
        """
        Apply replay updates with phase-appropriate learning rate.
        """
        lr = self.online_lr if batch.phase == "online" else self.offline_lr
        
        for episode in batch.episodes:
            # Compute TD error or other update signal
            # (This would come from MetaValueModel in practice)
            update_fn(episode, lr)
            
    def get_replay_stats(self) -> dict[str, Any]:
        """Get replay statistics."""
        counts = list(self.replay_counts.values())
        return {
            "total_episodes_replayed": len(self.replay_counts),
            "avg_replay_count": np.mean(counts) if counts else 0.0,
            "max_replay_count": max(counts) if counts else 0,
            "min_replay_count": min(counts) if counts else 0,
        }
        
    def _get_recent_episodes(
        self,
        store: Any,
        task_family: str,
        max_age_iters: int,
        limit: int,
    ) -> list[dict]:
        """Get recent episodes from a task family."""
        # This would query the EpisodeStore
        # Placeholder implementation
        return []
        
    def _get_episodes_by_family(
        self,
        store: Any,
        task_family: str,
        limit: int,
    ) -> list[dict]:
        """Get episodes from a specific task family."""
        # This would query the EpisodeStore
        # Placeholder implementation
        return []
        
    def _prioritize_by_td_error(self, episodes: list[dict]) -> list[dict]:
        """Sort episodes by TD error (high error first)."""
        # Compute proxy for TD error from episode data
        def td_error_proxy(ep: dict) -> float:
            delta = ep.get("delta", 0.0)
            improvement = ep.get("improvement", 0.0)
            # High error = unexpected outcomes
            return abs(delta - improvement)
            
        return sorted(episodes, key=td_error_proxy, reverse=True)


class ConsolidationScheduler:
    """
    Schedules offline consolidation ('sleep') phases.
    
    Triggers offline replay when:
    - Sufficient new episodes accumulated
    - Performance plateau detected
    - Scheduled interval reached
    """
    
    def __init__(
        self,
        episodes_per_consolidation: int = 100,
        consolidation_interval_s: float = 3600,  # 1 hour
        force_on_plateau: bool = True,
    ):
        self.episodes_per_consolidation = episodes_per_consolidation
        self.consolidation_interval_s = consolidation_interval_s
        self.force_on_plateau = force_on_plateau
        
        self.episodes_since_last: int = 0
        self.last_consolidation_time: float = 0.0
        self.consolidation_count: int = 0
        
    def should_consolidate(
        self,
        current_time: float,
        plateau_detected: bool = False,
    ) -> bool:
        """Determine if consolidation should run now."""
        # Trigger 1: Episode count threshold
        if self.episodes_since_last >= self.episodes_per_consolidation:
            return True
            
        # Trigger 2: Time interval
        if current_time - self.last_consolidation_time >= self.consolidation_interval_s:
            return True
            
        # Trigger 3: Plateau detected
        if self.force_on_plateau and plateau_detected:
            return True
            
        return False
        
    def mark_episode_added(self):
        """Record that a new episode was added."""
        self.episodes_since_last += 1
        
    def mark_consolidation_done(self, current_time: float):
        """Record that consolidation completed."""
        self.episodes_since_last = 0
        self.last_consolidation_time = current_time
        self.consolidation_count += 1
        
    def get_stats(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "consolidations_run": self.consolidation_count,
            "episodes_since_last": self.episodes_since_last,
            "time_since_last_s": 0.0,  # Would compute from current time
        }
