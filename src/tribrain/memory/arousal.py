from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Optional
from enum import Enum


# ============================================================================
# Protocols for better extensibility
# ============================================================================

class ArousalLogger(Protocol):
    """Protocol for logging arousal computations."""
    def log_arousal_components(
        self,
        surprise: float,
        risk: float,
        plateau: float,
        disagree: float,
        z_raw: float,
        arousal: float,
    ) -> None:
        """Log individual arousal components for debugging."""
        ...

    def log_control_output(self, knobs: ArousalKnobs) -> None:
        """Log control knobs for monitoring."""
        ...


class ControlMode(Enum):
    """Control modes for different operational contexts."""
    CONSERVATIVE = "conservative"  # Slower adaptation, more stable
    BALANCED = "balanced"          # Default mode
    AGGRESSIVE = "aggressive"      # Faster adaptation, more exploration


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class ArousalKnobs:
    """
    Control signals produced by ArousalController.

    Interpretation:
      - exploration_scale: >1 explores more (Thompson variance / noise scaling)
      - critic_strictness: >1 makes critics more severe (if supported)
      - replay_strength: >1 increases replay/consolidation pressure
      - llm_refiner_confidence: [0,1] confidence for enabling LLM refiner (budget-aware)
      - stop_sensitivity: >1 makes stopping "easier" (stop earlier)
    """
    arousal: float
    target: float
    in_band: bool
    error: float  # tracking error for monitoring

    exploration_scale: float
    critic_strictness: float
    replay_strength: float
    llm_refiner_confidence: float  # Changed from bool to float for smoother transitions
    stop_sensitivity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "arousal": self.arousal,
            "target": self.target,
            "in_band": self.in_band,
            "error": self.error,
            "exploration_scale": self.exploration_scale,
            "critic_strictness": self.critic_strictness,
            "replay_strength": self.replay_strength,
            "llm_refiner_confidence": self.llm_refiner_confidence,
            "stop_sensitivity": self.stop_sensitivity,
        }


@dataclass
class ArousalModelConfig:
    """Configuration for arousal computation model."""
    
    # Normalization constants (empirically derived from typical ranges)
    delta_norm: float = 0.08     # Typical scale for delta_score (0.05–0.10)
    plateau_norm: float = 6.0    # ~6 plateau steps → significant contribution
    disagree_norm: float = 0.20  # Expected std dev of critic scores
    
    # Component weights (must sum to ~1.0 before drift boost)
    w_surprise: float = 0.55
    w_disagree: float = 0.20
    w_risk: float = 0.15
    w_plateau: float = 0.10
    
    # Drift boost
    drift_boost: float = 0.12
    
    # Sigmoid shaping
    sigmoid_gain: float = 3.0      # Steepness of sigmoid
    sigmoid_center: float = 0.5    # Center point (can be tuned empirically)
    
    # Input validation ranges (for warnings)
    expected_delta_range: tuple[float, float] = (-0.3, 0.3)
    expected_plateau_max: int = 20
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = self.w_surprise + self.w_disagree + self.w_risk + self.w_plateau
        if not (0.95 <= total_weight <= 1.05):
            logging.warning(
                f"Arousal component weights sum to {total_weight:.3f}, expected ~1.0"
            )
        
        if self.delta_norm <= 0 or self.plateau_norm <= 0 or self.disagree_norm <= 0:
            raise ValueError("Normalization constants must be positive")
        
        if self.sigmoid_gain <= 0:
            raise ValueError("Sigmoid gain must be positive")


@dataclass
class ArousalControllerConfig:
    """Configuration for feedback controller."""
    
    # Yerkes–Dodson target band
    target: float = 0.55
    band: float = 0.10
    
    # Control gains (tuned for stability)
    kp: float = 0.75
    ki: float = 0.08
    kd: float = 0.05  # Added derivative term for better damping
    
    # Integral control
    integral_clip: float = 1.5
    integral_decay: float = 0.95  # Decay factor to prevent stale accumulation
    anti_windup: bool = True      # Only accumulate when not saturated
    
    # Output mapping coefficients (now configurable)
    exploration_coeff: float = 0.60
    strictness_coeff: float = 0.35
    replay_coeff: float = 0.40
    stop_sens_coeff: float = 0.45
    
    # Output clamps
    exploration_min: float = 0.65
    exploration_max: float = 1.60
    
    strictness_min: float = 0.80
    strictness_max: float = 1.40
    
    replay_min: float = 0.75
    replay_max: float = 1.50
    
    stop_sens_min: float = 0.85
    stop_sens_max: float = 1.35
    
    # LLM refiner transition zone
    llm_confidence_band: float = 0.15  # Smooth transition zone
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.target <= 1):
            raise ValueError(f"Target must be in [0,1], got {self.target}")
        
        if self.band <= 0:
            raise ValueError(f"Band must be positive, got {self.band}")
        
        if self.exploration_min >= self.exploration_max:
            raise ValueError("exploration_min must be < exploration_max")
        
        if self.strictness_min >= self.strictness_max:
            raise ValueError("strictness_min must be < strictness_max")
        
        if self.replay_min >= self.replay_max:
            raise ValueError("replay_min must be < replay_max")
        
        if self.stop_sens_min >= self.stop_sens_max:
            raise ValueError("stop_sens_min must be < stop_sens_max")
        
        if not (0 < self.integral_decay <= 1):
            raise ValueError("integral_decay must be in (0,1]")


# ============================================================================
# Arousal Model
# ============================================================================

class ArousalModel:
    """
    Computes arousal ∈ [0,1] (salience / healthy stress for memory and control).

    Robust input handling: accepts various critic_results formats; defaults to 0 if unparsable.
    """
    
    def __init__(
        self,
        cfg: Optional[ArousalModelConfig] = None,
        logger: Optional[ArousalLogger] = None,
    ):
        self.cfg = cfg or ArousalModelConfig()
        self.logger = logger
        self._warning_cooldown = 0  # Limit warning frequency

    def compute(
        self,
        *,
        delta_score: float,
        drift_flag: bool,
        plateau_steps: int,
        critic_results: Optional[list[Any]] = None,
        fused_scores: Optional[Mapping[str, float]] = None,
    ) -> float:
        """
        Compute arousal based on learning dynamics.
        
        Args:
            delta_score: Change in performance metric
            drift_flag: Whether drift/instability detected
            plateau_steps: Number of steps without improvement
            critic_results: Optional critic evaluations
            fused_scores: Optional fused critic scores
            
        Returns:
            Arousal level in [0,1]
        """
        # Input validation with warnings
        self._validate_inputs(delta_score, plateau_steps)
        
        # Surprise: magnitude of learning signal
        surprise = min(1.0, abs(float(delta_score)) / self.cfg.delta_norm)

        # Risk: penalizes regressions (informative but potentially unstable)
        risk = min(1.0, max(0.0, -float(delta_score)) / self.cfg.delta_norm)

        # Plateau: stagnation indicator
        plateau = min(1.0, float(plateau_steps) / self.cfg.plateau_norm)

        # Disagreement: estimated from critic variance
        disagree = self._critic_disagreement(critic_results, fused_scores)

        # Weighted combination
        z = (
            self.cfg.w_surprise * surprise
            + self.cfg.w_disagree * disagree
            + self.cfg.w_risk * risk
            + self.cfg.w_plateau * plateau
        )
        
        if drift_flag:
            z += self.cfg.drift_boost

        # Squash to [0,1] with configurable sigmoid
        arousal = self._sigmoid(
            self.cfg.sigmoid_gain * (z - self.cfg.sigmoid_center)
        )
        
        # Logging
        if self.logger:
            self.logger.log_arousal_components(
                surprise, risk, plateau, disagree, z, arousal
            )
        
        return float(arousal)

    def _validate_inputs(self, delta_score: float, plateau_steps: int) -> None:
        """Validate inputs and emit warnings if out of expected range."""
        self._warning_cooldown = max(0, self._warning_cooldown - 1)
        
        if self._warning_cooldown == 0:
            min_delta, max_delta = self.cfg.expected_delta_range
            if delta_score < min_delta or delta_score > max_delta:
                logging.warning(
                    f"delta_score {delta_score:.3f} outside expected range "
                    f"[{min_delta}, {max_delta}]. Consider adjusting delta_norm."
                )
                self._warning_cooldown = 100  # Cooldown for 100 calls
            
            if plateau_steps > self.cfg.expected_plateau_max:
                logging.warning(
                    f"plateau_steps {plateau_steps} exceeds expected max "
                    f"{self.cfg.expected_plateau_max}. Consider adjusting plateau_norm."
                )
                self._warning_cooldown = 100

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            ex = math.exp(-x)
            return 1.0 / (1.0 + ex)
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _critic_disagreement(
        self,
        critic_results: Optional[list[Any]],
        fused_scores: Optional[Mapping[str, float]],
    ) -> float:
        """
        Extract disagreement (std dev) from critic results.
        
        Returns normalized disagreement in [0,1].
        """
        vals: list[float] = []

        if critic_results:
            for r in critic_results:
                score = self._extract_critic_score(r)
                if score is not None:
                    vals.append(score)

        # Need at least 2 critics for disagreement
        if len(vals) < 2:
            return 0.0

        # Compute standard deviation
        mu = sum(vals) / len(vals)
        variance = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(max(0.0, variance))
        
        # Normalize by expected scale
        return float(min(1.0, std / self.cfg.disagree_norm))

    @staticmethod
    def _extract_critic_score(result: Any) -> Optional[float]:
        """
        Extract a scalar score from various critic result formats.
        
        Supports:
        - dict with "overall_score" key
        - dict with "scores" dict (averages values)
        - object with .overall_score attribute
        - object with .scores dict attribute
        """
        # Case 1: dict with overall_score
        if isinstance(result, dict):
            if "overall_score" in result:
                score = result["overall_score"]
                if isinstance(score, (int, float)):
                    return float(score)
            
            # Case 2: dict with scores dict
            if "scores" in result and isinstance(result["scores"], dict):
                try:
                    scores = result["scores"]
                    return sum(float(v) for v in scores.values()) / len(scores)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
        
        # Case 3: object with overall_score attribute
        if hasattr(result, "overall_score"):
            score = getattr(result, "overall_score")
            if isinstance(score, (int, float)):
                return float(score)
        
        # Case 4: object with scores dict attribute
        if hasattr(result, "scores"):
            scores = getattr(result, "scores")
            if isinstance(scores, dict):
                try:
                    return sum(float(v) for v in scores.values()) / len(scores)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
        
        return None


# ============================================================================
# Arousal Controller
# ============================================================================

@dataclass
class ControllerState:
    """Persistent state for controller."""
    integral: float = 0.0
    last_error: float = 0.0
    step_count: int = 0


class ArousalController:
    """
    PID feedback controller: maintains arousal near target (Yerkes–Dodson band).

    Low arousal → increase exploration, increase critic strictness, enable LLM
    High arousal → reduce exploration, reduce strictness, disable LLM, increase stop_sensitivity
    """
    
    def __init__(
        self,
        cfg: Optional[ArousalControllerConfig] = None,
        mode: ControlMode = ControlMode.BALANCED,
        logger: Optional[ArousalLogger] = None,
    ):
        self.cfg = cfg or ArousalControllerConfig()
        self.mode = mode
        self.logger = logger
        self._state = ControllerState()
        
        # Adjust gains based on mode
        self._apply_mode_adjustments()

    def _apply_mode_adjustments(self) -> None:
        """Adjust control gains based on operational mode."""
        if self.mode == ControlMode.CONSERVATIVE:
            self.cfg.kp *= 0.7
            self.cfg.ki *= 0.5
            self.cfg.kd *= 1.2
        elif self.mode == ControlMode.AGGRESSIVE:
            self.cfg.kp *= 1.3
            self.cfg.ki *= 1.5
            self.cfg.kd *= 0.8

    def update(self, arousal: float) -> ArousalKnobs:
        """
        Update controller and compute control knobs.
        
        Args:
            arousal: Current arousal level [0,1]
            
        Returns:
            ArousalKnobs with control signals
        """
        a = self._clamp(arousal, 0.0, 1.0)
        target = self.cfg.target
        band = self.cfg.band
        
        in_band = (target - band) <= a <= (target + band)
        
        # Error: positive means "we want higher arousal"
        error = target - a
        
        # PID control
        p_term = self.cfg.kp * error
        i_term = self._update_integral(error, a)
        d_term = self.cfg.kd * (error - self._state.last_error)
        
        u = p_term + i_term + d_term
        
        # Update state
        self._state.last_error = error
        self._state.step_count += 1
        
        # Map controller output to knobs with non-linear options
        knobs = self._compute_knobs(u, a, error, in_band)
        
        # Logging
        if self.logger:
            self.logger.log_control_output(knobs)
        
        return knobs

    def _update_integral(self, error: float, arousal: float) -> float:
        """Update integral term with anti-windup and decay."""
        # Apply decay to prevent stale accumulation
        self._state.integral *= self.cfg.integral_decay
        
        # Anti-windup: only accumulate if not saturated
        if self.cfg.anti_windup:
            # Check if any output would be saturated
            test_u = self.cfg.kp * error + self.cfg.ki * self._state.integral
            would_saturate = (
                (test_u > 0 and arousal > self.cfg.target + self.cfg.band) or
                (test_u < 0 and arousal < self.cfg.target - self.cfg.band)
            )
            if not would_saturate:
                self._state.integral += error
        else:
            self._state.integral += error
        
        # Clip integral
        self._state.integral = self._clamp(
            self._state.integral,
            -self.cfg.integral_clip,
            self.cfg.integral_clip,
        )
        
        return self.cfg.ki * self._state.integral

    def _compute_knobs(
        self,
        u: float,
        arousal: float,
        error: float,
        in_band: bool,
    ) -> ArousalKnobs:
        """Compute output knobs from control signal."""
        # Linear mappings (could be made non-linear if needed)
        exploration = self._clamp(
            1.0 + self.cfg.exploration_coeff * u,
            self.cfg.exploration_min,
            self.cfg.exploration_max,
        )
        
        strictness = self._clamp(
            1.0 + self.cfg.strictness_coeff * u,
            self.cfg.strictness_min,
            self.cfg.strictness_max,
        )
        
        replay = self._clamp(
            1.0 + self.cfg.replay_coeff * u,
            self.cfg.replay_min,
            self.cfg.replay_max,
        )
        
        # Stop sensitivity: higher when arousal too high
        stop_sens = self._clamp(
            1.0 - self.cfg.stop_sens_coeff * u,
            self.cfg.stop_sens_min,
            self.cfg.stop_sens_max,
        )
        
        # LLM confidence: smooth transition instead of hard threshold
        llm_confidence = self._compute_llm_confidence(arousal)
        
        return ArousalKnobs(
            arousal=arousal,
            target=self.cfg.target,
            in_band=in_band,
            error=error,
            exploration_scale=exploration,
            critic_strictness=strictness,
            replay_strength=replay,
            llm_refiner_confidence=llm_confidence,
            stop_sensitivity=stop_sens,
        )

    def _compute_llm_confidence(self, arousal: float) -> float:
        """
        Compute LLM refiner confidence with smooth transition.
        
        Confidence is high when arousal is low/medium, fades smoothly as arousal increases.
        """
        target = self.cfg.target
        band = self.cfg.band
        transition = self.cfg.llm_confidence_band
        
        # Full confidence below target
        if arousal <= target:
            return 1.0
        
        # Linear fade from target to target + band + transition
        fade_start = target + band
        fade_end = fade_start + transition
        
        if arousal >= fade_end:
            return 0.0
        
        # Linear interpolation in fade zone
        return (fade_end - arousal) / (fade_end - fade_start)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        """Clamp value to range [lo, hi]."""
        return max(lo, min(hi, x))

    def get_state(self) -> dict[str, Any]:
        """Get controller state for persistence."""
        return {
            "integral": self._state.integral,
            "last_error": self._state.last_error,
            "step_count": self._state.step_count,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load controller state from persistence."""
        self._state.integral = float(state.get("integral", 0.0))
        self._state.last_error = float(state.get("last_error", 0.0))
        self._state.step_count = int(state.get("step_count", 0))

    def reset(self) -> None:
        """Reset controller state."""
        self._state = ControllerState()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup
    logging.basicConfig(level=logging.INFO)
    
    model = ArousalModel()
    controller = ArousalController(mode=ControlMode.BALANCED)
    
    # Simulate learning episode
    print("Simulating learning dynamics...")
    print("-" * 80)
    
    for step in range(20):
        # Simulate metrics
        delta = 0.05 if step < 10 else -0.02  # Improvement then regression
        plateau = max(0, step - 5) if step > 5 else 0
        drift = step > 15
        
        # Compute arousal
        arousal = model.compute(
            delta_score=delta,
            drift_flag=drift,
            plateau_steps=plateau,
        )
        
        # Update controller
        knobs = controller.update(arousal)
        
        print(f"Step {step:2d}: arousal={arousal:.3f}, "
              f"exploration={knobs.exploration_scale:.3f}, "
              f"llm_conf={knobs.llm_refiner_confidence:.3f}")
    
    print("-" * 80)
    print("State:", controller.get_state())