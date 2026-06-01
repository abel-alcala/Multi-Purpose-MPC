from dataclasses import dataclass
import numpy as np

"""
This module provides classes for processing actions into digestible formats for the SAC agent.
"""


@dataclass(frozen=True)
class ActionBounds:
    """
    Real TrackingEnv action bounds.

    Action:
        [velocity, steeringAngle]

    velocity:
        [0, 1] m/s

    steeringAngle:
        [-0.66, 0.66] rad
    """

    velocity_min: float = 0.0
    velocity_max: float = 1.0
    delta_min: float = -0.66
    delta_max: float = 0.66


class ActionProcessor:
    """
    Converts between SAC normalized actions and TrackingEnv raw actions.

    SAC actor output:
        normalized_action in [-1, 1]^2

    Environment action:
        raw_action = [velocity, steeringAngle]
        velocity in [0, 1]
        steeringAngle in [-0.66, 0.66]
    """

    def __init__(self, bounds: ActionBounds = ActionBounds()):
        self.bounds = bounds

        self.low = np.array(
            [bounds.velocity_min, bounds.delta_min],
            dtype=np.float32,
        )

        self.high = np.array(
            [bounds.velocity_max, bounds.delta_max],
            dtype=np.float32,
        )

        self.action_dim = 2

    def validate_raw(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError(f"Expected raw action shape (2,), got {action.shape}")

        if not np.all(np.isfinite(action)):
            raise ValueError(f"Raw action contains non-finite values: {action}")

        if np.any(action < self.low) or np.any(action > self.high):
            raise ValueError(
                f"Raw action {action} outside bounds. "
                f"Expected low={self.low}, high={self.high}"
            )

    def validate_normalized(self, normalized_action: np.ndarray) -> None:
        normalized_action = np.asarray(normalized_action, dtype=np.float32)

        if normalized_action.shape != (2,):
            raise ValueError(
                f"Expected normalized action shape (2,), got {normalized_action.shape}"
            )

        if not np.all(np.isfinite(normalized_action)):
            raise ValueError(
                f"Normalized action contains non-finite values: {normalized_action}"
            )

        if np.any(normalized_action < -1.0) or np.any(normalized_action > 1.0):
            raise ValueError(
                f"Normalized action {normalized_action} outside [-1, 1]^2"
            )

    def scale_from_normalized(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Map normalized action from [-1, 1]^2 to real TrackingEnv action bounds.

        normalized velocity:
            -1 -> 0.0 m/s
             1 -> 1.0 m/s

        normalized steering:
            -1 -> -0.66 rad
             1 ->  0.66 rad
        """
        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        self.validate_normalized(normalized_action)

        raw_action = self.low + 0.5 * (normalized_action + 1.0) * (self.high - self.low)
        return raw_action.astype(np.float32)

    def normalize_raw(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Map raw TrackingEnv action back to normalized [-1, 1]^2.
        """
        raw_action = np.asarray(raw_action, dtype=np.float32)
        self.validate_raw(raw_action)

        normalized_action = 2.0 * (raw_action - self.low) / (self.high - self.low) - 1.0
        return normalized_action.astype(np.float32)

    def clip_raw(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Clip raw action to environment bounds.
        Useful before passing action to env.step(action).
        """
        raw_action = np.asarray(raw_action, dtype=np.float32)
        return np.clip(raw_action, self.low, self.high).astype(np.float32)

    def clip_normalized(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Clip normalized SAC action to [-1, 1]^2.
        """
        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        return np.clip(normalized_action, -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    processor = ActionProcessor()

    normalized_action = np.array([0.0, 0.5], dtype=np.float32)
    raw_action = processor.scale_from_normalized(normalized_action)
    reconstructed_normalized_action = processor.normalize_raw(raw_action)

    print("Normalized action:              ", normalized_action)
    print("Raw TrackingEnv action:          ", raw_action)
    print("Reconstructed normalized action: ", reconstructed_normalized_action)
    print("Action dim:                      ", processor.action_dim)