from dataclasses import dataclass
import numpy as np

"""
This module provides classes for processing states into digestible formats for the SAC agent.
"""


@dataclass(frozen=True)
class StateNormalizationConfig:
    """
    Normalization constants for TrackingEnv observations.

    Observation:
        [devFromRef, headingError, kappaAhead, lastVelocity, clearance, upcomingRefShift, upcomingMinWidth]

    Notes:
        - headingError is already bounded by [-pi, pi].
        - lastVelocity is bounded by [0, 1].
        - devFromRef, kappaAhead, clearance, upcomingRefShift, upcomingMinWidth are technically declared unbounded, so we clip them to practical ranges.
    """

    max_lateral_error: float = 1.5
    max_abs_kappa: float = 10.0
    max_clearance: float = 1.5
    max_velocity: float = 1.0
    max_ref_shift: float = 0.3    # ±0.3 m covers the full corridor shift range
    max_corridor_width: float = 0.46  # full track max width (meters)


class StateProcessor:
    """
    Converts TrackingEnv raw observations into normalized observations
    suitable for SAC neural networks.

    Raw observation:
        [devFromRef, headingError, kappaAhead, lastVelocity, clearance, upcomingRefShift, upcomingMinWidth]

    Normalized observation:
        roughly in [-1, 1]^7
    """

    def __init__(self, config: StateNormalizationConfig = StateNormalizationConfig()):
        self.config = config
        self.state_dim = 7

    def validate_raw(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.shape != (7,):
            raise ValueError(f"Expected observation shape (7,), got {obs.shape}")

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation contains non-finite values: {obs}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize raw TrackingEnv observation to approximately [-1, 1]^7.
        """
        obs = np.asarray(obs, dtype=np.float32)
        self.validate_raw(obs)

        lateral_error = obs[0]
        heading_error = obs[1]
        kappa_ahead = obs[2]
        last_velocity = obs[3]
        clearance = obs[4]
        ref_shift = obs[5]
        min_width = obs[6]

        c = self.config

        lateral_error_norm = np.clip(
            lateral_error / c.max_lateral_error,
            -1.0,
            1.0,
        )

        heading_error_norm = np.clip(
            heading_error / np.pi,
            -1.0,
            1.0,
        )

        kappa_ahead_norm = np.clip(
            kappa_ahead / c.max_abs_kappa,
            -1.0,
            1.0,
        )

        # Map velocity from [0, max_velocity] to [-1, 1].
        last_velocity_clipped = np.clip(last_velocity, 0.0, c.max_velocity)
        last_velocity_norm = 2.0 * (last_velocity_clipped / c.max_velocity) - 1.0

        clearance_norm = np.clip(
            clearance / c.max_clearance,
            -1.0,
            1.0,
        )

        # Upcoming corridor reference shift: positive = reference shifts right, negative = left.
        ref_shift_norm = np.clip(ref_shift / c.max_ref_shift, -1.0, 1.0)

        # Upcoming minimum corridor width: 0 = extremely narrow, 1 = full track width.
        min_width_norm = np.clip(min_width / c.max_corridor_width, 0.0, 1.0)

        normalized = np.array(
            [
                lateral_error_norm,
                heading_error_norm,
                kappa_ahead_norm,
                last_velocity_norm,
                clearance_norm,
                ref_shift_norm,
                min_width_norm,
            ],
            dtype=np.float32,
        )

        return normalized

    def normalize_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of observations.

        Expected shape:
            (batch_size, 7)
        """
        obs_batch = np.asarray(obs_batch, dtype=np.float32)

        if obs_batch.ndim != 2 or obs_batch.shape[1] != 7:
            raise ValueError(f"Expected observation batch shape (batch_size, 7), got {obs_batch.shape}")
        return np.stack([self.normalize(obs) for obs in obs_batch]).astype(np.float32)


if __name__ == "__main__":
    processor = StateProcessor()

    raw_obs = np.array(
        [
            0.05,   # devFromRef
            0.10,   # headingError
            1.50,   # kappaAhead
            0.40,   # lastVelocity
            0.12,   # clearance
            0.08,   # upcomingRefShift
            0.20,   # upcomingMinWidth
        ],
        dtype=np.float32,
    )

    normalized_obs = processor.normalize(raw_obs)

    print("Raw observation:        ", raw_obs)
    print("Normalized observation: ", normalized_obs)
    print("State dim:              ", processor.state_dim)