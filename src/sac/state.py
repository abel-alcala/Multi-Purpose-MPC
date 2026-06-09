from dataclasses import dataclass
import numpy as np

"""
This module provides classes for processing states into digestible formats for the SAC agent.
"""


@dataclass(frozen=True)
class StateNormalizationConfig:
    """
    Normalization constants for TrackingEnv observations.

    Raw observation:
        [
            devFromRef,
            headingError,
            kappaAhead,
            lastVelocity,
            clearance,
            upcomingRefShift,
            upcomingMinWidth,
            lidar_0,
            lidar_1,
            ...
        ]

    Notes:
        - headingError is bounded by [-pi, pi].
        - lastVelocity is bounded by [0, 1].
        - lidar ranges are bounded by [0, max_lidar_range].
        - devFromRef, kappaAhead, clearance, upcomingRefShift, upcomingMinWidth
          are clipped to practical ranges.
    """

    max_lateral_error: float = 1.5
    max_abs_kappa: float = 10.0
    max_clearance: float = 1.5
    max_velocity: float = 1.0
    max_ref_shift: float = 0.3
    max_corridor_width: float = 0.46

    max_lidar_range: float = 2.0  # MUST MATCH TRACKING ENV
    num_lidar_measurements: int = 19  # MUST MATCH TRACKING ENV


class StateProcessor:
    """
    Converts TrackingEnv raw observations into normalized observations
    suitable for SAC neural networks.

    Raw observation:
        [
            devFromRef,
            headingError,
            kappaAhead,
            lastVelocity,
            clearance,
            upcomingRefShift,
            upcomingMinWidth,
            lidar_0,
            lidar_1,
            ...
        ]

    Normalized observation:
        roughly in [-1, 1]^(7 + num_lidar_measurements)
    """

    BASE_STATE_DIM = 7

    def __init__(self, config: StateNormalizationConfig = StateNormalizationConfig()):
        self.config = config
        self.state_dim = self.BASE_STATE_DIM + config.num_lidar_measurements

    def validate_raw(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.shape != (self.state_dim,):
            raise ValueError(
                f"Expected observation shape {(self.state_dim,)}, got {obs.shape}. "
                f"Expected 7 base features + {self.config.num_lidar_measurements} lidar measurements."
            )

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation contains non-finite values: {obs}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize raw TrackingEnv observation to approximately [-1, 1]^state_dim.
        """
        obs = np.asarray(obs, dtype=np.float32)
        self.validate_raw(obs)

        dev_from_ref = obs[0]
        heading_error = obs[1]
        kappa_ahead = obs[2]
        last_velocity = obs[3]
        clearance = obs[4]
        ref_shift = obs[5]
        min_width = obs[6]
        lidar_ranges = obs[7:]

        c = self.config

        dev_from_ref_norm = np.clip(
            dev_from_ref / c.max_lateral_error,
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

        # Upcoming corridor reference shift:
        # positive = reference shifts right, negative = left.
        ref_shift_norm = np.clip(
            ref_shift / c.max_ref_shift,
            -1.0,
            1.0,
        )

        # Upcoming minimum corridor width:
        # Maps [0, max_corridor_width] to [-1, 1].
        #
        # -1 means extremely narrow.
        #  1 means full expected width.
        min_width_clipped = np.clip(min_width, 0.0, c.max_corridor_width)
        min_width_norm = 2.0 * (min_width_clipped / c.max_corridor_width) - 1.0

        # Lidar ranges:
        # Maps [0, max_lidar_range] to [-1, 1].
        #
        # -1 means obstacle extremely close.
        #  1 means no obstacle detected within max range.
        lidar_clipped = np.clip(lidar_ranges, 0.0, c.max_lidar_range)
        lidar_norm = 2.0 * (lidar_clipped / c.max_lidar_range) - 1.0

        base_normalized = np.array(
            [
                dev_from_ref_norm,
                heading_error_norm,
                kappa_ahead_norm,
                last_velocity_norm,
                clearance_norm,
                ref_shift_norm,
                min_width_norm,
            ],
            dtype=np.float32,
        )

        normalized = np.concatenate(
            [base_normalized, lidar_norm.astype(np.float32)]
        ).astype(np.float32)

        return normalized

    def normalize_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of observations.

        Expected shape:
            (batch_size, state_dim)
        """
        obs_batch = np.asarray(obs_batch, dtype=np.float32)

        if obs_batch.ndim != 2 or obs_batch.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected observation batch shape (batch_size, {self.state_dim}), "
                f"got {obs_batch.shape}"
            )

        return np.stack([self.normalize(obs) for obs in obs_batch]).astype(np.float32)


if __name__ == "__main__":
    processor = StateProcessor()

    base_obs = np.array(
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

    lidar_obs = np.linspace(
        0.0,
        processor.config.max_lidar_range,
        processor.config.num_lidar_measurements,
        dtype=np.float32,
    )

    raw_obs = np.concatenate([base_obs, lidar_obs]).astype(np.float32)

    normalized_obs = processor.normalize(raw_obs)

    print("Raw observation shape:        ", raw_obs.shape)
    print("Normalized observation shape: ", normalized_obs.shape)
    print("State dim:                    ", processor.state_dim)
    print("Raw observation:              ", raw_obs)
    print("Normalized observation:       ", normalized_obs)