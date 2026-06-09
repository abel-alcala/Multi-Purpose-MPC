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
            lateralError,
            headingError,
            kappaAhead,
            lastVelocity,
            clearance,
            lidar_0,
            lidar_1,
            ...
        ]

    Notes:
        - headingError is bounded by [-pi, pi].
        - lastVelocity is bounded by [0, 1].
        - lidar ranges are bounded by [0, max_lidar_range].
        - lateralError, kappaAhead, and clearance are clipped to practical ranges.
    """

    max_lateral_error: float = 1.5
    max_abs_kappa: float = 10.0
    max_clearance: float = 1.5
    max_velocity: float = 1.0

    max_lidar_range: float = 2.0  # MUST MATCH TrackingEnv lidar range
    num_lidar_measurements: int = 19  # MUST MATCH TrackingEnv lidar measurement count


class StateProcessor:
    """
    Converts TrackingEnv raw observations into normalized observations
    suitable for SAC neural networks.

    Raw observation:
        [
            lateralError,
            headingError,
            kappaAhead,
            lastVelocity,
            clearance,
            lidar_0,
            lidar_1,
            ...
        ]

    Normalized observation:
        roughly in [-1, 1]^(5 + num_lidar_measurements)
    """

    BASE_STATE_DIM = 5

    def __init__(self, config: StateNormalizationConfig = StateNormalizationConfig()):
        self.config = config
        self.base_state_dim = self.BASE_STATE_DIM
        self.state_dim = self.base_state_dim + config.num_lidar_measurements

    def validate_raw(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.shape != (self.state_dim,):
            raise ValueError(
                f"Expected observation shape {(self.state_dim,)}, got {obs.shape}. "
                f"Expected 5 base features + "
                f"{self.config.num_lidar_measurements} lidar measurements."
            )

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation contains non-finite values: {obs}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize raw TrackingEnv observation to approximately [-1, 1]^state_dim.
        """
        obs = np.asarray(obs, dtype=np.float32)
        self.validate_raw(obs)

        lateral_error = obs[0]
        heading_error = obs[1]
        kappa_ahead = obs[2]
        last_velocity = obs[3]
        clearance = obs[4]
        lidar_ranges = obs[5:]

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

        last_velocity_clipped = np.clip(last_velocity, 0.0, c.max_velocity)
        last_velocity_norm = 2.0 * (last_velocity_clipped / c.max_velocity) - 1.0

        clearance_norm = np.clip(
            clearance / c.max_clearance,
            -1.0,
            1.0,
        )

        # Lidar ranges:
        #   0.0             -> -1.0  obstacle very close
        #   max_lidar_range ->  1.0  no obstacle within max range
        lidar_clipped = np.clip(lidar_ranges, 0.0, c.max_lidar_range)
        lidar_norm = 2.0 * (lidar_clipped / c.max_lidar_range) - 1.0

        base_normalized = np.array(
            [
                lateral_error_norm,
                heading_error_norm,
                kappa_ahead_norm,
                last_velocity_norm,
                clearance_norm,
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

    mpc_obs = np.array(
        [
            0.05,   # lateralError
            0.10,   # headingError
            1.50,   # kappaAhead
            0.40,   # lastVelocity
            0.12,   # clearance
        ],
        dtype=np.float32,
    )

    lidar_obs = np.linspace(
        0.0,
        processor.config.max_lidar_range,
        processor.config.num_lidar_measurements,
        dtype=np.float32,
    )

    raw_obs = np.concatenate([mpc_obs, lidar_obs]).astype(np.float32)
    normalized_obs = processor.normalize(raw_obs)

    print("Raw observation shape:        ", raw_obs.shape)
    print("Normalized observation shape: ", normalized_obs.shape)
    print("State dim:                    ", processor.state_dim)
    print("Raw observation:              ", raw_obs)
    print("Normalized observation:       ", normalized_obs)