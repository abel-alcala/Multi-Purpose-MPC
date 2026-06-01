import numpy as np
import torch

"""
This module implements a transition replay buffer for storing and sampling transitions in reinforcement learning.
"""

class ReplayBuffer:
    """
    Fixed-size replay buffer for SAC.

    Stores:
        state:      normalized observation, shape (state_dim,)
        action:     normalized action, shape (action_dim,)
        reward:     scalar
        next_state: normalized next observation, shape (state_dim,)
        done:       scalar, 1.0 if episode ended, else 0.0

    For TrackingEnv:
        state_dim = 5
        action_dim = 2
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        capacity: int = 1_000_000,
        device: str = "cpu",
    ):
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = torch.device(device)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the replay buffer.
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape {(self.state_dim,)}, got {state.shape}"
            )

        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {action.shape}"
            )

        if next_state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected next_state shape {(self.state_dim,)}, got {next_state.shape}"
            )

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; buffer only has {self.size} items"
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "states": torch.as_tensor(
                self.states[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "next_states": torch.as_tensor(
                self.next_states[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "dones": torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=self.device,
            ),
        }

        return batch

    def __len__(self) -> int:
        return self.size


if __name__ == "__main__":
    buffer = ReplayBuffer(
        state_dim=5,
        action_dim=2,
        capacity=10,
        device="cpu",
    )

    state = np.array([0.0, 0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    action = np.array([0.5, -0.5], dtype=np.float32)
    reward = 1.0
    next_state = np.array([0.1, 0.2, 0.3, -0.2, 0.5], dtype=np.float32)
    done = False

    buffer.add(state, action, reward, next_state, done)

    print("Buffer size:", len(buffer))

    batch = buffer.sample(batch_size=1)

    print("States shape:     ", batch["states"].shape)
    print("Actions shape:    ", batch["actions"].shape)
    print("Rewards shape:    ", batch["rewards"].shape)
    print("Next states shape:", batch["next_states"].shape)
    print("Dones shape:      ", batch["dones"].shape)