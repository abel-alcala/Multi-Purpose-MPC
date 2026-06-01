import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from networks import Actor, Critic


@dataclass
class SACConfig:
    state_dim: int = 5
    action_dim: int = 2
    hidden_dim: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # If True, SAC learns alpha automatically.
    automatic_entropy_tuning: bool = True

    # Used only if automatic_entropy_tuning is False.
    alpha: float = 0.2

    # Common SAC default: target entropy = -action_dim
    target_entropy: float | None = None

    device: str = "cpu"


class SACAgent:
    """
    Soft Actor-Critic agent.

    For your TrackingEnv:
        state_dim = 5
        action_dim = 2

    The agent assumes:
        states are normalized
        actions are normalized to [-1, 1]^2
    """

    def __init__(self, config: SACConfig = SACConfig()):
        self.config = config
        self.device = torch.device(config.device)

        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.gamma = config.gamma
        self.tau = config.tau

        # Actor/policy network.
        self.actor = Actor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Two critic networks.
        self.critic1 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.critic2 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Target critics are slowly updated copies of the critics.
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        # Target critics are not trained directly by gradient descent.
        for p in self.target_critic1.parameters():
            p.requires_grad = False

        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Optimizers.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=config.critic_lr,
        )

        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=config.critic_lr,
        )

        # Entropy coefficient alpha.
        self.automatic_entropy_tuning = config.automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            if config.target_entropy is None:
                self.target_entropy = -float(config.action_dim)
            else:
                self.target_entropy = config.target_entropy

            # We optimize log_alpha instead of alpha so alpha remains positive.
            self.log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self.device,
            )

            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=config.alpha_lr,
            )
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = torch.tensor(
                config.alpha,
                dtype=torch.float32,
                device=self.device,
            )

    @property
    def alpha(self) -> torch.Tensor:
        """
        Current entropy coefficient.
        """
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return self._fixed_alpha

    def select_action(self, state, deterministic: bool = False):
        """
        Select a normalized action in [-1, 1]^action_dim.

        Input:
            state: normalized state, shape (5,)

        Output:
            normalized action, shape (2,)
        """
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_tensor)
            else:
                action, _ = self.actor.sample(state_tensor)

        return action.squeeze(0).cpu().numpy()

    def update(self, replay_buffer, batch_size: int) -> dict[str, float]:
        """
        Perform one SAC gradient update.

        Returns a dictionary of losses for logging.
        """
        batch = replay_buffer.sample(batch_size)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # ------------------------------------------------------------
        # 1. Critic update
        # ------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)

            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            soft_target_q = target_q - self.alpha.detach() * next_log_probs

            q_target = rewards + self.gamma * (1.0 - dones) * soft_target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ------------------------------------------------------------
        # 2. Actor update
        # ------------------------------------------------------------
        new_actions, log_probs = self.actor.sample(states)

        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------
        # 3. Alpha / entropy-temperature update
        # ------------------------------------------------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_value = self.alpha.item()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_value = self.alpha.item()
            alpha_loss_value = 0.0

        # ------------------------------------------------------------
        # 4. Soft update target critics
        # ------------------------------------------------------------
        self.soft_update_targets()

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_value,
            "alpha": alpha_value,
        }

    def soft_update_targets(self) -> None:
        """
        Slowly update target critics toward the current critics.

        target = tau * online + (1 - tau) * target
        """
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic1.parameters(),
                self.critic1.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for target_param, param in zip(
                self.target_critic2.parameters(),
                self.critic2.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "automatic_entropy_tuning": self.automatic_entropy_tuning,
                "log_alpha": self.log_alpha.detach().cpu()
                if self.log_alpha is not None
                else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

        if self.automatic_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))


if __name__ == "__main__":
    import numpy as np

    from replay_buffer import ReplayBuffer

    config = SACConfig(
        state_dim=5,
        action_dim=2,
        device="cpu",
    )

    agent = SACAgent(config)

    buffer = ReplayBuffer(
        state_dim=5,
        action_dim=2,
        capacity=1000,
        device="cpu",
    )

    # Add fake transitions so we can test one update.
    for _ in range(300):
        state = np.random.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
        action = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
        done = np.random.rand() < 0.05

        buffer.add(state, action, reward, next_state, done)

    test_state = np.zeros(5, dtype=np.float32)
    test_action = agent.select_action(test_state)

    print("Selected normalized action:", test_action)
    print("Action shape:", test_action.shape)

    losses = agent.update(buffer, batch_size=64)
    print("Losses:", losses)