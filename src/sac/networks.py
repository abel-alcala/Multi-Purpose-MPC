import torch
import torch.nn as nn
# import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """
    Simple MLP used by both actor and critic.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    """
    SAC stochastic actor.

    Input:
        state, shape (batch_size, state_dim)

    Output:
        normalized action in [-1, 1]^action_dim

    For your environment:
        state_dim = 5
        action_dim = 2

    The actor outputs a Gaussian distribution, samples from it,
    then applies tanh so the final action is bounded in [-1, 1].
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Gaussian mean and log standard deviation.
        """
        x = self.net(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using the reparameterization trick.

        Returns:
            action:
                tanh-squashed action in [-1, 1], shape (batch_size, action_dim)

            log_prob:
                corrected log probability, shape (batch_size, 1)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        # rsample allows gradients to flow through the sampled action.
        z = normal.rsample()

        # Squash to [-1, 1].
        action = torch.tanh(z)

        # Log probability before tanh correction.
        log_prob = normal.log_prob(z)

        # Tanh correction.
        # Without this, SAC's entropy term is mathematically wrong.
        log_prob -= torch.log(1.0 - action.pow(2) + EPSILON)

        # Sum across action dimensions.
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.
        Uses tanh(mean) instead of sampling.
        """
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    """
    SAC Q-function approximator.

    Input:
        state, shape (batch_size, state_dim)
        action, shape (batch_size, action_dim)

    Output:
        Q-value, shape (batch_size, 1)

    For your environment:
        state_dim = 5
        action_dim = 2
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.q_net = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        q_value = self.q_net(x)
        return q_value


if __name__ == "__main__":
    state_dim = 5
    action_dim = 2
    batch_size = 4

    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)

    states = torch.randn(batch_size, state_dim)

    actions, log_probs = actor.sample(states)
    q_values = critic(states, actions)

    deterministic_actions = actor.deterministic(states)

    print("States shape:                 ", states.shape)
    print("Sampled actions shape:        ", actions.shape)
    print("Sampled actions min/max:      ", actions.min().item(), actions.max().item())
    print("Log probs shape:              ", log_probs.shape)
    print("Q-values shape:               ", q_values.shape)
    print("Deterministic actions shape:  ", deterministic_actions.shape)