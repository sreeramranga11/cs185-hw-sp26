import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence
from networks.rl_networks import LogParam


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())

        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)

        self.log_alpha = LogParam()

        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=3e-4)

        self.to(ptu.device)

    @property
    def alpha(self):
        return self.log_alpha()

    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        actor = self.target_bc_flow_actor
        action = noises
        dt = 1.0 / float(self.flow_steps)
        for k in range(self.flow_steps):
            times = torch.full(
                (observations.shape[0], 1),
                fill_value=float(k) / float(self.flow_steps),
                device=observations.device,
                dtype=observations.dtype,
            )
            velocity = actor(observations, action, times)
            action = action + dt * velocity
        return torch.clamp(action, -1, 1)

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        z = self.noise_actor(observations).rsample()
        scaled_z = self.noise_scale * z
        return self.sample_flow_actions(observations, scaled_z)
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        q = self.critic(observations, actions)
        with torch.no_grad():
            next_z = self.noise_actor(next_observations).rsample()
            next_actions = self.sample_flow_actions(next_observations, self.noise_scale * next_z)
            next_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            target_q = rewards + self.discount * (1.0 - dones.float()) * next_q
        loss = torch.mean((q - target_q[None]) ** 2)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def update_qz(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update z_critic."""
        del actions
        z = torch.randn(observations.shape[0], self.action_dim, device=observations.device, dtype=observations.dtype)
        scaled_z = self.noise_scale * z
        with torch.no_grad():
            a_bc = self.sample_flow_actions(observations, scaled_z)
            target = self.critic(observations, a_bc).mean(dim=0)
        qz = self.z_critic(observations, scaled_z)
        loss = torch.mean((qz - target[None]) ** 2)

        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()

        return {
            "qz_loss": loss,
            "qz_mean": qz.mean(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        z = torch.randn_like(actions)
        times = torch.rand((actions.shape[0], 1), device=actions.device, dtype=actions.dtype)
        action_mid = (1.0 - times) * z + times * actions
        target_velocity = actions - z
        pred_velocity = self.bc_flow_actor(observations, action_mid, times)
        loss = torch.mean((pred_velocity - target_velocity) ** 2)

        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        return {
            "loss": loss,
        }
    
    def update_noise_actor(
        self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        dist = self.noise_actor(observations)
        z = dist.rsample()
        scaled_z = self.noise_scale * z
        qz = self.z_critic(observations, scaled_z).mean(dim=0)
        loss = (self.alpha.detach() * dist.log_prob(z) - qz).mean()

        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()

        return {
            "loss": loss,
            "entropy": (-dist.log_prob(z)).mean(),
        }

    def update_alpha(self, observations: torch.Tensor) -> dict:
        """Update alpha."""
        dist = self.noise_actor(observations)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        loss = -(self.alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

        return {
            "loss": loss,
            "alpha": self.alpha,
        }

    def update_target_critic(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.target_update_rate)
                target_param.data.add_(self.target_update_rate * param.data)

    def update_target_bc_flow_actor(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_bc_flow_actor.parameters(), self.bc_flow_actor.parameters()):
                target_param.data.mul_(1.0 - self.target_update_rate)
                target_param.data.add_(self.target_update_rate * param.data)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations, actions)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v.item() for k, v in metrics_qz.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics
