import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        online_training: bool = False,
        num_samples: int = 32,
        expectile: float = 0.9,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.actor = make_actor_flow(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_optimizer = make_actor_flow_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile
        self.to(ptu.device)

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        return weight * (adv ** 2)

    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        v = self.value(observations)
        with torch.no_grad():
            q = self.target_critic(observations, actions).min(dim=0).values
        adv = q - v
        loss = self.expectile_loss(adv, self.expectile).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        batch_size = observations.shape[0]
        tiled_obs = observations[:, None, :].expand(
            batch_size, self.num_samples, observations.shape[-1]
        )
        flat_obs = tiled_obs.reshape(batch_size * self.num_samples, observations.shape[-1])
        flat_noise = torch.randn(
            batch_size * self.num_samples,
            self.action_dim,
            device=observations.device,
            dtype=observations.dtype,
        )
        flat_actions = self.get_flow_action(flat_obs, flat_noise)
        q = self.critic(flat_obs, torch.clamp(flat_actions, -1, 1)).min(dim=0).values
        q = q.view(batch_size, self.num_samples)
        actions = flat_actions.view(batch_size, self.num_samples, self.action_dim)
        best_idx = q.argmax(dim=1)
        return actions[torch.arange(batch_size, device=observations.device), best_idx]

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        action = noise
        dt = 1.0 / float(self.flow_steps)
        for k in range(self.flow_steps):
            times = torch.full(
                (observation.shape[0], 1),
                fill_value=float(k) / float(self.flow_steps),
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity = self.actor(observation, action, times)
            action = action + dt * velocity
        return torch.clamp(action, -1, 1)

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        q = self.critic(observations, actions)
        with torch.no_grad():
            target_v = self.value(next_observations)
            target_q = rewards + self.discount * (1.0 - dones.float()) * target_v
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


    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        noise = torch.randn_like(actions)
        times = torch.rand((actions.shape[0], 1), device=actions.device, dtype=actions.dtype)
        action_mid = (1.0 - times) * noise + times * actions
        target_velocity = actions - noise
        pred_velocity = self.actor(observations, action_mid, times)
        loss = torch.mean((pred_velocity - target_velocity) ** 2)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"loss": loss}


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
        metrics_v = self.update_value(observations, actions)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.target_update_rate)
                target_param.data.add_(self.target_update_rate * param.data)
