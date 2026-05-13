from typing import Sequence

import numpy as np
import torch

import infrastructure.pytorch_util as ptu
from agents.fql_agent import FQLAgent


class CustomAgent(FQLAgent):
    """FQL with critic-ranked action sampling and online actor freezing.

    The method keeps FQL's offline expressive-policy training, then uses the
    critic to choose the best action among multiple one-step policy samples.
    During online finetuning, the critic can adapt to new data while the actor
    stays fixed, which avoids the large policy drift seen in naive online FQL.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
        num_action_samples: int = 64,
        target_num_action_samples: int = 8,
        online_start_step: int = 500000,
        freeze_actor_online: bool = True,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_dim=action_dim,
            make_bc_actor=make_bc_actor,
            make_bc_actor_optimizer=make_bc_actor_optimizer,
            make_onestep_actor=make_onestep_actor,
            make_onestep_actor_optimizer=make_onestep_actor_optimizer,
            make_critic=make_critic,
            make_critic_optimizer=make_critic_optimizer,
            discount=discount,
            target_update_rate=target_update_rate,
            flow_steps=flow_steps,
            alpha=alpha,
        )
        self.num_action_samples = num_action_samples
        self.target_num_action_samples = target_num_action_samples
        self.online_start_step = online_start_step
        self.freeze_actor_online = freeze_actor_online

    @torch.no_grad()
    def select_actions(
        self,
        observations: torch.Tensor,
        num_samples: int,
        critic=None,
    ) -> torch.Tensor:
        batch_size = observations.shape[0]
        critic = self.critic if critic is None else critic

        tiled_observations = observations[:, None].expand(
            batch_size,
            num_samples,
            observations.shape[-1],
        )
        flat_observations = tiled_observations.reshape(
            batch_size * num_samples,
            observations.shape[-1],
        )
        flat_noise = torch.randn(
            batch_size * num_samples,
            self.action_dim,
            device=observations.device,
            dtype=observations.dtype,
        )
        flat_actions = self.onestep_actor(flat_observations, flat_noise)
        flat_actions = torch.clamp(flat_actions, -1, 1)

        q_values = critic(flat_observations, flat_actions).min(dim=0).values
        q_values = q_values.view(batch_size, num_samples)
        actions = flat_actions.view(batch_size, num_samples, self.action_dim)
        best_indices = q_values.argmax(dim=1)
        return actions[torch.arange(batch_size, device=observations.device), best_indices]

    def get_action(self, observation: np.ndarray):
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.select_actions(observation, self.num_action_samples)
        return ptu.to_numpy(action[0])

    @ptu.maybe_compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        q = self.critic(observations, actions)
        with torch.no_grad():
            next_actions = self.select_actions(
                next_observations,
                self.target_num_action_samples,
                critic=self.target_critic,
            )
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

        online_phase = step >= self.online_start_step
        if self.freeze_actor_online and online_phase:
            metrics = {
                **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
                "bc_actor/frozen": 1.0,
                "onestep_actor/frozen": 1.0,
            }
        else:
            metrics_bc_actor = self.update_bc_actor(observations, actions)
            metrics_onestep_actor = self.update_onestep_actor(observations, actions)
            metrics = {
                **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
                **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
                **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
            }

        self.update_target_critic()
        return metrics
