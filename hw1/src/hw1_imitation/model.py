"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        out_dim = action_dim * chunk_size

        lyrs: list[nn.Module] = []
        in_dim = state_dim
        for hiden_dim in hidden_dims:
            lyrs.append(nn.Linear(in_dim, hiden_dim))
            lyrs.append(nn.ReLU())
            in_dim = hiden_dim
        lyrs.append(nn.Linear(in_dim, out_dim))

        self.network = nn.Sequential(*lyrs)
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_acts = self.network(state)
        targ_acts = action_chunk.reshape(action_chunk.shape[0], -1)
        return nn.functional.mse_loss(pred_acts, targ_acts)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        del num_steps
        pred_acts = self.network(state)
        return pred_acts.reshape(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        act_chunk_dim = action_dim * chunk_size
        in_dim = state_dim + act_chunk_dim + 1  # +1 for flow timestep tau

        lyrs: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            lyrs.append(nn.Linear(in_dim, hidden_dim))
            lyrs.append(nn.ReLU())
            in_dim = hidden_dim
        lyrs.append(nn.Linear(in_dim, act_chunk_dim))
        self.network = nn.Sequential(*lyrs)

    def _predict_velocity(
        self,
        state: torch.Tensor,
        noisy_action_chunk: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        flat_noisy_actions = noisy_action_chunk.reshape(batch_size, -1)
        tau = tau.reshape(batch_size, 1)
        inp = torch.cat((state, flat_noisy_actions, tau), dim=-1)
        pred_velocity = self.network(inp)
        return pred_velocity.reshape(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(batch_size, 1, 1, device=state.device, dtype=state.dtype)
        inter_act = tau * action_chunk + (1.0 - tau) * noise

        pred_vel = self._predict_velocity(
            state=state,
            noisy_action_chunk=inter_act,
            tau=tau,
        )
        targ_vel = action_chunk - noise
        return nn.functional.mse_loss(pred_vel, targ_vel)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        batch_sz = state.shape[0]
        act_chunk = torch.randn(
            batch_sz,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        dt = 1.0 / num_steps

        for step in range(num_steps):
            tau_value = step / num_steps
            tau = torch.full(
                (batch_sz, 1, 1),
                fill_value=tau_value,
                device=state.device,
                dtype=state.dtype,
            )
            pred_velocity = self._predict_velocity(
                state=state,
                noisy_action_chunk=act_chunk,
                tau=tau,
            )
            act_chunk = act_chunk + dt * pred_velocity

        return act_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
