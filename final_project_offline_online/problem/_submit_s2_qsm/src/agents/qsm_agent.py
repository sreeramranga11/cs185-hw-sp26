import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps

        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_hats", torch.cumprod(1.0 - betas, dim=0))
        beta_hats = []
        for t in range(flow_steps):
            if t == 0:
                beta_hats.append(betas[t])
            else:
                beta_hats.append(((1.0 - self.alpha_hats[t - 1]) / (1.0 - self.alpha_hats[t])) * betas[t])
        self.register_buffer("beta_hats", torch.stack(beta_hats))

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        s = 0.08
        steps = torch.arange(timesteps + 1, dtype=torch.float32, device=ptu.device)
        f_t = torch.cos((((steps / timesteps) + s) / (1 + s)) * torch.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999)
    
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """
        x_t = noise
        for t in reversed(range(self.flow_steps)):
            times = torch.full(
                (observations.shape[0], 1),
                fill_value=float(t) / max(1, self.flow_steps - 1),
                device=observations.device,
                dtype=observations.dtype,
            )
            eps_pred = self.actor(observations, x_t, times)
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alpha_hats[t]
            beta_hat_t = self.beta_hats[t]
            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)) * eps_pred
            )
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            x_t = mu + torch.sqrt(beta_hat_t) * z
        return torch.clamp(x_t, -1, 1)
    
    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn((observation.shape[0], self.action_dim), device=observation.device)
        action = self.ddpm_sampler(observation, noise)
        return ptu.to_numpy(action[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Critic
        """
        q = self.critic(observations, actions)
        with torch.no_grad():
            noise = torch.randn_like(actions)
            next_actions = self.ddpm_sampler(next_observations, noise)
            next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
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
        
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """
        batch_size = observations.shape[0]
        t_indices = torch.randint(0, self.flow_steps, (batch_size,), device=observations.device)
        alpha_hat_t = self.alpha_hats[t_indices].unsqueeze(-1)
        z = torch.randn_like(actions)
        noisy_actions = torch.sqrt(alpha_hat_t) * actions + torch.sqrt(1.0 - alpha_hat_t) * z
        times = (t_indices.float() / max(1, self.flow_steps - 1)).unsqueeze(-1)
        eps_pred = self.actor(observations, noisy_actions, times)

        action_for_grad = actions.detach().requires_grad_(True)
        q = self.critic(observations, action_for_grad).min(dim=0).values
        grad_q = torch.autograd.grad(q.sum(), action_for_grad, create_graph=False)[0].detach()

        qsm_loss = torch.mean((eps_pred - self.inv_temp * grad_q) ** 2)
        ddpm_loss = torch.mean((z - eps_pred) ** 2)
        loss = qsm_loss + self.alpha * ddpm_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss": loss,
            "qsm_loss": qsm_loss,
            "ddpm_loss": ddpm_loss,
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
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
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
