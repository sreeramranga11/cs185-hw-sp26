import argparse
import csv
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import configs
from agents import agents
from infrastructure import pytorch_util as ptu
from infrastructure import utils


VARIANTS = (
    "default",
    "prior_target",
    "prior_current",
    "best_q_prior_target",
    "best_q_prior_current",
    "best_qz_prior_target",
    "best_qz_prior_current",
    "best_q_noise_current",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--num_eval_trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=Path, default=Path("exp/debug_dsrl_variant_eval.csv"))
    parser.add_argument("--no_gpu", action="store_true")
    return parser.parse_args()


def build_agent_from_run(run_dir: Path):
    flags = json.loads((run_dir / "flags.json").read_text())
    config = configs.configs["dsrl"](
        flags["env_name"],
        hidden_size=flags.get("hidden_size", 512),
        num_layers=flags.get("num_layers", 4),
        learning_rate=flags.get("learning_rate", 3e-4),
        flow_steps=flags["agent_kwargs"].get("flow_steps", 10),
        noise_scale=flags["agent_kwargs"].get("noise_scale", 1.0),
    )
    env, dataset = config["make_env_and_dataset"]()
    example_batch = dataset.sample(1)
    agent = agents["dsrl"](
        example_batch["observations"].shape[1:],
        example_batch["actions"].shape[-1],
        **config["agent_kwargs"],
    )
    agent.load_state_dict(torch.load(run_dir / "agent.pt", map_location=ptu.device))
    agent.eval()
    return env, agent


def flow_actions(agent, observations, scaled_z, use_current_actor: bool):
    actor = agent.bc_flow_actor if use_current_actor else agent.target_bc_flow_actor
    action = scaled_z
    dt = 1.0 / float(agent.flow_steps)
    for k in range(agent.flow_steps):
        times = torch.full(
            (observations.shape[0], 1),
            fill_value=float(k) / float(agent.flow_steps),
            device=observations.device,
            dtype=observations.dtype,
        )
        action = action + dt * actor(observations, action, times)
    return torch.clamp(action, -1, 1)


def make_variant_get_action(agent, variant: str, num_candidates: int):
    @torch.no_grad()
    def get_action(self, observation):
        obs = ptu.from_numpy(np.asarray(observation))[None]
        batch_size = obs.shape[0]
        action_dim = self.action_dim

        if variant == "default":
            return ptu.to_numpy(self.sample_actions(obs)[0])

        if variant in {"prior_target", "prior_current"}:
            z = torch.randn((batch_size, action_dim), device=obs.device, dtype=obs.dtype)
            action = flow_actions(self, obs, self.noise_scale * z, variant == "prior_current")
            return ptu.to_numpy(action[0])

        obs_many = obs.expand(num_candidates, obs.shape[-1])
        use_current = variant.endswith("_current")

        if variant == "best_q_noise_current":
            dist = self.noise_actor(obs)
            z = dist.rsample((num_candidates,)).reshape(num_candidates, action_dim)
        else:
            z = torch.randn((num_candidates, action_dim), device=obs.device, dtype=obs.dtype)

        scaled_z = self.noise_scale * z
        actions = flow_actions(self, obs_many, scaled_z, use_current)

        if variant.startswith("best_qz"):
            scores = self.z_critic(obs_many, scaled_z).mean(dim=0)
        else:
            scores = self.critic(obs_many, actions).mean(dim=0)

        best_idx = torch.argmax(scores)
        return ptu.to_numpy(actions[best_idx])

    return MethodType(get_action, agent)


def evaluate(env, agent, num_eval_trajectories: int):
    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    trajectories = utils.sample_n_trajectories(env, agent, num_eval_trajectories, ep_len)
    successes = [traj["episode_statistics"]["s"] for traj in trajectories]
    return float(np.mean(successes))


def main():
    args = parse_args()
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=0)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in args.run_dirs:
        env, agent = build_agent_from_run(run_dir)
        for variant in args.variants:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            agent.get_action = make_variant_get_action(agent, variant, args.num_candidates)
            success_rate = evaluate(env, agent, args.num_eval_trajectories)
            row = {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "variant": variant,
                "num_candidates": args.num_candidates,
                "num_eval_trajectories": args.num_eval_trajectories,
                "success_rate": success_rate,
            }
            print(row)
            rows.append(row)

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
