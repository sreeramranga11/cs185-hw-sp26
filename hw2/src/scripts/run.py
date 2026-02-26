import argparse
import json
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

MAX_NVIDEO = 2


def get_hw2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_default_logs_txt_name(env_name: str) -> str:
    if "CartPole" in env_name:
        return "section3_logs.txt"
    if "HalfCheetah" in env_name:
        return "section4_logs.txt"
    if "LunarLander" in env_name:
        return "section5_logs.txt"
    if "InvertedPendulum" in env_name:
        return "section6_logs.txt"
    return "experiment_logs.txt"


def get_logs_txt_path(env_name: str, logs_txt_name: str | None) -> str:
    if logs_txt_name is None:
        logs_txt_name = get_default_logs_txt_name(env_name)
    if os.path.isabs(logs_txt_name):
        return logs_txt_name
    return os.path.join(get_hw2_root(), logs_txt_name)


def append_run_logs_to_txt(args, exp_name: str) -> None:
    csv_log_path = os.path.join(args.save_dir, "log.csv")
    if not os.path.exists(csv_log_path):
        print(f"Warning: expected log file at {csv_log_path}, skipping txt export.")
        return

    txt_log_path = get_logs_txt_path(args.env_name, args.logs_txt_name)
    run_config = dict(vars(args))
    run_config["save_dir"] = os.path.abspath(args.save_dir)

    with open(csv_log_path, "r", encoding="utf-8") as csv_file:
        csv_contents = csv_file.read().rstrip()

    with open(txt_log_path, "a", encoding="utf-8") as txt_file:
        txt_file.write("\n" + "=" * 88 + "\n")
        txt_file.write(f"Experiment: {exp_name}\n")
        txt_file.write(
            f"Logged at: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        txt_file.write(f"Config: {json.dumps(run_config, sort_keys=True)}\n")
        txt_file.write("-" * 88 + "\n")
        txt_file.write(csv_contents + "\n")
        txt_file.write("=" * 88 + "\n")

    print(f"Appended run log to {txt_log_path}")


def run_training_loop(logger, args):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            logger.log(logs, itr)
            print("Done logging...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    dump_log(agent, logger, args, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument(
        "--logs_txt_name",
        type=str,
        default=None,
        help=(
            "Output txt file used to append each run's csv logs. If omitted, "
            "defaults to section3_logs.txt for CartPole, section4_logs.txt for "
            "HalfCheetah, section5_logs.txt for LunarLander, and "
            "section6_logs.txt for InvertedPendulum."
        ),
    )

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))

    run_training_loop(logger, args)
    logger.close()
    append_run_logs_to_txt(args, exp_name)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
