from typing import List, Optional

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer


def make_sac_her(
    env: gym.Env,
    policy: str = "MultiInputPolicy",
    learning_rate: float = 1e-3,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.95,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = "future",
    net_arch: Optional[List[int]] = None,
    experiment_name: str = "default_experiment",
    device: str = "auto",
    verbose: int = 1,
) -> SAC:
    if net_arch is None:
        net_arch = [256, 256, 256]
    else:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(net_arch, (ListConfig, DictConfig)):
            net_arch = OmegaConf.to_container(net_arch, resolve=True)

    return SAC(
        policy=policy,
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        ),
        verbose=verbose,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=f"logs/{experiment_name}",
        device=device,
    )
