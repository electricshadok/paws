import argparse

import gymnasium as gym
import gymnasium_robotics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def inspect_agent(cfg: DictConfig):
    # Register the environment
    gym.register_envs(gymnasium_robotics)

    # Instantiate the environment
    # Override render_mode to None mostly, unless specified otherwise
    print("\n--- Environment Usage ---")
    env = instantiate(cfg.env)

    # 1. Inspect Environment
    obs, info = env.reset()
    print("Observation Keys:", obs.keys() if isinstance(obs, dict) else "Not a dict")

    # Check for Mujoco internals
    if hasattr(env.unwrapped, 'data') and hasattr(env.unwrapped, 'model'):
        print("\nMujoco Data Found!")
        model = env.unwrapped.model
        # data = env.unwrapped.data

        # List all site names
        site_names = [model.site(i).name for i in range(model.nsite)]
        print("Site Names:", site_names)

        # List body names (first 20)
        body_names = [model.body(i).name for i in range(model.nbody)]
        print("Body Names:", body_names[:20])
    else:
        print("\nNo Mujoco data found in env.unwrapped")

    # 2. Verify Reward
    print("\n--- Verifying Rewards ---")
    print("Stepping environment for 10 steps...")
    try:
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step {i+1}: Reward = {reward}")
            if terminated or truncated:
                obs, info = env.reset()
        print(f"Total Reward (10 steps): {total_reward}")
    except Exception as e:
        print(f"Error during stepping: {e}")

    # 3. Verify Policy
    print("\n--- Verifying Policy Architecture ---")
    if "model" in cfg:
        try:
            # We instantiate the model using the config, but we might not load weights if we just want architecture
            # However, initializing PPO requires an env.
            # cfg.model generally looks like {_target_: stable_baselines3.PPO, policy: MlpPolicy, ...}

            # We can instantiate it directly
            model = instantiate(cfg.model, env=env)

            print("\nPolicy Architecture:")
            print(model.policy)

            print("\nNetwork Details:")
            # Stable Baselines 3 structure checking
            if hasattr(model.policy, 'mlp_extractor'):
                print(f"Policy Network: {model.policy.mlp_extractor.policy_net}")
                print(f"Value Network: {model.policy.mlp_extractor.value_net}")
            else:
                print("Could not find mlp_extractor (might be using a different policy type).")

        except Exception as e:
            print(f"Failed to instantiate model for inspection: {e}")
    else:
        print("No 'model' section found in config.")

    env.close()
    print("\nInspection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the environment, policy, and rewards.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    inspect_agent(cfg)
