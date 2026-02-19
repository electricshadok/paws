import argparse

import gymnasium as gym
import gymnasium_robotics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def visualize_env(cfg: DictConfig):
    # Register the environment
    gym.register_envs(gymnasium_robotics)

    # Instantiate the environment
    # Force render_mode to human for visualization
    if "env" in cfg and "env" in cfg.env:
         # Handling nested structure if present, or direct
         # The config usually has env: {_target_: ..., env: {_target_: gym.make, ...}}
         # We need to inject render_mode='human' into the gym.make call args
         pass

    # To be safe and generic with Hydra instantiation, we might need to modify the config object
    # But since we use instantiate(cfg.env), we should verify if cfg.env.env has render_mode
    # or if cfg.env has it.

    # Let's try to update the config dynamically for visualization
    # We assume the structure relies on `gym.make` being somewhere.
    # Based on AdroitHandDoor.yaml:
    # env:
    #   _target_: paws.components.rewards.TouchDoorReward
    #   env:
    #     _target_: gymnasium.make
    #     render_mode: null

    try:
        if "env" in cfg.env and "render_mode" in cfg.env.env:
            cfg.env.env.render_mode = "human"
            print("Set render_mode to 'human' for visualization.")
        elif "render_mode" in cfg.env:
             cfg.env.render_mode = "human"
             print("Set render_mode to 'human' for visualization.")
    except Exception as e:
        print(f"Warning: Could not automatically set render_mode. Error: {e}")

    print("Instantiating environment...")
    env = instantiate(cfg.env)

    obs, info = env.reset()
    print("Environment created. Press Ctrl+C to stop.")

    try:
        while True:
            # Render is handled by the environment's render_mode='human'
            # But specific envs might need explicit render() call if mode is not human or if it's old gym
            # Gymnasium with render_mode='human' usually handles it in step() or separate loop?
            # Actually, usually step() renders if human, but let's call render() to be sure if needed
            # For gymnasium, if render_mode is human, render() might just function or be called implicitly.
            # Let's call it explicitly as a fallback or main driver.
            env.render()

            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the environment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    visualize_env(cfg)
