import argparse
import os

import gymnasium as gym
import gymnasium_robotics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def train(cfg: DictConfig):
    # Create directories
    # Hydra changes the working directory, so we can just use the current directory
    # or rely on cfg.hydra.run.dir if set to .

    # Instantiate the environment
    # The config has nested env: _target_: TouchDoorReward, env: {_target_: gym.make, ...}
    gym.register_envs(gymnasium_robotics)
    env = instantiate(cfg.env)

    # Initialize the model
    # Instantiate the model, passing the env
    model = instantiate(cfg.model, env=env)

    # Train the model
    print(f"Training for {cfg.training.timesteps} timesteps...")
    model.learn(total_timesteps=cfg.training.timesteps, progress_bar=True)

    # Save the model
    # Use the model directory from config
    model_dir = cfg.training.model_dir

    # Default filename if not specified or available
    filename = "model"

    if model.logger and model.logger.dir:
        # e.g. model.logger.dir is ".../logs/PPO_1"
        run_name = os.path.basename(model.logger.dir)
        # New path: "models/PPO_1/experiment_name"
        save_path = os.path.join(model_dir, run_name, filename)
    else:
        # Fallback if logger not available
        save_path = os.path.join(model_dir, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at {save_path}.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg)
