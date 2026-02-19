import argparse
import os

import gymnasium as gym
import gymnasium_robotics
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf


def get_trained_model_path(cfg: DictConfig) -> str:
    # The model path structure is models/<experiment_name>/<run_name>/model.zip
    # e.g. models/android_hand_door/PPO_1/model.zip

    base_model_dir = cfg.training.model_dir
    if not os.path.exists(base_model_dir):
        raise FileNotFoundError(f"Model directory not found at {base_model_dir}. Please run train.py first.")

    # Determine the run directory
    if cfg.evaluation.run_name:
        run_dir = os.path.join(base_model_dir, cfg.evaluation.run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found at {run_dir}. Please check the run_name.")
        print(f"Using specified run: {run_dir}")
    else:
        # Find the latest run if specific run is not provided

        runs = [f.path for f in os.scandir(base_model_dir) if f.is_dir()]

        if not runs:
            raise FileNotFoundError(f"No runs found in {base_model_dir}.")

        # Sort by modification time, newest first
        run_dir = max(runs, key=os.path.getmtime)
        print(f"Using latest run: {run_dir}")

    model_path = os.path.join(run_dir, "model")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip. Please run train.py first.")

    return model_path


def evaluate(cfg: DictConfig):
    # Instantiate the environment
    # Override render_mode to human for visualization
    if "env" in cfg.env and "render_mode" in cfg.env.env:
        cfg.env.env.render_mode = "human"

    gym.register_envs(gymnasium_robotics)
    env = instantiate(cfg.env)

    try:
        model_path = get_trained_model_path(cfg)
    except FileNotFoundError as e:
        print(e)
        return

    # Resolve the model class
    # The config has model: {_target_: stable_baselines3.PPO, ...}
    try:
        model_class = get_class(cfg.model._target_)
        model = model_class.load(model_path)
        print("Loaded model. Press Ctrl+C to stop.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    obs, info = env.reset()

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg)
