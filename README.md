# Paws : Robotic AI Playground 

![Paws Logo](https://raw.githubusercontent.com/electricshadok/paws/refs/heads/main/docs/paws_logo.png)

Paws is an experimental sandbox dedicated to mastering modern practices in Physical AI and Reinforcement Learning. This repository serves as a modular playground for testing agent behaviors across various simulation technologies.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yaml
conda activate paws
```

## Commands

This project includes setup for the **Adroit Hand** environment (Door task).

### Visualization
Run the visualization script to verify the environment setup.
```bash
python paws/scripts/visualize_env.py --config configs/AdroitHandDoor.yaml
python paws/scripts/visualize_env.py --config configs/FetchPickAndPlace.yaml
```

### Inspection
Inspect the environment observations, rewards, and agent architecture:
```bash
python paws/scripts/inspect_agent.py --config configs/AdroitHandDoor.yaml
python paws/scripts/inspect_agent.py --config configs/FetchPickAndPlace.yaml
```

### Training
Train an agent
```bash
python paws/scripts/train.py --config configs/AdroitHandDoor.yaml
python paws/scripts/train.py --config configs/FetchPickAndPlace.yaml
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluation
Watch the trained agent in action:
```bash
python paws/scripts/eval.py --config configs/AdroitHandDoor.yaml
python paws/scripts/eval.py --config configs/FetchPickAndPlace.yaml
```

---

## 🛠️ Technical Stack

This project utilizes a modern stack for Physical AI and Robotics research:

*   **Gymnasium:** Standard API for reinforcement learning environments (formerly OpenAI Gym).
*   **Gymnasium Robotics:** Collection of robotics environments for Gymnasium (includes Adroit Hand).
*   **Stable Baselines 3:** Reliable implementations of reinforcement learning algorithms (PPO, SAC, etc.).
*   **MuJoCo:** Physics engine for detailed, efficient rigid body simulations.
*   **JAX:** High-performance numerical computing library (included for future accelerated simulation/training).
*   **Warp (NVIDIA):** Differentiable simulation framework (optional/experimental).

