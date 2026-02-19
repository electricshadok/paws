
import torch.nn as nn


# Define the custom network architecture
# This is a dictionary that Stable Baselines3 accepts
# 'pi' is the policy network (actor)
# 'vf' is the value function network (critic)
# [256, 256] means two hidden layers with 256 neurons each
# activation_fn defines the activation function (e.g., ReLU, Tanh)

policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

# You can also use a simpler list if policy and value networks share the same architecture
# policy_kwargs = dict(net_arch=[256, 256])
