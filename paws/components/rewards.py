
import gymnasium as gym
import numpy as np


class TouchDoorReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Access Mujoco data
        # We need to access the unwrapped environment to get to the data
        # 'S_grasp' is likely the palm site
        # 'S_handle' is the handle site
        try:
            data = self.unwrapped.data
            palm_pos = data.site('S_grasp').xpos
            handle_pos = data.site('S_handle').xpos

            # Calculate distance
            distance = np.linalg.norm(palm_pos - handle_pos)

            # Define custom reward
            # Negative distance to encourage getting closer
            # Scale it to be reasonable
            custom_reward = -distance

            # Optional: Add bonus for touching (distance < threshold)
            if distance < 0.1:
                custom_reward += 1.0

            # You can decide to replace the original reward or add to it
            # custom_reward += reward
            # For now, let's just use the custom reward + a bit of the original to keep some task structure
            # reward = custom_reward

            # Let's simply replace it for this exercise as requested "get then hand touching the door"
            reward = custom_reward

        except AttributeError:
            # Fallback if unwrapped data is not accessible (e.g. not mujoco)
            pass

        return obs, reward, terminated, truncated, info
