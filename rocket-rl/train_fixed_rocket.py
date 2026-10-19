
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class ProgressRewardWrapper(gym.Wrapper):
    """Reward progress toward the landing area."""

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.previous_distance = (
            observation[0] ** 2 + observation[1] ** 2
        ) ** 0.5

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        distance = (
            observation[0] ** 2 + observation[1] ** 2
        ) ** 0.5

        # Reward progress toward the target
        progress = self.previous_distance - distance
        reward += 5.0 * progress

        # Small penalty for every simulation step
        reward -= 0.01

        self.previous_distance = distance

        return observation, reward, terminated, truncated, info


# Create the original LunarLander simulation
base_env = gym.make("LunarLander-v3")

# Apply our custom reward
env = ProgressRewardWrapper(base_env)

# Check that the previously trained model exists
if not os.path.exists("rocket_fixed_model.zip"):
    raise FileNotFoundError(
        "rocket_fixed_model.zip not found. "
        "Run the initial corrected-model training first."
    )

# Load the existing model and connect it to this environment
model = PPO.load("rocket_fixed_model", env=env)

# Create a folder for training checkpoints
os.makedirs("checkpoints", exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="rocket_fixed"
)

print("Continuing corrected-model training...")
print("Additional timesteps: 500,000")

model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)

# Save the final trained model
model.save("rocket_fixed_model")

print("Training complete!")
print("Updated model saved as rocket_fixed_model.zip")
print("Checkpoints saved in the checkpoints folder.")

env.close()

