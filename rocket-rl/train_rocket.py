
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

# Load the AI we already trained
model = PPO.load("rocket_model", env=env)

print("Continuing training from saved model...")

# Train for another 100,000 steps
model.learn(
    total_timesteps=100_000,
    reset_num_timesteps=False
)

# Save the improved model
model.save("rocket_model")

print("Training complete. Updated model saved.")

env.close()
