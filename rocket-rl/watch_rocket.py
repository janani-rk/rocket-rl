
import gymnasium as gym
from stable_baselines3 import PPO
import time

# Open the rocket simulation window
env = gym.make("LunarLander-v3", render_mode="human")

# Load the AI we trained
model = PPO.load("rocket_fixed_model")

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    done = False

    print(f"\nEpisode {episode + 1} started")

    while not done:
        # Let the trained AI choose an action
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print(f"Episode reward: {total_reward:.2f}")
    time.sleep(2)

env.close()
print("\nAll episodes completed.")
