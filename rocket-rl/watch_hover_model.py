
import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("rocket_hover_model")

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    done = False

    print(f"Episode {episode + 1}")

    while not done:
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("Episode reward:", round(total_reward, 2))
    time.sleep(2)

env.close()
