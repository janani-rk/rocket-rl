
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


def evaluate(model_name, episodes=10):
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_name)

    rewards = []

    for episode in range(episodes):
        observation, info = env.reset(seed=100 + episode)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(
            f"{model_name} | Episode {episode + 1}: "
            f"{total_reward:.2f}"
        )

    print(f"\n{model_name}")
    print("Average reward:", round(float(np.mean(rewards)), 2))
    print("Best reward:", round(float(np.max(rewards)), 2))
    print("Worst reward:", round(float(np.min(rewards)), 2))

    env.close()


evaluate("rocket_model")
evaluate("rocket_hover_model")
evaluate("rocket_fixed_model")
