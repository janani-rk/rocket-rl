
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# Create a folder for recorded videos
env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda episode_id: True,
    name_prefix="rocket"
)

# Load our trained AI
model = PPO.load("rocket_model")

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    done = False

    print(f"Recording episode {episode + 1}...")

    while not done:
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode reward: {total_reward:.2f}")

env.close()
print("Recording complete. Check the videos folder.")
