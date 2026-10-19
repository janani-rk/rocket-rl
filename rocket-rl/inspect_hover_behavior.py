
import gymnasium as gym
from stable_baselines3 import PPO

class HoverRewardWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        slow = abs(observation[2]) < 0.1 and abs(observation[3]) < 0.1
        upright = abs(observation[4]) < 0.1
        hovering_state = slow and upright

        if hovering_state:
            reward += 1.0

        info["hovering_state"] = hovering_state
        return observation, reward, terminated, truncated, info


env = HoverRewardWrapper(
    gym.make("LunarLander-v3", render_mode="human")
)
model = PPO.load("rocket_hover_model")

observation, info = env.reset(seed=42)
done = False
steps = 0
hover_steps = 0
total_reward = 0

while not done:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    steps += 1
    total_reward += reward
    if info["hovering_state"]:
        hover_steps += 1

    done = terminated or truncated

print("Episode steps:", steps)
print("Hovering-state steps:", hover_steps)
print("Hovering-state percentage:", round(100 * hover_steps / steps, 1))
print("Reward with hover bonus:", round(total_reward, 2))

env.close()
