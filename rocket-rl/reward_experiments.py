
import gymnasium as gym
from stable_baselines3 import PPO


class HoverRewardWrapper(gym.Wrapper):
    """A deliberately flawed reward function."""

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        horizontal_speed = abs(observation[2])
        vertical_speed = abs(observation[3])
        angle = abs(observation[4])

        # Incorrect incentive: reward hovering upright
        if (
            horizontal_speed < 0.1
            and vertical_speed < 0.1
            and angle < 0.1
        ):
            reward += 1.0

        return observation, reward, terminated, truncated, info


env = HoverRewardWrapper(gym.make("LunarLander-v3"))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]},
)

print("Training the deliberately flawed reward model...")

model.learn(total_timesteps=100_000)

model.save("rocket_hover_model")

print("Saved model as rocket_hover_model.zip")

env.close()


