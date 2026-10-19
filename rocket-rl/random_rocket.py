
import gymnasium as gym

env = gym.make("LunarLander-v3")

observation, info = env.reset()

print("\nROCKET OBSERVATIONS")
print("-------------------")

names = [
    "Horizontal position",
    "Height",
    "Horizontal speed",
    "Vertical speed",
    "Rocket angle",
    "Rotation speed",
    "Left leg contact",
    "Right leg contact",
]

for name, value in zip(names, observation):
    print(f"{name}: {value:.3f}")

print("\nAvailable actions:", env.action_space)
print("Observation size:", env.observation_space.shape)

env.close()


