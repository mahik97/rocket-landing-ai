
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("rocket_hover_bonus")

for episode in range(3):
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    print(
        f"Episode {episode + 1}: "
        f"Original reward = {total_reward:.2f}"
    )

env.close()
