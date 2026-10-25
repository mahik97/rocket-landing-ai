
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("rocket_position_penalty")

for episode in range(5):
    observation, info = env.reset(seed=episode)

    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action, _ = model.predict(
            observation, deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}"
    )

env.close()
