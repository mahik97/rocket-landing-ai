import gymnasium as gym
from stable_baselines3 import PPO

# Open the simulation window
env = gym.make("LunarLander-v3", render_mode="human")

# Load the trained model
model = PPO.load("rocket_baseline")

# Watch five attempts
for episode in range(5):
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # Let the trained AI choose an action
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
        f"Total reward = {total_reward:.2f}"
    )

env.close()
print("Watching complete!")