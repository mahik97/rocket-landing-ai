import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("LunarLander-v3")

# Create the PPO learning algorithm
model = PPO("MlpPolicy", env, verbose=1)

# Train the AI
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("rocket_baseline")

env.close()

print("Training complete!")