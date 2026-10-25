import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrapper import RewardWrapper

# Choose the experiment to train
REWARD_MODE = "position_velocity"

# Create the simulation with modified rewards
env = gym.make("LunarLander-v3")
env = RewardWrapper(env, mode=REWARD_MODE)

# Create the AI
model = PPO("MlpPolicy", env, verbose=1)

# Train for one million timesteps
model.learn(total_timesteps=1_000_000)

# Save this experiment separately
model.save(f"rocket_{REWARD_MODE}")

env.close()

print(f"Training complete: {REWARD_MODE}")