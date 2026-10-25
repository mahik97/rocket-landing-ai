import gymnasium as gym

# Create the LunarLander simulation
env = gym.make("LunarLander-v3", render_mode="human")

# Run three random attempts
for episode in range(3):
    observation, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Choose a random action
        action = env.action_space.sample()

        # Apply the action and receive feedback
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

    print(f"Episode {episode + 1} finished")

env.close()
print("All episodes completed!")
