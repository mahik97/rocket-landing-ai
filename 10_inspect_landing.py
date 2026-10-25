
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_NAME = "rocket_position_velocity"

LANDING_TOLERANCE = 0.2
ANGLE_TOLERANCE = 0.2

env = gym.make("LunarLander-v3")
model = PPO.load(MODEL_NAME)

for episode in range(10):
    observation, info = env.reset(seed=episode)

    terminated = False
    truncated = False
    total_reward = 0.0
    final_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        final_reward = reward

    x = float(observation[0])
    angle = float(observation[4])
    left_leg = observation[6] > 0.5
    right_leg = observation[7] > 0.5

    inside_zone = abs(x) < LANDING_TOLERANCE
    upright = abs(angle) < ANGLE_TOLERANCE
    both_legs = left_leg and right_leg

    # Candidate success classification.
    # We will inspect the results before using this
    # definition for the full evaluation.
    successful = (
        terminated
        and not truncated
        and final_reward > 0
        and inside_zone
        and upright
        and both_legs
    )

    print(f"\nEpisode {episode + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final step reward: {final_reward:.2f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Horizontal position: {x:.3f}")
    print(f"Angle: {angle:.3f}")
    print(f"Left leg: {left_leg}")
    print(f"Right leg: {right_leg}")
    print(f"Inside zone: {inside_zone}")
    print(f"Upright: {upright}")
    print(f"Candidate successful landing: {successful}")

env.close()
