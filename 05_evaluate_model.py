
import gymnasium as gym
from stable_baselines3 import PPO
import statistics

models = [
    "rocket_baseline",
    "rocket_time_penalty",
    "rocket_hover_bonus",
    "rocket_fixed",
    "rocket_position_penalty",
    "rocket_position_velocity",
]

LANDING_TOLERANCE = 0.2
ANGLE_TOLERANCE = 0.2

for model_name in models:
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_name)

    rewards = []
    successful_landings = 0
    negative_reward_episodes = 0
    truncated_episodes = 0

    for episode in range(100):
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

        rewards.append(total_reward)

        x = float(observation[0])
        angle = float(observation[4])

        left_leg = observation[6] > 0.5
        right_leg = observation[7] > 0.5

        inside_zone = abs(x) < LANDING_TOLERANCE
        upright = abs(angle) < ANGLE_TOLERANCE
        both_legs = left_leg and right_leg

        successful = (
            terminated
            and not truncated
            and final_reward > 0
            and inside_zone
            and upright
            and both_legs
        )

        if successful:
            successful_landings += 1

        if total_reward < 0:
            negative_reward_episodes += 1

        if truncated:
            truncated_episodes += 1

    print(f"\n{'=' * 45}")
    print(f"Model: {model_name}")
    print(f"{'=' * 45}")
    print(f"Average reward: {statistics.mean(rewards):.2f}")
    print(f"Reward standard deviation: {statistics.pstdev(rewards):.2f}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Worst reward: {min(rewards):.2f}")
    print(f"Reward >= 200: {sum(r >= 200 for r in rewards)}/100")
    print(f"Candidate successful landings: {successful_landings}/100")
    print(f"Negative reward episodes: {negative_reward_episodes}/100")
    print(f"Time-limit truncations: {truncated_episodes}/100")

    env.close()

print("\nAll evaluations completed!")
