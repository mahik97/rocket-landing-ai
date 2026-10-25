
import gymnasium as gym


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, mode="baseline"):
        super().__init__(env)
        self.mode = mode

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        if self.mode in ("time_penalty", "fixed"):
            reward -= 0.01

        elif self.mode == "hover_bonus":
            angle = observation[4]
            horizontal_speed = observation[2]
            vertical_speed = observation[3]

            upright = abs(angle) < 0.1
            slow = (
                abs(horizontal_speed) < 0.1
                and abs(vertical_speed) < 0.1
            )

            if upright and slow:
                reward += 1.0

        elif self.mode == "position_penalty":
            horizontal_position = observation[0]
            reward -= 0.5 * abs(horizontal_position)

        elif self.mode == "position_velocity":
            horizontal_position = observation[0]
            horizontal_velocity = observation[2]

            reward -= 0.5 * abs(horizontal_position)
            reward -= 0.2 * abs(horizontal_velocity)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
