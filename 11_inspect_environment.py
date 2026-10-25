
import inspect
import gymnasium.envs.box2d.lunar_lander as lunar_lander

print("LunarLander source file:")
print(inspect.getfile(lunar_lander))

print("\nRelevant termination and landing logic:\n")

source = inspect.getsource(lunar_lander.LunarLander.step)
lines = source.splitlines()

keywords = (
    "terminated",
    "truncated",
    "game_over",
    "lander.awake",
    "helipad",
    "contact",
    "reward",
)

for number, line in enumerate(lines, start=1):
    if any(word in line.lower() for word in keywords):
        start = max(0, number - 3)
        end = min(len(lines), number + 2)

        print(f"\n--- Around source line {number} ---")
        for index in range(start, end):
            print(f"{index + 1}: {lines[index]}")
