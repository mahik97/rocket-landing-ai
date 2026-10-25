# Rocket Landing AI — Experimental Results

## Objective

Train an AI agent to land a simulated lunar lander using reinforcement learning, then compare different reward designs.

## Experimental Setup

- Environment: Gymnasium `LunarLander-v3`
- Algorithm: Proximal Policy Optimization (PPO)
- Training duration: 1,000,000 timesteps per trained experiment
- Evaluation: 100 episodes per model
- Reset seeds: 0–99, shared across all models
- Policy actions: deterministic during evaluation

## Final Evaluation Results

All six models were evaluated for 100 episodes each using the same reset seeds (0–99) and deterministic actions.

| Model               | Average Reward | Reward Std. Dev. | Reward >= 200 | Candidate Successful Landings 
| Baseline            | 176.19         | 74.47            | 47/100        | 22/100 
| Time Penalty        | 177.69         | 86.64            | 59/100        | 26/100 
| Hover Bonus         | 21.32          | 88.79            | 2/100         | 16/100 
| Fixed               | 138.32         | 108.01           | 47/100        | 22/100 
| Position Penalty    | 178.46         | 94.40            | 57/100        | 41/100 
| Position + Velocity | 271.81         | 27.64            | 98/100        | 90/100 

### Conclusion

The position-and-velocity reward experiment performed best among the six tested models. It achieved an average reward of 271.81 and passed the candidate landing criteria in 90 out of 100 episodes.

Its custom reward penalizes horizontal displacement from the landing center and horizontal velocity. The results suggest that these penalties helped the agent learn more effective landing behavior in the tested environment.

### Evaluation Limitation

Candidate success was defined using the final environment reward, final horizontal position, final angle, both leg-contact indicators, and natural episode termination. This is an approximate measure, not an official Gymnasium success flag or a complete physical safety guarantee. Further validation with additional seeds and a more rigorous landing detector is needed.
