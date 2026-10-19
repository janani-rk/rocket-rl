# Rocket Landing AI Using Reinforcement Learning

## Project Overview

This project trains an AI agent to control a simulated lunar lander using reinforcement learning. The agent learns through repeated interaction with the LunarLander-v3 environment and improves its actions based on reward feedback.

## Objectives

- Train an AI agent to land a simulated rocket safely.
- Compare a baseline reinforcement learning model with reward-shaped models.
- Study how reward design influences agent behavior.
- Evaluate model performance across multiple episodes.

## Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PPO (Proximal Policy Optimization)
- LunarLander-v3 simulation

## Reward Shaping

Three approaches were compared:

1. **Baseline model:** Trained using the environment's original reward.
2. **Hover-reward model:** Added a reward for maintaining low velocity and an upright angle. This model performed poorly in evaluation.
3. **Corrected model:** Added a reward for moving closer to the landing area and a small penalty per step.

## Evaluation Results

| Model | Average Reward |
|---|---:|
| Baseline model | 63.70 |
| Hover-reward model | -163.59 |
| Corrected model | 198.48 |

The corrected model improved the average reward by approximately 134.78 points compared with the baseline model.

In a separate three-episode recording run, the corrected model achieved rewards of 218.75, 250.84, and 251.76. Results can vary between episodes.

## Project Structure

- `random_rocket.py` - Random-action experiment
- `inspect_rocket.py` - Observation and action-space inspection
- `train_rocket.py` - Baseline model training
- `reward_experiments.py` - Reward-shaping experiments
- `train_fixed_rocket.py` - Corrected reward model training
- `evaluate_rocket.py` - Model evaluation
- `watch_rocket.py` - Visual simulation
- `record_fixed_rocket.py` - Final video recording
- `rocket_model.zip` - Baseline model
- `rocket_hover_model.zip` - Hover-reward model
- `rocket_fixed_model.zip` - Corrected model
- `videos/final/` - Final recorded episodes

## Run the Project

Create and activate a Python virtual environment, install Gymnasium with Box2D support and Stable-Baselines3, then run:

```bash
python evaluate_rocket.py
```

To record the corrected model:

```bash
python record_fixed_rocket.py
```

## Conclusion

The experiment demonstrates that reward shaping can substantially affect reinforcement learning performance. The corrected model achieved a much higher average reward than the baseline model, while the hover-focused model performed poorly under the evaluation setup. The corrected model still showed occasional weak episodes, so further training and testing would be needed to establish reliable landing performance.
