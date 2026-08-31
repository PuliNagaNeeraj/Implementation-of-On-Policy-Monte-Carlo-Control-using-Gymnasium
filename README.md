# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the FrozenLake-v1 environment provided by Gymnasium. FrozenLake is a grid-based reinforcement learning environment in which an agent starts from a fixed starting position and must reach the goal while avoiding holes. The environment is stochastic when is_slippery=True, meaning the agent may move in a direction different from the intended action.

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
1. Create the FrozenLake-v1 environment.
2. Initialize the Q-table with zeros.
3. Set the values of α (learning rate), γ (discount factor), and ε (exploration rate).
4. Generate an episode using the epsilon-greedy policy.
5. Store the states, actions, and rewards of the episode.
6. Calculate the return G from the end of the episode.
7. Update the Q-value using:
Q(s,a)←Q(s,a)+α[G−Q(s,a)]

9. Reduce ε gradually to allow more exploitation.
10. Repeat the process for many episodes.
11. Select the action with the highest Q-value for each state.
12. Display the Q-table, state-value function, learned policy, and average reward.

## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)
env = env.unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    # Exploration
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    # Exploitation
    return np.argmax(Q[state])
# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current
    epsilon-greedy policy.

    Returns:
        List of (state, action, reward)
    """
    episode = []
    state, info = env.reset()
    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(
            state,
            epsilon
        )
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append(
            (state, action, reward)
        )
        state = next_state
        if terminated or truncated:
            break
    return episode

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start
for episode_number in range(num_episodes):
    # Generate a complete episode
    episode = generate_episode(epsilon)
    # Store total reward
    total_reward = sum(
        reward for _, _, reward in episode
    )
    episode_rewards.append(total_reward)
    # Return
    G = 0
    # Process episode backwards
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = reward + gamma * G
        # Incremental Monte Carlo update
        Q[state, action] += alpha * (
            G - Q[state, action]
        )
    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }
    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name: PULI NAGA NEERAJ")
    print("Register Number: 212223240130")
    print("\nLearned Policy:")
    print(policy_grid)

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )
print("\nFinal Q-table:")
print(np.round(Q, 3))
print_value_function(state_values)
print_policy(optimal_policy)

# -------------------------------------------------
# Average Reward
# -------------------------------------------------

success_rate = np.mean(
    episode_rewards[-1000:]
)
print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)
plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title(
    "Monte Carlo Control Learning Curve"
)
plt.grid(True)
plt.show()
env.close()

```

---

## Output

<img width="415" height="607" alt="image" src="https://github.com/user-attachments/assets/2467081b-a5f3-403b-b36a-b32aa40d671c" />

<img width="988" height="527" alt="image" src="https://github.com/user-attachments/assets/7245bb7b-bfcb-4a6f-92e9-0b4aeaffcb0a" />

---

## Result

On-Policy Monte Carlo Control was successfully implemented using the Gymnasium FrozenLake-v1 environment. The algorithm generated complete episodes using an epsilon-greedy policy and estimated the action-value function Q(s,a) from the observed returns. The Q-table was progressively updated using the Monte Carlo incremental update rule, and the final greedy policy was obtained by selecting the action with the highest Q-value for each state.

---

## Inference

Monte Carlo Control learns the optimal policy directly from complete episodes without requiring an explicit model of the environment. The epsilon-greedy strategy allows the agent to explore different actions while gradually exploiting actions with higher estimated Q-values. Since FrozenLake is stochastic, multiple episodes are required to obtain reliable estimates. The learning curve shows how the agent's performance changes as more episodes are experienced. The final Q-table provides action-value estimates, while the greedy policy represents the learned behavior of the agent.

---
