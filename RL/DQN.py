import numpy as np
import gym

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Set the parameters
render_10=True #do you want to render the animation
num_episodes = 2000
max_steps = 100
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# Initialize the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

print(env.action_space.sample())

# Training the agent
for episode in range(num_episodes):
    state = env.reset()

    if isinstance(state, tuple):
        state = state[0]
    done = False
    for step in range(max_steps):
        # Choose an action (epsilon-greedy policy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Perform the action
        result = env.step(action)
        new_state = result[0]
        if isinstance(new_state, np.ndarray):
            new_state = new_state[0]
        reward = result[1]
        done = result[2]

        # Update the Q-value
        Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state

        if done:
            break

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

# Testing the agent
total_rewards = 0
num_test_episodes = 100

for _ in range(num_test_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    done = False
    while not done:
        action = np.argmax(Q[state, :])
        result = env.step(action)
        new_state = result[0]
        if isinstance(new_state, np.ndarray):
            new_state = new_state[0]
        reward = result[1]
        done = result[2]
        total_rewards += reward
        state = new_state

print(f"Average reward over {num_test_episodes} test episodes: {total_rewards / num_test_episodes}")

if render_10:
    # Render the optimal policy
    env = gym.make("FrozenLake-v1", render_mode="human")
    state = env.reset()

    if isinstance(state, tuple):
        state = state[0]
    env.render()
    done = False
    while not done:
        action = np.argmax(Q[state, :])
        result = env.step(action)
        new_state = result[0]
        if isinstance(new_state, tuple):
            new_state = new_state[0]
        reward = result[1]
        done = result[2]
        env.render()
        state = new_state
