from AIr.dqn.core import DQN
import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
state, info = env.reset()

n_actions = env.action_space.n
n_state = env.observation_space.shape[0]
print(n_actions)
print(n_state)

agent = DQN(n_state, n_actions)

episodes = 1000

for episode in range(episodes):
    state, info = env.reset()
    state = np.asarray(state, dtype=np.float32)
    print(state)

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Your own DQN chooses an action
        action = agent.action(state)

        # Execute action in Gymnasium
        next_state, reward, terminated, truncated, info = env.step(action)

        next_state = np.asarray(next_state, dtype=np.float32)

        # Store experience in your replay buffer
        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        # Train your network
        agent.train()

        state = next_state
        total_reward += reward

    print(f"Episode {episode}: reward = {total_reward}")

env.close()
