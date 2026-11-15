from copy import deepcopy

from AIr.dqn.core import DQN
import gymnasium as gym
import numpy as np

# env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")
envs = gym.make_vec("LunarLander-v3", num_envs=1, vectorization_mode="sync", render_mode="human", enable_wind=False)

state, info = envs.reset()

n_actions = envs.action_space.shape[0]
n_state = envs.observation_space.shape[1]

N = 4
agent = DQN(n_state, N, epsilon=0.05, epsilon_decay_interval=5000, sync_interval=1000, batch_size=64, train_interval=1, eta=0.0001)
agent.load("lunar.npz")


episodes = 3000

try:
    for episode in range(episodes):
        state, info = envs.reset()
        # state = np.asarray(state, dtype=np.float32)
        # print(state)

        total_reward = 0
        state, reward, terminated, truncated, info = envs.step(envs.action_space.sample())

        while not (terminated.all() or truncated.all()):

            # Your own DQN chooses an action
            action = agent.action(state)

            # Execute action in Gymnasium
            next_state, reward, terminated, truncated, info = envs.step(action)

            # next_state = np.asarray(next_state, dtype=np.float32)

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

        print(f"Episode {episode}: reward = {total_reward}, epsilon = {agent.epsilon}")

except KeyboardInterrupt:
    print("saving")
    agent.save("lunar.npz")

envs.close()
