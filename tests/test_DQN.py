import pytest
import gymnasium as gym

from AIr.dqn.core import DQN
import numpy as np

def test_predict():
    dqn = DQN(3, 5)

    try:
        predict = dqn.predict(np.array([-1.4, 0.1, 0.3]))
        print(predict)
        print(dqn.online_nn.output)
    except:
        self.fail("forward prop not working")

def test_train():
    env = gym.make("LunarLander-v3", render_mode="human")
    state, info = env.reset(32)

    n_actions = env.action_space.n
    n_state = env.observation_space.shape[0]

    agent = DQN(n_actions, n_state)

    episodes = 1000

    for episode in range(episodes):
        state, info = env.reset()
        state = np.asarray(state, dtype=np.float32)

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
