import numpy as np
import random
from AIr.nn.core import NeuralNetwork
from copy import deepcopy

class DQN:
    def __init__(self, n_input, n_actions):
        self.n_input = n_input
        self.n_actions = n_actions
        schema = {"n_input": n_input,
                  "layers": [{"n": 100, "activation": "LeakyReLU", },
                             {"n": 50, "activation": "ReLU", },
                             {"n": n_actions, "activation": "Linear"}],
                  }
        self.target_nn = NeuralNetwork(schema)
        self.online_nn = deepcopy(self.target_nn)
        self.experiences = []
        self.sync_count = 0
        self.train_count = 0

    def predict(self, state, target=False):
        if target is False:
            self.online_nn.forward(state)
            return np.argmax(self.online_nn.output)
        else:
            self.target_nn.forward(state)
            return np.max(self.target_nn.output)

    def experience_to_training(self, experiences, size, gamma):
        X = np.empty((size, self.n_input))
        Y = np.empty((size, self.n_actions))
        for i in range(size):
            exp = random.choice(experiences)
            X[i] = exp[0]
            y = exp[2] + gamma * self.predict(exp[3], target=True)
            Y[i] = self.predict(exp[0])
            Y[i, exp[1]] = y
        return X,Y

    def remember(self, state, action, reward, next_state, terminated):
        self.experiences.append((state, action, reward, next_state, terminated))

    def action(self, state):
        return self.predict(state)

    # environment should be a function that takes in (state, action) and returns (return, new_state)
    def train(self, mini_batch_size=5, train_interval=10, sync_interval=20, state=None, alpha=0.1, gamma=0.1, epsilon=0.8, epsilon_decay=0.99):
        if self.train_count > train_interval:
            self.train_count = 0
            X,Y = self.experience_to_training(self.experiences, mini_batch_size, gamma)
            self.online_nn.train((X,Y))

        if self.sync_count > sync_interval:
            self.sync_count = 0
            self.target_nn = deepcopy(self.online_nn)

        self.train_count += 1
        self.sync_count += 1
