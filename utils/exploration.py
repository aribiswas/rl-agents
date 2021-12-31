import numpy as np


class OUNoise:

    def __init__(self, n=1, theta=0.15, sigma=0.2, sigma_decay=0.001, sigma_min=0.01, mu=0):
        self.n = n
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.mu = mu
        self.state = np.zeros(n)

    def step(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.n)
        self.sigma *= (1-self.sigma_decay)
        if self.sigma < self.sigma_min:
            self.sigma = self.sigma_min
        return self.state

