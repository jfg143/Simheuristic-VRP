import numpy as np
from Distribution import Distribution


class PoissonDistribution(Distribution):
    def __init__(self, lam):
        self.lam = lam

    def random_value(self) -> float:
        return np.random.poisson(self.lam)

    def mean_value(self):
        return self.lam

class LogNormalDistribution(Distribution):

    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def random_value(self) -> float:
        return np.random.lognormal(np.log(self.mean), self.var)

    def mean_value(self):
        return np.exp(self.mean + self.var / 2)
