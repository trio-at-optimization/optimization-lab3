import numpy as np
import random


class Generator:
    def __init__(self, f):
        """
        Creates instance of dataset generator.
        :param density:
        :param variance:
        :param dist:
        :param f: takes **two** parameters: `f(x, w)`
        """
        self.f = f

    def generate(self, dots_count, dist, density, variance, w):
        X = np.linspace(-dist, dist, density)
        Y = np.array([self.f(x, w) for x in X])
        Dataset_X = np.random.rand(dots_count, 1) * 2 * dist - dist
        Dataset_Y = np.array([(self.f(x, w) + random.uniform(-1, 1) * variance) for x in Dataset_X])

        return X, Y, Dataset_X, Dataset_Y