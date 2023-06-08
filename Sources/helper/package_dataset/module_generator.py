import torch
import numpy as np
import random


class Generator:
    def __init__(self, f):
        """
        Creates instance of dataset generator.
        :param f: takes **two** parameters: `f(x, w)`
        """
        self.f = f

    def generate_simple(self, dots_count, dist, density, variance, w):
        """
        Creates instance of dataset generator.
        :param dots_count:
        :param dist:
        :param density:
        :param variance:
        :param w:
        """

        X = np.linspace(-dist, dist, density)
        Y = np.array([self.f(x, w) for x in X])
        Dataset_X = np.random.rand(dots_count, 1) * 2 * dist - dist
        Dataset_Y = np.array([(self.f(x, w) + random.uniform(-1, 1) * variance) for x in Dataset_X])

        return X, Y, Dataset_X, Dataset_Y

    def generate(self, dots_count, dist, density, radius, weights, isMulti=None):
        if isMulti is None:
            isMulti = False

        """
        Creates instance of dataset generator.
        :param dots_count:
        :param dist:
        :param radius:
        :param density:
        :param weights:
        :param isMulti:
        """

        X = np.linspace(-dist, dist, density)
        Y = np.array([self.f(x, weights) for x in X])

        if isMulti:
            X = np.array([X])

        return X, Y, self.generate_without_XY(X, Y, dots_count, dist, density, radius, weights, isMulti)

    def generate_without_XY(self, X, Y, dots_count, dist, density, radius, weights, isMulti=None):
        if isMulti is None:
            isMulti = False

        dataset_X = []
        dataset_Y = []

        x_min = min(X) - radius
        y_min = min(Y) - radius
        x_max = max(X) + radius
        y_max = max(Y) + radius

        method = 'cpu'
        if torch.cuda:
            if torch.cuda.is_available():
                method = 'cuda'

        # print(method)
        device = torch.device(method)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

        while len(dataset_X) < dots_count:
            x_rand = torch.empty(1).uniform_(x_min, x_max).to(device)
            y_rand = torch.empty(1).uniform_(y_min, y_max).to(device)

            within_radius = (x_rand - X_tensor) ** 2 + (y_rand - Y_tensor) ** 2 <= radius ** 2
            if torch.any(within_radius):
                if isMulti:
                    dataset_X.append([x_rand.item()])
                    dataset_Y.append([y_rand.item()])
                else:
                    dataset_X.append(x_rand.item())
                    dataset_Y.append(y_rand.item())

        return np.array(dataset_X), np.array(dataset_Y)

def generate_func(f, dots_count, dist, density, radius, weights, isMulti=None):
    g = Generator(f)
    return g.generate(dots_count, dist, density, radius, weights, isMulti)

def generate_full_dataset(X, Y, dots_count, radius):
    dataset = []

    x_min = min(X) - radius
    y_min = min(Y) - radius
    x_max = max(X) + radius
    y_max = max(Y) + radius

    method = 'cpu'
    if torch.cuda:
        if torch.cuda.is_available():
            method = 'cuda'

    # print(method)
    device = torch.device(method)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    while len(dataset) < dots_count:
        x_rand = torch.empty(1).uniform_(x_min, x_max).to(device)
        y_rand = torch.empty(1).uniform_(y_min, y_max).to(device)

        within_radius = (x_rand - X_tensor) ** 2 + (y_rand - Y_tensor) ** 2 <= radius ** 2
        if torch.any(within_radius):
            dataset.append([x_rand.item(), y_rand.item()])

    return np.array(dataset)
