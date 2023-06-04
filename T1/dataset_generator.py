import torch
import numpy as np

class Generator:
    def __init__(self, f):
        """
        Creates instance of dataset generator.
        :param f: takes **two** parameters: `f(x, w)`
        """
        self.f = f

    def generate(self, dots_count, dist, density, radius, weights):
        """
        Creates instance of dataset generator.
        :param dots_count:
        :param dist:
        :param radius:
        :param density:
        :param weights:
        """

        X = np.linspace(-dist, dist, density)
        Y = np.array([self.f(x, weights) for x in X])

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
                dataset_X.append(x_rand.item())
                dataset_Y.append(y_rand.item())

        return X, Y, np.array(dataset_X), np.array(dataset_Y)



    def generate_multi(self, dots_count, dist, density, radius, weights):
        """
        Creates instance of dataset generator.
        :param dots_count:
        :param dist:
        :param radius:
        :param density:
        :param weights:
        """

        X = np.linspace(-dist, dist, density)
        Y = np.array([self.f(x, weights) for x in X])

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
                dataset_X.append([x_rand.item()])
                dataset_Y.append([y_rand.item()])

        return np.array([X]), Y, np.array(dataset_X), np.array(dataset_Y)
    
