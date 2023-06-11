from scipy.optimize import line_search
from collections import deque
import numpy as np


def lbfgs(f, dataset_X, dataset_Y, initial_w, max_iter=100, m=10, epsilon=1e-10):
    # print(dataset_X.shape, dataset_X)
    # print(dataset_Y.shape, dataset_Y)

    def mse_loss(w):
        y_pred = f(dataset_X, w)
        mse = np.mean((dataset_Y - y_pred) ** 2)
        return mse

    delta = 1e-9

    def mse_loss_grad(w):
        grad = np.zeros(len(w))

        for j in range(len(w)):
            w[j] += delta
            f_plus = mse_loss(w)
            w[j] -= delta
            w[j] -= delta
            f_minus = mse_loss(w)
            w[j] += delta
            grad[j] = np.divide(f_plus - f_minus, 2 * delta)

        return grad

    # result = [np.copy(initial_w)]
    current_x = np.copy(initial_w)
    n = len(initial_w)
    H = 1

    y = deque(maxlen=m)
    s = deque(maxlen=m)
    rho = deque(maxlen=m)


    cur_iter = 0
    for i in range(max_iter):

        if abs(mse_loss(current_x)) < epsilon:
            cur_iter = i + 1
            break

        q = mse_loss_grad(current_x)

        alpha = np.zeros(m)

        if i > m:
            for i in reversed(range(len(y))):
                alpha[i] = rho[i] * s[i].T @ q
                q = q - alpha[i] * y[i]

        r = H * q

        if i > m:
            for i in range(len(y)):
                beta = rho[i] * y[i].T @ r
                r = r + s[i] * (alpha[i] - beta)

        p = -r

        # Wolfe
        a, _, _, _, _, _ = line_search(mse_loss, mse_loss_grad, current_x, p, amax=100)

        if a is None:
            a = 1e-2

        new_x = current_x + a * p

        # if np.linalg.norm(new_x - current_x, 2) < tolerance:
        #     print("too small")
        #     break


        s.append(new_x - current_x)
        y.append(mse_loss_grad(new_x) - q)
        rho.append(1 / (y[-1].T @ s[-1] + 1e-10))

        H = s[-1].T @ y[-1] / (y[-1].T @ y[-1])

        current_x = new_x
        # result.append(current_x)

    return current_x, cur_iter
