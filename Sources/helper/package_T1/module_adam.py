import numpy as np
import math

from ..package_dataset import mse_loss

# ===================== ADAM =====================


def mse_loss_adam(f, X, Y, w):
    return mse_loss(X, Y, w, f)


def mse_loss_grad_adam(f, X, y, w, batch_size, delta=1e-6):
    # Choose n random data points from the training set without replacement
    indices = np.random.choice(X.shape[0], batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]

    grad = np.zeros(len(w))

    for j in range(len(w)):
        w[j] += delta
        f_plus = mse_loss_adam(f, X_batch, y_batch, w)
        w[j] -= delta
        w[j] -= delta
        f_minus = mse_loss_adam(f, X_batch, y_batch, w)
        w[j] += delta
        grad[j] = np.divide(f_plus - f_minus, 2 * delta)

    return grad


def adam(
        f
        , X
        , Y
        , x0
        , initial_lr=2.5
        , max_iter=100
        , eps=4e-4
        , epsilon=2e-2
        , eps_Adam=1e-8
        , minimum=0.0
        , apply_min=True
        , apply_value=True
):
    B1 = 0.9
    B2 = 0.999

    batch_size = len(X)

    x = np.copy(x0)
    # points = [x.copy()]
    value = mse_loss_adam(f, X, Y, x)
    G = eps_Adam
    moment = 0.0

    cur_iter = 0
    for i in range(1, max_iter):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                cur_iter = i
                break
        else:
            if apply_min and abs(value - minimum) < eps:
                cur_iter = i
                break

        grad_x = mse_loss_grad_adam(f, X, Y, x, batch_size)
        # print(f'adam gradient: {grad_x}\nx:{x}')
        moment = moment * B1 + (1 - B1) * grad_x
        G = G * B2 + (1 - B2) * (grad_x.dot(grad_x))
        # new_x = x - lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_Adam)) * moment

        moment_more = moment / (1 - B1 ** i)
        G_more = G / (1 - B2 ** i)
        new_x = x - initial_lr * (moment_more) / (math.sqrt(G_more + eps_Adam))

        # print(f'step adam: {initial_lr * (moment_more) / (math.sqrt(G_more + eps_Adam))}\n newx: {new_x}')

        if apply_value:
            new_value = mse_loss_adam(f, X, Y, new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        # points.append(x.copy())

    return x, cur_iter

# # ===================== ADAM =====================
