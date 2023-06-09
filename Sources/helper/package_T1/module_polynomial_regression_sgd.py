from matplotlib import pyplot as plt
import math
import copy
import numpy as np


def custom_gradient_descent_with_lr_scheduling(
        f
        , X
        , Y
        , x0
        , max_iter=100
        , initial_lr=0.1
        , eps=1e-6
        , minimum=0.0
        , constant_lambda=0.0009
        , apply_min=False
        , apply_value=True
):
    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    batch_size = int(0.9 * len(X))

    if apply_value:
        value = mse_loss(X, Y, x, constant_lambda)

    for iteration in range(1, max_iter):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                return points, iteration
        else:
            if apply_min and abs(mse_loss(X, Y, x, constant_lambda) - minimum) < eps:
                return points, iteration

        grad_x = mse_loss_grad(X, Y, x, batch_size, constant_lambda)
        new_x = x - grad_x * initial_lr

        if np.linalg.norm(grad_x) < eps:
            return points, iteration

        if apply_value:
            new_value = mse_loss(X, Y, new_x, constant_lambda)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points, max_iter


def func(x, w):
    pows = [x ** i for i in range(len(w))]
    return w.dot(pows)


def mse_loss(X, Y, w, constant_lambda):
    y_pred = np.array([func(x, w) for x in X])
    result = sum((Y - y_pred) * (Y - y_pred)) / len(Y)
    result += np.sum(np.abs(w) * constant_lambda)
    return result


def mse_loss_grad(X, Y, w, batch_size, constant_lambda):
    # Choose n random data points from the training set without replacement
    indices = np.random.choice(X.shape[0], batch_size, replace=False)
    # Getting data from dataset according indices
    X_batch = X[indices]
    y_batch = Y[indices]

    # Compute the gradient of the MSE loss with respect to x for the chosen data points
    y_pred = np.array([func(x, w) for x in X_batch])
    X_batch_pow = [X_batch ** i for i in range(len(w))]

    grad = (-2 / batch_size) * np.array([X_batch_pow[i].dot(y_batch - y_pred) for i in range(len(w))])

    grad += constant_lambda

    return grad
