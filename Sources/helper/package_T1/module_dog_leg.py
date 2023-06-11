import math
import copy
import numpy as np

def func(f, X, Y, w, j):
    x = X[j]
    value = Y[j]

    return f(x, w) - value


def derivative(f, X, Y, w, i, j, delta=1e-6):
    w1 = np.copy(w)
    w2 = np.copy(w)

    w1[i] -= delta
    w2[i] += delta

    obj1 = func(f, X, Y, w1, j)
    obj2 = func(f, X, Y, w2, j)

    return (obj2 - obj1) / (2 * delta)


def jacobian(f, X, Y, w, delta):
    rowNum = len(X)
    colNum = len(w)

    Jac = np.zeros((rowNum, colNum))

    for i in range(rowNum):
        for j in range(colNum):
            Jac[i][j] = derivative(f, X, Y, w, j, i, delta)

    return Jac


def dog_leg(f, X, Y, initial_params, max_iter=100, epsilon=2e-2, delta=1e-6, radius=1.5):
    e1 = epsilon  # 1e-12
    e2 = epsilon  # 1e-12
    e3 = epsilon  # 1e-12

    current_params = np.copy(initial_params)

    obj = f(X, current_params) - Y
    Jac = jacobian(f, X, Y, current_params, delta)
    gradient = Jac.T @ obj

    if np.linalg.norm(obj) <= e3 or np.linalg.norm(gradient) <= e1:
        return current_params, -1

    for iteration in range(max_iter):
        obj = f(X, current_params) - Y
        Jac = jacobian(f, X, Y, current_params, delta)
        gradient = Jac.T @ obj

        if np.linalg.norm(gradient) <= e1:
            # print("stop F'(x) = g(x) = 0 for a global minimizer optimizer.")
            return current_params, iteration + 1
        elif np.linalg.norm(obj) <= e3:
            # print("stop f(x) = 0 for f(x) is so small")
            return current_params, iteration + 1

        alpha = np.linalg.norm(gradient) ** 2 / np.linalg.norm(Jac @ gradient) ** 2
        stepest_descent = -alpha * gradient
        gauss_newton = -1 * np.linalg.inv(Jac.T @ Jac) @ Jac.T @ obj

        beta = 0.0
        dog_leg = np.zeros(len(current_params))

        if np.linalg.norm(gauss_newton) <= radius:
            # print('Now GAUSS_NEWTON')
            dog_leg = np.copy(gauss_newton)
        elif alpha * np.linalg.norm(stepest_descent) >= radius:
            dog_leg = (radius / np.linalg.norm(stepest_descent)) * stepest_descent
        else:
            a = alpha * stepest_descent
            b = np.copy(gauss_newton)
            c = a.T @ (b - a)

            if c <= 0:
                beta = (math.sqrt(math.fabs(
                    c * c + np.linalg.norm(b - a, 2) * (radius * radius - np.linalg.norm(a) ** 2))) - c) / (np.linalg.norm(
                    b - a, 2) + e3)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a, 2) * max(0, radius * radius - np.linalg.norm(a, 2))) - c + e3)
            dog_leg = alpha * stepest_descent + (gauss_newton - alpha * stepest_descent) * beta

        # print(f'dog-leg: {dog_leg}')

        if np.linalg.norm(dog_leg) <= e2 * (np.linalg.norm(current_params) + e2):
            return current_params, iteration + 1

        new_params = current_params + dog_leg

        # print(f'new parameter is: {new_params}\n')

        obj_new = f(X, new_params) - Y

        deltaF = np.linalg.norm(obj, 2) / 2 - np.linalg.norm(obj_new, 2) / 2

        delta_l = 0.0

        if np.linalg.norm(gauss_newton) <= radius:
            delta_l = np.linalg.norm(obj, 2) / 2
        elif alpha * np.linalg.norm(stepest_descent) >= radius:
            delta_l = radius * (2 * alpha * np.linalg.norm(gradient) - radius) / (2.0 * alpha)
        else:
            a = stepest_descent * alpha
            b = copy.copy(gauss_newton)
            c = a.T @ (b - a)

            if c <= 0:
                beta = (math.sqrt(math.fabs(
                    c * c + np.linalg.norm(b - a, 2) * (radius * radius - np.linalg.norm(a, 2)))) - c) / (np.linalg.norm(
                    b - a, 2) + e3)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a) ** 2 * max(0, radius * radius - np.linalg.norm(a, 2))) - c + e3)

            delta_l = alpha * (1 - beta) * (1 - beta) * (np.linalg.norm(gradient) ** 2) / 2.0 + beta * (
                    2.0 - beta) * np.linalg.norm(obj, 2) / 2

        roi = deltaF / delta_l

        if roi > 0:
            current_params = np.copy(new_params)
        if roi > 0.75:
            radius = max(radius, 3.0 * np.linalg.norm(dog_leg))
        elif roi < 0.25:
            radius /= 2.0

            if radius <= e2 * (np.linalg.norm(current_params) + e2):
                # print("trust region radius is too small.")
                return current_params, iteration + 1

    return current_params, max_iter + 1
