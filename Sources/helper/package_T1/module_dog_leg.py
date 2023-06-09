import numpy as np
import math
import copy


def func_without_f(x, w):
    return w[0] * np.sin(x ** 2) + np.cos(w[1] * x)
    # return (1 - x * w[0]) ** 2 + 100 * (x * w[1] - x ** 2) ** 2


def func(X, Y, w, j):
    x = X[j]
    f = Y[j]

    return func_without_f(x, w) - f


def objF(X, Y, w):
    obj = np.zeros(len(X))

    for i in range(len(X)):
        obj[i] = func(X, Y, w, i)

    return obj


def derivative(X, Y, w, i, j, delta=1e-6):
    w1 = np.copy(w)
    w2 = np.copy(w)

    w1[i] -= delta
    w2[i] += delta

    obj1 = func(X, Y, w1, j)
    obj2 = func(X, Y, w2, j)

    return (obj2 - obj1) / (2 * delta)


def jacobian(X, Y, w):
    rowNum = len(X)
    colNum = len(w)

    Jac = np.zeros((rowNum, colNum))

    for i in range(rowNum):
        for j in range(colNum):
            Jac[i][j] = derivative(X, Y, w, j, i)

    return Jac


def dog_leg(X, Y, initial_params, max_iter=1000, radius=1.0, e1=1e-12, e2=1e-12, e3=1e-12):
    result = [initial_params]
    current_params = np.copy(initial_params)

    obj = objF(X, Y, current_params)
    Jac = jacobian(X, Y, current_params)
    gradient = Jac.T @ obj

    if np.linalg.norm(obj) <= e3 or np.linalg.norm(gradient) <= e1:
        return

    for i in range(max_iter):
        obj = objF(X, Y, current_params)
        Jac = jacobian(X, Y, current_params)
        gradient = Jac.T @ obj

        if np.linalg.norm(gradient) <= e1:
            print("stop F'(x) = g(x) = 0 for a global minimizer optimizer.")
            break
        elif np.linalg.norm(obj) <= e3:
            print("stop f(x) = 0 for f(x) is so small")
            break

        alpha = np.linalg.norm(gradient, 2) / np.linalg.norm(Jac * gradient, 2)
        stepest_descent = -alpha * gradient
        gauss_newton = -1 * np.linalg.inv(Jac.T @ Jac) @ Jac.T @ obj

        beta = 0.0
        dog_leg = np.zeros(len(current_params))

        if np.linalg.norm(gauss_newton) <= radius:
            dog_leg = np.copy(gauss_newton)
        elif alpha * np.linalg.norm(stepest_descent) >= radius:
            dog_leg = (radius / np.linalg.norm(stepest_descent)) * stepest_descent
        else:
            a = alpha * stepest_descent
            b = np.copy(gauss_newton)
            c = a.T @ (b - a)

            if c <= 0:
                beta = (math.sqrt(
                    c * c + np.linalg.norm(b - a, 2) * (
                                radius * radius - np.linalg.norm(a, 2))) - c) / np.linalg.norm(
                    b - a, 2)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a, 2) * abs(radius * radius - np.linalg.norm(a, 2))) - c)
            dog_leg = alpha * stepest_descent + (gauss_newton - alpha * stepest_descent) * beta

        print(f'dog-leg: {dog_leg}')

        if np.linalg.norm(dog_leg) <= e2 * (np.linalg.norm(current_params) + e2):
            break

        new_params = current_params + dog_leg

        print(f'new parameter is: {new_params}\n')

        obj = objF(X, Y, current_params)
        obj_new = objF(X, Y, new_params)

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
                beta = (math.sqrt(
                    c * c + np.linalg.norm(b - a, 2) * (
                                radius * radius - np.linalg.norm(a, 2))) - c) / np.linalg.norm(
                    b - a, 2)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a, 2) * abs(radius * radius - np.linalg.norm(a, 2))) - c)

            delta_l = alpha * (1 - beta) * (1 - beta) * np.linalg.norm(gradient, 2) / 2.0 + beta * (
                        2.0 - beta) * np.linalg.norm(obj, 2) / 2

        roi = deltaF / delta_l

        if roi > 0:
            current_params = np.copy(new_params)
        if roi > 0.75:
            radius = max(radius, 3.0 * np.linalg.norm(dog_leg))
        elif roi < 0.25:
            radius /= 2.0

            if radius <= e2 * (np.linalg.norm(current_params) + e2):
                print("trust region radius is too small.")
                break

        result.append(current_params)
        print(f'radius: {radius}')

    return result
