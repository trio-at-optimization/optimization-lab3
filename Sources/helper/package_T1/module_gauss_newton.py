import numpy as np


def gauss_newton_fast(f, X, Y, initial_params, max_iter=100, epsilon=2e-2, delta=1e-6):
    params = np.array(initial_params)

    for iteration in range(max_iter):
        # ==========================================================================
        n_samples, n_features = X.shape[0], len(params)
        jacobian = np.zeros((n_samples, n_features), dtype=float)

        for i in range(n_samples):
            jacobian[i] = np.zeros(n_features, dtype=float)

            for j in range(n_features):
                params[j] += delta
                f_plus = f(X[i], params)
                params[j] -= delta
                params[j] -= delta
                f_minus = f(X[i], params)
                params[j] += delta
                jacobian[i][j] = np.divide(f_plus - f_minus, 2 * delta)
        # ==========================================================================

        residuals = Y - f(X, params)
        jacobian_T = jacobian.T

        # ==========================================================================

        update = (np.linalg.inv(jacobian_T @ jacobian) @ jacobian_T) @ residuals
        params += update

        if np.linalg.norm(update) < epsilon:
            return params, iteration + 1

    return params, max_iter + 1
