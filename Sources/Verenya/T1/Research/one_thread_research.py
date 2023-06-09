import sys
import time
import traceback
import numpy as np
from tqdm import tqdm
sys.path.append('../../../')
import helper


def f(x, w):
    return_value = np.sin(w[0] * x) + w[1]
    return return_value


def gauss_newton_fast(x, y, initial_params, max_iter=100, epsilon=2e-2, delta=1e-6):
    params = np.array(initial_params)

    for iteration in range(max_iter):
        # ==========================================================================

        n_samples, n_features = x.shape[0], len(params)
        jacobian = np.zeros((n_samples, n_features), dtype=float)

        for i in range(n_samples):
            jacobian[i] = np.zeros(n_features, dtype=float)

            for j in range(n_features):
                params[j] += delta
                f_plus = f(x[i], params)
                params[j] -= delta
                params[j] -= delta
                f_minus = f(x[i], params)
                params[j] += delta
                jacobian[i][j] = np.divide(f_plus - f_minus, 2 * delta)
        # ==========================================================================

        residuals = y - f(x, params)
        jacobian_T = jacobian.T

        # ==========================================================================

        update = (np.linalg.inv(jacobian_T @ jacobian) @ jacobian_T) @ residuals
        params += update

        if np.linalg.norm(update) < epsilon:
            return params, iteration

    return params, max_iter


def get_func(argument):
    switch_dict = {
        'gauss-newton': gauss_newton_fast,
    }

    result = switch_dict.get(argument)
    if result is None:
        raise ValueError("Unknown method")
    
    return result
    

def main(num_thread, method, dataset_filename, filename_part, result_filename):
    func = get_func(method)
    X, Y, datasets = helper.load_datasets(dataset_filename)
    current_part = helper.load_matrix(filename_part)

    test_count = len(datasets)
    datasets_X = []
    datasets_Y = []
    for i in range(test_count):
        datasets_X.append(datasets[i][:, 0])
        datasets_Y.append(datasets[i][:, 1])

    progress_bar = tqdm(total=len(current_part) * test_count
                        , desc="Thread: " + str(num_thread))

    results = []
    for i in range(len(current_part)):
            # sum_mse_result = 0
            sum_step_count = 0
            init_weights = np.array([current_part[i][0], current_part[i][1]], dtype=float)
            for k in range(test_count):
                result_weights, count_step = func(datasets_X[k], datasets_Y[k], init_weights
                                                               , epsilon=2e-2, max_iter=100)
                # result_loss = mse_loss(datasets_X[k], datasets_Y[k], result_weights, f)
                # sum_mse_result += result_loss
                sum_step_count += count_step
                progress_bar.update(1)
            # sum_mse_result /= test_count
            sum_step_count /= test_count
            results.append(np.append(current_part[i], sum_step_count))

    results = np.array(results)
    progress_bar.close()
    helper.save_matrix(results, result_filename)


if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        print("Start thread", args[0])

        main(int(args[0])
             , args[1]
             , args[2]
             , args[3]
             , args[4]
             )

        print("End", args[0])
    except Exception as e:
        print("Exception caught!")
        print("Type of exception:", type(e).__name__)
        print("Error Message:", str(e))
        print("Stack trace:")
        traceback.print_exc()
        for i in tqdm(range(100), desc="Time before close"):
            time.sleep(1)
