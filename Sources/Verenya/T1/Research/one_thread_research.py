import sys
import time
import traceback
import types
import numpy as np
from tqdm import tqdm
sys.path.append('../../../')
import helper


def get_func_method(argument):
    switch_dict = {
        'gauss-newton': helper.gauss_newton_fast,
        'dog-leg': helper.dog_leg,
    }

    result = switch_dict.get(argument)
    if result is None:
        raise ValueError("Unknown method")
    
    return result


def get_func_research(f_label):
    compiled_function = compile(f_label, "<string>", "exec")
    exec(compiled_function)

    # Создаем объект функции из скомпилированного кода
    return types.FunctionType(compiled_function.co_consts[0], globals())


def main(num_thread, method, dataset_name, filename_part, result_filename):
    func_method = get_func_method(method)
    dataset_params = helper.get_params_dataset(dataset_name)
    dataset_filename = helper.get_filenames_datasets()[dataset_name]
    f = get_func_research(dataset_params['f_label'])

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
                result_weights, count_step = func_method(f, datasets_X[k], datasets_Y[k], init_weights
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
