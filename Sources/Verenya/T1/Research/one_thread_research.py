import sys
import time
import traceback
import json
import numpy as np
from tqdm import tqdm
sys.path.append('../../../')
import helper


def main(num_thread, method, dataset_name, filename_part, result_filename):
    func_method = helper.get_func_method(method)
    dataset_params = helper.get_params_dataset(dataset_name)
    dataset_filename = helper.get_filenames_datasets()[dataset_name]
    f = helper.get_func_research(dataset_params['f_label'])

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
        current_point_result = {"init_weights": [current_part[i][0], current_part[i][1]], "results": []}
        init_weights = np.array([current_part[i][0], current_part[i][1]], dtype=float)
        for k in range(test_count):
            result_weights, count_step = func_method(f, datasets_X[k], datasets_Y[k], init_weights
                                                           , epsilon=2e-2, max_iter=100)
            current_point_result["results"].append([result_weights.tolist(), count_step])
            progress_bar.update(1)

        results.append(current_point_result)

    progress_bar.close()
    helper.save_json(results, result_filename, indent=0)


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
