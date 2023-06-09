import subprocess
import time
import sys
import os
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
sys.path.append('../../../')
import helper


def research_thread(num_thread, script_path, dataset_name, method, result_filename, filename_part):

    process = subprocess.Popen(['cmd', '/c', 'python', script_path
                                   , str(num_thread)
                                   , method
                                   , dataset_name
                                   , filename_part
                                   , result_filename
                                ], creationflags=subprocess.CREATE_NEW_CONSOLE)
    print(f"Task start: {num_thread}")
    start = time.perf_counter()
    process.wait()
    finish = time.perf_counter()
    print(f"Task completed: {num_thread}, time {finish - start: .2f}")
    return num_thread


def main():
    result_name = '4'
    params = {
        'dataset_name': '1',
        'method': 'gauss-newton',
        'init_dist_x': 0,
        'init_dist_y': 10,
        'init_density_x': 1,
        'init_density_y': 10001
    }
    count_threads = max(cpu_count() - 2, 1)

    script_path = 'one_thread_research.py'
    dataset_params = helper.get_params_dataset(params['dataset_name'])

    start = time.perf_counter()
    print(f'Start Research')
    print('params', params)
    print('Generate linspace')
    init_x = np.linspace(dataset_params['w0'] - params['init_dist_x'], dataset_params['w0'] + params['init_dist_x'], params['init_density_x'])
    init_y = np.linspace(dataset_params['w1'] - params['init_dist_y'], dataset_params['w1'] + params['init_dist_y'], params['init_density_y'])
    print('Generate meshgrid')
    X, Y = np.meshgrid(init_x, init_y)
    print('Generate combined')
    combined = list(zip(X.flatten(), Y.flatten()))
    print('Split combined')
    split_parts = np.array_split(combined, count_threads)
    filenames_parts = []
    results_filenames = []
    print('Save parts to a file')
    for i in range(len(split_parts)):
        filenames_parts.append('part ' + str(i))
        helper.save_matrix(split_parts[i], filenames_parts[i])
        results_filenames.append('result ' + str(i))

    count_tasks = params['init_density_x'] * params['init_density_y'] * dataset_params['test_count']
    print(f'Scheduling tasks - {count_tasks}')
    with ProcessPoolExecutor() as executor:
        futures = []

        for i in range(count_threads):
            executor.submit(research_thread, i, script_path, params['dataset_name'], params['method'], results_filenames[i], filenames_parts[i])

        for future in concurrent.futures.as_completed(futures):
            num_thread = future.result()

    print('Delete parts files')
    for file_path in filenames_parts:
        os.remove(file_path)

    print('Combining results')
    results = []
    for file_path in results_filenames:
        result = helper.load_matrix(file_path)
        results.append(result)
        os.remove(file_path)

    results = np.concatenate((*results,))
    helper.save_matrix(results, 'temp')
    helper.add_result(result_name, params, 'temp')

    finish = time.perf_counter()
    print(f'It took {finish - start: .2f} second(s) to finish')


if __name__ == '__main__':
    main()
