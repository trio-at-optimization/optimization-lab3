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

count_threads = max(cpu_count() - 2, 1)
script_path = 'one_thread_research.py'
dataset_filename = "sin(w[0]x + w[1]) weights=[2. 3.] density=8000 dots_count=1000 radius=0.1 dist=2.5 test_count=10.txt"
test_count = 10


def research_thread(num_thread, result_filename, filename_part):

    process = subprocess.Popen(['cmd', '/c', 'python', script_path
                                   , str(num_thread)
                                   , dataset_filename
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
    x = 2
    y = 3
    init_dist_x = 0
    init_dist_y = 50
    init_density_x = 1
    init_density_y = 10001
    label = 'gauss-newton '
    label += 'init_dist_x=' + str(init_dist_x) + ' '
    label += 'init_dist_y=' + str(init_dist_y) + ' '
    label += 'init_density_x=' + str(init_density_x) + ' '
    label += 'init_density_y=' + str(init_density_y) + ' '

    start = time.perf_counter()
    print(f'Start Research')
    print(f'params: init_dist_x {init_dist_x}, init_dist_y {init_dist_y}, init_density_x {init_density_x}, init_density_y {init_density_y}')
    print('Generate linspace')
    init_x = np.linspace(x - init_dist_x, x + init_dist_x, init_density_x)
    init_y = np.linspace(y - init_dist_y, y + init_dist_y, init_density_y)
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

    print(f'Scheduling tasks - {init_density_x * init_density_y * test_count}')
    with ProcessPoolExecutor() as executor:
        futures = []

        for i in range(count_threads):
            executor.submit(research_thread, i, results_filenames[i], filenames_parts[i])

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
    helper.save_matrix(results, 'result ' + label + dataset_filename)

    finish = time.perf_counter()
    print(f'It took {finish - start: .2f} second(s) to finish')


if __name__ == '__main__':
    main()
