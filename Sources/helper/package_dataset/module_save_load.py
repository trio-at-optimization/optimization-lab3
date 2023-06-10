import numpy as np
import glob
import os
import json
import shutil


def save_datasets(X, Y, datasets, filename):
    with open(filename, 'w') as file:
        file.write(f'{len(X)}\n')
        np.savetxt(file, X, delimiter=' ')

        file.write(f'{len(Y)}\n')
        np.savetxt(file, Y, delimiter=' ')

        for dataset in datasets:
            rows, cols = dataset.shape
            file.write(f'{rows} {cols}\n')
            np.savetxt(file, dataset, delimiter=' ')


def load_datasets(filename):
    datasets = []
    with open(filename, 'r') as file:
        line = file.readline().strip()
        rows_x = int(line)
        X = np.loadtxt(file, delimiter=' ', max_rows=rows_x)

        line = file.readline().strip()
        rows_y = int(line)
        Y = np.loadtxt(file, delimiter=' ', max_rows=rows_y)

        line = file.readline().strip()
        while line:
            rows, cols = map(int, line.split())
            dataset = np.loadtxt(file, delimiter=' ', max_rows=rows)
            datasets.append(dataset)
            line = file.readline().strip()
    return X, Y, datasets


def save_matrix(result_matrix, filename):
    with open(filename, 'w') as file:
        rows, cols = result_matrix.shape
        file.write(f'{rows} {cols}\n')
        np.savetxt(file, result_matrix, delimiter=' ')


def load_matrix(filename):
    with open(filename, 'r') as file:
        line = file.readline().strip()
        rows, cols = map(int, line.split())
        result_matrix = np.loadtxt(file, delimiter=' ', max_rows=rows)
    return result_matrix


def save_json(data, filename, indent=4):
    with open(filename, "w") as file:
        json.dump(data, file, indent=indent)


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# ======================================================================================================================
Datasets_folder_name = 'Datasets'
Results_folder_name = 'Results'

Dataset_file_name = 'dataset.txt'
Result_file_name = 'result.json'
Params_file_name = 'params.json'


def get_filenames(path, folder_path, filename):
    file_paths = glob.glob(path + folder_path + '/*/' + filename)

    file_dict = {}

    for file_path in file_paths:
        parent_folder = os.path.basename(os.path.dirname(file_path))
        file_dict[parent_folder] = file_path

    return file_dict


def get_filenames_datasets(path=''):
    return get_filenames(path, Datasets_folder_name, Dataset_file_name)


def get_filenames_results(path=''):
    return get_filenames(path, Results_folder_name, Result_file_name)


def add_folder(folder_path, name, params, current_filename, need_filename):
    os.makedirs(folder_path + '/' + name, exist_ok=True)
    shutil.move(current_filename, folder_path + '/' + name + '/' + need_filename)
    set_params(folder_path, params, name)


def add_dataset(name, params, current_filename):
    add_folder(Datasets_folder_name, name, params, current_filename, Dataset_file_name)


def add_result(name, params, current_filename):
    add_folder(Results_folder_name, name, params, current_filename, Result_file_name)


def get_params(folder_path, name, path):
    file_path = path + folder_path + "/" + name + "/" + Params_file_name

    with open(file_path, "r") as json_file:
        parameters = json.load(json_file)

    return parameters


def get_params_dataset(name, path=''):
    return get_params(Datasets_folder_name, name, path)


def get_params_results(name, path=''):
    return get_params(Results_folder_name, name, path)


def set_params(folder_path, params, name):
    file_path = folder_path + "/" + name + "/" + Params_file_name

    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)


def set_params_dataset(params, name):
    set_params(Datasets_folder_name, params, name)


def set_params_results(params, name):
    set_params(Results_folder_name, params, name)
