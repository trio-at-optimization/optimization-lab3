import numpy as np


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
