import matplotlib.pyplot as plt


def print_generated(X, Y, dataset, answer=None):
    # Plot style:
    plt.style.use('default')
    _ = plt.figure(figsize=(8, 8))
    # ===========
    plt.axis('equal')

    x = dataset[:, 0]
    y = dataset[:, 1]

    plt.scatter(x, y, label='Data', color='gray', alpha=0.5, s=20.8, antialiased=True)
    plt.plot(X, Y, label='Real', color='lime', antialiased=True, linewidth=1.7)
    if answer is not None:
        plt.plot(X, answer, label='Answer', color='red', antialiased=True, linewidth=1.5)
    plt.legend()
    plt.show()
