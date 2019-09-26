import matplotlib.pyplot as plt
import numpy as np
def plot_loss_per_iteration(tr_loss, val_loss, save_path):
    t = np.arange(1, len(tr_loss) + 1, 1)
    plt.figure()
    plt.plot(t, tr_loss, color='red')
    plt.plot(t, val_loss, color='blue')
    plt.title('Loss')
    plt.legend(['Training loss', 'Validation Loss'])

    plt.savefig(save_path)


def plot_scatter_positions(data, index_start_pos, save_path=None):
    """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
    x_data   =   data[:, index_start_pos]
    y_data   =   data[:, index_start_pos + 1]
    z_data   =   data[:, index_start_pos + 2]

    """ Initialize subplot 1 x 3 """
    plt.figure(figsize=(12, 4))
    

    """ Distribution X-Y"""
    plt.subplot(1, 3, 1)
    plt.scatter(x_data, y_data, alpha=0.2, marker='o', s=5, color='blue')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

    """ Distribution of Position in X-Z"""
    plt.subplot(1, 3, 2)
    plt.scatter(x_data, z_data, alpha=0.2, marker='o', s=5, color='red')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

    """ Distribution of Position in Y-Z"""
    plt.subplot(1, 3, 3)
    plt.scatter(y_data, z_data, alpha=0.2, marker='o', s=5, color='gray')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

    """ Final commands """
    plt.tight_layout()
    plt.show() if save_path is None else plt.savefig(save_path)