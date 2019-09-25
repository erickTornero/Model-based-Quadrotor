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