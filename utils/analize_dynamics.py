import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def plot_error_map(data, traj_len=15, cmap_name='OrRd', _vmin=0.0, _vmax=5.0):
    fig, ax = plt.subplots(1)
    p = ax.pcolormesh(data, cmap=plt.get_cmap(cmap_name), vmin=_vmin,vmax=_vmax)
    fig.colorbar(p, ax=ax)
    plt.show()

#dummydata = np.random.normal(0.0, 1.0, (15,15))
#plot_error_map(dummydata)

