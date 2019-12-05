import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def plot_error_map(data, traj_len=15, cmap_name='OrRd', _vmin=0.0, _vmax=5.0, fig =None, ax=None):
    allowplot    =   False
    if ax is None and fig is None:
        fig , ax   =   plt.subplots(1)
        allowplot   =   True
    p = ax.pcolormesh(data, cmap=plt.get_cmap(cmap_name), vmin=_vmin,vmax=_vmax)
    fig.colorbar(p, ax=ax)
    
    if allowplot:
        plt.show()
    else:
        return fig, ax 

#dummydata = np.random.normal(0.0, 1.0, (15,15))
#plot_error_map(dummydata)

def plot_multiple_error_map(data, traj_len=15, cmap_name='OrRd', _vmin=0.0, _vmax=5.0):
    assert data.ndim == 3
    #for i in range(data.shape[2]):
    #    plt.subplot(1, i, data.shape[2])
    #    p   =   plt.pcolormesh(data[:,:,i], cmap=plt.get_cmap(cmap_name), vmin=_vmin, vmax=_vmax)
    #    plt.colorbar(p)
    #plt.show()
    ndynamics   =   data.shape[2]
    fig , axs   =   plt.subplots(1, ndynamics)
    
    for i in range(data.shape[2]):
        p   =   axs[i].pcolormesh(data[:,:,i], cmap=plt.get_cmap(cmap_name), vmin=_vmin, vmax=_vmax)
        fig.colorbar(p, ax=axs[i])
    
    plt.show()
    
    