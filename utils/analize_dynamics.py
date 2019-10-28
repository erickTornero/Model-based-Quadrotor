import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

data = np.random.randn(10,10) * 20
cdic    =   {'red':((0.0,0.0,0.0),(0.5,1.0,1.0),(1.0,0.8,0.8)),
             'green':((0.0,0.8,0.8),(0.5,1.0,1.0),(1.0,0.0,0.0)),
             'blue':((0.0,0.0,0.0),(0.5,1.0,1.0),(1.0,0.0,0.0))
            }
gnrd=colors.LinearSegmentedColormap('bn',cdic)
#cmap = colors.ListedColormap(['red', 'blue'])



#bounds = [0, 10, 20]

#norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1)
dummydata = np.random.randn(15,15) *6.-3.
p = ax.pcolormesh(dummydata, cmap=gnrd, vmin=-3,vmax=3)
fig.colorbar(p,ax=ax)
plt.show()