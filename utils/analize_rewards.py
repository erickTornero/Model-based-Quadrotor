import joblib
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
def plot_rewards(list_paths, smooth_factor=0.5):
    assert smooth_factor < 0.99, 'Smooth factor must be between 0.0 & 0.99'
    dirs    =   []
    for path in list_paths:
        if os.path.exists(path): dirs.append(path)
    
    #set_trace()
    sample_rws  =   [glob.glob(os.path.join(path,'rewards/rewards_it_*.pkl')) for path in dirs]

    n_samples   =   [len(s_rws) for s_rws in sample_rws]
    #sample_rws  =   [sorted(sample) for sample in sample_rws]

    rws_paths   =   []
    for nsample, dirname in zip(n_samples, dirs):
        rewards =   []
        for fn_it in range(1, nsample+1):
            rw_it   =   joblib.load(os.path.join(dirname, 'rewards/rewards_it_'+str(fn_it)+'.pkl'))
            rewards.append(np.mean(rw_it))
        rws_paths.append(rewards)

    if smooth_factor > 0.0:
        lasts       =   [sample_rw[0] for sample_rw in rws_paths]
        smooth_samples  =   []
        for sample_rw, last in zip(rws_paths, lasts):
            smooth  =   []
            for rw in sample_rw:
                smooth_val  =   last * smooth_factor + (1 - smooth_factor) * rw
                smooth.append(smooth_val)
                last    =   smooth_val
            smooth_samples.append(smooth)
        rws_paths   =   smooth_samples

    for rws in rws_paths:
        plt.plot(rws)

    plt.grid(True)
    plt.show()

l = ['data/sample25','data/sample26', 'data/sample27']

plot_rewards(l, 0.8)