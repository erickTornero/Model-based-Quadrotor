import joblib
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from utils.plotter import drawStdPlot
def plot_rewards(list_paths, smooth_factor=0.5, history=None):
    assert smooth_factor < 0.99, 'Smooth factor must be between 0.0 & 0.99'
    dirs    =   []
    for path in list_paths:
        if os.path.exists(path): dirs.append(path)
    
    #set_trace()
    sample_rws  =   [glob.glob(os.path.join(path,'rewards/rewards_it_*.pkl')) for path in dirs]

    n_samples   =   [len(s_rws) for s_rws in sample_rws]
    #sample_rws  =   [sorted(sample) for sample in sample_rws]
    set_trace()
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
    
    if history is not None: plt.legend(history)

    plt.grid(True)
    plt.show()



def get_rewards_lists(list_paths, smooth_factor=0.5, history=None):
    assert smooth_factor < 0.99, 'Smooth factor must be between 0.0 & 0.99'
    dirs    =   []
    for path in list_paths:
        if os.path.exists(path): dirs.append(path)
    
    #set_trace()
    sample_rws  =   [glob.glob(os.path.join(path,'rewards/rewards_it_*.pkl')) for path in dirs]

    n_samples   =   [len(s_rws) for s_rws in sample_rws]
    max_length  =   min(n_samples)
    n_samples   =   [max_length for _ in n_samples]
    #sample_rws  =   [sorted(sample) for sample in sample_rws]
    set_trace()
    rws_paths   =   []
    for nsample, dirname in zip(n_samples, dirs):
        rewards =   []
        for fn_it in range(1, nsample+1):
            rw_it   =   joblib.load(os.path.join(dirname, 'rewards/rewards_it_'+str(fn_it)+'.pkl'))
            rewards.append(np.mean(rw_it))
        rws_paths.append(rewards)
    
    return rws_paths

def plotStdPlotComparison(list_paths, smooth_factor=0.5, history=None):
    rewards_list    =   get_rewards_lists(list_paths, smooth_factor, history)
    axes            =   drawStdPlot(rewards_list, 'title', 'x','y')
    plt.show()

#l = ['data/sample16','data/sample25','data/sample26', 'data/sample27', 'data/sample28', 'data/sample29']

#l = ['data/sample4','data/sample5','data/sample6', 'data/sample7', 'data/sample12', 'data/sample13']
#l = ['data/sample5','data/sample6', 'data/sample7']
l = ['data/sample40', 'data/sample47']
#plot_rewards(l, 0.75 , l)
plotStdPlotComparison(l)
