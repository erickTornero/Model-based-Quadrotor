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


def plot_reward_bar_distributions(folder_dir, n_bars=20, rmin=0, rmax=1000.0, iterations=None, save_path=None):
    import os
    import glob
    all_files =   glob.glob(os.path.join(folder_dir, 'rewards_it_*.pkl'))
    if iterations is not None:
        from braceexpand import braceexpand
        iterations_set_itr = list_set_to_str(iterations)
        files_candidates  =   list(braceexpand(folder_dir+'rewards_it_'+iterations_set_itr+'.pkl'))
        all_files_set = set(all_files)
        files_  =   []
        for f in files_candidates:
            if f in all_files_set: files_.append(f)
        
        all_files = files_
    
    print(all_files)
    import joblib

    #rewards = [np.ceil(joblib.load(namef)/((rmax-rmin)/float(n_bars))).astype(int) for namef in all_files]
    rewards = [joblib.load(namef) for namef in all_files]
    stats = [np.unique(r, return_counts=True) for r in rewards]

    plt.figure(figsize=(12, 4))
    #from IPython.core.debugger import set_trace
    #set_trace()
    
    for i, st in enumerate(stats):
        plt.subplot(len(stats), 1, i+1)
        #plt.bar(st[0], st[1])
        plt.hist(rewards[i], np.arange(rmin, rmax, (rmax-rmin)/float(n_bars)), rwidth=0.85, alpha=0.6, color='#0504aa')
        if i==0:
            plt.title('Frequency of Rewards over iterations')
        plt.ylabel('Iteration '+ str(iterations[i]))
        plt.xlabel('Reward')
        #n, bins, patches = plt.hist(st[1], st[0], density=1, facecolor='g', alpha=0.75)
        #print(stats)
    #plt.tight_layout()
    
    plt.show()

        


    pass
def list_set_to_str(data_):
    data    =   list(data_)
    nplots  =   len(data)
    str_data    =   '{' if nplots > 1 else ''
    for i, d in enumerate(data): 
        str_data += str(d)
        if i < (nplots-1): str_data+=','

    str_data += '}' if nplots > 1 else ''
    return str_data