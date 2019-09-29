from IPython.core.debugger import set_trace
import joblib
import os
import numpy as np
import glob
import matplotlib.pyplot as plt


def compute_restore_file(fold_path, id_str):
    """ Return the path direction to the file of paths if it exists else return: None
        fold_path:  Folder path of training example (./data/sample5/)
        id_str:     Id rolls running
    """

    fold_path       =   os.path.join(fold_path, 'rolls'+id_str)
    file_path       =   os.path.join(fold_path,'paths.pkl')

    files_list      =   glob.glob(file_path)
    
    return (file_path if len(files_list) > 0 else None) 


def plot_trajectory(fold, id_ex, list_paths=None):
    """ 
        Plot specific trajectories given in list_paths else
        plots all the trajectories

        The plots are shown in scattering way and shows 3 subplots (3, 1)
        Where the plot:
        (3,1,1): X-Y
        (3,1,2): X-Z
        (3,1,3): Y-Z
    """
    path_name   =   compute_restore_file(fold, id_ex)
    assert path_name is not None, 'Not file of paths founded'
    
    paths       =   joblib.load(path_name)
    

    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))

    nfigures    =   len(list_paths)
    #set_trace()
    plt.figure(figsize=(12,4))
    index_start_pos =   63 # Index where position starts
    for i, i_path  in enumerate(list_paths):
        #plt.subplot(nfigures, 3, i + 1)
        
        data    =   paths[i_path]['observation']

        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        x_data   =   data[:, index_start_pos]
        y_data   =   data[:, index_start_pos + 1]
        z_data   =   data[:, index_start_pos + 2]

        """ Initialize subplot 1 x 3 """
        #plt.figure(figsize=(12, 4))
        

        """ Distribution X-Y"""
        plt.subplot(1, 3, 1)
        plt.scatter(x_data, y_data, alpha=0.6, marker='o', s=5)
        #plt.plot(x_data, y_data)
        #circ1 = plt.Circle((x_data[0], y_data[0]), radius=20, color='red')
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        #plt.plot(circ1)
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)

        """ Distribution of Position in X-Z"""
        plt.subplot(1, 3, 2)
        plt.scatter(x_data, z_data, alpha=0.6, marker='o', s=5)
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)

        """ Distribution of Position in Y-Z"""
        plt.subplot(1, 3, 3)
        plt.scatter(y_data, z_data, alpha=0.6, marker='o', s=5)
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)

    plt.legend(['Path '+ str(i_path) for i_path in list_paths])

    plt.show()

# Example:
#plot_trajectory('./data/sample5/','1', [0, 1, 2])

def plot_pos_over_time(fold, id_ex, max_path_length=250, list_paths=None):
    """
        Shows the behavior of position in x-y-z over time
        This trajectory must converge to the target point
    """
    path_name   =   compute_restore_file(fold, id_ex)


    assert path_name is not None, 'Not file of paths founded'

    paths           =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))

    #index_start_pos =   63
    index_start_pos =   27
    plt.figure(figsize=(12, 4))
    for i_path in list_paths:
        data    =   paths[i_path]['observation']
        targets =   paths[i_path]['target']

        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        x_data   =   data[:max_path_length, index_start_pos]
        y_data   =   data[:max_path_length, index_start_pos + 1]
        z_data   =   data[:max_path_length, index_start_pos + 2]

        """target data"""
        x_target    =   targets[:max_path_length, 0]
        y_target    =   targets[:max_path_length, 1]
        z_target    =   targets[:max_path_length, 2]

        x_data  =   x_data + x_target
        y_data  =   y_data + y_target
        z_data  =   z_data + z_target
        """
            PLOT X POS
        """
        plt.subplot(1, 3, 1)
        plt.plot(np.arange(len(x_data)), x_data, color='red')
        plt.plot(np.arange(len(x_target)), x_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Y POS
        """
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(len(y_data)), y_data, color='blue')
        plt.plot(np.arange(len(y_target)), y_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Z POS
        """
        plt.subplot(1, 3, 3)
        plt.plot(np.arange(len(z_data)), z_data, color='gray')
        plt.plot(np.arange(len(z_target)), z_target, '-', color='olive',linestyle='dashed')

    axis    =   ['X','Y','Z']    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        #plt.plot(np.arange(max_path_length), max_path_length * [0.0], '-', color='olive', linestyle='dashed')
        plt.xlabel('timestep')
        plt.ylabel('position (m)')

        plt.title('Position in '+axis[i]+' axis')
        plt.xlim(0,max_path_length)
        plt.ylim(-2,2)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #total_rollouts  =   len(data_pos)
        plt.grid(True)
        plt.legend(['roll '+ str(idx) for idx in list_paths])
        plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()

# Example of its usage
plot_pos_over_time('./data/sample6/', '1', list_paths=[4])
