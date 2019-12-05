from IPython.core.debugger import set_trace
import joblib
import os
import numpy as np
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import json
from utils.plotter import drawStdPlot
from itertools import count

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

    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)

    nstack          =   config_experiment['nstack']
    #index_start_pos =   18*(config_experiment['nstack']-1) + 9# Index where position starts
    for i, i_path  in enumerate(list_paths):
        #plt.subplot(nfigures, 3, i + 1)
        #set_trace()
        data    =   paths[i_path]['observation']
        index_start_pos =   (data.shape[1]//nstack)*(nstack-1) + 9
        targets =   paths[i_path]['target']
        x_target    =   targets[:, 0]
        y_target    =   targets[:, 1]
        z_target    =   targets[:, 2]
        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        x_data   =   data[:, index_start_pos]       +   x_target
        y_data   =   data[:, index_start_pos + 1]   +   y_target
        z_data   =   data[:, index_start_pos + 2]   +   z_target

        """ Initialize subplot 1 x 3 """
        #plt.figure(figsize=(12, 4))
        

        """ Distribution X-Y"""
        plt.subplot(1, 3, 1)
        plt.scatter(x_data, y_data, alpha=0.6, marker='o', s=5)
        plt.plot(x_target, y_target, 'black')
        #plt.plot(x_data, y_data)
        #circ1 = plt.Circle((x_data[0], y_data[0]), radius=20, color='red')
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        #plt.plot(circ1)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

        """ Distribution of Position in X-Z"""
        plt.subplot(1, 3, 2)
        plt.scatter(x_data, z_data, alpha=0.6, marker='o', s=5)
        plt.plot(x_target, z_target, 'black')
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

        """ Distribution of Position in Y-Z"""
        plt.subplot(1, 3, 3)
        plt.scatter(y_data, z_data, alpha=0.6, marker='o', s=5)
        plt.plot(y_target, z_target, 'black')
        #plt.scatter(x_data[0], y_data[0], marker='o', s=40, color='red')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

    plt.legend(['Path '+ str(i_path) for i_path in list_paths])

    plt.show()

# Example:
#plot_trajectory('./data/sample16/','11', [19])

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
    #index_start_pos =   27
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    nstack          =   config_experiment['nstack']
    dt              =   config_experiment['dt']
    #index_start_pos =   21*(nstack-1) + 9
    max_path_length =   config_experiment['max_path_length']
    plt.figure(figsize=(12, 4))
    for i_path in list_paths:
        data    =   paths[i_path]['observation']
        index_start_pos =   (data.shape[1]//nstack)*(nstack-1) + 9
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
        plt.plot(np.arange(len(x_data))*dt, x_data, color='red')
        plt.plot(np.arange(len(x_target))*dt, x_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Y POS
        """
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(len(y_data)) * dt, y_data, color='blue')
        plt.plot(np.arange(len(y_target)) * dt, y_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Z POS
        """
        plt.subplot(1, 3, 3)
        plt.plot(np.arange(len(z_data)) * dt, z_data, color='gray')
        plt.plot(np.arange(len(z_target)) * dt, z_target, '-', color='olive',linestyle='dashed')

    axis    =   ['X','Y','Z']    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        #plt.plot(np.arange(max_path_length), max_path_length * [0.0], '-', color='olive', linestyle='dashed')
        plt.xlabel('time (s)')
        plt.ylabel('position (m)')

        plt.title('Position in '+axis[i]+' axis')
        plt.xlim(0,max_path_length * dt)
        plt.ylim(-2,2)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #total_rollouts  =   len(data_pos)
        plt.grid(True)
        plt.legend(['roll '+ str(idx) for idx in list_paths])
        plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()

def get_positions_otime(fold, id_ex, max_path_length=250, list_paths=None):
    """
        Shows the behavior of position in x-y-z over time
        This trajectory must converge to the target point
    """
    path_name   =   compute_restore_file(fold, id_ex)


    assert path_name is not None, 'Not file of paths founded'

    paths           =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))

    #index_start_pos =   63
    #index_start_pos =   27
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    nstack          =   config_experiment['nstack']
    dt              =   config_experiment['dt']
    max_path_length =   config_experiment['max_path_length']
    positions       =   []
    for i_path in list_paths:
        data    =   paths[i_path]['observation']
        index_start_pos =   (data.shape[1]//nstack)*(nstack-1) + 9
        targets =   paths[i_path]['target']
        #set_trace()
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

        data_pos    =   {}
        data_pos['t']   =   np.arange(len(x_data))*dt
        data_pos['x']   =   x_data
        data_pos['y']   =   y_data
        data_pos['z']   =   z_data
        data_pos['x_target']   =   x_target
        data_pos['y_target']   =   y_target
        data_pos['z_target']   =   z_target

        positions.append(data_pos)

    return positions

def plot_comparison_trajectories(input_data:list, colors:list, legend:list):
    x_max           =   0.0
    plt.figure(figsize=(12, 4))
    for _data, col in zip(input_data, colors[:-1]):
        set_trace()
        positions   =   get_positions_otime(_data['fold'], _data['id_ex'], _data['max_path_length'],  _data['list_paths'])
        drawStdPlot(positions, 't','x','y')
        plt.show()
        _target     = None
        for _pos in positions:
            t           =   _pos['t']
            x_data      =   _pos['x']
            y_data      =   _pos['y']
            z_data      =   _pos['z']
            x_target    =   _pos['x_target']
            y_target    =   _pos['y_target']
            z_target    =   _pos['z_target']

            if t[-1] > x_max: 
                _t        = t
                _target_x = x_target
                _target_y = y_target
                _target_z = z_target

            plt.subplot(1, 3, 1)
            #plt.plot(t, x_data, color='red')
            #plt.plot(t, x_target, '-', color='olive',linestyle='dashed')
            plt.plot(t, x_data, color=col)
            #plt.plot(t, x_target, '-', color=colors[-1],linestyle='dashed')
            """
                PLOT Y POS
            """
            plt.subplot(1, 3, 2)
            #plt.plot(t, y_data, color='blue')
            #plt.plot(t, y_target, '-', color='olive',linestyle='dashed')
            plt.plot(t, y_data, color=col)
            #plt.plot(t, y_target, '-', color=colors[-1],linestyle='dashed')
            """
                PLOT Z POS
            """
            plt.subplot(1, 3, 3)
            #plt.plot(t, z_data, color='gray')
            #plt.plot(t, z_target, '-', color='olive',linestyle='dashed')
            plt.plot(t, z_data, color=col)
            #plt.plot(t, z_target, '-', color=colors[-1],linestyle='dashed')

    """" plot targets """
    plt.subplot(1, 3, 1)
    plt.plot(_t, _target_x, '-', color=colors[-1],linestyle='dashed')
    plt.subplot(1, 3, 2)
    plt.plot(_t, _target_y, '-', color=colors[-1],linestyle='dashed')
    plt.subplot(1, 3, 3)
    plt.plot(_t, _target_z, '-', color=colors[-1],linestyle='dashed')
    """"""
    axis    =   ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        #plt.plot(np.arange(max_path_length), max_path_length * [0.0], '-', color='olive', linestyle='dashed')
        plt.xlabel('timestep')
        plt.ylabel('position (m)')

        plt.title('Position in '+axis[i]+' axis')
        plt.xlim(0,_t[-1])
        plt.ylim(-2,2)
        #plt.xticks(np.arange(0, 12.5, 2))
        #plt.yticks(np.arange(-2, 2, 0.64))
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #total_rollouts  =   len(data_pos)
        
        #plt.axis([0,_t[-1], -2, 2], 'equal')
        plt.grid(True)
        
        
        #plt.legend(['roll '+ str(idx) for idx in list_paths])
        #plt.subplots_adjust(wspace=0.01)
        #plt.tight_layout()
    plt.legend(legend)
    plt.tight_layout()
    plt.show()

dict2 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '13',
            list_paths  =   [0,1, 2, 3, 4, 5]
        )

dict1 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '9',
            list_paths  =   [0, 1, 2]
        )

lll = [dict1, dict2]
#plot_comparison_trajectories(lll, ['blue', 'red', 'olive'],['fault-free', 'fault-rotor', 'ground-truth'])
# Example of its usage
#plot_pos_over_time('./data/sample16/', '11', list_paths=[19])

def plot_ang_velocity(fold, id_ex, list_paths=None):
    path_name   =   compute_restore_file(fold, id_ex)
    assert path_name is not None, 'Not file of paths founded'
    
    paths       =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))

    nfigures    =   len(list_paths)
    #set_trace()
    plt.figure(figsize=(12,4))
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)

    nstack          =   config_experiment['nstack']-1
    
    #index_start_pos =   18*(config_experiment['nstack']-1) + 15

    for i_path in list_paths:
        data    =   paths[i_path]['observation']
        index_start_pos =   (data.shape[1]//nstack)*(nstack-1) + 15
        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        vx_data   =   data[:, index_start_pos]
        vy_data   =   data[:, index_start_pos + 1]
        vz_data   =   data[:, index_start_pos + 2]

        """ Distribution X-Y"""
        plt.subplot(1, 3, 1)
        plt.scatter(vx_data, vy_data, alpha=0.6, marker='o', s=5)
        
        #plt.xlim(-3.2, 3.2)
        #plt.ylim(-3.2, 3.2)

        """ Distribution X-Z"""
        plt.subplot(1, 3, 2)
        plt.scatter(vx_data, vz_data, alpha=0.6, marker='o', s=5)

        """ Distribution Y-Z"""
        plt.subplot(1, 3, 3)
        plt.scatter(vy_data, vz_data, alpha=0.6, marker='o', s=5)
    
    plt.show()

#plot_ang_velocity('./data/sample34/', '5', [0, 1, 2])
def plot_roll_pitch_angle_otime(fold, id_ex, list_paths=None):
    path_name   =   compute_restore_file(fold, id_ex)
    assert path_name is not None, 'Not file of paths founded'
    
    paths       =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))
    
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    nstack              =   config_experiment['nstack']
    
    #index_SyawCroll     =   18*(config_experiment['nstack']-1) + 3
    #index_Sroll         =   18*(config_experiment['nstack']-1) + 6
    #index_SpitchCroll   =   18*(config_experiment['nstack']-1) + 7

    for i_path in list_paths:
        data                =   paths[i_path]['observation']
        index_orientation   =   (data.shape[1]//nstack) * (nstack - 1) + 18
        roll_rad            =   data[:, index_orientation]
        pitch_rad           =   data[:, index_orientation + 1]
        yaw_rad             =   data[:, index_orientation + 2]
        #roll_rad            =   np.arcsin(-data[:, index_Sroll])
        #cosroll             =   np.cos(roll_rad)
        #roll_rad            =   np.arctan2(-data[:, index_Sroll], np.sqrt(data[:, index_Sroll+1]*data[:, index_Sroll+1]+ data[:, index_Sroll+2]*data[:, index_Sroll+2]))
        #pitch_rad           =   np.arctan2(data[:, index_Sroll + 1], data[:, index_Sroll + 2])
        #yaw_rad             =   np.arctan2(data[:, index_SyawCroll], data[:, index_SyawCroll-3])
        #cosroll             =   np.cos(roll_rad)
        #pitch_rad           =   data[:, index_SpitchCroll]/cosroll
        #yaw_rad             =   data[:, index_SyawCroll]/cosroll
        #set_trace()
        roll_rad            =   roll_rad * 180.0/np.pi
        pitch_rad           =   pitch_rad * 180.0/np.pi
        yaw_rad             =   yaw_rad * 180.0/np.pi
        """
            PLOT pitch POS
        """
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(pitch_rad)), pitch_rad, color='red')
        plt.ylim(-90.0, 90.0)
        #plt.plot(pitch_rad, np.arange(len(pitch_rad))/1250, color='blue')
        #ax.set_thetamin(-45.0)
        #ax.set_thetamax(45.0)
        plt.xlim(0.0)
        #plt.grid(True)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #plt.plot(np.arange(len(x_target)), x_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Y POS
        """
        #plt.subplot(3, 1, 2, projection='polar')
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(len(roll_rad)), roll_rad, color='blue')
        #plt.plot( roll_rad, np.arange(len(roll_rad))/1250, color='blue')
        plt.ylim(-90.0, 90.0)
        plt.xlim(0.0)
        #plt.grid(True)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #plt.plot(np.arange(len(y_target)), y_target, '-', color='olive',linestyle='dashed')
        """
            PLOT Z POS
        """
        plt.subplot(3, 1, 3)
        #plt.plot(yaw_rad, np.arange(len(yaw_rad))/1250, color='gray')
        plt.plot(np.arange(len(yaw_rad)), yaw_rad, color='gray')
        plt.ylim(-200.0, 200.0)
        plt.xlim(0.0)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #plt.grid(True)
        #plt.plot(np.arange(len(z_target)), z_target, '-', color='olive',linestyle='dashed')
    plt.show()

def plot_ang_velocity_otime(fold, id_ex, list_paths=None):
    path_name   =   compute_restore_file(fold, id_ex)
    assert path_name is not None, 'Not file of paths founded'
    
    paths       =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))
    
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    nstack              =   config_experiment['nstack']
    dt              =   config_experiment['dt']
    #index_start_pos =   21*(nstack-1) + 9
    max_path_length =   config_experiment['max_path_length']
    plt.figure(figsize=(12, 4))
    for i_path in list_paths:
        data    =   paths[i_path]['observation']
        index_start_angvel =   (data.shape[1]//nstack)*(nstack-1) + 15
        
        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        x_data   =   data[:max_path_length, index_start_angvel]
        y_data   =   data[:max_path_length, index_start_angvel + 1]
        z_data   =   data[:max_path_length, index_start_angvel + 2]
        #set_trace()
        """
            PLOT X POS
        """
        plt.subplot(1, 3, 1)
        plt.plot(np.arange(len(x_data))*dt, x_data, color='red')
        """
            PLOT Y POS
        """
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(len(y_data)) * dt, y_data, color='blue')
        """
            PLOT Z POS
        """
        plt.subplot(1, 3, 3)
        plt.plot(np.arange(len(z_data)) * dt, z_data, color='gray')

    axis    =   ['Pitch','Roll','Yaw']    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        #plt.plot(np.arange(max_path_length), max_path_length * [0.0], '-', color='olive', linestyle='dashed')
        plt.xlabel('timestep')
        plt.ylabel('speed (rad/s)')

        plt.title('Angular speed in '+axis[i])
        plt.xlim(0,max_path_length * dt)
        plt.ylim(-100,100)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')
        #total_rollouts  =   len(data_pos)
        plt.grid(True)
        plt.legend(['roll '+ str(idx) for idx in list_paths])
        plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()
#plot_ang_velocity_otime('./data/sample15/', '2', [3, 11, 19])
#plot_ang_velocity_otime('./data/sample15/', '2', [3])
#plot_roll_pitch_angle_otime('./data/sample6/','1', list_paths=[0])

def plot_forces(fold, id_ex, list_paths=None):
    path_name   =   compute_restore_file(fold, id_ex)
    assert path_name is not None, 'Not file of paths founded'
    
    paths       =   joblib.load(path_name)
    list_paths  =   list_paths if list_paths is not None else list(np.arange(len(paths)))

    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    mask        =   np.ones((4,), dtype=np.float32)
    fault_rotor =   config_experiment['crippled_rotor']
    if fault_rotor is not None:
        mask[fault_rotor]   =   0.0
    #set_trace()
    start_action    =   4 * (config_experiment['nstack']-1)
    for id_path in list_paths:
        actions     =   paths[id_path]['actions'][:, start_action:]
        actions     =   actions * mask
        forces  =   1.5618e-4*actions*actions + 1.0395e-2*actions + 0.13894
        len_path    =   len(actions)
        plt.subplot(2, 2, 1)
        len_path = min(250,len_path)
        plt.plot(np.arange(len_path), forces[:len_path,0])
        plt.xticks(np.arange(0, len_path, 10))
        plt.grid()
        plt.ylim(0.0, 3.0)
        plt.title('Motor 1')

        plt.subplot(2, 2, 2)
        plt.plot(np.arange(len_path), forces[:len_path,1])
        plt.xticks(np.arange(0, len_path, 10))
        plt.grid()
        plt.ylim(0.0, 3.0)
        plt.title('Motor 2')
        
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(len_path), forces[:len_path,2])
        plt.xticks(np.arange(0, len_path, 10))
        plt.grid()
        plt.ylim(0.0, 3.0)
        plt.title('Motor 3')

        plt.subplot(2, 2, 4)
        plt.plot(np.arange(len_path), forces[:len_path,3])
        plt.xticks(np.arange(0, len_path, 10))
        plt.grid()
        plt.ylim(0.0, 3.0)
        plt.title('Motor 4')

    plt.tight_layout()
    plt.show()

#plot_forces('./data/sample15/', '2', [3])


def plot_3Dtrajectory(fold, id_ex, cols, figure=None, axes=None, list_paths=None):
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
    #plt.figure(figsize=(12,4))

    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)

    #index_start_pos =   18*(config_experiment['nstack']-1) + 9# Index where position starts
    nstack          =   config_experiment['nstack']
    fig =   plt.figure() if figure is None else figure
    ax  =   fig.gca(projection='3d') if axes is None else axes
    #set_trace()
    for i, i_path  in enumerate(list_paths):
        #plt.subplot(nfigures, 3, i + 1)
        
        data    =   paths[i_path]['observation']
        index_start_pos =   (data.shape[1]//nstack)*(nstack-1) + 9
        target  =   paths[i_path]['target']
        """ Plot distributions of positions in X-Y, X-Z, Y-Z"""
        x_data   =   data[:, index_start_pos]     + target[:, 0]
        y_data   =   data[:, index_start_pos + 1] + target[:, 1]
        z_data   =   data[:, index_start_pos + 2] + target[:, 2]
        startpos    =   np.array([x_data[0], y_data[0], z_data[0]], dtype=np.float32)
        """ Initialize subplot 1 x 3 """
        #plt.figure(figsize=(12, 4))
        #set_trace()
        ax.plot([startpos[0]], [startpos[1]], [startpos[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
        ax.text(startpos[0], startpos[1], startpos[2], 'Start-Point', color='k')
        
        ax.plot(x_data, y_data, z_data, color=cols[0], linewidth=3)
        ax.plot(target[:,0],target[:,1],target[:,2], cols[1], linestyle='dashed')
    
    return fig, ax

def plot_comparison_3Dtrajectory(data_config, colors, legend):
    fig, ax =   None, None
    set_trace()
    for _data, col in zip(data_config, colors[:-1]):
        fig, ax =   plot_3Dtrajectory(_data['fold'], _data['id_ex'], [col, colors[-1]],fig, ax, _data['list_paths'])
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-1.5, 1.5)
    fig.show()

def make3danimation(input_data, colors, skipsteps=1):
    positions   =   []
    for _data in input_data:
        pos   =   get_positions_otime(_data['fold'], _data['id_ex'], None, _data['list_paths'])
        positions.append(pos)
    fig =   plt.figure()
    ax  =   fig.gca(projection='3d')
    ax.set_xlim(-1.0, 1.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=10., azim=40)
    #if legend is not None: ax.legend(legend)
    #set_trace()
    x_target    =   positions[0][0]['x_target']
    y_target    =   positions[0][0]['y_target']
    z_target    =   positions[0][0]['z_target']
    ax.plot(x_target,y_target,z_target, colors[-1], linestyle='dashed')
    startpos  =   np.array([positions[0][0]['x'][0], positions[0][0]['y'][0], positions[0][0]['z'][0]], dtype=np.float32)
    ax.plot([startpos[0]], [startpos[1]], [startpos[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
    ax.text(startpos[0], startpos[1], startpos[2], 'Start-Point', color='k')    
    #data_pos['t']   =   np.arange(len(x_data))*dt
    #data_pos['x']   =   x_data
    #data_pos['y']   =   y_data
    #data_pos['z']   =   z_data
    #data_pos['x_target']   =   x_target
    #data_pos['y_target']   =   y_target
    #data_pos['z_target']   =   z_target
    length  =   len(positions[0][0]['t'])
    all_data    =   [[[], [], []] for _ in range(len(positions))]
    for ts in range(0, length, skipsteps):
        poss    =   [[_p[0]['x'][ts], _p[0]['y'][ts],_p[0]['z'][ts]] for _p in positions]
        ax.view_init(elev=30.+ts*0.01, azim=65.0+ts*0.01)
        for _data3d, colp, cumm_data in zip(poss, colors[:-1], all_data):
            cumm_data[0].append(_data3d[0])
            cumm_data[1].append(_data3d[1])
            cumm_data[2].append(_data3d[2])
            ax.plot(cumm_data[0],cumm_data[1],cumm_data[2], color=colp, linewidth=2.5)
        fig.savefig('./animations/100_'+str(1000+ts//10))
        
        
#plot_3Dtrajectory('./data/sample16/','11', list_paths=[19])

def plot_trajectory_comparison(input_data, colors, title=None, legend=['fault-free', 'fault-rotor', 'Ground-Truth']):
    def callback(_pos):
        return _pos['t'].shape[0]
    fig, axs = plt.subplots(2,2, figsize=(12, 8), sharey=True)
    axs     =   axs.flatten()
    #set_trace()
    xlim_max    =   0.0
    xlim_min    =   np.inf
    #_target     =   []
    for _data, col in zip(input_data, colors[:-1]):
        #set_trace()
        positions   =   get_positions_otime(_data['fold'], _data['id_ex'], _data['max_path_length'],  _data['list_paths'])
        dt = positions[0]['t'][1] - positions[0]['t'][0]
        _target=[positions[0]['x_target'], positions[0]['y_target'], positions[0]['z_target']]
        
        xlim_min    =   positions[0]['t'][0]
        xlim_max    =   positions[0]['t'][-1]
        pos_x = []
        pos_y = []
        pos_z = []
        max_path_length =   min([_pos['t'].shape[0] for _pos in positions])
        for _pos in positions:
            pos_x.append(_pos['x'][:max_path_length])
            pos_y.append(_pos['y'][:max_path_length])
            pos_z.append(_pos['z'][:max_path_length])
        axs[0] = drawStdPlot(pos_x, None,None,'X-axis', color=col, axes=axs[0],  scalefactor=dt)
        axs[1] = drawStdPlot(pos_y, title,'time (s)','Y-axis', color=col, axes=axs[1],  scalefactor=dt)
        axs[2] = drawStdPlot(pos_z, None,None,'Z-axis', color=col, axes=axs[2],  scalefactor=dt)
        #axs[1,1] = drawStdPlot(pos_z, None,None,'Z-axis', color=col, axes=axs[1,1],  scalefactor=dt)
    plt.ylim(-2.0, 2.0)
    for idx, _targ in zip(count(), _target):
        axs[idx].plot(np.arange(len(_targ))*dt, _targ, color=colors[-1], linestyle='dashed')
        axs[idx].set_xlim(xlim_min, xlim_max)
        axs[idx].grid(which='major', color='#CCCCCC', linestyle='--')
        axs[idx].grid(which='minor', color='#CCCCCC', linestyle=':')
    axs[-1].axis('off')
    #axs[-1].legend(legend)  
    plt.tight_layout()
    fig.legend(legend, loc='center', bbox_to_anchor=(0.75, 0.25), fontsize=25)
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    
    
    

        
dict1 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '11',
            list_paths  =   [0]
        )
dict2 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '15',
            list_paths  =   [0]
        )
#plot_comparison_3Dtrajectory([dict1,dict2], ['b','y','orange'],None)
#make3danimation([dict1, dict2], ['b','y','orange'], 10)
legend=['fault-free', 'fault-rotor', 'Ground-Truth']
#title='Position (m) over time in a Circular path. Fault-free vs fault-case'
title=None
#plot_trajectory_comparison([dict1, dict2], ['b','r','g'], title,legend)

dict3 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '10',
            list_paths  =   None
        )

dict4 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '11',
            list_paths  =   None
        )
legend  =   ['10ms', '50ms','Ground-Truth']
title=None
#plot_trajectory_comparison([dict4, dict3], ['b','r','g'], title, legend)

dict5 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            id_ex       =   '17',
            list_paths  =   None
        )
dict6 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '16',
            list_paths  =   None
        )
legend=['fault-free', 'fault-rotor', 'Ground-Truth']
#title='Position (m) over time in a Fixed Point path. Fault-free vs fault-case'
title=None
#plot_trajectory_comparison([dict5, dict6], ['b','r','g'], title,legend)

""" Hover at fixed point"""
dict7 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '12',
            list_paths  =   None
        )
dict8 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '16',
            list_paths  =   None
        )
legend=['fault-free', 'fault-rotor', 'Ground-Truth']
#make3danimation([dict7, dict8], ['b','y','orange'], 10)
#title='Position (m) over time in a Circular path. Fault-free vs fault-case'
title=None
#plot_trajectory_comparison([dict7, dict8], ['b','r','g'], title,legend)

""" Helicoid Trjaectory"""
dict9 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '13',
            list_paths  =   [0]
        )
dict10 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '18',
            list_paths  =   [2]
        )
    
#make3danimation([dict9, dict10], ['b','y','orange'], 10)

""" Sin-Vertical Trjaectory"""
dict9 = dict(
            fold  =   './data/sample42',
            max_path_length = 20,
            id_ex       =   '14',
            list_paths  =   [0]
        )
dict10 = dict(
            fold  =   './data/sample40',
            max_path_length = 20,
            #id_ex       =   '10',
            #list_paths  =   [12]
            id_ex       =   '19',
            list_paths  =   [8]
        )
    
#make3danimation([dict9, dict10], ['b','y','orange'], 10)

def sanity_check_path(fold, id_ex, ipath):
    set_trace()
    from mbrl.network import Dynamics
    from utils.sanity_check import SanityCheck
    import torch
    from utils.analize_dynamics import plot_error_map
    device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_name   =   compute_restore_file(fold, id_ex)
    
    assert path_name is not None, 'Not file of paths founded'

    paths           =   joblib.load(path_name)

    #index_start_pos =   63
    #index_start_pos =   27
    with open(os.path.join(fold, 'rolls'+id_ex+'/experiment_config.json'), 'r') as fp:
        config_experiment   =   json.load(fp)
    
    nstack          =   config_experiment['nstack']
    dt              =   config_experiment['dt']
    horizon         =   config_experiment['horizon']
    #index_start_pos =   21*(nstack-1) + 9
    max_path_length =   config_experiment['max_path_length']

    env_class       =   config_experiment['env_name']
    
    path    =   paths[ipath]

    state_sz     =   path['observation'].shape[1]//nstack
    action_sz    =   path['actions'].shape[1]//nstack
    
    dynamics            =   Dynamics((state_sz, ), (action_sz,), nstack, False)
    checkpoint  =   torch.load(fold +'/params_high.pkl')
    dynamics.load_state_dict(checkpoint['model_state_dict'])
    dynamics.mean_input =   checkpoint['mean_input']
    dynamics.std_input  =   checkpoint['std_input']
    dynamics.epsilon    =   checkpoint['epsilon']
    dynamics.to(device)
    
    set_trace()
    matrix  =   SanityCheck.get_errors_matrixes_from_path(path, action_sz, state_sz, horizon, nstack, dynamics, device)
    
    fig, axs = plt.subplots(2,1, figsize=(12, 8))

    fig, axs[1]    =   plot_error_map(matrix, 1250, _vmax=20, fig=fig, ax=axs[1])
    
    pos_otime       =    get_positions_otime(fold, id_ex, list_paths=[ipath])
    x_time          =   pos_otime[0]['x']
    t_              =   pos_otime[0]['t']
    axs[0].plot(t_, x_time)

    plt.show()

#sanity_check_path('./data/sample42', '1', 12)

plot_pos_over_time('./data/sample40', '13', list_paths=[0,1,2,3,4,5,6])