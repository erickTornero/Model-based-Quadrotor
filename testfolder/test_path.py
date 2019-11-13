from utils.analize_paths import *
import json
sample_run  =   './data/sample36'
nroll       =   '4'
paths_list  =   [6]

path_name   =   compute_restore_file(sample_run, nroll)
with open(os.path.join(sample_run, 'rolls'+nroll+'/experiment_config.json'), 'r') as fp:
    config_experiment   =   json.load(fp)

max_path_length =   config_experiment['max_path_length']

plot_roll_pitch_angle_otime(sample_run, nroll, paths_list)

plot_ang_velocity(sample_run, nroll, paths_list)

plot_trajectory(sample_run, nroll, list_paths=paths_list)

plot_pos_over_time(sample_run, nroll, max_path_length, paths_list)

plot_3Dtrajectory(sample_run, nroll, list_paths=paths_list)


# Good result
#sample_run  =   './data/sample36'
#nroll       =   '4'
#paths_list  =   [6]