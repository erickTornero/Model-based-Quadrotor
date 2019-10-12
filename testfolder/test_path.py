from utils.analize_paths import *

sample_run  =   './data/sample16'
id_run      =   '11'
paths_list  =   [19]

plot_trajectory(sample_run, id_run, paths_list)

plot_pos_over_time(sample_run, id_run, paths_list)

plot_3Dtrajectory(sample_run, id_run, paths_list)
