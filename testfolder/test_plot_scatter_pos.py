from utils.plots import plot_scatter_positions

import joblib

restore_file = './data/sample4/observations_it_55.pkl'

data = joblib.load(restore_file)


plot_scatter_positions(data, 63)