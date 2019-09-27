from utils.plots import plot_reward_bar_distributions


import joblib

folder = './data/sample4/'

plot_reward_bar_distributions(folder, 100, 0.0, 1000.0, iterations=[1,20,40,55,60])
