import numpy as np

class DataProcessor:
    def __init__(self, discount):
        self.discount   =   discount


    def process(self, paths):
        """
            Process Data in a list of paths
            Each path contains a dict of: Observations, actions, rewards, dones, nex_obs
        """
        sample_data = dict()

        sample_data['observations'] =   np.vstack([path['observations'] for path in paths])
        sample_data['next_obs']     =   np.vstack([path['next_obs'] for path in paths])
        sample_data['actions']      =   np.vstack([path['actions'] for path in paths])

        return sample_data