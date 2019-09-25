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
        sample_data['delta_obs']      =   np.vstack([path['delta_obs'] for path in paths])

        sample_data['rewards']      =   self.reward_process([path['rewards'] for path in paths])

        return sample_data
    
    def reward_process(self, rewards):
        """
            Return the total rewards per Rollout
        """
        total_rewards   =   np.array([np.sum(r, axis=0) for r in rewards])
        #mean_rewards    =   np.mean(total_rewards)
        #std_rewards     =   np.std(total_rewards)
        #min_reward      =   np.min(total_rewards)
        #max_reward      =   np.max(total_rewards)

        return total_rewards


# TODO: Hacer una prueba de ablacion para ver si mejora el resultado next_obcuando no se toma
#       En cuenta el estado inicial en el dataset de entrenamiento (cuando el stack no
#       esa lleno) Si se considera este TODO debe realizarse en la función process

# TODO: Falta devolver el reward, sin embargo no es necesario para el MBRL
