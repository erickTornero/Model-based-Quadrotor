from mbrl.mpc import RandomShooter
from mbrl.network import Dynamics
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.runner import StackStAct

#import ie
def rollouts(dynamics:Dynamics, env:QuadrotorEnv, mpc:RandomShooter, n_rolls=20, max_path_length=250):
    nstack = dynamics.stack_n

    for i_roll in range(1, n_rolls+1):
        obs = env.reset()
        stack_as = StackStAct(env.action_space.shape, env.observation_space.shape, n=nstack, init_st=obs)
        done = False
        timestep    =   0
        cum_reward  =   0.0
        while not done and timestep < max_path_length:
            action = mpc.get_action(stack_as)

            next_obs, reward, done, env_info =   env.step(action)

            stack_as.append(obs=next_obs, acts=action)
            cum_reward  +=  reward
            timestep += 1

        print('{} rollout, reward-> {} in {} timesteps'.format(i_roll, cum_reward, timestep))

