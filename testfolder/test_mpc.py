
from mbrl.network import Dynamics
from mbrl.parallel_env import ParallelVrepEnv
from mbrl.runner import Runner
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter

from IPython.core.debugger import set_trace

env_ = QuadrotorEnv(port=27001)

vecenv=ParallelVrepEnv(ports=[25001,26001],num_rollouts=10, max_path_length=250, envClass=QuadrotorEnv)

horizon = 15
candidates = 1000



state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=True)

rs = RandomShooter(horizon, candidates, env_=env_, dynamics=dyn, discount=0.99)
print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, rs, 250, 100)

print('running with policy...')

paths = runner.run(random=False)

set_trace()
