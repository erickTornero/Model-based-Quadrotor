from multiprocessing import Process, Pipe
import numpy as np

class ParallelVrepEnv(object):
    """
    Wrap multiples instances of vrep without loss the id connection
    """
    
    def __init__(self, num_rollouts:int, max_path_length:int, ports:list, envClass):
        """
        Initialize Pipes and Process
        """
        self.n_parallel =   len(ports)
        
        self._num_envs  =   self.n_parallel
        self.ports      =   ports
        self.env_       =   None
        self.envClass   =   envClass

        #assert num_rollouts == self._num_envs
        assert num_rollouts % self.n_parallel == 0

        self.samples_per_proc   =  max_path_length * (num_rollouts/self.n_parallel)


        self.remotes, self.work_remotes =   zip(*[Pipe() for _ in range(self.n_parallel)])
        seeds = np.random.choice(range(10**6), size=self.n_parallel, replace=False)   
        self.ps = [
            Process(target=self.worker, args=(work_remote, remote, max_path_length, self.samples_per_proc, seed, port)) 
            for work_remote, remote, seed, port in zip(self.work_remotes, self.remotes, seeds, ports)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()
        
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_):
        
        """
        Step for each environment
        """

        for remote, action_list in zip(self.remotes, actions_):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]
        obs, rws, dones, env_infos = map(lambda x: x, zip(*results))
        
        return obs, rws, dones, env_infos
    
    def reset(self):
        """
        Reset all environments
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        observations = [np.asarray(remote.recv(), np.float32) for remote in self.remotes]
        
        return observations

    def worker(self, remote, parent_remote, max_path_length, samples_pp, seed, port_):
        
        env = self.envClass(port=port_)
        total_samples   =   samples_pp

        if port_ == self.ports[0]:
            self.env_ = env
        np.random.seed(seed)
        
        ts = 0
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                action  = data
                nextobs, rw, done, info = env.step(action)
                ts = ts + 1
                if done or ts >= max_path_length:
                    done = True
                    env.reset()
                    ts = 0
                """Send the next observation"""
                remote.send((nextobs, rw, done, info))
            elif cmd =='reset':
                """
                Reset the environment associated with the worker
                """
                obs = env.reset()
                remote.send(obs)
            else:
                print('Warning: Receiving unknown command!!')

    @property
    def getenv(self):
        return self.env_

    @property
    def num_envs(self):
        return self._num_envs