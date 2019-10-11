import numpy as np

class Trajectory:
    def __init__(self, npoints, nrounds=2.0):
        self.dt         =   float(1.0/npoints)
        #self.wave       =   wave
        self.npoints    =   npoints
        self.nrounds    =   nrounds
        assert nrounds >= 0, 'nrounds must be a possitive value'

    def gen_points(self, wave):
        t   =   np.arange(self.npoints+1)
        if wave == 'sin-vertical':
            x   =   t * self.dt
            y   =   t * 0.0
            z   =   np.sin(2*np.pi*t*self.dt*self.nrounds)

        elif wave   ==  'circle':
            x   =   np.cos(2 * np.pi * t * self.dt)
            y   =   np.sin(2 * np.pi * t * self.dt)
            z   =   np.ones(np.shape(t))*0.0

        
        elif wave  ==  'helicoid':
            """ Elicoid trajectory, two rounds"""
            x   =   np.cos(2*np.pi*t*self.dt*self.nrounds)
            y   =   np.sin(2*np.pi*t*self.dt*self.nrounds)
            z   =   t/self.npoints
        elif wave  ==  'stepped':
            """ An stepped Trajectory will be generated"""
            x   =   np.ones_like(t, dtype=np.float32) * 0.8
            y   =   np.ones_like(t, dtype=np.float32) * 0.8
            z   =   np.ones_like(t, dtype=np.float32) * 0.8
            i_step  =   (self.npoints * 12)//25   
            x[i_step:]  =   0.0
            y[i_step:]  =   0.0
            z[i_step:]  =   0.0

        else:
            assert False, 'Trajectory not defined'
        
        x       =   x.reshape(-1,1)
        y       =   y.reshape(-1,1)
        z       =   z.reshape(-1,1)

        position    =   np.concatenate((x,y), axis=1)
        position    =   np.concatenate((position, z), axis=1)
        return position
        