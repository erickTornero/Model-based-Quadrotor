import numpy as np
import math
# Return the rotation through Euler angles 
# R = Rx(alpha)*Ry(beta)*Rz(gamma)
def GetRotationMatrix(eulerAngles):
    alpha, beta, gamma = eulerAngles
    rx  =   np.array([[1,   0,  0],
                      [0, math.cos(alpha), -math.sin(alpha)],
                      [0, math.sin(alpha),  math.cos(alpha)]
                    ])
    ry  =   np.array([[math.cos(beta), 0, math.sin(beta)],
                      [0             , 1, 0             ],
                      [-math.sin(beta),0, math.cos(beta)]
                    ])
    rz  =   np.array([[math.cos(gamma), -math.sin(gamma), 0],
                      [math.sin(gamma),  math.cos(gamma), 0],
                      [0              , 0,                1]
                    ])
    R   =   np.dot(rz, np.dot(ry, rx))

    return R   

def GetFlatRotationMatrix(eulerAngles):
    rotmat  =   GetRotationMatrix(eulerAngles)
    
    return np.reshape(rotmat, (9, ))
    #return np.append(rotmat[0,:], rotmat[1:3,:])