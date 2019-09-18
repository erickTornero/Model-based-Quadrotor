import vrep
from utility import *
import sys

vrep.simxFinish(-1)
clientID    =   vrep.simxStart('192.168.0.19',19999, True, True, 5000, 5)
if clientID != -1:
    print('Conexion Establecida!')
else:
    sys.exit('Error: No se puede conectar!')

_, left_motor       =   vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
_, right_motor      =   vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)

_, pioneer_handler  =   vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)

while True:
    position        =   vrep.simxGetObjectPosition(clientID, pioneer_handler, -1, vrep.simx_opmode_streaming)
    orientation     =   vrep.simxGetObjectOrientation(clientID, pioneer_handler, -1, vrep.simx_opmode_streaming)
    velocity        =   vrep.simxGetObjectVelocity(clientID, pioneer_handler, vrep.simx_opmode_streaming)
    quaternion      =   vrep.simxGetObjectQuaternion(clientID, pioneer_handler, -1, vrep.simx_opmode_streaming)
    
    #print('p>:\t', position)
    #print('o>:\t', orientation)
    #print('q>:\t', quaternion)
    #print(velocity[1])
    RotMat          =   GetFlatRotationMatrix(orientation[1])
    rowdata         =   np.append(RotMat, position[1])
    rowdata         =   np.append(rowdata, velocity[1])
    rowdata         =   np.append(rowdata, velocity[2])
    #print(np.shape(rowdata))
    print(rowdata)
vrep.simxFinish(-1)