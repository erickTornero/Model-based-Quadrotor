from wrapper_vrep import *



env = VREPQuad(ip='192.168.0.36',port=19999)


for i in range(10):

    ob      = env.reset()
    done    = False
    cum_rw  = 0.0
    while not done:
        act = np.random.uniform(0,1, 4) + 3.5
        ob, rw, done, _ = env.step(act)
        cum_rw += rw
    
    print(i+1, cum_rw)