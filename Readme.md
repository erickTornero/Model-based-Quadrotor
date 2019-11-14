# Model-Based Reinforcement Learning for Quadrotor

Implementation of Quadrotor Model-based Learning in pytorch and VREP Simulator based of the following papers: 

[**Low-Level control of a quadrotor with deep model-based reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/8769882)

[**Learning to adapt in dynamic, real-world environments through meta-reinforcement learning**](https://arxiv.org/abs/1803.11347)

# Current Progress
We are testing separately Fault-Free Case, and Fault-Motor 1 case for same trajectory.

#### Circle Trajectory
Here, we show trajectory followed by quadrotor in a *Circle* trajectory

**Fault-Free Case, trajectory over time**
![Fault-Free Case](./showimages/_circle_traj_faultfree_s36_r6_p11_otime.png)

**Fault-M1 Case, trajectory over time** 

![Fault-M1 Case](./showimages/_circle_traj_fault_s15r13p3_otime.png)

**Same Comparison in 3D dim, Left Fault free, Right Fault Motor 1**

![3D Trajectories](./showimages/_circle_ffree_vs_f.jpg)

#### Point Trajectory

**Fault-Free Case, trajectory over time**
![Fault-Free Case](./showimages/_point_traj_ffree_s36r5p8_otime.png)

**Fault-M1 Case, trajectory over time** 

![Fault-M1 Case](./showimages/_point_traj_fault_s15r14p1_otime.png)

**Same Comparison in 3D dim, Left Fault free, Right Fault Motor 1**

![3D Trajectories](./showimages/_point_ffree_vs_f.png)