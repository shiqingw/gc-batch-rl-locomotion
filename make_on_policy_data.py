import gym
import time
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
from scipy.spatial.transform import rotation

# env = gym.make("HexapodFewerStatesDataCollection-v1")
# observation = env.reset()

input_file = "/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-on-policy-samples-1e7.pkl"
output_file = "/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-on-policy-samples-4e6.pkl"
training_file = open(input_file, 'rb')
aug_file = open(output_file, 'a+b')


for i in range(0, int(5.88e6)):
    old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
    if (i+1)%int(5.88e5) == 0:
        print('Number of samples passed out of 7.88e6: ', (i+1)/5.88e6)

for i in range(int(4e6)):
    
    old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
    pickle.dump([old_s, old_xy, new_s, new_xy, action], aug_file)

    if (i+1)%int(1e5) == 0:
        print('Number of saved samples out of 4e6: ', (i+1)/4e6)


    # print("dim(qpos)", env.model.nq) # 7 + 8 = 15 = number of generalized coordinates = dim(qpos)
    # print("dim(qvel)", env.model.nv) # 6 + 8 = 14 = number of degrees of freedom = dim(qvel)
    # print("max possible contacts",env.model.nconmax) # max possible contacts
    # print("dim(ctrl)", env.model.nu) # number of actuators/controls = dim(ctrl)
    # print("number of joints", env.model.njnt) # number of joints
    # print("number of bodies", env.model.nbody) # number of bodies
    # print(len(env.sim.data.qpos.flat))
    # print(len(env.sim.data.qvel.flat))
    # print(len(env.sim.data.cfrc_ext.flat))
    # print(env.sim.data.cfrc_ext)
    # print(env.sim.data.qpos.flat.copy()[7:25])
print("Done.")
# env.close()
training_file.close()
aug_file.close()