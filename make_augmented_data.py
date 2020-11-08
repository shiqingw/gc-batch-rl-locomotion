import gym
import time
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
from scipy.spatial.transform import rotation

env = gym.make("HexapodFewerStatesDataCollection-v1")
observation = env.reset()

input_file = "/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-on-policy-samples-1e7.pkl"
output_file = "/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-augmented-samples-4e6-bound-rotation/raw.pkl"
training_file = open(input_file, 'rb')
aug_file = open(output_file, 'a+b')
# pickle.dump([np.array(self.last_state), np.array(self.last_pos_xy), np.array(observation), np.array(self.sim.data.qpos.flat.copy()[0:2]),
#             np.array(action)], fp)
# fp.close()

for i in range(0, int(7.88e6)):
    old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
    if (i+1)%int(7.88e5) == 0:
        print('Number of samples passed out of 7.88e6: ', (i+1)/7.88e6)

for i in range(int(2e6)):
    theta = np.random.uniform(-np.pi, np.pi)
    random_rot = Rotation.from_euler('z', theta, degrees=False).as_matrix()

    old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
    # save original data
    pickle.dump([old_s, old_xy, new_s, new_xy, action], aug_file)

    aug_old_joint_position = old_s[7:25]
    old_xyz = np.concatenate((old_xy, [old_s[0]]))
    aug_old_xyz = np.dot(random_rot, old_xyz)

    old_rot_pose = Rotation.from_euler('xyz', old_s[4:7], degrees=False)
    aug_old_rot_pose = Rotation.from_matrix(np.dot(random_rot, old_rot_pose.as_matrix()))
    [x, y, z, w] = aug_old_rot_pose.as_quat()
    aug_old_wxyz = [w, x, y, z]
    aug_old_qpos = np.concatenate((aug_old_xyz, aug_old_wxyz, aug_old_joint_position))

    old_translational_velocity = old_s[1:4]
    aug_old_translational_velocity = np.dot(random_rot, old_translational_velocity)
    aug_old_rotational_velocity = [0]*3
    aug_old_joint_speed = old_s[25:43]
    aug_old_qvel = np.concatenate((aug_old_translational_velocity, aug_old_rotational_velocity, aug_old_joint_speed))
    
    aug_old_z = aug_old_xyz[2]
    aug_old_rpy = aug_old_rot_pose.as_euler('xyz', degrees=False)
    aug_old_feet_contact = old_s[43:49]
    aug_old_s = np.concatenate(([aug_old_z], aug_old_translational_velocity, aug_old_rpy, aug_old_joint_position, aug_old_joint_speed, aug_old_feet_contact))
    aug_old_xy = aug_old_xyz[0:2]

    # specific rest and step the env
    env.set_state(aug_old_qpos, aug_old_qvel)
    aug_observation, reward, done, info = env.step(action)

    aug_new_s = aug_observation
    aug_new_xy = env.sim.data.qpos.flat.copy()[0:2]

    pickle.dump([aug_old_s, aug_old_xy, aug_new_s, aug_new_xy, action], aug_file)
    if (i+1)%int(1e5) == 0:
        print('Number of saved samples out of 2e6: ', (i+1)/2e6)


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
env.close()
training_file.close()
aug_file.close()