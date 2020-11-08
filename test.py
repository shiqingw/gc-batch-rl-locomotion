import gym
import time
from scipy.spatial.transform import Rotation

env = gym.make("HexapodFewerStates-v1")  # 创建游戏环境
# env = gym.make("Ant-v4")

observation = env.reset()  # 游戏回到初始状态
for _ in range(1000):
    # time.sleep(2)
    env.render()  # 显示当前时间戳的游戏画面
    # action = env.action_space.sample()  # 随机生成一个动作
    # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
    # print(action)
    
    observation, reward, done, info = env.step(18*[0])
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
    print(env.sim.data.qpos.flat.copy()[7:25])
    
    
    
    if done:  # 游戏回合结束，复位状态
        observation = env.reset()
env.close()