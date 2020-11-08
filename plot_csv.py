import numpy as np
import matplotlib.pyplot as plt

def loadCSVfile2(csv_file):
    tmp = np.loadtxt(csv_file, dtype=np.str, delimiter=",")
    data = tmp[1:,0:].astype(np.float)#加载数据部分
    label = tmp[0,:]#加载类别标签部分
    return data, label #返回array类型的数据


csv_file = '/home/shiqing/gc-batch-rl-locomotion/logs/2020-10-27_00-32-27_HexapodFewerStatesDataCollection-v1_data-collection_ppo2_1.000000e+07/log/progress.csv'
data, label = loadCSVfile2(csv_file)
# ['eplenmean' 'eprewmean' 'fps' 'loss/approxkl' 'loss/clipfrac'
#  'loss/policy_entropy' 'loss/policy_loss' 'loss/value_loss'
#  'misc/explained_variance' 'misc/nupdates' 'misc/serial_timesteps'
#  'misc/time_elapsed' 'misc/total_timesteps']
plt.plot(data[:,-1], data[:,1])
plt.show()