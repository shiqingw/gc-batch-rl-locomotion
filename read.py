import pickle
import os
from datetime import datetime

# input_file = "/home/shiqing/gc-batch-rl-locomotion/samples/ant-augmented-samples.pkl"
# training_file = open(input_file, 'rb')
# for i in range(0, 10):
#     old_s, old_g, new_s, new_g, action = pickle.load(training_file)
#     print(old_s.shape)
#     print(old_g.shape)
#     print(new_s.shape)
#     print(new_g.shape)
#     print(action.shape)
#     # (111,)
#     # (3,)
#     # (111,)
#     # (3,)
#     # (8,)
#     print(old_s)
#     print(old_g)
#     print(new_s)
#     print(new_g)
#     print(action)
#     print()

input_file = "/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-on-policy-samples-1e7.pkl"
training_file = open(input_file, 'rb')
count = 0
while True:
    old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
    count += 1
    if count % int(5e5) == 0:
        print(count)

# for i in range(0, 10):
#     old_s, old_xy, new_s, new_xy, action = pickle.load(training_file)
#     # print(target.shape)
#     print(old_s.shape)
#     print(old_xy.shape)
#     print(new_s.shape)
#     print(new_xy.shape)
#     print(action.shape)
#     # # (111,)
#     # # (3,)
#     # # (111,)
#     # # (3,)
#     # # (8,)
#     # print(target)
#     print(old_s)
#     print(old_xy)
#     print(new_s)
#     print(new_xy)
#     print(action)
#     print()
