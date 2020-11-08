import os
from datetime import datetime
import argparse


class Options():
    def __init__(self):
        """
        Create an argparse.ArgumentParser for run_mujoco.py.
        """
        # env = 'Hexapod-v1'
        env = 'HexapodFewerStatesDataCollection-v1'
        alg = 'ppo2'
        purpose = "data-collection"
        num_timesteps = 0
        if_load = True
        play = True

        base_path = "/home/shiqing/gc-batch-rl-locomotion/"
        date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        storage_path = os.path.join(base_path, "logs", date_str + "_" + env + "_" + purpose + "_" + alg + "_" + str("%e"%num_timesteps))
        if play: storage_path = storage_path + "_" + "play"
        save_path = os.path.join(storage_path, "save")
        log_path = os.path.join(storage_path, "log")
        os.mkdir(storage_path)
        os.mkdir(save_path)
        os.mkdir(log_path)
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--env', help='environment ID', type=str, default=env)
        parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
        parser.add_argument('--seed', help='RNG seed', type=int, default=None)
        parser.add_argument('--alg', help='Algorithm', type=str, default=alg)
        parser.add_argument('--num_timesteps', type=float, default=num_timesteps),
        parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
        parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
        parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
        parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
        parser.add_argument('--save_path', help='Path to save trained model to', default=save_path+'/'+ 'model_weights', type=str)
        parser.add_argument('--load_path', help='Path to load trained model from', default='/home/shiqing/gc-batch-rl-locomotion/logs/2020-10-27_00-32-27_HexapodFewerStatesDataCollection-v1_data-collection_ppo2_1.000000e+07/save/model_weights', type=str)
        parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
        parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
        parser.add_argument('--log_path', help='Directory to save learning curve data.', default=log_path, type=str)
        parser.add_argument('--play', default=play)
        parser.add_argument('--if_load', default=if_load)

        self.args = parser.parse_args()

        #########Extra args (defaults)########
        # extra_args= dict(
        # nsteps=2048,
        # nminibatches=32,
        # lam=0.95,
        # gamma=0.99,
        # noptepochs=10,
        # log_interval=1,
        # ent_coef=0.0,
        # lr=lambda f: 3e-4 * f,
        # cliprange=0.2,
        # value_network='copy')
    
  