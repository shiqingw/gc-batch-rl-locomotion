import sys
import numpy as np
sys.path.append("/home/shiqing/gc-batch-rl-locomotion") 
import gym
import torch.nn as nn
from torch.optim import Adam
import argparse
import os
import errno
import pickle
import time
from random import shuffle
from datetime import datetime
sys.path.append("/home/shiqing/gc-batch-rl-locomotion/scripts") 
from utils import set_global_seed, save_checkpoint
from utils import convert_to_variable, distance, rotate_point
from utils import preprocess_goal, find_norm, prepare_dir, normalize
from dataset import Dataset
from augmented_dataset import AugDataset
from helper_functions import load_checkpoint
from network import Policy
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description='Naive goal-conditioned policy')
parser.add_argument('-e', '--env_name', default='HexapodFewerStatesDataCollection-v1',
                    help='Environment name')
parser.add_argument('-i', '--input_folder', default='/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-augmented-samples-4e6',
                    help='Path to file containing trajectories.')
parser.add_argument('--resume', default=False,
                    help='If the training is to be resumed from a checkpoint')
parser.add_argument('--checkpoint_folder', default='/home/shiqing/gc-batch-rl-locomotion/logs/2020-10-31_02-16-14_HexapodFewerStatesDataCollection-v1_naive-gcp_on-policy',
                    help='Previous checkpoint path, from which training is to '
                    'be resumed.')
parser.add_argument('--test_only', default=False)
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Seed for initializing the network.')
parser.add_argument('--no_gpu', default=False)
parser.add_argument('--dir_name', default='', help='')
parser.add_argument('--visualize', default=False)
parser.add_argument('--n_test_steps', type=int, default=10,
                    help='Number of test trajectories')
parser.add_argument('--log_perf_file', default='perf',
                    help='File in which results of current run are to be '
                    ' stored')
parser.add_argument('--min_distance', type=float, default=3.9,
                    help='Min. distance to target')
parser.add_argument('--max_distance', type=float, default=4.0,
                    help='Max. distance to target')
parser.add_argument('--threshold', type=float, default=0.1,
                    help='Threshold distance for navigation to be considered '
                    'successful')
parser.add_argument('--y_range', type=float, default=0.01,
                    help='Max. distance along +ve and -ve y-axis that the '
                    'target can lie')
parser.add_argument('--n_training_samples', type=int, default=4000000,
                    help='Number of samples used to train the policy')
parser.add_argument('--start_index', type=int, default=0,
                    help='Starting index of transitions in pickle file')
parser.add_argument('--exp_name', default='exp-1.2.3',
                    help='Alias for the experiment')
parser.add_argument('--batch_size', default=512, type=int,
                    help='Batch size')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--n_epochs', type=int, default=400,
                    help='Number of epochs')

def main():
    args = parser.parse_args()
    env_name = args.env_name
    resume = args.resume
    checkpoint_folder = args.checkpoint_folder
    test_only = args.test_only
    seed = args.seed
    no_gpu = args.no_gpu
    dir_name = args.dir_name
    visualize = args.visualize
    n_test_steps = args.n_test_steps
    log_perf_file = args.log_perf_file
    min_distance = args.min_distance
    max_distance = args.max_distance
    threshold = args.threshold
    y_range = args.y_range
    n_training_samples = args.n_training_samples
    start_index = args.start_index
    exp_name = args.exp_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs

    if test_only: 
        env  = gym.make('HexapodFewerStatesWithTarget-v1')
    else:
        env = gym.make(env_name)

    set_global_seed(seed)
    env.seed(seed)

    # input_shape = env.observation_space.shape[0] + 2
    # output_shape = env.action_space.shape[0]
    
    input_shape = 51
    output_shape = 18

    net = Policy(input_shape, output_shape)
    if not no_gpu:
        net = net.cuda()
    optimizer = Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    epochs = 0

    if test_only and not(checkpoint_folder):
        print('ERROR: You have not entered a checkpoint file.')
        return
    elif test_only and checkpoint_folder:
        storage_path = checkpoint_folder
        checkpoint_file = storage_path + '/checkpoints/400.pth.tar'
        norms_file = storage_path + '/norms/normalization_factors.pkl'
        checkpoints_path = os.path.join(storage_path, "checkpoints")
        tensorboard_path = os.path.join(storage_path, "tensorboard")
        norms_path = os.path.join(storage_path, "norms")
        perf_path = os.path.join(storage_path, "performance")
        epochs, net, optimizer = load_checkpoint(checkpoint_file,
                                                 net,
                                                 optimizer)


    if not test_only:
        # Set up folder structure
        base_path = "/home/shiqing/gc-batch-rl-locomotion/"
        date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        storage_path = os.path.join(base_path, "logs", date_str + "_" + env_name + "_naive-gcp")
        checkpoints_path = os.path.join(storage_path, "checkpoints")
        tensorboard_path = os.path.join(storage_path, "tensorboard")
        norms_path = os.path.join(storage_path, "norms")
        perf_path = os.path.join(storage_path, "performance")
        os.mkdir(storage_path)
        os.mkdir(checkpoints_path)
        os.mkdir(tensorboard_path)
        os.mkdir(norms_path)
        os.mkdir(perf_path)
        input_folder = args.input_folder

        # myDataset = Dataset(input_folder, batch_size, total_number=n_training_samples, first_load=False)
        myDataset = AugDataset(input_folder, batch_size, total_number=n_training_samples, first_load=False, distinguish=False)
        
        state_mean, state_std = myDataset.get_state_normalization_factors()
        step_mean, step_std = myDataset.get_step_normalization_factors()
        normalization_factors = {'state':
                                 [state_mean, state_std],
                                 'distance_per_step':
                                 [step_mean, step_std]}
        n_file = open(norms_path + '/' + 'normalization_factors.pkl', 'wb')
        pickle.dump(normalization_factors, n_file)
        n_file.close()

        # Summary writer for tensorboardX
        writer = {}
        writer['writer'] = SummaryWriter(logdir=tensorboard_path)

        train_batch_number = myDataset.train_batch_number
        val_batch_number = myDataset.val_batch_number
        for e in range(epochs, n_epochs):
            ep_loss = []
            # Train network
            for i in range(train_batch_number):
                inp, target = myDataset.get_next_train_batch()
                out = net(convert_to_variable(inp,
                                              grad=False,
                                              gpu=(not no_gpu)))
                target = convert_to_variable(np.array(target),
                                             grad=False,
                                             gpu=(not no_gpu))
                loss = criterion(out, target)
                optimizer.zero_grad()
                ep_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = []
            for i in range(val_batch_number):
                inp, target = myDataset.get_next_val_batch()
                out = net(convert_to_variable(inp,
                                              grad=False,
                                              gpu=(not no_gpu)))
                target = convert_to_variable(np.array(target),
                                             grad=False,
                                             gpu=(not no_gpu))
                loss = criterion(out, target)
                val_loss.append(loss.item())

            writer['iter'] = e + 1
            writer['writer'].add_scalar('data/val_loss',
                                        np.array(val_loss).mean(),
                                        e + 1)
            writer['writer'].add_scalar('data/training_loss',
                                        np.array(ep_loss).mean(),
                                        e + 1)
            if (e + 1)%10 == 0:
                save_checkpoint({'epochs': (e + 1),
                                'state_dict': net.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                filename=os.path.join(checkpoints_path,
                                                    str(e + 1) + '.pth.tar'))

            print('Epoch:', e + 1)
            print('Training loss:', np.array(ep_loss).mean())
            print('Val loss:', np.array(val_loss).mean())
            print('')
        writer['writer'].close()
    # Now we use the trained net to see how the agent reaches a different
    # waypoint from the current one.

    success = 0
    failure = 0

    closest_distances = []
    time_to_closest_distances = []

    if test_only:
        f = open(norms_file, 'rb')
    else:
        f = open(norms_path + '/' + 'normalization_factors.pkl', 'rb')
    normalization_factors = pickle.load(f)
    average_distance = normalization_factors['distance_per_step'][0]

    for i in range(n_test_steps):
        state = env.reset()
        if env_name == 'HexapodFewerStatesDataCollection-v1':
            obs = env.unwrapped.get_body_com('torso')[0:2]
            target_obs = [obs[0] + np.random.uniform(min_distance, max_distance),
                          obs[1] + np.random.uniform(-y_range, y_range)]
            # theta = np.random.uniform(-np.pi/4, np.pi/4)
            position = env.sim.data.qpos.flat.copy() # 25 = 7 + 18
            w, x, y, z = position[3:7]
            rot = Rotation.from_quat([x, y, z, w])
            rpy = rot.as_euler('xyz', degrees=False) #[roll, pitch, yaw]
            target_obs = rotate_point(target_obs, rpy[2])
            env.unwrapped.sim.model.body_pos[-1] = np.concatenate((target_obs, [0]), axis=0)

        else:
            print('Env name not expected')
            assert False
        steps = 0
        done = False
        closest_d = distance(obs, target_obs)
        closest_t = 0
        while distance(obs, target_obs) > threshold and not done:
            goal = preprocess_goal(target_obs - obs)
            state = normalize(np.array(state), norms_path + '/' + 'normalization_factors.pkl')
            inp = np.concatenate([np.squeeze(state),
                                  goal])
            inp = convert_to_variable(inp, grad=False, gpu=(not no_gpu))
            action = net(inp).cpu().detach().numpy()
            state, _, done, _ = env.step(action)
            steps += 1
            if env_name == 'HexapodFewerStatesDataCollection-v1':
                obs = env.unwrapped.get_body_com('torso')[0:2]
            if distance(obs, target_obs) < closest_d:
                closest_d = distance(obs, target_obs)
                closest_t = steps
            if visualize:
                env.render()

        if distance(obs, target_obs) <= threshold:
            success += 1
        elif done:
            failure += 1

        if visualize:
            time.sleep(2)

        closest_distances.append(closest_d)
        time_to_closest_distances.append(closest_t)

    print('Successes: %d, Failures: %d, '
          'Closest distance: %f, Time to closest distance: %d'
          % (success, failure, np.mean(closest_distances),
             np.mean(time_to_closest_distances)))

    f = open(perf_path + '/'+ log_perf_file, 'a+')
    f.write(exp_name + ':Seed-' + str(seed) + ',Success-' +
            str(success) + ',Failure-' + str(failure) +
            ',Closest_distance-' + str(closest_distances) +
            ',Time_to_closest_distance-' + str(time_to_closest_distances)
            + ',Closest distance-' + str(np.mean(closest_distances)) + ',Time to closest distance-' + str(np.mean(time_to_closest_distances))+ '\n')
    f.close()


if __name__ == '__main__':
    main()
