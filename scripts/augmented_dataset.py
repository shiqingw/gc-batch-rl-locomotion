import numpy as np
import math
import pickle
import random
from utils import set_global_seed, save_checkpoint
from utils import convert_to_variable, distance, rotate_point
from utils import preprocess_goal, find_norm, prepare_dir, normalize

class AugDataset():
    def __init__(self, data_folder_path, batch_size, total_number, first_load=False, distinguish=True):
        self.data_folder_path = data_folder_path
        self.raw_path = self.data_folder_path + '/' + 'raw.pkl'
        self.train_path = self.data_folder_path + '/' + 'train.pkl'
        self.val_path = self.data_folder_path + '/' + 'val.pkl'
        self.normalization_factors_path = self.data_folder_path + '/' + 'normalization_factors.pkl'
        self.total_number = total_number/2.0 #number of pairs
        self.batch_size = batch_size/2.0 #number of pairs
        self.val_number = int(self.total_number/5)
        self.train_number = self.total_number - self.val_number
        if first_load:
            self.initialize()
        else:
            self.read_normalization_factors()
        self.train_file = open(self.train_path, 'rb')
        self.val_file = open(self.val_path, 'rb')
        self.train_batch_number = int(self.train_number/self.batch_size)
        self.val_batch_number = int(self.val_number/self.batch_size)
        self.next_train_batch_index = 0
        self.next_val_batch_index = 0
        self.distinguish = distinguish
        
    def initialize(self):
        # divide train and  val datasets, calculate normalization factors
        print('First load. Start initializing.')
        train_file = open(self.train_path, 'a+b')
        val_file = open(self.val_path, 'a+b')
        raw_file = open(self.raw_path, 'rb')
        indices = np.arange(self.total_number)
        random.shuffle(indices)
        val_indices = indices[:self.val_number]
        val_indices.sort() # sorted indices for val
        count = 0
        count_val = 0
        batch_state_mean = []
        batch_state_std = []
        batch_step_mean = []
        batch_step_std = []
        batch_action_mean = []
        batch_action_std = []
        batch_number = math.ceil(self.total_number/self.batch_size)
        batch_size_list = [self.batch_size]*(batch_number-1) + [self.total_number - self.batch_size*(batch_number-1)]
        assert int(self.total_number) == int(sum(batch_size_list))
        for i in range(batch_number):
            if i%int(batch_number/20) == 0:
                print('Number of batches loaded: {}/{}'.format(i, batch_number)) 
            old_states = []
            aug_old_states = []
            norms = []
            goals = []
            aug_goals = []
            actions = []
            for _ in range(int(batch_size_list[i])):
                old_s, old_g, new_s, new_g, action = pickle.load(raw_file)
                aug_old_s, aug_old_g, aug_new_s, aug_new_g, _= pickle.load(raw_file) # same action
                if count_val < self.val_number and count == int(val_indices[count_val]):
                    pickle.dump([old_s, old_g, new_s, new_g, action], val_file)
                    pickle.dump([aug_old_s, aug_old_g, aug_new_s, aug_new_g, action], val_file)
                    count_val += 1
                else:
                    pickle.dump([old_s, old_g, new_s, new_g, action], train_file)
                    pickle.dump([aug_old_s, aug_old_g, aug_new_s, aug_new_g, action], train_file)
                count += 1
                
                old_states.append(np.squeeze(np.array(old_s)))
                aug_old_states.append(np.squeeze(np.array(aug_old_s)))
                norms.append(find_norm(np.squeeze(np.array(new_g) -
                                                  np.array(old_g))))
                norms.append(find_norm(np.squeeze(np.array(aug_new_g) -
                                                  np.array(aug_old_g))))
                goals.append(preprocess_goal(np.squeeze(
                                                    np.array(new_g) -
                                                    np.array(old_g))))
                aug_goals.append(preprocess_goal(np.squeeze(
                                                        np.array(aug_new_g) -
                                                        np.array(aug_old_g))))
                actions.append(np.squeeze(np.array(action)))

            old_states = np.array(old_states)
            aug_old_states = np.array(aug_old_states)
            norms = np.array(norms)
            goals = np.array(goals)
            aug_goals = np.array(aug_goals)
            actions = np.array(actions)

            batch_state_mean.append(np.concatenate((old_states, aug_old_states),axis=0).mean(axis=0))
            batch_state_std.append(np.concatenate((old_states, aug_old_states),axis=0).std(axis=0))
            batch_step_mean.append(norms.mean(axis=0))
            batch_step_std.append(norms.std(axis=0))
            batch_action_mean.append(actions.mean(axis=0))
            batch_action_std.append(actions.std(axis=0))
        assert count == self.total_number
        assert count_val == self.val_number
        raw_file.close()
        train_file.close()
        val_file.close()
        # list to array
        batch_state_mean = np.array(batch_state_mean)
        batch_state_std = np.array(batch_state_std)
        batch_step_mean = np.array(batch_step_mean)
        batch_step_std = np.array(batch_step_std)
        batch_action_mean = np.array(batch_action_mean)
        batch_action_std = np.array(batch_action_std)
        # state normalization factors
        self.state_mean = np.dot(np.array(batch_size_list)*2, batch_state_mean)/(self.total_number*2)
        A = np.dot(np.array(batch_size_list)*2, np.square(batch_state_std))
        B = np.dot(np.array(batch_size_list)*2, np.square(batch_state_mean - self.state_mean))
        self.state_std = np.sqrt((A+B)/(self.total_number*2))
        # step normalization factors
        self.step_mean = np.dot(np.array(batch_size_list)*2, batch_step_mean)/(self.total_number*2)
        A = np.dot(np.array(batch_size_list)*2, np.square(batch_step_std))
        B = np.dot(np.array(batch_size_list)*2, np.square(batch_step_mean - self.step_mean))
        self.step_std = np.sqrt((A+B)/(self.total_number*2))
        # action normalization factors
        self.action_mean = np.dot(np.array(batch_size_list), batch_action_mean)/self.total_number
        A = np.dot(np.array(batch_size_list), np.square(batch_action_std))
        B = np.dot(np.array(batch_size_list), np.square(batch_action_mean - self.action_mean))
        self.action_std = np.sqrt((A+B)/self.total_number)
        #save normalization factors
        normalization_factors = {'state':
                                 [self.state_mean, self.state_std],
                                 'distance_per_step':
                                 [self.step_mean, self.step_std],
                                 'action':
                                 [self.action_mean, self.action_std]}
        n_file = open(self.normalization_factors_path, 'wb')
        pickle.dump(normalization_factors, n_file)
        n_file.close()
        print('Initialization done.')

    def read_normalization_factors(self):
        n_file = open(self.normalization_factors_path, 'rb')
        normalization_factors = pickle.load(n_file)
        n_file.close()
        self.state_mean, self.state_std = normalization_factors['state']
        self.step_mean, self.step_std = normalization_factors['distance_per_step']
        self.action_mean, self.action_std = normalization_factors['action']

    def get_state_normalization_factors(self):
        return self.state_mean, self.state_std

    def get_step_normalization_factors(self):
        return self.step_mean, self.step_std
    
    def get_action_normalization_factors(self):
        return self.action_mean, self.action_std
    
    def get_next_train_batch(self):
        if self.next_train_batch_index >= self.train_batch_number:
            self.next_train_batch_index = 0
            self.train_file.close()
            self.train_file = open(self.train_path, 'rb')
        self.next_train_batch_index += 1
        old_states = []
        aug_old_states = []
        goals = []
        aug_goals = []
        actions = []
        for _ in range(int(self.batch_size)):
            old_s, old_g, _, new_g, action = pickle.load(self.train_file)
            aug_old_s, aug_old_g, aug_new_s, aug_new_g, _= pickle.load(self.train_file)
            old_states.append(np.squeeze(np.array(old_s)))
            aug_old_states.append(np.squeeze(np.array(aug_old_s)))
            goals.append(preprocess_goal(np.squeeze(
                                                    np.array(new_g) -
                                                    np.array(old_g))))
            aug_goals.append(preprocess_goal(np.squeeze(
                                                        np.array(aug_new_g) -
                                                        np.array(aug_old_g))))
            actions.append(np.squeeze(np.array(action)))

        indices = np.arange(self.batch_size)
        random.shuffle(indices)
        indices = indices.astype(int)
        old_states = np.array(old_states)[indices]
        aug_old_states = np.array(aug_old_states)[indices]
        goals = np.array(goals)[indices]
        aug_goals = np.array(aug_goals)[indices]
        actions = np.array(actions)[indices]

        old_states = (old_states - self.state_mean) / (self.state_std + 1e-6)
        aug_old_states = (aug_old_states - self.state_mean) / (self.state_std + 1e-6)
        
        inp = np.concatenate((old_states, goals), axis=1)
        aug_inp = np.concatenate((aug_old_states, aug_goals), axis=1)
        labels = actions
        if self.distinguish: 
            return inp, aug_inp, labels
        else:
            return np.concatenate((inp, aug_inp), axis=0), np.concatenate((labels, labels), axis=0)

    def get_next_val_batch(self):
        if self.next_val_batch_index >= self.val_batch_number:
            self.next_val_batch_index = 0
            self.val_file.close()
            self.val_file = open(self.val_path, 'rb')
        self.next_val_batch_index += 1
        old_states = []
        aug_old_states = []
        goals = []
        aug_goals = []
        actions = []
        for _ in range(int(self.batch_size)):
            old_s, old_g, _, new_g, action = pickle.load(self.val_file)
            aug_old_s, aug_old_g, aug_new_s, aug_new_g, _= pickle.load(self.val_file)
            old_states.append(np.squeeze(np.array(old_s)))
            aug_old_states.append(np.squeeze(np.array(aug_old_s)))
            goals.append(preprocess_goal(np.squeeze(
                                                    np.array(new_g) -
                                                    np.array(old_g))))
            aug_goals.append(preprocess_goal(np.squeeze(
                                                        np.array(aug_new_g) -
                                                        np.array(aug_old_g))))
            actions.append(np.squeeze(np.array(action)))
        indices = np.arange(self.batch_size)
        random.shuffle(indices)
        indices = indices.astype(int)
        old_states = np.array(old_states)[indices]
        aug_old_states = np.array(aug_old_states)[indices]
        goals = np.array(goals)[indices]
        aug_goals = np.array(aug_goals)[indices]
        actions = np.array(actions)[indices]

        old_states = (old_states - self.state_mean) / (self.state_std + 1e-6)
        aug_old_states = (aug_old_states - self.state_mean) / (self.state_std + 1e-6)
        
        inp = np.concatenate((old_states, goals), axis=1)
        aug_inp = np.concatenate((aug_old_states, aug_goals), axis=1)
        labels = actions
        if self.distinguish: 
            return inp, aug_inp, labels
        else:
            return np.concatenate((inp, aug_inp), axis=0), np.concatenate((labels, labels), axis=0)
