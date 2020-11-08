from dataset import Dataset

input_folder = '/home/shiqing/gc-batch-rl-locomotion/samples/Hexapod-on-policy-samples-4e6'
batch_size = 512
myDataset = Dataset(input_folder, batch_size, total_number=4e6, first_load=False)
_, _ = myDataset.get_next_train_batch()
print(myDataset.next_train_batch_index)
_, _ = myDataset.get_next_val_batch()
print(myDataset.next_val_batch_index)