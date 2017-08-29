import features
import h5py
import os

data_path = '/home/nobackup/ch/coref'
ana_fmap_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphMapping.txt')
test_ana_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphTestFeats.txt')
pw_fmap_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-pwMapping.txt')
test_pw_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-pwTestFeats.txt')

print('Loading test data...')
training_set = features.load_text_data(test_ana_file, ana_fmap_file,
                                       test_pw_file, pw_fmap_file,
                                       None)

print('Saving...')
with h5py.File('/home/nobackup/ch/coref/test2.h5', 'w') as h5:
    training_set.save_to_hdf5(h5)
