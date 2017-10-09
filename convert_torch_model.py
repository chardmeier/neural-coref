import h5py

torch_file = '/homeappl/home/chardmei/coref/nn_coref/nn/models/vanilla.h5'
out_file = '/wrk/chardmei/mr/mr-torch-vanilla.model'
with h5py.File(torch_file, 'r') as torch_h5, h5py.File(out_file, 'w') as mr_h5:
    mr = mr_h5.create_group('MentionRankingModel')
    mr.attrs['net-config'] = '{"dropout_h_comb": 0.4, "ha_size": 128, "hp_size": 700, "g2_size": null, ' + \
                             '"one_based_features": true, "anaphoricity_fsize": 14200, "pairwise_fsize": 28394}'

    na = torch_h5['na']
    pw = torch_h5['pw']

    mr.create_dataset('ana_model.ana_scoring_model.1.bias', data=pw['6'])
    mr.create_dataset('ana_model.ana_scoring_model.1.weight', data=pw['5'])
    mr.create_dataset('ana_model.hp_model.bias', data=pw['2'])
    mr.create_dataset('ana_model.hp_model.embedding.weight', data=pw['1'])
    mr.create_dataset('eps_model.eps_scoring_model.bias', data=na['4'])
    mr.create_dataset('eps_model.eps_scoring_model.weight', data=na['3'])
    mr.create_dataset('eps_model.ha_model.bias', data=na['2'])
    mr.create_dataset('eps_model.ha_model.embedding.weight', data=na['1'])