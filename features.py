# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import contextlib
import h5py
import itertools
import numpy
import os
import torch


class CorefCorpus:
    def __init__(self, anaphoricity_fmap, pairwise_fmap, docs):
        self.anaphoricity_fmap = anaphoricity_fmap
        self.pairwise_fmap = pairwise_fmap
        self.docs = docs

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return CorefCorpus(self.anaphoricity_fmap, self.pairwise_fmap, self.docs[key])
        else:
            return self.docs[key]

    def save_to_hdf5(self, h5_group):
        enc_fmap = [ft.encode('utf-8') for ft in self.anaphoricity_fmap]
        maxlen = max(len(ft) for ft in enc_fmap)
        dtype = 'S%d' % maxlen
        h5_group.create_dataset('anaphoricity_fmap', dtype=dtype, data=enc_fmap)

        enc_fmap = [ft.encode('utf-8') for ft in self.pairwise_fmap]
        maxlen = max(len(ft) for ft in enc_fmap)
        dtype = 'S%d' % maxlen
        h5_group.create_dataset('pairwise_fmap', dtype=dtype, data=enc_fmap)

        doc_nmentions = numpy.array([d.nmentions for d in self.docs], dtype=numpy.int32)
        h5_group.create_dataset('doc_nmentions', dtype=numpy.int32, data=doc_nmentions)

        ext_doc_nmentions = numpy.concatenate([numpy.zeros((1,), dtype=numpy.int32), doc_nmentions])
        ana_idx = ext_doc_nmentions.cumsum()
        pw_idx = numpy.cumsum(ext_doc_nmentions * (ext_doc_nmentions - 1) // 2)

        ana_group = h5_group.create_group('anaphoricity_features')
        _store_features(ana_group,
                        [d.anaphoricity_features for d in self.docs],
                        [d.anaphoricity_offsets for d in self.docs])

        pw_group = h5_group.create_group('pairwise_features')
        _store_features(pw_group,
                        [d.pairwise_features for d in self.docs],
                        [d.pairwise_offsets for d in self.docs])

        opc_m2c = numpy.concatenate([d.mention_to_opc.numpy() for d in self.docs])
        h5_group.create_dataset('mention_to_opc', dtype=numpy.int32, data=opc_m2c)


class CorefDocument:
    def __init__(self, anaphoricity_dim, anaphoricity_features, anaphoricity_offsets,
                 pairwise_dim, pairwise_features, pairwise_offsets, opc_m2c):
        self.nmentions = len(anaphoricity_features)
        self.anaphoricity_dim = anaphoricity_dim
        self.pairwise_dim = pairwise_dim

        self.anaphoricity_features = anaphoricity_features
        self.anaphoricity_offsets = anaphoricity_offsets
        self.pairwise_features = pairwise_features
        self.pairwise_offsets = pairwise_offsets
        self.mention_to_opc = opc_m2c

        # Note: This must be  a list of *sorted* lists
        self.opc_clusters = [[] for _ in range(self.mention_to_opc.max() + 1)]
        for mention_idx, cluster_idx in enumerate(self.mention_to_opc):
            self.opc_clusters[cluster_idx].append(mention_idx)

        self.solution_mask = torch.FloatTensor(self.nmentions, self.nmentions)
        for c in self.opc_clusters:
            cluster_mask = torch.zeros(self.nmentions)
            cluster_mask[torch.LongTensor(c)] = 1.0
            for m in c:
                self.solution_mask[m, :] = cluster_mask
        self.solution_mask.tril_(-1)
        eps = torch.eq(self.solution_mask.sum(dim=1, keepdim=False), 0).float()
        self.solution_mask[torch.eye(self.nmentions).byte()] = eps

    def is_anaphoric(self, mention):
        cluster_id = self.mention_to_opc[mention]
        return mention > self.opc_clusters[cluster_id][0]

    def anaphoricity_labels(self):
        # returns N x 1 matrix
        return torch.FloatTensor([[1] if self.is_anaphoric(m) else [-1] for m in range(self.nmentions)])


def _store_features(group, features, mention_offsets):
    n_features = sum(f.shape[0] for f in features)
    n_mention_offsets = sum(f.shape[0] for f in mention_offsets)
    feature_ds = group.create_dataset('features', shape=(n_features,), dtype=numpy.int32)
    mention_ds = group.create_dataset('mention_offsets', shape=(n_mention_offsets,), dtype=numpy.int32)
    doc_ds = group.create_dataset('doc_offsets', shape=(len(features) + 1,), dtype=numpy.int32)
    fidx = 0
    oidx = 0
    for i, (d, o) in enumerate(zip(features, mention_offsets)):
        feature_ds[fidx:(fidx + d.shape[0])] = d.numpy()
        mention_ds[oidx:(oidx + o.shape[0])] = o.numpy()
        doc_ds[i, :] = (fidx, oidx)
        fidx += d.shape[0]
        oidx += o.shape[0]
    doc_ds[-1, :] = (fidx, oidx)


def _load_features(group, docno):
    from_f, from_o = group['doc_offsets'][docno, :]
    to_f, to_o = group['doc_offsets'][docno + 1, :]
    features = torch.from_numpy(group['features'][from_f:to_f])
    offsets = torch.from_numpy(group['mention_offsets'][from_o:to_o])
    return features, offsets


def load_from_hdf5(h5_group):
    anaphoricity_fmap = h5_group['anaphoricity_fmap']
    pairwise_fmap = h5_group['pairwise_fmap']
    doc_nmentions = h5_group['doc_nmentions']
    anaphoricity_features = h5_group['anaphoricity_features']
    pairwise_features = h5_group['pairwise_features']
    mention_to_opc = h5_group['mention_to_opc']

    anaphoricity_dim = anaphoricity_fmap.shape[0]
    pairwise_dim = pairwise_fmap.shape[0]

    ndocs = doc_nmentions.shape[0]

    ext_doc_nmentions = numpy.pad(doc_nmentions, (1, 0), 'constant')
    ana_idx = ext_doc_nmentions.cumsum()
    pw_idx = numpy.cumsum(ext_doc_nmentions * (ext_doc_nmentions - 1) // 2)

    docs = []
    for i in range(ndocs):
        ana_features, ana_offsets = _load_features(anaphoricity_features, i)
        pw_features, pw_offsets = _load_features(pairwise_features, i)
        opc_m2c = torch.from_numpy(mention_to_opc[ana_idx[i]:ana_idx[i + 1]])
        docs.append(CorefDocument(anaphoricity_dim, ana_features, ana_offsets,
                                  pairwise_dim, pw_features, pw_offsets, opc_m2c))

    return CorefCorpus(numpy.copy(anaphoricity_fmap), numpy.copy(pairwise_fmap), docs)


def vocabulary_sizes_from_hdf5(h5_group):
    anaphoricity_fmap = h5_group['anaphoricity_fmap']
    pairwise_fmap = h5_group['pairwise_fmap']
    return len(anaphoricity_fmap), len(pairwise_fmap)


def load_feature_map(fmap_file):
    with open(fmap_file, 'r') as f:
        feature_list = [line.rstrip('\n').split(' : ')[1] for line in f]
    maxlen = max(len(f) for f in feature_list)
    feature_dtype = 'U%d' % maxlen
    return numpy.array(feature_list, dtype=feature_dtype)


def load_text_data(ana_file, ana_fmap_file, pw_file, pw_fmap_file, opc_file):
    docs = []
    ana_fmap = load_feature_map(ana_fmap_file)
    pw_fmap = load_feature_map(pw_fmap_file)
    with contextlib.ExitStack() as stack:
        ana_f = stack.enter_context(open(ana_file, 'r'))
        pw_f = stack.enter_context(open(pw_file, 'r'))

        if opc_file:
            opc_f = stack.enter_context(open(opc_file, 'r'))
        else:
            opc_f = itertools.repeat(None)

        lineno = 0
        for ana_line, pw_line, opc_line in zip(ana_f, pw_f, opc_f):
            lineno += 1
            if lineno % 100 == 0:
                print(lineno)

            # anaphoricity features
            mentions = ana_line.rstrip('\n').split('|')
            nmentions = len(mentions) - 1
            ftlist = []
            offsets = []
            for m in mentions[1:]:
                offsets.append(len(ftlist))
                ftlist.extend(int(ft) for ft in m.split(' '))
            ana_features = torch.IntTensor(ftlist)
            ana_offsets = torch.IntTensor(offsets)

            # pairwise features
            mention_pairs = pw_line.rstrip('\n').split('|')
            split_mention_pairs = [m.split(' ') for m in mention_pairs[1:]]
            curr_m = 0
            ant_m = 0
            i = 0
            ftlist = []
            offsets = []
            for m in split_mention_pairs:
                if curr_m != ant_m:
                    offsets.append(len(ftlist))
                    ftlist.extend(int(ft) for ft in m)
                    i += 1

                if curr_m == ant_m:
                    curr_m += 1
                    ant_m = 0
                else:
                    ant_m += 1
            pw_features = torch.IntTensor(ftlist)
            pw_offsets = torch.IntTensor(offsets)

            # oracle predicted clusters
            if opc_line:
                opc_m2c = torch.IntTensor(nmentions)
                for i, sc in enumerate(opc_line.rstrip('\n').split('|')):
                    if sc:
                        for m in sc.split(' '):
                            opc_m2c[int(m)] = i
            else:
                opc_m2c = torch.zeros(nmentions).int()

            docs.append(CorefDocument(len(ana_fmap), ana_features, ana_offsets,
                                      len(pw_fmap), pw_features, pw_offsets,
                                      opc_m2c))

    return CorefCorpus(ana_fmap, pw_fmap, docs)


def main():
    data_path = '/wrk/chardmei/DONOTREMOVE/conll-features'
    out_path = '/homeappl/home/chardmei/coref/neural-coref/exp/data'

    ana_fmap_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-anaphMapping.txt')
    train_ana_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt')
    dev_ana_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt')
    test_ana_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-anaphTestFeats.txt')
    pw_fmap_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-pwMapping.txt')
    train_pw_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-pwTrainFeats.txt')
    dev_pw_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-pwDevFeats.txt')
    test_pw_file = os.path.join(data_path, 'SMALL-FINAL+MOARANAPH+MOARPW-pwTestFeats.txt')
    train_opc_file = os.path.join(data_path, 'SMALLTrainOPCs.txt')
    dev_opc_file = os.path.join(data_path, 'SMALLDevOPCs.txt')

    training_h5 = os.path.join(out_path, 'training.h5')
    dev_h5 = os.path.join(out_path, 'dev.h5')
    test_h5 = os.path.join(out_path, 'test.h5')

    print('Loading training data...')
    training_set = load_text_data(train_ana_file, ana_fmap_file,
                                  train_pw_file, pw_fmap_file,
                                  train_opc_file)

    print('Saving...')
    with h5py.File(training_h5, 'w') as h5:
        training_set.save_to_hdf5(h5)

    print('Loading development data...')
    dev_set = load_text_data(dev_ana_file, ana_fmap_file,
                             dev_pw_file, pw_fmap_file,
                             dev_opc_file)

    print('Saving...')
    with h5py.File(dev_h5, 'w') as h5:
        dev_set.save_to_hdf5(h5)

    print('Loading test data...')
    test_set = load_text_data(test_ana_file, ana_fmap_file,
                              test_pw_file, pw_fmap_file,
                              None)

    print('Saving...')
    with h5py.File(test_h5, 'w') as h5:
        test_set.save_to_hdf5(h5)


if __name__ == '__main__':
    main()

