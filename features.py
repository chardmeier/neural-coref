# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import contextlib
import h5py
import itertools
import numpy
import os
import torch


#def convert_anaph(infile, fmap_file):
#    with open(fmap_file, 'r') as f:
#        feature_list = [line.rstrip('\n').split(' : ')[1] for line in f]
#    maxlen = max(len(f) for f in feature_list)
#    feature_dtype = 'U%d' % maxlen
#    feature_map = numpy.array(feature_list, dtype=feature_dtype)
#
#    nfeatures = feature_map.shape[0]
#    attributes = {
#        'nfeatures': nfeatures,
#        'feature_map': feature_map
#    }
#
#    docs = []
#    with open(infile, 'r') as f:
#        for line in f:
#            mentions = line.rstrip('\n').split('|')
#            nmentions = len(mentions) - 1
#            coo_indices_list = []
#            for i, m in enumerate(mentions[1:]):
#                if m:
#                    for ft in m.split(' '):
#                        coo_indices_list.append([i, int(ft)])
#            coo_indices = torch.LongTensor(coo_indices_list).transpose(0, 1)
#            coo_values = torch.FloatTensor([1.0] * len(coo_indices_list))
#            matrix = torch.sparse.FloatTensor(coo_indices, coo_values, torch.Size([nmentions, nfeatures]))
#            docs.append(matrix)
#
#    return AnaphoricityFeatures(docs, attributes=attributes)


# def store_anaph(anaph, outfile):
#     metadata = []
#     docstart = 0
#     for m in anaph.docs:
#         nmentions, maxfeats = m.shape
#         metadata.append((docstart, nmentions, maxfeats))
#         docstart += nmentions * maxfeats
#     flattened = numpy.concatenate([numpy.ravel(m.numpy()) for m in anaph.docs])
#     with h5py.File(outfile, 'w') as h5:
#         group = h5.create_group('anaphoricity_features')
#         for key, val in anaph.attributes.items():
#             group.attrs.create(key, val)
#         group.create_dataset('metadata', (len(metadata), 3), dtype=numpy.int64, data=metadata)
#         group.create_dataset('features', (len(flattened),), dtype=numpy.int32, data=flattened)
#
#
# def load_anaph(h5_file):
#     with h5py.File(h5_file, 'r') as h5:
#         metadata = h5['/anaphoricity_features/metadata']
#         flat_features = h5['/anaphoricity_features/features']
#         attributes = dict(h5['/anaphoricity_features'].attrs)
#         docs = []
#         for i in range(metadata.shape[0]):
#             from_idx, nmentions, maxfeats = metadata[i, :]
#             to_idx = from_idx + nmentions * maxfeats
#             docs.append(torch.from_numpy(flat_features[from_idx:to_idx].reshape(nmentions, maxfeats).copy()))
#     return AnaphoricityFeatures(docs, attributes=attributes)

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

        max_ana_features = max(d.anaphoricity_features.size()[1] for d in self.docs)
        fmatrix = numpy.zeros((ana_idx[-1], max_ana_features), dtype=numpy.int32)
        for i, d in enumerate(self.docs):
            fmatrix[ana_idx[i]:ana_idx[i + 1], :d.anaphoricity_features.size()[1]] = d.anaphoricity_features.numpy()
        h5_group.create_dataset('anaphoricity_features', dtype=numpy.int32, data=fmatrix)

        max_pw_features = max(d.pairwise_features.size()[1] for d in self.docs)
        fmatrix = numpy.zeros((pw_idx[-1], max_pw_features), dtype=numpy.int32)
        for i, d in enumerate(self.docs):
            fmatrix[pw_idx[i]:pw_idx[i + 1], :d.pairwise_features.size()[1]] = d.pairwise_features.numpy()
        h5_group.create_dataset('pairwise_features', dtype=numpy.int32, data=fmatrix)

        opc_m2c = numpy.concatenate([d.mention_to_opc.numpy() for d in self.docs])
        h5_group.create_dataset('mention_to_opc', dtype=numpy.int32, data=opc_m2c)


class CorefDocument:
    def __init__(self, anaphoricity_dim, anaphoricity_features, pairwise_dim, pairwise_features, opc_m2c):
        self.nmentions = len(anaphoricity_features)
        self.anaphoricity_dim = anaphoricity_dim
        self.pairwise_dim = pairwise_dim

        self.anaphoricity_features = anaphoricity_features
        self.pairwise_features = pairwise_features
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
        eps = torch.eq(self.solution_mask.sum(dim=0), 0).float()
        self.solution_mask[torch.eye(self.nmentions).byte()] = eps

    def is_anaphoric(self, mention):
        cluster_id = self.mention_to_opc[mention]
        return mention > self.opc_clusters[cluster_id][0]

    def anaphoricity_labels(self):
        return torch.FloatTensor([1 if self.is_anaphoric(m) else -1 for m in range(self.nmentions)])


def _convert_and_truncate(inp):
    sums = numpy.absolute(inp).sum(axis=0)  # abs is for safety only, should be positive anyway
    maxlen = sums.nonzero()[0].max() + 1
    return torch.from_numpy(inp[:, :maxlen])


def load_from_hdf5(h5_group):
    anaphoricity_fmap = h5_group['anaphoricity_fmap']
    pairwise_fmap = h5_group['pairwise_fmap']
    doc_nmentions = h5_group['doc_nmentions']
    anaphoricity_features = h5_group['anaphoricity_features']
    pairwise_features = h5_group['pairwise_features']
    mention_to_opc = h5_group['mention_to_opc']

    # add 1 because index 0 is used for padding
    anaphoricity_dim = anaphoricity_fmap.shape[0] + 1
    pairwise_dim = pairwise_fmap.shape[0] + 1

    ndocs = doc_nmentions.shape[0]

    ext_doc_nmentions = numpy.pad(doc_nmentions, (1, 0), 'constant')
    ana_idx = ext_doc_nmentions.cumsum()
    pw_idx = numpy.cumsum(ext_doc_nmentions * (ext_doc_nmentions - 1) // 2)

    docs = []
    for i in range(ndocs):
        ana_features = _convert_and_truncate(anaphoricity_features[ana_idx[i]:ana_idx[i + 1], :])
        pw_features = _convert_and_truncate(pairwise_features[pw_idx[i]:pw_idx[i + 1], :])
        opc_m2c = torch.from_numpy(mention_to_opc[ana_idx[i]:ana_idx[i + 1]])
        docs.append(CorefDocument(anaphoricity_dim, ana_features, pairwise_dim, pw_features, opc_m2c))

    return CorefCorpus(numpy.copy(anaphoricity_fmap), numpy.copy(pairwise_fmap), docs)


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
            for m in mentions[1:]:
                if m:
                    # note: all features get converted to 1-based
                    ftlist.append([int(ft) + 1 for ft in m.split(' ')])
                else:
                    ftlist.append([])
            maxfeat = max(len(ft) for ft in ftlist)
            ana_features = torch.IntTensor(nmentions, maxfeat).zero_()
            for i, ft in enumerate(ftlist):
                if ft:
                    ana_features[i, :len(ft)] = torch.IntTensor(ft)

            # pairwise features
            mention_pairs = pw_line.rstrip('\n').split('|')
            split_mention_pairs = [m.split(' ') for m in mention_pairs[1:]]
            max_pw_features = max(len(m) for m in split_mention_pairs)
            pw_features = torch.IntTensor(nmentions * (nmentions - 1) // 2, max_pw_features).zero_()
            curr_m = 0
            ant_m = 0
            i = 0
            for m in split_mention_pairs:
                if curr_m != ant_m:
                    if m:
                        # note: all features get converted to 1-based
                        conv_feats = torch.IntTensor([int(ft) + 1 for ft in m])
                        pw_features[i, :len(m)] = conv_feats
                    i += 1
                if curr_m == ant_m:
                    curr_m += 1
                    ant_m = 0
                else:
                    ant_m += 1

            # oracle predicted clusters
            if opc_line:
                opc_m2c = torch.IntTensor(nmentions)
                for i, sc in enumerate(opc_line.rstrip('\n').split('|')):
                    if sc:
                        for m in sc.split(' '):
                            opc_m2c[int(m)] = i
            else:
                opc_m2c = torch.zeros(nmentions).int()

            docs.append(CorefDocument(len(ana_fmap), ana_features, len(pw_fmap), pw_features, opc_m2c))

    return CorefCorpus(ana_fmap, pw_fmap, docs)


def main():
    data_path = '/home/nobackup/ch/coref'
    ana_fmap_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphMapping.txt')
    train_ana_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt')
    dev_ana_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt')
    pw_fmap_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-pwMapping.txt')
    train_pw_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-pwTrainFeats.txt')
    dev_pw_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-pwDevFeats.txt')
    train_opc_file = os.path.join(data_path, 'TrainOPCs.txt')
    dev_opc_file = os.path.join(data_path, 'DevOPCs.txt')

    print('Loading training data...')
    training_set = load_text_data(train_ana_file, ana_fmap_file,
                                  train_pw_file, pw_fmap_file,
                                  train_opc_file)

    print('Saving...')
    with h5py.File('/home/nobackup/ch/coref/training.h5', 'w') as h5:
        training_set.save_to_hdf5(h5)

    print('Loading development data...')
    dev_set = load_text_data(dev_ana_file, ana_fmap_file,
                             dev_pw_file, pw_fmap_file,
                             dev_opc_file)

    print('Saving...')
    with h5py.File('/home/nobackup/ch/coref/dev.h5', 'w') as h5:
        dev_set.save_to_hdf5(h5)


if __name__ == '__main__':
    main()


#class AnaphoricityFeatures:
#    def __init__(self, docs, attributes=None):
#        self.docs = docs
#        if attributes is None:
#            self.attributes = {'nfeatures': int(max(d.max() for d in docs)) + 1}
#        else:
#            self.attributes = attributes
#
#    def nfeatures(self):
#        return self.attributes['nfeatures']


#class OraclePredictedClustering:
#    def __init__(self, opc_file):
#        self.clusters = []
#        self.mention_to_cluster = []
#        with open(opc_file, 'r') as f:
#            for doc, line in enumerate(f):
#                self.clusters.append([])
#                doc_m2c = {}
#                for sc in line.rstrip('\n').split('|'):
#                    cluster = {int(m) for m in sc.split(' ')}
#                    cluster_idx = len(self.clusters[-1])
#                    self.clusters[-1].append(cluster)
#                    for m in cluster:
#                        doc_m2c[m] = cluster_idx
#                self.mention_to_cluster.append([doc_m2c[i] for i in range(len(doc_m2c))])
#                assert len(self.mention_to_cluster[-1]) == len(doc_m2c)




