# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import features
import util

import argparse
import copy
import h5py
import json
import logging
import numpy
import sys
import torch
import torch.nn as nn

from torch.autograd import Variable
# This doesn't seem to be accessible without from ... import
from torch.nn import init as nn_init

from mention_ranking import HModel


class AnaphoricityModel(nn.Module):
    def __init__(self, nfeatures, size_ha):
        super(AnaphoricityModel, self).__init__()
        self.ha_model = HModel(nfeatures, size_ha)
        self.anaphoricity_layer = nn.Linear(size_ha, 1)

    def forward(self, phi_a):
        # noinspection PyCallingNonCallable
        h_a = self.ha_model(phi_a)
        predictions = self.anaphoricity_layer(h_a)
        return predictions


class AnaphoricityLoss(nn.Module):
    def __init__(self, delta_a):
        super(AnaphoricityLoss, self).__init__()
        self.delta_a = torch.autograd.Variable(delta_a)

    def forward(self, predictions, labels):
        class_weights = torch.index_select(self.delta_a, 0, (labels > 0).long().squeeze())
        model_loss = torch.sum(class_weights * torch.clamp(1 - labels * predictions, min=0).squeeze())

        return model_loss


def train(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
    epoch_size = len(training_set)
    dot_interval = epoch_size // 80
    logging.info('%d documents per epoch' % epoch_size)

    for name, p in model.named_parameters():
        # Sparse initialisation similar to Sutskever et al. (ICML 2013)
        # For tanh units, use std 0.25 and set biases to 0.5
        if name.endswith('bias'):
            nn_init.constant(p, 0.5)
        else:
            util.sparse(p, sparsity=0.1, std=0.25)

    if cuda:
        model = model.cuda()

    opt = torch.optim.Adagrad(params=model.parameters(), lr=train_config['learning_rate'])
    loss_fn = AnaphoricityLoss(torch.FloatTensor(train_config['delta_a']))

    for epoch in range(train_config['nepochs']):
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)

            if cuda:
                phi_a = Variable(training_set[idx].anaphoricity_features.long().pin_memory()).cuda(async=True)
            else:
                phi_a = Variable(training_set[idx].anaphoricity_features.long())

            opt.zero_grad()

            predictions = model(phi_a).cpu()
            labels = Variable(training_set[idx].anaphoricity_labels())
            model_loss = loss_fn(predictions, labels)

            reg_loss = sum(p.abs().sum() for p in model.parameters()).cpu()
            loss = model_loss + train_config['l1reg'] * reg_loss

            train_loss_unreg += model_loss.data[0] / phi_a.size()[0]
            train_loss_reg += loss.data[0] / phi_a.size()[0]

            loss.backward()
            opt.step()
        print(flush=True)

        if checkpoint:
            with h5py.File('%s-%03d' % (checkpoint, epoch), 'w') as h5:
                util.save_model(h5, model)

        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        for doc in dev_set:
            if cuda:
                phi_a = Variable(doc.anaphoricity_features.long().pin_memory(), volatile=True).cuda(async=True)
            else:
                phi_a = Variable(doc.anaphoricity_features.long(), volatile=True)
            labels = Variable(doc.anaphoricity_labels(), volatile=True)
            docsize = phi_a.size()[0]
            predictions = model(phi_a).cpu()
            dev_correct += torch.sum((predictions * labels) > 0).data[0]
            dev_total += docsize
            dev_loss += loss_fn(predictions, labels).data[0] / docsize

        dev_acc = dev_correct / dev_total
        logging.info('Epoch %d: train_loss_reg %g / train_loss_unreg %g / dev_loss %g / dev_acc %g' %
                     (epoch, train_loss_reg, train_loss_unreg, dev_loss, dev_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train_file', help='Training corpus (HDF5).', required=True)
    parser.add_argument('--dev', dest='dev_file', help='Development corpus (HDF5).', required=True)
    parser.add_argument('--train-config', dest='train_config', help='Training configuration file.')
    parser.add_argument('--net-config', dest='net_config', help='Network configuration file.')
    parser.add_argument('--model', dest='model_file', help='File name for the trained model.')
    parser.add_argument('--checkpoint', dest='checkpoint', help='File name stem for training checkpoints.')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, format='%(asctime)-15s %(message)s', level=logging.DEBUG)

    cuda = torch.cuda.is_available()

    with h5py.File(args.train_file, 'r') as h5:
        training_set = features.load_from_hdf5(h5)

    with h5py.File(args.dev_file, 'r') as h5:
        dev_set = features.load_from_hdf5(h5)

    # Default settings
    train_config = {
        'nepochs': 20,
        'delta_a': [1.35, 1.0],
        'learning_rate': 0.01,
        'l1reg': 0.001
    }

    if args.train_config:
        with open(args.train_config, 'r') as f:
            util.recursive_dict_update(train_config, json.load(f))

    print(json.dumps(train_config), file=sys.stderr)

    net_config = {
        'ha_size': 128
    }

    if args.net_config:
        with open(args.net_config, 'r') as f:
            util.recursive_dict_update(net_config, json.load(f))

    print(json.dumps(net_config), file=sys.stderr)

    model = AnaphoricityModel(len(training_set.anaphoricity_fmap), net_config['ha_size'])

    train(model, train_config, training_set, dev_set, checkpoint=args.checkpoint, cuda=cuda)

    if args.model_file:
        with h5py.File(args.model_file, 'w') as h5:
            util.save_model(h5, model)


if __name__ == '__main__':
    main()

