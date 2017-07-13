import features

import numpy
import os
import torch
import torch.nn as nn

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
        class_weights = torch.index_select(self.delta_a, 0, (labels > 0).long())
        model_loss = torch.sum(class_weights * torch.clamp(1 - labels * predictions, min=0))

        return model_loss


def train(model, train_config, train_features, train_labels, dev_features, dev_labels):
    opt = torch.optim.Adagrad(params=model.parameters())
    loss_fn = AnaphoricityLoss(torch.FloatTensor(train_config['delta_a']))
    epoch_size = len(train_features.docs)
    dot_interval = epoch_size // 80
    print('%d documents per epoch' % epoch_size)

    for p in model.parameters():
        # Sparse initialisation similar to Sutskever et al. (ICML 2013)
        # For tanh units, use std 0.25 and set biases to 0.5
        if p.dim() == 2:
            nn_init.sparse(p, sparsity=0.1, std=0.25)
        else:
            nn_init.constant(p, 0.5)

    for epoch in range(train_config['nepochs']):
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)
            phi_a = torch.autograd.Variable(train_features.docs[idx].to_dense())
            labels = torch.autograd.Variable(train_labels[idx])
            opt.zero_grad()

            predictions = model(phi_a)
            model_loss = loss_fn(predictions, labels)

            reg_loss = sum(p.abs().sum() for p in model.parameters())
            loss = model_loss + train_config['l1reg'] * reg_loss

            train_loss_unreg += model_loss.data[0] / phi_a.size()[0]
            train_loss_reg += loss.data[0] / phi_a.size()[0]

            loss.backward()
            opt.step()
        print()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        for t_phi_a, t_labels in zip(dev_features.docs, dev_labels):
            phi_a = torch.autograd.Variable(t_phi_a.to_dense())
            labels = torch.autograd.Variable(t_labels)
            docsize = phi_a.size()[0]
            predictions = model(phi_a)
            dev_correct += torch.sum((predictions.data * t_labels) > 0)
            dev_total += docsize
            dev_loss += loss_fn(predictions, labels).data[0] / docsize

        dev_acc = dev_correct / dev_total
        print('Epoch %d: train_loss_reg %g / train_loss_unreg %g / dev_loss %g / dev_acc %g' %
              (epoch, train_loss_reg, train_loss_unreg, dev_loss, dev_acc))


def main():
    data_path = '/home/nobackup/ch/coref'
    fmap_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphMapping.txt')
    train_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphTrainFeats.txt')
    dev_file = os.path.join(data_path, 'NONE-FINAL+MOARANAPH+MOARPW-anaphDevFeats.txt')
    train_opc_file = os.path.join(data_path, 'TrainOPCs.txt')
    dev_opc_file = os.path.join(data_path, 'DevOPCs.txt')

    train_features = features.convert_anaph(train_file, fmap_file)
    dev_features = features.convert_anaph(dev_file, fmap_file)

    train_opc = features.OraclePredictedClustering(train_opc_file)
    dev_opc = features.OraclePredictedClustering(dev_opc_file)

    train_config = {
        'nepochs': 100,
        'delta_a': [1, 1],
        'l1reg': 0.001
    }
    model = AnaphoricityModel(train_features.nfeatures(), 200)

    train(model, train_config, train_features, train_opc.anaphoricity_labels(), dev_features, dev_opc.anaphoricity_labels())


if __name__ == '__main__':
    main()

