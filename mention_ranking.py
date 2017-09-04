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

from torch.autograd import Variable
# This doesn't seem to be accessible without from ... import
from torch.nn import init as nn_init


class HModel(torch.nn.Module):
    def __init__(self, nfeatures, size_ha):
        super(HModel, self).__init__()
        self.size_ha = size_ha
        self.embedding = torch.nn.Embedding(nfeatures + 1, size_ha, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.FloatTensor(size_ha))

    def forward(self, phi_a):
        batch_size = phi_a.size()[0]
        ft_embed = self.embedding(phi_a)
        sum_embed = torch.sum(ft_embed, dim=1).squeeze(1)
        out = torch.tanh(sum_embed + self.bias.expand(batch_size, self.size_ha))
        return out


class MentionRankingModel(torch.nn.Module):
    def __init__(self, phi_a_size, phi_p_size, ha_size, hp_size, hidden_size=None, cuda=False):
        super(MentionRankingModel, self).__init__()

        self.with_cuda = cuda

        self.ha_size = ha_size
        self.hp_size = hp_size
        self.ha_model = HModel(phi_a_size, ha_size)
        self.hp_model = HModel(phi_p_size, hp_size)

        if hidden_size:
            self.ana_scoring_model = torch.nn.Sequential(
                    torch.nn.Linear(ha_size + hp_size, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, 1)
                )
        else:
            self.ana_scoring_model = torch.nn.Linear(ha_size + hp_size, 1)

        self.eps_scoring_model = torch.nn.Linear(ha_size, 1)

    def forward(self, phi_a, all_phi_p):
        nmentions = phi_a.size()[0]
        ncands = all_phi_p.size()[0]

        h_a = self.ha_model(phi_a)

        eps_scores = self.eps_scoring_model(h_a)

        if phi_a.is_cuda and all_phi_p.is_cuda:
            h_combined = torch.autograd.Variable(torch.cuda.FloatTensor(ncands, self.ha_size + self.hp_size))
        else:
            h_combined = torch.autograd.Variable(torch.FloatTensor(ncands, self.ha_size + self.hp_size))

        i = 0
        for j in range(1, nmentions):
            h_combined[i:(i + j), :self.ha_size] = h_a[j, :].expand(j, self.ha_size)
            i += j

        h_combined[:, self.ha_size:] = self.hp_model(all_phi_p)
        ana_scores = self.ana_scoring_model(h_combined)

        all_scores = torch.autograd.Variable(torch.zeros(nmentions, nmentions))

        # Put epsilon scores on the main diagonal
        eps_idx = torch.eye(nmentions).byte()
        all_scores[eps_idx] = eps_scores.cpu()

        # Put anaphoric scores in the triangular part below the diagonal
        ana_idx = torch.tril(torch.ones(nmentions, nmentions).byte(), -1)
        all_scores[ana_idx] = ana_scores.cpu()

        return all_scores


class MentionRankingLoss(torch.nn.Module):
    def __init__(self, costs):
        super(MentionRankingLoss, self).__init__()
        self.false_new_cost = costs['false_new']
        self.link_costs = Variable(torch.FloatTensor([[costs['false_link']], [costs['wrong_link']]]))

    def forward(self, scores, solution_mask):
        # we can't use infinity here because otherwise multiplication by 0 is NaN
        minimum_score = scores.min()
        solution_scores = solution_mask * scores + (1.0 - solution_mask) * minimum_score.expand_as(scores)
        best_correct = solution_scores.max(dim=1)[0]

        # The following calculations create a tensor in which each component contains the right loss penalty:
        # 0 for correct predictions, cost[0..2] for false link, false new and wrong link, respectively
        non_anaphoric = torch.diag(solution_mask)
        anaphoricity_selector = torch.stack([non_anaphoric, 1 - non_anaphoric], dim=1).float()
        # tril() suppresses cataphoric and self-references
        potential_costs = torch.tril(torch.mm(anaphoricity_selector, self.link_costs.expand(2, scores.size()[1])), -1)
        potential_costs[torch.eye(scores.size()[0]).byte()] = self.false_new_cost
        cost_matrix = (1.0 - solution_mask) * potential_costs

        loss = torch.sum(cost_matrix * (1.0 + scores - best_correct.expand_as(scores)))

        return loss


def train(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
    loss_fn = MentionRankingLoss(train_config['error_costs'])

    epoch_size = len(training_set)
    dot_interval = max(epoch_size // 80, 1)
    logging.info('%d documents per epoch' % epoch_size)

    logging.info('Initialising parameters...')
    for p in model.parameters():
        # Sparse initialisation similar to Sutskever et al. (ICML 2013)
        # For tanh units, use std 0.25 and set biases to 0.5
        if p.dim() == 2:
            util.sparse(p, sparsity=0.1, std=0.25)
        else:
            nn_init.constant(p, 0.5)

    if cuda:
        model = model.cuda()

    opt = torch.optim.Adagrad(params=model.parameters())

    logging.info('Starting training...')
    for epoch in range(train_config['nepochs']):
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        skipped = 0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)

            if training_set[idx].nmentions > train_config['maxsize_gpu']:
                skipped += 1
                continue

            solution_mask = Variable(training_set[idx].solution_mask)

            opt.zero_grad()

            if cuda:
                phi_a = Variable(training_set[idx].anaphoricity_features.long().pin_memory()).cuda(async=True)
                phi_p = Variable(training_set[idx].pairwise_features.long().pin_memory()).cuda(async=True)
                scores = model(phi_a, phi_p).cpu()
            else:
                phi_a = Variable(training_set[idx].anaphoricity_features.long())
                phi_p = Variable(training_set[idx].pairwise_features.long())
                scores = model(phi_a, phi_p).cpu()

            model_loss = loss_fn(scores, solution_mask)

            reg_loss = sum(p.abs().sum() for p in model.parameters()).cpu()
            loss = model_loss + train_config['l1reg'] * reg_loss

            train_loss_unreg += model_loss.data[0] / phi_a.size()[0]
            train_loss_reg += loss.data[0] / phi_a.size()[0]

            loss.backward()
            opt.step()

            del loss
            del model_loss
            del reg_loss

        print(flush=True)
        logging.info('Skipped %d/%d documents.' % (skipped, len(training_set)))

        if cuda:
            cpu_model = copy.deepcopy(model).cpu()
        else:
            cpu_model = model

        if checkpoint:
            logging.info('Saving checkpoint...')
            with open('%s-%03d' % (checkpoint, epoch), 'wb') as f:
                torch.save(cpu_model, f)

        logging.info('Computing devset performance...')
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        for doc in dev_set:
            if cuda and doc.nmentions <= train_config['maxsize_gpu']:
                phi_a = Variable(doc.anaphoricity_features.long().pin_memory(), volatile=True).\
                    cuda(async=True)
                phi_p = Variable(doc.pairwise_features.long().pin_memory(), volatile=True).\
                    cuda(async=True)
                predictions = model(phi_a, phi_p).cpu()
            else:
                phi_a = Variable(doc.anaphoricity_features.long(), volatile=True)
                phi_p = Variable(doc.pairwise_features.long(), volatile=True)
                predictions = cpu_model(phi_a, phi_p)

            solution_mask = Variable(doc.solution_mask, volatile=True)
            docsize = phi_a.size()[0]
            dev_correct += torch.sum((predictions * solution_mask) > 0).data[0]
            dev_total += docsize
            dev_loss += loss_fn(predictions, solution_mask).data[0] / docsize

        dev_acc = dev_correct / dev_total
        logging.info('Epoch %d: train_loss_reg %g / train_loss_unreg %g / dev_loss %g / dev_acc %g' %
                     (epoch, train_loss_reg, train_loss_unreg, dev_loss, dev_acc))


def predict(model, test_set, cuda=False, maxsize_gpu=None):
    if maxsize_gpu is None:
        maxsize_gpu = 10000

    if cuda:
        cpu_model = copy.deepcopy(model).cpu()
        model.cuda()
    else:
        cpu_model = model

    predictions = []
    for doc in test_set:
        if cuda and doc.nmentions <= maxsize_gpu:
            phi_a = Variable(doc.anaphoricity_features.long().pin_memory(), volatile=True). \
                cuda(async=True)
            phi_p = Variable(doc.pairwise_features.long().pin_memory(), volatile=True). \
                cuda(async=True)
            doc_pred = model(phi_a, phi_p).cpu()
        else:
            phi_a = Variable(doc.anaphoricity_features.long(), volatile=True)
            phi_p = Variable(doc.pairwise_features.long(), volatile=True)
            doc_pred = cpu_model(phi_a, phi_p)

        n_doc_pred = doc_pred.data.numpy()
        n_doc_pred[numpy.triu_indices_from(n_doc_pred, 1)] = float('-inf')
        argmax = n_doc_pred.argmax(axis=1)
        predictions.append([x for x in argmax])

    return predictions


def training_mode(args, cuda):
    logging.info('Loading training data...')
    with h5py.File(args.train_file, 'r') as h5:
        training_set = features.load_from_hdf5(h5)

    logging.info('Loading development data...')
    with h5py.File(args.dev_file, 'r') as h5:
        dev_set = features.load_from_hdf5(h5)

    # Default settings
    train_config = {
        'nepochs': 100,
        'delta_a': [1, 1],
        'l1reg': 0.001,
        'error_costs': {
            'false_link': 0.5,
            'false_new': 1.2,
            'wrong_link': 1.0
        },
        'maxsize_gpu': 350
    }

    if args.train_config:
        with open(args.train_config, 'r') as f:
            util.recursive_dict_update(train_config, json.load(f))

    model = MentionRankingModel(len(training_set.anaphoricity_fmap), len(training_set.pairwise_fmap), 200, 200,
                                cuda=cuda)

    logging.info('Training model...')
    train(model, train_config, training_set, dev_set, checkpoint=args.checkpoint, cuda=cuda)

    if args.model_file:
        logging.info('Saving model...')
        with open(args.model_file, 'wb') as f:
            torch.save(model, f)

    return model


def test_mode(args, model, cuda):
    if model is None:
        if args.model_file is None:
            logging.error('No model given.')
            sys.exit(1)

        with open(args.model_file, 'rb') as f:
            model = torch.load(f)
    elif args.model_file:
        logging.warning('Using new model for prediction, ignoring --model argument.')

    logging.info('Loading test data...')
    with h5py.File(args.test_file, 'r') as h5:
        test_set = features.load_from_hdf5(h5)
    
    logging.info('Predicting...')
    predictions = predict(model, test_set, cuda, 350)

    for doc in predictions:
        print(' '.join(str(m) for m in doc))
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', dest='test_file', help='Test corpus to predict on (HDF5).')
    parser.add_argument('--train', dest='train_file', help='Training corpus (HDF5).')
    parser.add_argument('--dev', dest='dev_file', help='Development corpus (HDF5).')
    parser.add_argument('--train-config', dest='train_config', help='Training configuration file.')
    parser.add_argument('--model', dest='model_file', help='File name for the trained model.')
    parser.add_argument('--checkpoint', dest='checkpoint', help='File name stem for training checkpoints.')
    args = parser.parse_args()

    if args.test_file is None and args.train_file is None:
        print('Either --predict or --train is required.', file=sys.stderr)
        sys.exit(1)

    cuda = torch.cuda.is_available()

    logging.basicConfig(stream=sys.stderr, format='%(asctime)-15s %(message)s', level=logging.DEBUG)

    model = None
    if args.train_file:
        model = training_mode(args, cuda)

    if args.test_file:
        test_mode(args, model, cuda)


if __name__ == '__main__':
    main()

