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
        best_correct = solution_scores.max(dim=1)[0].unsqueeze(1)

        # The following calculations create a tensor in which each component contains the right loss penalty:
        # 0 for correct predictions, cost[0..2] for false link, false new and wrong link, respectively
        non_anaphoric = torch.diag(solution_mask)
        anaphoricity_selector = torch.stack([non_anaphoric, 1 - non_anaphoric], dim=1).float()
        # tril() suppresses cataphoric and self-references
        potential_costs = torch.tril(torch.mm(anaphoricity_selector, self.link_costs.expand(2, scores.size()[1])), -1)
        potential_costs[torch.eye(scores.size()[0]).byte()] = self.false_new_cost
        cost_matrix = (1.0 - solution_mask) * potential_costs

        loss_values = cost_matrix * (1.0 + scores - best_correct.expand_as(scores))
        loss_per_example = torch.max(loss_values, dim=1)[0]
        loss = torch.sum(loss_per_example)

        return loss


class AntecedentRankingPretrainingModel(torch.nn.Module):
    def __init__(self, phi_p_size, hp_size):
        super(AntecedentRankingPretrainingModel, self).__init__()
        self.hp_model = HModel(phi_p_size, hp_size)
        self.ana_scoring_model = torch.nn.Linear(hp_size, 1)

    # phi_p can be a CUDA tensor, costs and sizes should always be on the CPU
    # result is on the CPU
    def forward(self, phi_p, solutions, sizes):
        h_p = self.hp_model(phi_p)
        ana_scores = self.ana_scoring_model(h_p).cpu()

        loss = Variable(torch.zeros(1))
        idx = 0
        for sol, sz in zip(solutions, sizes):
            m_scores = ana_scores[idx:(idx + sz)]
            idx = idx + sz

            best_score, best_idx = torch.max(m_scores, 0)
            if not sol[best_idx].data[0]:
                best_correct = torch.max(m_scores[sol])
                loss += 1.0 + best_correct - best_score

        return loss


def init_parameters(model, ha_pretrain=None, hp_pretrain=None):
    pretrained_params = {}

    if ha_pretrain:
        for name, p in ha_pretrain.items():
            if name.startswith('ha_model.'):
                pretrained_params[name] = p

    if hp_pretrain:
        for name, p in hp_pretrain.items():
            if name.startswith('hp_model.'):
                pretrained_params[name] = p

    for name, p in model.named_parameters():
        if name in pretrained_params:
            p.data = pretrained_params[name]
        else:
            # Sparse initialisation similar to Sutskever et al. (ICML 2013)
            # For tanh units, use std 0.25 and set biases to 0.5
            if name.endswith('bias'):
                nn_init.constant(p, 0.5)
            else:
                util.sparse(p, sparsity=0.1, std=0.25)


def filter_for_pretrain_hp(corpus):
    feats = []
    sizes = []
    solutions = []
    for i, doc in enumerate(corpus):
        ncands = doc.nmentions * (doc.nmentions - 1) // 2
        anaphoric_mask = torch.zeros(ncands).long()
        doc_solutions = []
        doc_sizes = []
        k = 0
        for j in range(doc.nmentions):
            if doc.is_anaphoric(j):
                anaphoric_mask[k:(k + j)] = 1
                doc_sizes.append(j)
                sol = torch.ByteTensor(j).zero_()
                cluster_id = doc.mention_to_opc[j]
                for l in doc.opc_clusters[cluster_id]:
                    if l >= j:
                        break
                    sol[l] = 1
                doc_solutions.append(sol)
            k = k + j

        filtered_phi_p = doc.pairwise_features[anaphoric_mask, :].long()

        feats.append(filtered_phi_p)
        sizes.append(doc_sizes)
        solutions.append(doc_solutions)

    return feats, sizes, solutions


def pretrain_hp(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
    epoch_size = len(training_set)

    dot_interval = max(epoch_size // 80, 1)
    logging.info('%d documents per epoch' % epoch_size)

    opt = torch.optim.Adagrad(params=model.parameters(), lr=train_config['learning_rate'][1])

    logging.info('Filtering corpora for pretraining...')
    train_features, train_sizes, train_solutions = filter_for_pretrain_hp(training_set)
    dev_features, dev_sizes, dev_solutions = filter_for_pretrain_hp(dev_set)

    logging.info('Starting training...')
    for epoch in range(train_config['nepochs']):
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)

            if len(train_sizes[idx]) == 0:
                # no anaphoric mentions in document
                continue

            opt.zero_grad()

            if cuda:
                phi_p = Variable(train_features[idx].pin_memory()).cuda(async=True)
            else:
                phi_p = Variable(train_features[idx])

            solutions = [Variable(sol) for sol in train_solutions[idx]]
            model_loss = model(phi_p, solutions, train_sizes[idx])

            reg_loss = sum(p.abs().sum() for p in model.parameters()).cpu()
            loss = model_loss + train_config['l1reg'] * reg_loss

            train_loss_unreg += model_loss.data[0] / len(train_sizes[idx])
            train_loss_reg += loss.data[0] / len(train_sizes[idx])

            loss.backward()
            opt.step()

            del loss
            del model_loss
            del reg_loss

        print(flush=True)

        if cuda:
            cpu_model = copy.deepcopy(model).cpu()
        else:
            cpu_model = model

        if checkpoint:
            logging.info('Saving checkpoint...')
            with open('%s-%03d' % (checkpoint, epoch), 'wb') as f:
                torch.save(cpu_model.state_dict(), f)

        logging.info('Computing devset performance...')
        dev_loss = 0.0
        for docft, docsz, docsol in zip(dev_features, dev_sizes, dev_solutions):
            if cuda:
                phi_p = Variable(docft.pin_memory(), volatile=True).cuda(async=True)
            else:
                phi_p = Variable(docft, volatile=True)

            solutions = [Variable(sol, volatile=True) for sol in docsol]
            dev_loss += model(phi_p, solutions, docsz).data[0]

        logging.info('Epoch %d: train_loss_reg %g / train_loss_unreg %g / dev_loss %g' %
                     (epoch, train_loss_reg, train_loss_unreg, dev_loss))


def train(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
    loss_fn = MentionRankingLoss(train_config['error_costs'])

    epoch_size = len(training_set)
    dot_interval = max(epoch_size // 80, 1)
    logging.info('%d documents per epoch' % epoch_size)

    if cuda:
        model = model.cuda()

    embedding_layers = []
    deep_layers = []
    for name, p in model.named_parameters():
        if name.startswith('ha_model.') or name.startswith('hp_model.'):
            embedding_layers.append(p)
        else:
            deep_layers.append(p)

    opt_params = [
        {
            'params': embedding_layers,
            'lr': train_config['learning_rate'][0]
        },
        {
            'params': deep_layers,
            'lr': train_config['learning_rate'][1]
        }
    ]

    opt = torch.optim.Adagrad(params=opt_params)

    training_set, truncated = training_set.truncate_docs(train_config['maxsize_gpu'])
    logging.info('Truncated %d/%d documents.' % (truncated, len(training_set)))

    logging.info('Starting training...')
    for epoch in range(train_config['nepochs']):
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)

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

        if cuda:
            cpu_model = copy.deepcopy(model).cpu()
        else:
            cpu_model = model

        if checkpoint:
            logging.info('Saving checkpoint...')
            with open('%s-%03d' % (checkpoint, epoch), 'wb') as f:
                torch.save(cpu_model.state_dict(), f)

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


def load_net_config(file):
    net_config = {
        'ha_size': 128,
        'hp_size': 700,
        'g2_size': None
    }

    if file:
        with open(file, 'r') as f:
            util.recursive_dict_update(net_config, json.load(f))

    return net_config


def load_train_config(file):
    train_config = {
        'nepochs': 100,
        'l1reg': 0.001,
        'learning_rate': [0.01, 0.01], # for embedding layers and others
        'error_costs': {
            'false_link': 0.5,
            'false_new': 1.2,
            'wrong_link': 1.0
        },
        'maxsize_gpu': 230,
        'ha_pretrain': None,
        'hp_pretrain': None
    }

    if file:
        with open(file, 'r') as f:
            util.recursive_dict_update(train_config, json.load(f))

    return train_config


def pretraining_mode(args, cuda):
    net_config = load_net_config(args.net_config)
    print('net_config ' + json.dumps(net_config), file=sys.stderr)

    train_config = load_train_config(args.train_config)
    print('train_config ' + json.dumps(train_config), file=sys.stderr)

    logging.info('Loading training data...')
    with h5py.File(args.train_file, 'r') as h5:
        training_set = features.load_from_hdf5(h5)

    logging.info('Loading development data...')
    with h5py.File(args.dev_file, 'r') as h5:
        dev_set = features.load_from_hdf5(h5)

    pairwise_fsize = len(training_set.pairwise_fmap)

    model = AntecedentRankingPretrainingModel(pairwise_fsize, net_config['hp_size'])
    pretrain_hp(model, train_config, training_set, dev_set, checkpoint=args.checkpoint, cuda=cuda)

    if args.model_file:
        logging.info('Saving model...')
        with open(args.model_file, 'wb') as f:
            torch.save(model.state_dict(), f)

    return model


def create_model(args, cuda):
    net_config = load_net_config(args.net_config)
    print('net_config ' + json.dumps(net_config), file=sys.stderr)

    if args.train_file:
        h5_file = args.train_file
    elif args.test_file:
        h5_file = args.test_file
    elif args.dev_file:
        h5_file = args.dev_file
    else:
        logging.error('Cannot determine vocabulary size without corpus.')
        sys.exit(1)

    with h5py.File(h5_file, 'r') as h5:
        anaphoricity_fsize, pairwise_fsize = features.vocabulary_sizes_from_hdf5(h5)

    model = MentionRankingModel(anaphoricity_fsize, pairwise_fsize,
                                net_config['ha_size'], net_config['hp_size'],
                                hidden_size=net_config['g2_size'],
                                cuda=cuda)

    return model


def training_mode(args, model, cuda):
    train_config = load_train_config(args.train_config)
    print('train_config ' + json.dumps(train_config), file=sys.stderr)

    logging.info('Loading training data...')
    with h5py.File(args.train_file, 'r') as h5:
        training_set = features.load_from_hdf5(h5)

    logging.info('Loading development data...')
    with h5py.File(args.dev_file, 'r') as h5:
        dev_set = features.load_from_hdf5(h5)

    if train_config['ha_pretrain']:
        logging.info('Loading pretrained weights for h_a layer...')
        with open(train_config['ha_pretrain'], 'rb') as f:
            ha_pretrain = torch.load(f)
    else:
        ha_pretrain = None

    if train_config['hp_pretrain']:
        logging.info('Loading pretrained weights for h_p layer...')
        with open(train_config['hp_pretrain'], 'rb') as f:
            hp_pretrain = torch.load(f)
    else:
        hp_pretrain = None

    logging.info('Initialising parameters...')
    init_parameters(model, ha_pretrain=ha_pretrain, hp_pretrain=hp_pretrain)

    logging.info('Training model...')
    train(model, train_config, training_set, dev_set, checkpoint=args.checkpoint, cuda=cuda)

    if args.model_file:
        logging.info('Saving model...')
        with open(args.model_file, 'wb') as f:
            torch.save(model.state_dict(), f)

    return model


def test_mode(args, model, cuda):
    if args.model_file:
        logging.info('Loading model...')
        with open(args.model_file, 'rb') as f:
            model.load_state_dict(torch.load(f))

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
    parser.add_argument('--pretrain-hp', dest='pretrain_hp', action='store_true',
                        help='Pretrain h_p model instead of training full model.')
    parser.add_argument('--train-config', dest='train_config', help='Training configuration file.')
    parser.add_argument('--net-config', dest='net_config', help='Network configuration file.')
    parser.add_argument('--model', dest='model_file', help='File name for the trained model.')
    parser.add_argument('--checkpoint', dest='checkpoint', help='File name stem for training checkpoints.')
    args = parser.parse_args()

    if args.test_file is None and args.train_file is None:
        print('Either --predict or --train is required.', file=sys.stderr)
        sys.exit(1)

    cuda = torch.cuda.is_available()

    logging.basicConfig(stream=sys.stderr, format='%(asctime)-15s %(message)s', level=logging.DEBUG)

    if args.pretrain_hp:
        pretraining_mode(args, cuda)
    else:
        model = create_model(args, cuda)

        if args.train_file:
            training_mode(args, model, cuda)

        if args.test_file:
            test_mode(args, model, cuda)


if __name__ == '__main__':
    main()

