# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import features
import pretrain
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
from util import to_cpu


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


class EpsilonScoringModel(torch.nn.Module):
    def __init__(self, phi_a_size, ha_size, cuda=False):
        super(EpsilonScoringModel, self).__init__()

        self.ha_size = ha_size
        self.ha_model = HModel(phi_a_size, ha_size)

        self.eps_scoring_model = torch.nn.Linear(ha_size, 1)

        if cuda:
            self.factory = util.CudaFactory()
        else:
            self.factory = util.CPUFactory()

    def forward(self, phi_a, batchsize=None):
        nmentions = phi_a.size()[0]

        if batchsize is None:
            batchsize = nmentions

        eps_scores = Variable(self.factory.float_tensor(nmentions))
        h_a = Variable(self.factory.float_tensor(nmentions, self.ha_size))

        for batch_start in range(0, nmentions, batchsize):
            this_batchsize = min(batchsize, nmentions - batch_start)
            this_phi_a = phi_a[batch_start:(batch_start + this_batchsize)]

            this_h_a = self.ha_model(this_phi_a)
            h_a[batch_start:(batch_start + this_batchsize), :] = this_h_a
            eps_scores[batch_start:(batch_start + this_batchsize)] = self.eps_scoring_model(this_h_a)

        return eps_scores, h_a


class AntecedentScoringModel(torch.nn.Module):
    def __init__(self, phi_p_size, hp_size, ha_size, hidden_size=None, dropout=None, cuda=False):
        super(AntecedentScoringModel, self).__init__()

        self.with_cuda = cuda

        self.ha_size = ha_size
        self.hp_size = hp_size
        self.hp_model = HModel(phi_p_size, hp_size)

        layers = []

        if dropout:
            layers.append(torch.nn.Dropout(dropout))

        if hidden_size:
            layers.append(torch.nn.Linear(ha_size + hp_size, hidden_size))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_size, 1))
        else:
            layers.append(torch.nn.Linear(ha_size + hp_size, 1))

        self.ana_scoring_model = torch.nn.Sequential(*layers)

        if cuda:
            self.factory = util.CudaFactory()
        else:
            self.factory = util.CPUFactory()

    def forward(self, h_a, all_phi_p, cand_subset=None, batchsize=None):
        nmentions = h_a.size()[0]
        ncands = all_phi_p.size()[0]

        if batchsize is None:
            batchsize = ncands

        if cand_subset is None:
            cand_subset = self.factory.long_tensor(ncands)
            i = 0
            for j in range(1, nmentions):
                cand_subset[i:(i + j)] = j
                i += j

        ana_scores = Variable(self.factory.float_tensor(ncands))

        for batch_start in range(0, ncands, batchsize):
            this_batchsize = min(batchsize, ncands - batch_start)

            h_combined = Variable(self.factory.float_tensor(this_batchsize, self.ha_size + self.hp_size))

            this_phi_p = all_phi_p[batch_start:(batch_start + this_batchsize), :]
            this_cand_subset = cand_subset[batch_start:(batch_start + this_batchsize)]

            h_combined[:, :self.ha_size] = torch.index_select(h_a, 0, this_cand_subset)
            h_combined[:, self.ha_size:] = self.hp_model(this_phi_p)

            ana_scores[batch_start:(batch_start + this_batchsize)] = self.ana_scoring_model(h_combined)

        return ana_scores


class MentionRankingModel(torch.nn.Module):
    def __init__(self, eps_model, ana_model, cuda=False):
        super(MentionRankingModel, self).__init__()
        self.eps_model = eps_model
        self.ana_model = ana_model
        if cuda:
            self.factory = util.CudaFactory()
        else:
            self.factory = util.CPUFactory()
        self.false_new_cost = None
        self.link_costs = None

    def forward(self):
        # call one of the more specific methods instead
        raise NotImplementedError

    def set_error_costs(self, costs):
        self.false_new_cost = costs['false_new']
        self.link_costs = self.factory.to_device(torch.FloatTensor([[costs['false_link']], [costs['wrong_link']]]))

    def compute_loss(self, doc, batchsize=None):
        t_phi_a = self.factory.to_device(doc.anaphoricity_features.long())
        t_phi_p = self.factory.to_device(doc.pairwise_features.long())
        solution_mask = self.factory.to_device(doc.solution_mask)

        # First do the full computation without gradients
        phi_a = Variable(t_phi_a, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        all_eps_scores, h_a = self.eps_model(phi_a, batchsize=batchsize)
        all_ana_scores = self.ana_model(h_a, phi_p, batchsize=batchsize)
        margin_info = self.find_margin(all_eps_scores.data, all_ana_scores.data, solution_mask)

        best_correct_idx = margin_info['best_correct_idx']
        loss_idx = margin_info['loss_idx']
        cost_values = margin_info['cost_values']
        loss_per_example = margin_info['loss_per_example']

        # Then turn on gradients and run on loss-contributing elements only
        loss_contributing = torch.gt(loss_per_example, 0.0).unsqueeze(1)
        if torch.sum(loss_contributing) == 0:
            return Variable(torch.zeros(1), requires_grad=False)
        loss_contributing_idx = loss_contributing.nonzero()[:, 0]
        n_loss_contributing = loss_contributing_idx.size()[0]

        # In the second run, we just compute the scores for the two elements per example
        # that contribute to the margin loss. At most one of them can be an epsilon score.
        # The scores will be put in an nmentions x 2 matrix. The following code determines
        # which of the entries in this matrix come from the eps and the ana scorer, respectively,
        # and which examples must be fed to each of the scorers.
        cand_idx = torch.stack([best_correct_idx, loss_idx], dim=1)
        example_no = self.factory.long_arange(0, doc.nmentions).unsqueeze(1).expand_as(cand_idx)
        is_epsilon = torch.eq(cand_idx, example_no)
        sub_is_epsilon = is_epsilon[loss_contributing_idx]
        cand_mask = (1 - is_epsilon) * loss_contributing.expand_as(is_epsilon)
        sub_cand_mask = cand_mask[loss_contributing_idx]
        cand_subset = Variable(example_no[:sub_cand_mask.size()[0], :].masked_select(sub_cand_mask))
        example_offsets = torch.cumsum(torch.cat([self.factory.long_zeros(1, 2),
                                                  example_no[:(doc.nmentions - 1), :]]), 0)
        cand_idx_in_doc = cand_idx + example_offsets
        relevant_cands = cand_idx_in_doc[cand_mask]

        # Next, we compute the required scores.
        phi_a = Variable(t_phi_a, volatile=False, requires_grad=False)
        phi_p = Variable(t_phi_p, volatile=False, requires_grad=False)
        sub_phi_a = torch.index_select(phi_a, 0, loss_contributing_idx)
        sub_phi_p = torch.index_select(phi_p, 0, relevant_cands)
        sub_eps_scores, sub_h_a = self.eps_model(sub_phi_a, batchsize=batchsize)
        sub_ana_scores = self.ana_model(sub_h_a, sub_phi_p, cand_subset=cand_subset, batchsize=batchsize)

        # Then we store them in the right components of the scores matrix.
        scores = Variable(self.factory.zeros(n_loss_contributing, 2))
        scores[sub_cand_mask] = sub_ana_scores
        needs_eps = torch.gt(torch.sum(sub_is_epsilon, dim=1), 0)
        if self.factory.get_single(torch.sum(needs_eps)) > 0:
            eps_idx = Variable(example_no[:sub_cand_mask.size()[0], :].masked_select(1 - sub_cand_mask))
            scores[1 - sub_cand_mask] = sub_eps_scores[eps_idx]

        # The applicable rescaling weights can be taken from the first run. We now compute the scores.
        var_cost_values = Variable(cost_values, requires_grad=False)
        sub_loss_per_example = var_cost_values[loss_contributing_idx].squeeze() * (1.0 - scores[:, 0] + scores[:, 1])
        model_loss = to_cpu(torch.sum(sub_loss_per_example))

        # The loss values computed in the first and the second run should be equal, since the second
        # run only serves to obtain the gradients. In rare cases, there seems to be a discrepancy
        # between the scores. This needs more investigation.
        score_diff = abs(self.factory.get_single(model_loss) - self.factory.get_single(margin_info['loss']))
        if score_diff > 1e-4:
            logging.warning('Unexpected score difference: %g' % score_diff)

        return model_loss

    def compute_dev_scores(self, doc, batchsize=None):
        t_phi_a = self.factory.to_device(doc.anaphoricity_features.long())
        t_phi_p = self.factory.to_device(doc.pairwise_features.long())

        phi_a = Variable(t_phi_a, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        all_eps_scores, h_a = self.eps_model(phi_a, batchsize=batchsize)
        all_ana_scores = self.ana_model(h_a, phi_p, batchsize=batchsize)
        solution_mask = self.factory.to_device(doc.solution_mask)
        margin_info = self.find_margin(all_eps_scores.data, all_ana_scores.data, solution_mask)

        scores = margin_info['scores']
        cost_matrix = margin_info['cost_matrix']
        best_correct = margin_info['best_correct']
        cost_values = margin_info['cost_values']

        loss_values = cost_matrix * (1.0 + scores - best_correct.unsqueeze(1).expand_as(scores))
        loss_per_example, loss_idx = torch.max(loss_values, dim=1)

        loss = self.factory.get_single(torch.sum(loss_per_example))
        ncorrect = self.factory.get_single(torch.sum(torch.eq(cost_values, 0.0)))

        return loss, ncorrect

    def find_margin(self, eps_scores, ana_scores, solution_mask):
        # This finds the scores and rescaling weights defining the margin loss for each example.
        # We start by setting up the score matrix.
        scores = self.create_score_matrix(eps_scores, ana_scores)

        # Now find the highest-scoring element in the correct cluster (best_correct)
        # and the highest-scoring element overall.
        # we can't use infinity here because otherwise multiplication by 0 is NaN
        minimum_score = scores.min()
        solution_scores = solution_mask * scores + (1.0 - solution_mask) * minimum_score
        best_correct, best_correct_idx = solution_scores.max(dim=1)
        highest_scoring, highest_scoring_idx = scores.max(dim=1)

        # The following calculations create a tensor in which each component contains the right loss penalty:
        # 0 for correct predictions, cost[0..2] for false link, false new and wrong link, respectively
        non_anaphoric = torch.diag(solution_mask)
        anaphoricity_selector = torch.stack([non_anaphoric, 1 - non_anaphoric], dim=1).float()
        # tril() suppresses cataphoric and self-references
        potential_costs = torch.tril(torch.mm(anaphoricity_selector, self.link_costs.expand(2, scores.size()[1])), -1)
        potential_costs[self.factory.byte_eye(scores.size()[0])] = self.false_new_cost
        cost_matrix = (1.0 - solution_mask) * potential_costs

        # Compute the potential loss for each element and maximise.
        loss_values = cost_matrix * (1.0 + scores - best_correct.unsqueeze(1).expand_as(scores))
        loss_per_example, loss_idx = torch.max(loss_values, dim=1)

        # The cost_values are the rescaling coefficients corresponding to the applicable error type
        # in each example.
        cost_values = torch.gather(cost_matrix, 1, loss_idx.unsqueeze(1))
        loss = torch.sum(loss_per_example)

        # Return all kinds of results because different callers require different details.
        return {
            'scores': scores,
            'best_correct': best_correct,
            'best_correct_idx': best_correct_idx,
            'highest_scoring': highest_scoring,
            'highest_scoring_idx': highest_scoring_idx,
            'cost_matrix': cost_matrix,
            'cost_values': cost_values,
            'loss': loss,
            'loss_idx': loss_idx,
            'loss_per_example': loss_per_example
        }

    def predict(self, doc, batchsize=None):
        t_phi_a = self.factory.to_device(doc.anaphoricity_features.long())
        t_phi_p = self.factory.to_device(doc.pairwise_features.long())

        phi_a = Variable(t_phi_a, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        eps_scores, h_a = self.eps_model(phi_a, batchsize=batchsize)
        ana_scores = self.ana_model(h_a, phi_p, batchsize=batchsize)

        scores = self.create_score_matrix(eps_scores.data, ana_scores.data)

        return to_cpu(scores)

    def create_score_matrix(self, eps_scores, ana_scores):
        # This method takes the complete epsilon and antecedent scores for a document
        # and arranges them in a lower-triangular matrix with one row for each mention,
        # with the score for being non-anaphoric (epsilon) on the main diagonal and
        # the scores for each potential antecedent under the diagonal.
        nmentions = eps_scores.size()[0]

        all_scores = self.factory.zeros(nmentions, nmentions)

        # Put epsilon scores on the main diagonal
        eps_idx = self.factory.byte_eye(nmentions)
        all_scores[eps_idx] = eps_scores

        # Put anaphoric scores in the triangular part below the diagonal
        ana_idx = torch.tril(self.factory.byte_ones(nmentions, nmentions), -1)
        all_scores[ana_idx] = ana_scores

        return all_scores


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


def train(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
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

    # training_set, truncated = training_set.truncate_docs(train_config['maxsize_gpu'])
    # logging.info('Truncated %d/%d documents.' % (truncated, len(training_set)))

    model.set_error_costs(train_config['error_costs'])

    logging.info('Starting training...')
    for epoch in range(train_config['nepochs']):
        model.train()
        train_loss_reg = 0.0
        train_loss_unreg = 0.0
        for i, idx in enumerate(numpy.random.permutation(epoch_size)):
            if (i + 1) % dot_interval == 0:
                print('.', end='', flush=True)

            if training_set[idx].nmentions == 1:
                logging.info('Skipping document with only one mention.')
                continue

            opt.zero_grad()

            model_loss = model.compute_loss(training_set[idx], batchsize=train_config['batchsize'])

            reg_loss = to_cpu(sum(p.abs().sum() for p in model.parameters()))
            loss = model_loss + train_config['l1reg'] * reg_loss

            train_loss_unreg += model_loss.data[0] / training_set[idx].nmentions
            train_loss_reg += loss.data[0] / training_set[idx].nmentions

            loss.backward()
            opt.step()

            del loss
            del model_loss
            del reg_loss

        print(flush=True)

        cpu_model = copy.deepcopy(model).cpu()

        if checkpoint:
            logging.info('Saving checkpoint...')
            with open('%s-%03d' % (checkpoint, epoch), 'wb') as f:
                torch.save(cpu_model.state_dict(), f)

        logging.info('Computing devset performance...')
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        for doc in dev_set:
            loss, ncorrect = model.compute_dev_scores(doc, batchsize=train_config['batchsize'])

            dev_loss += loss
            dev_correct += ncorrect
            dev_total += doc.nmentions

        dev_acc = dev_correct / dev_total
        logging.info('Epoch %d: train_loss_reg %g / train_loss_unreg %g / dev_loss %g / dev_acc %g' %
                     (epoch, train_loss_reg, train_loss_unreg, dev_loss, dev_acc))


def predict(model, test_set, batchsize=None):
    model.eval()
    predictions = []
    for doc in test_set:
        doc_pred = model.predict(doc, batchsize=batchsize)
        n_doc_pred = doc_pred.numpy()
        n_doc_pred[numpy.triu_indices_from(n_doc_pred, 1)] = float('-inf')
        argmax = n_doc_pred.argmax(axis=1)
        predictions.append([x for x in argmax])

    return predictions


def load_net_config(file):
    net_config = {
        'dropout_h_comb': None,
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
        'batchsize': 30000,
        'ha_pretrain': None,
        'hp_pretrain': None
    }

    if file:
        with open(file, 'r') as f:
            util.recursive_dict_update(train_config, json.load(f))

    return train_config


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

    eps_model = EpsilonScoringModel(anaphoricity_fsize, net_config['ha_size'], cuda=cuda)
    ana_model = AntecedentScoringModel(pairwise_fsize,
                                       net_config['hp_size'], net_config['ha_size'],
                                       hidden_size=net_config['g2_size'],
                                       dropout=net_config['dropout_h_comb'],
                                       cuda=cuda)

    model = MentionRankingModel(eps_model, ana_model, cuda=cuda)

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
        if cuda:
            model.cuda()

    logging.info('Loading test data...')
    with h5py.File(args.test_file, 'r') as h5:
        test_set = features.load_from_hdf5(h5)

    logging.info('Predicting...')
    predictions = predict(model, test_set, batchsize=20000)

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
        pretrain.pretraining_mode(args, cuda)
    else:
        model = create_model(args, cuda)

        if args.train_file:
            training_mode(args, model, cuda)

        if args.test_file:
            test_mode(args, model, cuda)


if __name__ == '__main__':
    main()

