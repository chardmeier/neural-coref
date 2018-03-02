# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import features
import pretrain
import util

import argparse
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
    """A feature embedding followed by summation with bias and a tanh nonlinearity."""
    def __init__(self, nfeatures, size_ha):
        super(HModel, self).__init__()
        self.size_ha = size_ha
        self.embedding = torch.nn.EmbeddingBag(nfeatures, size_ha, mode='sum')
        self.bias = torch.nn.Parameter(torch.FloatTensor(size_ha))

    def forward(self, values, offsets):
        batch_size = offsets.size()[0]
        ft_embed = self.embedding(values, offsets)
        out = torch.tanh(ft_embed + self.bias.expand(batch_size, self.size_ha))
        return out


class EpsilonScoringModel(torch.nn.Module):
    """Submodel computing the scores for each mention being non-anaphoric."""
    def __init__(self, phi_a_size, ha_size, cuda=False):
        super(EpsilonScoringModel, self).__init__()

        self.n_embeddings = phi_a_size
        self.ha_size = ha_size
        self.ha_model = HModel(phi_a_size, ha_size)

        self.eps_scoring_model = torch.nn.Linear(ha_size, 1)

        if cuda:
            self.factory = util.CudaFactory()
        else:
            self.factory = util.CPUFactory()

    def forward(self, phi_a, phi_a_offsets, batchsize=None):
        nmentions = phi_a_offsets.size()[0] - 1

        if batchsize is None:
            batchsize = nmentions

        eps_scores = Variable(self.factory.float_tensor(nmentions))
        h_a = Variable(self.factory.float_tensor(nmentions, self.ha_size))

        for batch_start in range(0, nmentions, batchsize):
            this_batchsize = min(batchsize, nmentions - batch_start)
            start_idx = self.factory.get_single(phi_a_offsets[batch_start])
            end_idx = self.factory.get_single(phi_a_offsets[batch_start + this_batchsize])
            this_phi_a = phi_a[start_idx:end_idx]
            this_phi_a_offsets = phi_a_offsets[batch_start:(batch_start + this_batchsize)] - start_idx

            this_h_a = self.ha_model(this_phi_a, this_phi_a_offsets)
            h_a[batch_start:(batch_start + this_batchsize), :] = this_h_a
            eps_scores[batch_start:(batch_start + this_batchsize)] = self.eps_scoring_model(this_h_a)

        return eps_scores, h_a


class AntecedentScoringModel(torch.nn.Module):
    """Submodel computing a score for each antecedent candidate for each mention."""
    def __init__(self, phi_p_size, hp_size, ha_size, hidden_size=None, dropout=None, cuda=False):
        super(AntecedentScoringModel, self).__init__()

        self.with_cuda = cuda

        self.n_embeddings = phi_p_size

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

    def forward(self, h_a, phi_p, phi_p_offsets, cand_subset=None, batchsize=None):
        nmentions = h_a.size()[0]
        ncands = phi_p_offsets.size()[0] - 1

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

            start_idx = self.factory.get_single(phi_p_offsets[batch_start])
            end_idx = self.factory.get_single(phi_p_offsets[batch_start + this_batchsize])
            this_phi_p = phi_p[start_idx:end_idx]
            this_phi_p_offsets = phi_p_offsets[batch_start:(batch_start + this_batchsize)] - start_idx

            this_cand_subset = cand_subset[batch_start:(batch_start + this_batchsize)]

            h_combined[:, :self.hp_size] = self.hp_model(this_phi_p, this_phi_p_offsets)
            h_combined[:, self.hp_size:] = torch.index_select(h_a, 0, this_cand_subset)

            ana_scores[batch_start:(batch_start + this_batchsize)] = self.ana_scoring_model(h_combined)

        return ana_scores


class MentionRankingModel(torch.nn.Module):
    """Main class implementing the mention-ranking model by Wiseman et al. (ACL 2015)."""
    def __init__(self, net_config, eps_model, ana_model, cuda=False, one_based_features=False):
        super(MentionRankingModel, self).__init__()
        self.net_config = net_config
        self.eps_model = eps_model
        self.ana_model = ana_model
        if cuda:
            self.factory = util.CudaFactory()
        else:
            self.factory = util.CPUFactory()
        self.false_new_cost = None
        self.link_costs = None
        self.one_based_features = one_based_features

    def forward(self):
        """The forward method is not implemented in this class because the computations at training
        and test time are rather different. Call predict, compute_loss or compute_dev_scores instead."""
        raise NotImplementedError

    def set_error_costs(self, costs):
        """This sets the error cost for the slack-rescaled loss function. Must be called before training."""
        self.false_new_cost = costs['false_new']
        self.link_costs = self.factory.to_device(torch.FloatTensor([[costs['false_link']], [costs['wrong_link']]]))

    def predict(self, doc, batchsize=None):
        """Prediction method for use at test time. Returns a lower-triangular score matrix."""
        t_phi_a = self.factory.to_device(self._adjust_features(doc.anaphoricity_features.long(), self.eps_model))
        t_phi_a_offsets = self.factory.to_device(doc.anaphoricity_offsets.long())
        t_phi_p = self.factory.to_device(self._adjust_features(doc.pairwise_features.long(), self.ana_model))
        t_phi_p_offsets = self.factory.to_device(doc.pairwise_offsets.long())

        phi_a = Variable(t_phi_a, volatile=True)
        phi_a_offsets = Variable(t_phi_a_offsets, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        phi_p_offsets = Variable(t_phi_p_offsets, volatile=True)

        eps_scores, h_a = self.eps_model(phi_a, phi_a_offsets, batchsize=batchsize)
        ana_scores = self.ana_model(h_a, phi_p, phi_p_offsets, batchsize=batchsize)

        scores = self._create_score_matrix(eps_scores.data, ana_scores.data)

        return to_cpu(scores)

    def compute_loss(self, doc, batchsize=None):
        """Compute the training loss.

        The loss is computed in a two-step procedure that exploits the structure of the objective function,
        whose value only ever depends on two scores per mention (the highest-scoring predicted and the
        highest-scoring correct). In the first step, we run the whole network without computing gradients
        to identify the scores contributing to the loss function. In the second step, we recompute the
        scores for those items only and do backpropagation."""
        t_phi_a = self.factory.to_device(self._adjust_features(doc.anaphoricity_features.long(), self.eps_model))
        t_phi_a_offsets = self.factory.to_device(doc.anaphoricity_offsets.long())
        t_phi_p = self.factory.to_device(self._adjust_features(doc.pairwise_features.long(), self.ana_model))
        t_phi_p_offsets = self.factory.to_device(doc.pairwise_offsets.long())
        solution_mask = self.factory.to_device(doc.solution_mask)

        # First do the full computation without gradients
        phi_a = Variable(t_phi_a, volatile=True)
        phi_a_offsets = Variable(t_phi_a_offsets, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        phi_p_offsets = Variable(t_phi_p_offsets, volatile=True)
        all_eps_scores, h_a = self.eps_model(phi_a, phi_a_offsets, batchsize=batchsize)
        all_ana_scores = self.ana_model(h_a, phi_p, phi_p_offsets, batchsize=batchsize)
        margin_info = self._find_margin(all_eps_scores.data, all_ana_scores.data, solution_mask)

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
        phi_a_offsets = Variable(t_phi_a_offsets, volatile=False, requires_grad=False)
        phi_p = Variable(t_phi_p, volatile=False, requires_grad=False)
        phi_p_offsets = Variable(t_phi_p_offsets, volatile=False, requires_grad=False)

        sub_phi_a, sub_phi_a_offsets = self._select_features(phi_a, phi_a_offsets, loss_contributing_idx)
        sub_phi_p, sub_phi_p_offsets = self._select_features(phi_p, phi_p_offsets, relevant_cands)
        sub_eps_scores, sub_h_a = self.eps_model(sub_phi_a, sub_phi_a_offsets, batchsize=batchsize)
        sub_ana_scores = self.ana_model(sub_h_a, sub_phi_p, sub_phi_p_offsets,
                                        cand_subset=cand_subset, batchsize=batchsize)

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
        # The warning is silenced for nets with dropout until we've implemented consistent dropout masks
        # in the two-stage scoring process.
        score_diff = abs(self.factory.get_single(model_loss) - self.factory.get_single(margin_info['loss']))
        if score_diff > 1e-4 and self.net_config['dropout_h_comb'] is None:
            logging.warning('Unexpected score difference: %g' % score_diff)

        return model_loss

    def compute_dev_scores(self, doc, batchsize=None):
        """Compute scores on the validation set."""
        t_phi_a = self.factory.to_device(self._adjust_features(doc.anaphoricity_features.long(), self.eps_model))
        t_phi_a_offsets = self.factory.to_device(doc.anaphoricity_offsets.long())
        t_phi_p = self.factory.to_device(self._adjust_features(doc.pairwise_features.long(), self.ana_model))
        t_phi_p_offsets = self.factory.to_device(doc.pairwise_offsets.long())

        phi_a = Variable(t_phi_a, volatile=True)
        phi_a_offsets = Variable(t_phi_a_offsets, volatile=True)
        phi_p = Variable(t_phi_p, volatile=True)
        phi_p_offsets = Variable(t_phi_p_offsets, volatile=True)
        all_eps_scores, h_a = self.eps_model(phi_a, phi_a_offsets, batchsize=batchsize)
        all_ana_scores = self.ana_model(h_a, phi_p, phi_p_offsets, batchsize=batchsize)
        solution_mask = self.factory.to_device(doc.solution_mask)
        margin_info = self._find_margin(all_eps_scores.data, all_ana_scores.data, solution_mask)

        scores = margin_info['scores']
        cost_matrix = margin_info['cost_matrix']
        best_correct = margin_info['best_correct']
        cost_values = margin_info['cost_values']

        loss_values = cost_matrix * (1.0 + scores - best_correct.unsqueeze(1).expand_as(scores))
        loss_per_example, loss_idx = torch.max(loss_values, dim=1)

        loss = self.factory.get_single(torch.sum(loss_per_example))
        ncorrect = self.factory.get_single(torch.sum(torch.eq(cost_values, 0.0)))

        return loss, ncorrect

    def _find_margin(self, eps_scores, ana_scores, solution_mask):
        """Find the scores and rescaling weights defining the margin loss for each example."""
        # We start by setting up the score matrix.
        scores = self._create_score_matrix(eps_scores, ana_scores)

        # Now find the highest-scoring element in the correct cluster (best_correct)
        # and the highest-scoring element overall.
        # we can't use infinity here because otherwise multiplication by 0 is NaN
        minimum_score = scores.min()
        solution_scores = solution_mask * scores + (1.0 - solution_mask) * minimum_score
        best_correct, best_correct_idx = solution_scores.max(dim=1)
        highest_scoring, highest_scoring_idx = scores.max(dim=1)

        # The following calculations create a tensor in which each component contains the right loss penalty:
        # 0 for correct predictions, self.link_cost[0..1] for false link and wrong link, and
        # self.false_new_cost for false new, respectively.
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

    def _create_score_matrix(self, eps_scores, ana_scores):
        """Combine the output of the anaphoric and non-anaphoric submodels into a score matrix.

        This method takes the complete epsilon and antecedent scores for a document
        and arranges them in a lower-triangular matrix with one row for each mention,
        with the score for being non-anaphoric (epsilon) on the main diagonal and
        the scores for each potential antecedent under the diagonal."""
        nmentions = eps_scores.size()[0]

        all_scores = self.factory.zeros(nmentions, nmentions)

        # Put epsilon scores on the main diagonal
        eps_idx = self.factory.byte_eye(nmentions)
        all_scores[eps_idx] = eps_scores

        # Put anaphoric scores in the triangular part below the diagonal
        ana_idx = torch.tril(self.factory.byte_ones(nmentions, nmentions), -1)
        all_scores[ana_idx] = ana_scores

        return all_scores

    def _adjust_features(self, values, model):
        """Adjust the feature values if the weight matrix has been trained with one-based indexing.

        This is for debugging purposes only and allows the model to use weight matrices trained with
        Sam Wiseman's original Torch implementation."""
        if not self.one_based_features:
            return values
        else:
            # This is just for debugging, to be compatible with Wiseman's Torch code
            return (values - 1) % model.n_embeddings

    def _select_features(self, values, offsets, indices):
        """This is like torch.index_select, but works on a value/offset tensor pair with variable-length items."""
        start_offsets = torch.index_select(offsets, 0, indices)
        end_offsets = torch.index_select(offsets, 0, indices + 1)
        out_values = torch.cat([values[self.factory.get_single(a):self.factory.get_single(b)]
                                for a, b in zip(start_offsets, end_offsets)])
        zero = Variable(offsets.data.new(1).zero_(), requires_grad=False)
        out_offsets = torch.cat([zero, torch.cumsum(end_offsets - start_offsets, 0)])
        return out_values, out_offsets


def init_parameters(model, pretrained=None):
    """Initialise mention-ranking model parameters, randomly or with pretrained weights."""

    if not pretrained:
        pretrained = {}

    for name, p in model.named_parameters():
        if name in pretrained:
            p.data = pretrained[name]
        else:
            # Sparse initialisation similar to Sutskever et al. (ICML 2013)
            # For tanh units, use std 0.25 and set biases to 0.5
            if name.endswith('bias'):
                nn_init.constant(p, 0.5)
            else:
                util.sparse(p, sparsity=0.1, std=0.25)


def train(model, train_config, training_set, dev_set, checkpoint=None, cuda=False):
    """Main training loop."""
    epoch_size = len(training_set)
    dot_interval = max(epoch_size // 80, 1)
    logging.info('%d documents per epoch' % epoch_size)

    if cuda:
        model = model.cuda()

    embedding_lr, deep_lr = train_config['learning_rate']
    embedding_layers = []
    deep_layers = []
    logging.info('Learning rates:')
    for name, p in model.named_parameters():
        if name.startswith('eps_model.ha_model.') or name.startswith('ana_model.hp_model.'):
            logging.info('%g  %s  (embedding)' % (embedding_lr, name))
            embedding_layers.append(p)
        else:
            logging.info('%g  %s' % (deep_lr, name))
            deep_layers.append(p)

    opt_params = [
        {
            'params': embedding_layers,
            'lr': embedding_lr
        },
        {
            'params': deep_layers,
            'lr': deep_lr
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

        if checkpoint:
            logging.info('Saving checkpoint...')
            with h5py.File('%s-%03d' % (checkpoint, epoch), 'w') as h5:
                util.save_model(h5, model)

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
    """Output predictions for a test set."""
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
    """Load network configuration from a file. Set default values here."""
    net_config = {
        'dropout_h_comb': None,
        'ha_size': 128,
        'hp_size': 700,
        'g2_size': None,
        'one_based_features': False
    }

    if file:
        with open(file, 'r') as f:
            util.recursive_dict_update(net_config, json.load(f))

    return net_config


def load_train_config(file):
    """Load training configuration from a file. Set default values here."""
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


def setup_model(net_config, cuda):
    """Create a MentionRankingModel object for a given network configuration."""
    eps_model = EpsilonScoringModel(net_config['anaphoricity_fsize'], net_config['ha_size'], cuda=cuda)
    ana_model = AntecedentScoringModel(net_config['pairwise_fsize'],
                                       net_config['hp_size'], net_config['ha_size'],
                                       hidden_size=net_config['g2_size'],
                                       dropout=net_config['dropout_h_comb'],
                                       cuda=cuda)

    model = MentionRankingModel(net_config, eps_model, ana_model, cuda=cuda,
                                one_based_features=net_config['one_based_features'])

    return model


def create_model(args, cuda):
    """Create a MentionRankingModel object from command-line arguments."""
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

    net_config['anaphoricity_fsize'] = anaphoricity_fsize
    net_config['pairwise_fsize'] = pairwise_fsize

    model = setup_model(net_config, cuda)

    return model


def training_mode(args, model, cuda):
    """Do all that needs to be done for training the model."""
    train_config = load_train_config(args.train_config)
    print('train_config ' + json.dumps(train_config), file=sys.stderr)

    logging.info('Loading training data...')
    with h5py.File(args.train_file, 'r') as h5:
        training_set = features.load_from_hdf5(h5)

    logging.info('Loading development data...')
    with h5py.File(args.dev_file, 'r') as h5:
        dev_set = features.load_from_hdf5(h5)

    if train_config['pretrain']:
        logging.info('Loading pretrained weights...')
        with h5py.File(train_config['pretrain'], 'r') as h5:
            pretrained_params = {name: torch.from_numpy(numpy.array(p)) for name, p in h5.items()
                                 if name.startswith('eps_model.ha_model.') or name.startswith('ana_model.hp_model.')}
    else:
        pretrained_params = None

    logging.info('Initialising parameters...')
    init_parameters(model, pretrained=pretrained_params)

    logging.info('Training model...')
    train(model, train_config, training_set, dev_set, checkpoint=args.checkpoint, cuda=cuda)

    if args.model_file:
        logging.info('Saving model...')
        with h5py.File(args.model_file, 'w') as h5:
            util.save_model(h5, model)

    return model


def test_mode(args, model, cuda):
    """Do all that needs to be done for prediction."""
    if args.model_file:
        logging.info('Loading model...')
        with h5py.File(args.model_file, 'r') as h5:
            model = util.load_model(h5, cuda)

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

