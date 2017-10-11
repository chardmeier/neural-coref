# TODO: The pretraining code is currently untested and outdated and unlikely to work.
import features

import copy
import h5py
import json
import logging
import numpy
import sys
import torch

from mention_ranking import HModel, load_net_config, load_train_config
from torch.autograd import Variable
from util import to_cpu


class AntecedentRankingPretrainingLoss(torch.nn.Module):
    def forward(self, scores, solution_mask):
        # we can't use infinity here because otherwise multiplication by 0 is NaN
        minimum_score = scores.min()
        solution_scores = solution_mask * scores + (1.0 - solution_mask) * minimum_score.expand_as(scores)
        best_correct = solution_scores.max(dim=1)[0]

        # Pretraining loss penalty:
        # 0 for correct predictions and for errors involving non-anaphoric mentions,
        # 1 for incorrectly linked anaphoric mentions
        cost_matrix = torch.tril(1.0 - solution_mask, -1)
        non_anaphoric = torch.diag(solution_mask)
        cost_matrix[non_anaphoric, :] = 0.0

        loss = torch.sum(cost_matrix * (1.0 + scores - best_correct.expand_as(scores)))

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
        ana_scores = to_cpu(self.ana_scoring_model(h_p))

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

            reg_loss = to_cpu(sum(p.abs().sum() for p in model.parameters()))
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


