# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import os
import sys


def get_scores(fname):
    with open(fname, 'r') as f:
        in_totals = False
        for line in f:
            if line == '====== TOTALS =======\n':
                in_totals = True
            if in_totals and line.startswith('Coreference:'):
                tok = line.rstrip('\n').split()
                p = float(tok[10].rstrip('%'))
                r = float(tok[5].rstrip('%'))
                f = float(tok[12].rstrip('%'))
                return p, r, f

    print('No totals found in ' + fname, file=sys.stderr)
    return 0, 0, 0


def main():
    if len(sys.argv) != 2:
        print('Usage: sumup.py [dev|test]', file=sys.stderr)
        sys.exit(1)

    dataset = sys.argv[1]

    if dataset in {'dev', 'test'}:
        path = '/homeappl/home/chardmei/coref/neural-coref/exp/' + dataset
    else:
        print('Unknown dataset: ' + dataset, file=sys.stderr)
        sys.exit(1)

    scores = {}

    for fname in os.listdir(path):
        exp, ext = os.path.splitext(fname)
        if ext in {'.muc', '.bcub', '.ceafe'}:
            score_type = ext.lstrip('.')
            if exp not in scores:
                scores[exp] = {'exp': exp}
            p, r, f = get_scores(os.path.join(path, fname))
            scores[exp].update({score_type + '_p': p, score_type + '_r': r, score_type + '_f': f})

    print('                            MUC________________   B-Cube_____________   CEAF-E_____________   CoNLL')
    for exp, vals in sorted(scores.items()):
        vals['conll'] = (vals['muc_f'] + vals['bcub_f'] + vals['ceafe_f']) / 3
        print('%25s   %5.2f  %5.2f  %5.2f   %5.2f  %5.2f  %5.2f   %5.2f  %5.2f  %5.2f   %5.2f' %
              tuple(vals[x] for x in ['exp',
                                      'muc_p', 'muc_r', 'muc_f',
                                      'bcub_p', 'bcub_r', 'bcub_f',
                                      'ceafe_p', 'ceafe_r', 'ceafe_f',
                                      'conll']))


if __name__ == '__main__':
    main()
