import sh
import itertools
import os
import multiprocessing as mp
import re
import logging
import operator
import random
import datetime
from functools import partial
from collections import OrderedDict

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

def get_best_model_name(hyperparams, script_name):
    args, kwargs = hyperparams
    args = [''.join(s.split('/')) for s in args]
    script_name = ''.join(script_name.split('/'))
    kv = ['%s=%s' % (k, ''.join(str(kwargs[k]).split('/'))) for k in sorted(kwargs.keys())]
    return '-'.join([script_name] + args + kv) + '.pt'

# The GPU IDs to run on.  The same number of workers would spawn, one for each GPU ID specified.
gpu_ids = [0, 1, 2, 3]

# The script to run
#script = 'main_fism.py'
script = 'main_knn.py'

# Number of combinations to try, or None for complete grid search
n_combinations = None

date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# file that stores the metric output
outfile_path = 'result.' + script + '.log.' + date_str

# This is the hyperparameter grid.
# Each combination is passed into the script file as shell arguments.
hyperparam_grid = {
        'batch-size': [1024],
        'feature-size': [64, 128, 256],
        'weight-decay': [0],
        'lr': [1e-3, 1e-4],
        'num-workers': [8],
        'id-as-feature': [False, True],
        'n-negs': [20],
        'n-neighbors': [3, 10],
        'n-layers': [0, 1],
        'n-epoch': [20],
        'pretrain': [False],
        'n-traces': [10],
        'trace-len': [3],
        'neg-by-freq': [False, True],
        'neg-freq-max': [100],
        'neg-freq-min': [1],
        'data-pickle': ['bt.pkl'],
        'data-path': ['../DGL-RS/datasets/bio-techne'],
        'dataset': ['biotechne'],
        #'alpha': [1],
        }

def work(hyperparams, script_name, regex, better):
    identity = mp.current_process()._identity
    gpu_id = gpu_ids[(identity[0] - 1) if len(identity) > 0 else 0]
    newenv = os.environ.copy()
    newenv['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    filename = 'output.log.' + str(os.getpid())
    logger.info('Started %s on PID %d and GPU %d' % (hyperparams, os.getpid(), gpu_id))

    args, kwargs = hyperparams
    best = None
    model_path = 'model.pt.' + str(os.getpid())
    best_model_path = 'best_' + model_path
    kwargs['model-path'] = model_path
    sh.rm('-f', model_path)

    n_epoch = 0
    for l in sh.python3('-u', script_name, *args, _env=newenv, _iter=True, **kwargs):
        m = re.search(regex, l)
        if m is None:
            continue
        metric = float(m.group(1))
        if best is None or better(metric, best[0]):
            best = metric, l, n_epoch
            sh.cp(model_path, best_model_path)
        logger.info('PID %d %s Epoch %d: %s' % (os.getpid(), identity, n_epoch, l.strip()))
        n_epoch += 1

    logger.info('Finished %s on PID %d and GPU %d with best metric %s' % (hyperparams, os.getpid(), gpu_id, best))
    sh.cp(best_model_path, get_best_model_name(hyperparams, script_name))
    sh.rm(best_model_path)
    return hyperparams, best


def hyperparam_iterator(grid, n_combinations=None):
    grid = OrderedDict(grid)
    keys = list(grid.keys())
    values = list(grid.values())

    if n_combinations is None:
        product_iter = itertools.product(*values)
    else:
        product_iter = list(itertools.product(values))
        random.shuffle(product_iter)
        product_iter = product_iter[:n_combinations]

    for config in product_iter:
        args = []
        kwargs = {}
        for k, v in zip(keys, config):
            if isinstance(v, bool):
                if v:
                    args.append('--' + k)
            else:
                kwargs[k] = v
        yield args, kwargs

outfile = open(outfile_path, 'w')
with mp.Pool(len(gpu_ids)) as p:
    result = p.imap(
            partial(
                work,
                script_name=script,
                regex=r'HITS@10: ([0-9.]+)',
                better=operator.gt),
            hyperparam_iterator(hyperparam_grid, n_combinations))
    print(list(result), file=outfile)
#for hp in hyperparam_iterator(hyperparam_grid, n_combinations):
#    work(hp, script_name=script, regex=r'HTS@10: ([0-9.]+)', better=operator.gt)
outfile.close()
# One can read the result by the following:
# with open(outfile_path) as f:
#     result = eval(f.read())
