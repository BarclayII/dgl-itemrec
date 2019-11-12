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

gpu_ids = [1, 2, 3]

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


# This is the hyperparameter grid.
# Each combination is passed into the script file as shell arguments.
hyperparam_grid = {
        'batch-size': [1024],
        'feature-size': [16],
        'weight-decay': [1e-3],
        'lr': [1e-5],
        'num-workers': [8],
        'id-as-feature': [False],
        'n-negs': [20],
        'n-neighbors': [3, 10, 20],
        'n-layers': [1, 2],
        'n-epoch': [6],
        'pretrain': [False],
        'n-traces': [10, 50, 100],
        'trace-len': [1, 2, 3],
        #'data-pickle': ['bx.pkl'],
        #'data-path': ['../bookcrossing'],
        #'dataset': ['bx'],
        #'alpha': [1],
        }

def hyperparam_iterator(grid, sel=None):
    grid = OrderedDict(grid)
    keys = list(grid.keys())
    values = list(grid.values())

    if sel is None:
        product_iter = itertools.product(*values)
    else:
        product_iter = list(itertools.product(values))
        random.shuffle(product_iter)
        product_iter = product_iter[:sel]

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

#script = 'main_fism.py'
script = 'main_knn.py'
date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
outfile = open('result.' + script + '.log.' + date_str, 'w')
with mp.Pool(len(gpu_ids)) as p:
    result = p.imap(
            partial(
                work,
                script_name=script,
                regex=r'HITS@10: ([0-9.]+)',
                better=operator.gt),
            hyperparam_iterator(hyperparam_grid))
    print(list(result), file=outfile)
#for hp in hyperparam_iterator(hyperparam_grid):
#    print(work(hp, script, r'HITS@10: ([0-9.]+)', operator.gt))
outfile.close()
