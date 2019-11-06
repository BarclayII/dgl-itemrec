import sh
import itertools
import os
import multiprocessing as mp
import re
import logging
import operator
import random
from functools import partial
from collections import OrderedDict

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

def work(hyperparams, script_name, regex, better):
    identity = mp.current_process()._identity
    gpu_id = mp.current_process()._identity[0] if len(identity) > 0 else 1
    newenv = os.environ.copy()
    newenv['CUDA_VISIBLE_DEVICES'] = str(gpu_id - 1)

    filename = 'output.log.' + str(os.getpid())
    logger.info('Started %s on PID %d and GPU %d' % (hyperparams, os.getpid(), gpu_id))

    args, kwargs = hyperparams

    sh.python3('-u', script_name, *args, _out=filename, _env=newenv, **kwargs)
    logger.info('Finished %s on PID %d and GPU %d' % (hyperparams, os.getpid(), gpu_id))
    best = None

    with open(filename) as f:
        for l in f:
            m = re.search(regex, l)
            if m is None:
                continue
            metric = float(m.group(1))
            if best is None or better(metric, best):
                best = metric

    logger.info('Completed %s with best metric %s' % (hyperparams, best))
    return hyperparams, best


hyperparam_grid = {
        'batch-size': [1024],
        'feature-size': [8, 16, 32],
        'weight-decay': [1e-2, 1e-3, 1e-4],
        'lr': [3e-5],
        'num-workers': [4],
        'id-as-feature': [True],
        'n-negs': [4],
        'n-neighbors': [2, 3, 5, 7],
        'n-layers': [0, 1, 2],
        'n-epoch': [20]}

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

script = 'main_fism.py'
#script = 'main_knn.py'
outfile = open('result.' + script + '.log', 'w')
with mp.Pool(4) as p:
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
