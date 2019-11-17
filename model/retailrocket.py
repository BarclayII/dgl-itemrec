import os
import pickle
import numpy as np
import pandas as pd
import tqdm
import torch
from .utils import ComplementView

class RetailRocket(object):
    def __init__(self, path, neg_size=99):
        with open(os.path.join(path, 'retail_item_feats_100.pkl'), 'rb') as f:
            x = pickle.load(f)

        train_uv = pd.read_csv(os.path.join(path, 'train_uv.csv'), sep='\t')
        test_uv = pd.read_csv(os.path.join(path, 'test_uv.csv'), sep='\t')
        valid_uv = pd.read_csv(os.path.join(path, 'val_uv.csv'), sep='\t')

        self.users_train = train_uv['visitorid'].values.copy()
        self.movies_train = train_uv['itemid'].values.copy()
        valid_uv_target = valid_uv.groupby('visitorid').apply(lambda x: x.iloc[-1])
        test_uv_target = test_uv.groupby('visitorid').apply(lambda x: x.iloc[-1])

        all_uv = pd.concat([train_uv, test_uv, valid_uv], 0)

        self.users_valid = valid_uv_target['visitorid'].values.copy()
        self.movies_valid = valid_uv_target['itemid'].values.copy()
        self.users_test = test_uv_target['visitorid'].values.copy()
        self.movies_test = test_uv_target['itemid'].values.copy()

        self.train_size = len(self.users_train)
        self.valid_size = len(self.users_valid)
        self.test_size = len(self.users_test)

        assert all_uv['visitorid'].nunique() == all_uv['visitorid'].max() + 1
        assert all_uv['itemid'].nunique() == all_uv['itemid'].max() + 1
        self.num_users = all_uv['visitorid'].max() + 1
        self.num_movies = all_uv['itemid'].max() + 1

        self.neg_size = neg_size
        self._pos_train = [None] * self.num_users
        self.neg_valid = np.zeros((self.num_users, neg_size), dtype='int64')

        for u in tqdm.trange(self.num_users):
            interacted_movies = all_uv[all_uv['visitorid'] == u]['itemid'].values.copy()
            self._pos_train[u] = interacted_movies
            neg_samples = np.setdiff1d(np.arange(self.num_movies), interacted_movies)
            self.neg_valid[u] = np.random.choice(neg_samples, neg_size)

        all_query = all_uv.groupby('visitorid').apply(lambda x: x.iloc[-2])
        past_items = all_uv.groupby('visitorid').apply(lambda x: x.iloc[:-1])
        self.ctx_users = past_items['visitorid'].values.copy()
        self.ctx_movies = past_items['itemid'].values.copy()
        self.user_latest_item = {r['visitorid']: r['itemid'] for _, r in all_query.iterrows()}

        self.movie_data = {'x': torch.FloatTensor(x)}
        self.movie_count = train_uv['itemid'].value_counts().sort_index().values.copy()
        assert len(self.movie_count) == self.num_movies
        self.neg_train = ComplementView(self._pos_train, self.num_movies)

    @property
    def neg_test(self):
        return self.neg_valid

    @property
    def neg_test_complete(self):
        return self.neg_train
