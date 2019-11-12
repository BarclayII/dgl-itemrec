import pickle
import torch
import os
import numpy as np

class Yelp2018(object):
    def __init__(self, path):
        with open(os.path.join(path, 'yelp2018_entity_embed_features.pkl'), 'rb') as f:
            x_emb = pickle.load(f)
        with open(os.path.join(path, 'yelp2018_entity_id_features.pkl'), 'rb') as f:
            x_id = pickle.load(f)
        with open(os.path.join(path, 'yelp2018_eval.pkl'), 'rb') as f:
            valid, test = pickle.load(f)
        with open(os.path.join(path, 'yelp2018_neg.pkl'), 'rb') as f:
            valid_neg, test_neg = pickle.load(f)
        with open(os.path.join(path, 'yelp2018_orig_train.pkl'), 'rb') as f:
            train = pickle.load(f)

        train = train.tocoo()
        self.users_train, self.movies_train = train.row, train.col
        self.users_valid = np.arange(train.shape[0])
        self.movies_valid = valid.astype('int64')
        self.users_test = np.arange(train.shape[0])
        self.movies_test = test.astype('int64')
        self.train_size = len(self.users_train)
        self.valid_size = len(self.users_valid)
        self.test_size = len(self.users_test)
        self.num_users, self.num_movies = num_users, num_movies = train.shape

        train = train.tocsr()
        self.neg_train = [None] * num_users
        self.neg_valid = valid_neg
        self.neg_test = test_neg
        self.neg_size = valid_neg.shape[1]

        for u in range(num_users):
            interacted_movies = train[u].nonzero()[1]
            neg_samples = np.setdiff1d(np.arange(num_movies), interacted_movies)
            self.neg_train[u] = neg_samples

        self.movie_data = {
                'x_emb': x_emb,
                'x_id': x_id,
                }
        self.user_latest_item = None
        self.movie_count = train.sum(0).A.squeeze().astype('int64')
