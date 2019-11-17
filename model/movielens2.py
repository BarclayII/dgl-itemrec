import pickle
import numpy as np
from .utils import ComplementView

class MovieLens(object):
    def __init__(self, path):
        '''
        path: path to the MovieLens dataset pickle
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)

        train = data['train']
        train_coo = train.tocoo()
        valid = data['val'].tocoo()
        test = data['test'].tocoo()
        test_neg = data['test_negative']
        train_original = data['train_original']
        neg_size = int(test_neg.sum(1).min())
        assert neg_size == int(test_neg.sum(1).max())

        self.users_train, self.movies_train = train_coo.row, train_coo.col
        self.users_valid, self.movies_valid = valid.row, valid.col
        self.users_test, self.movies_test = test.row, test.col
        self.train_size = len(self.users_train)
        self.valid_size = len(self.users_valid)
        self.test_size = len(self.users_test)

        num_users, num_movies = train.shape

        self.neg_size = neg_size
        self._pos_train = [None] * num_users
        self._pos_test_complete = [None] * num_users
        self.neg_valid = np.zeros((num_users, neg_size), dtype='int64')
        self.neg_test = np.zeros((num_users, neg_size), dtype='int64')

        for u in range(num_users):
            interacted_movies = train[u].nonzero()[1]
            self._pos_train[u] = interacted_movies
            self._pos_test_complete[u] = np.concatenate([interacted_movies, [self.movies_valid[u]]])

            neg_samples = np.setdiff1d(np.arange(num_movies), interacted_movies)
            self.neg_valid[u] = np.random.choice(neg_samples, neg_size)
            self.neg_test[u] = test_neg[u].nonzero()[1]

        self.movie_data = {}

        self.user_latest_item = data['user_latest_item']
        self.num_users = num_users
        self.num_movies = num_movies
        self.movie_count = train_original.sum(0).A.squeeze().astype('int64')
        self.neg_train = ComplementView(self._pos_train, self.num_movies)
        self.neg_test_complete = ComplementView(self._pos_test_complete, self.num_movies)
