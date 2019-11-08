# Usage:
# Copy this file to the root directory of this repo:
# https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
# This will generate a file called "ml-1m.dataset"
import Conferences.WWW.NeuMF_our_interface.Movielens1M.Movielens1MReader as reader
import pandas as pd
import pickle

r = reader.Movielens1MReader()

# read original training set
train_table = pd.read_csv(
        'Conferences/WWW/NeuMF_github/Data/ml-1m.train.rating',
        sep='\t',
        header=None,
        names=['user_idx', 'movie_idx', 'rating', 'timestamp'])
user_latest_item_indices = train_table.groupby('user_idx')['timestamp'].transform(pd.Series.max) == train_table['timestamp']
user_latest_item = train_table[user_latest_item_indices]
user_latest_item = dict(
        zip(user_latest_item['user_idx'].values, user_latest_item['movie_idx'].values))

with open('ml-1m.dataset', 'wb') as f:
    pickle.dump({
        'train': r.URM_train,
        'val': r.URM_validation,
        'test': r.URM_test,
        'test_negative': r.URM_test_negative,
        'train_original': r.URM_train_original,
        'user_latest_item': user_latest_item,
        }, f)
