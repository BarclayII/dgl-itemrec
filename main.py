import dgl
from model import movielens

randomwalk_steps = 3
n_epoch = 200
batch_size = 32

data = movielens.MovieLens('data/ml-1m/ml-1m')
ratings = data.ratings
ratings_train = ratings[~(ratings['valid_mask'] | ratings['test_mask'])]
users_train = ratings_train['user_idx']
movies_train = ratings_train['movie_idx']
users_valid = ratings[ratings['valid_mask']]['user_idx']
movies_valid = ratings[ratings['valid_mask']]['movie_idx']
users_test = ratings[ratings['test_mask']]['user_idx']
movies_test = ratings[ratings['test_mask']]['movie_idx']
train_size = len(users_train)
valid_size = len(users_valid)
test_size = len(users_test)

HG = dgl.heterograph({
    ('user', 'um', 'movie'): (ratings_train['user_idx'], ratings_train['movie_idx']),
    ('movie', 'mu', 'user'): (ratings_train['movie_idx'], ratings_train['user_idx'])})


class Model(nn.Module):
    def __init__(self, pinsage):
        super().__init__()

        self.pinsage = pinsage

    def forward(self, HG, U, I):
        _, I_U = HG.out_edges(U, form='uv', etype='um')

for _ in range(n_epoch):
    train_indices = torch.arange(len(ratings_train))
    train_batches = train_indices.split(batch_size)

    for train_batch_indices in train_batches:
        U = users_train[train_batch_indices]
        I = movies_train[train_batch_indices]

        _, I_U = HG['um'].out_edges(U)
        N_U = HG['um'].out_degrees(U)
    paths = dgl.contrib.sampling.metapath_random_walk(
            HG, ['um', 'mu'] * randomwalk_steps, )
