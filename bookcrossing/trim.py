import fileinput
import numpy as np
import scipy.sparse as spsp

rows = []
cols = []
vals = []

for line in fileinput.input():
    row, col, val = line.split(' ')
    rows.append(int(row))
    cols.append(int(col))
    vals.append(float(val))

rows = np.array(rows)
cols = np.array(cols)
vals = np.array(vals)

mat = spsp.coo_matrix((vals, (rows, cols)))
mat = mat.tocsc()
print('size:', mat.shape)
print('nnz:', mat.nnz)
filter_idx = mat.sum(axis=0) > 5
filter_idx = np.array(filter_idx)
print('#frequent terms:', np.sum(filter_idx))
filter_idx = np.squeeze(filter_idx)

mat = mat[:,filter_idx]
print('size:', mat.shape)
print('nnz:', mat.nnz)
np.save('book_mat.npy', mat)
