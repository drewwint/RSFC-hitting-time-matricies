## Tests in speed for hitting time



### Hitting matricies 
import time

## Old version 
def hitting_matrix(correlation_matrix):
  start_time = time.perf_counter()
  # correlation_matrix = pd.DataFrame(correlation_matrix)
  ## input is the correlation matrix values are assumed to be > 0
  # returns H - number of hops
  # for i in correlation_matrix:                  ## added this to ensure diagonal is 0
  #   np.fill_diagonal(i.values,0)
  correlation_matrix = abs(correlation_matrix)  ## added this to only retain absolute values.
  L = np.size(correlation_matrix,axis = 0)
  A_matrix = np.array(correlation_matrix)
  D_matrix = np.zeros((L,L))
  for i in range(L):
      D_matrix[i,i] = np.sum(A_matrix[i]) # loop unnecessary - just sum the columns

  d_max = np.max(D_matrix)

  for j in range(L):
      if np.max(A_matrix[j,:]) < .05:
          A_matrix[j,j] = d_max - D_matrix[j,j] # no need to make a matrix with a values on the diagonal - just a list

  D_matrix = np.zeros((L,L))
  D_inv = np.zeros((L,L))
  D_sqrt = np.zeros((L,L))
  D_sqrt_inv = np.zeros((L,L))
  for i in range(L):
      D_matrix[i,i] = np.sum(A_matrix[i])
      D_inv[i,i] = 1./D_matrix[i,i]
      D_sqrt[i,i] = np.sqrt(D_matrix[i,i])
      D_sqrt_inv[i,i] = 1./D_sqrt[i,i]

  p_matrix = np.dot(D_inv, A_matrix)

  # Graph Laplacian
  eye_matrix = np.eye(L,L)
  eye_P = eye_matrix - p_matrix

  G_Lap = np.dot(D_sqrt,eye_P)
  G_Lap_n = np.dot(G_Lap, D_sqrt_inv)

  [eig_val, eig_vec] = np.linalg.eigh(G_Lap_n)
  H = np.zeros((L,L))
  d = np.sum(D_matrix)
  for i in range(L):
      for j in range(L):
          deg_i = D_matrix[i,i]
          deg_j = D_matrix[j,j]
          for k in range(L):
              if eig_val[k] != min(eig_val):
                  t_i = (eig_vec[i,k]*eig_vec[i,k])/deg_i
                  t_j = (eig_vec[j,k]*eig_vec[j,k])/deg_j
                  t_ij = (eig_vec[i,k]*eig_vec[j,k])/np.sqrt(deg_i*deg_j)
                  H[i,j] = H[i,j] + d*(1./(eig_val[k]))*(t_j-t_ij)

  H = np.transpose(H)
  end_time = time.perf_counter()
  print(f"total time: {end_time - start_time:.2f} seconds")
  return H


## Parallalized version


def hitting_matrix_p(correlation_matrix):
    start_time = time.perf_counter()
    from joblib import Parallel, delayed
    # Ensure absolute values
    correlation_matrix = abs(correlation_matrix)
    L = np.size(correlation_matrix, axis=0)
    A_matrix = np.array(correlation_matrix)

    # Degree matrices
    D_matrix = np.zeros((L, L))
    for i in range(L):
        D_matrix[i, i] = np.sum(A_matrix[i])

    d_max = np.max(D_matrix)

    for j in range(L):
        if np.max(A_matrix[j, :]) < 0.05:
            A_matrix[j, j] = d_max - D_matrix[j, j]

    D_matrix = np.zeros((L, L))
    D_inv = np.zeros((L, L))
    D_sqrt = np.zeros((L, L))
    D_sqrt_inv = np.zeros((L, L))
    for i in range(L):
        D_matrix[i, i] = np.sum(A_matrix[i])
        D_inv[i, i] = 1.0 / D_matrix[i, i]
        D_sqrt[i, i] = np.sqrt(D_matrix[i, i])
        D_sqrt_inv[i, i] = 1.0 / D_sqrt[i, i]

    p_matrix = np.dot(D_inv, A_matrix)

    # Graph Laplacian
    eye_matrix = np.eye(L, L)
    eye_P = eye_matrix - p_matrix

    G_Lap = np.dot(D_sqrt, eye_P)
    G_Lap_n = np.dot(G_Lap, D_sqrt_inv)

    eig_val, eig_vec = np.linalg.eigh(G_Lap_n)
    d = np.sum(D_matrix)

    # Parallel computation of rows
    def compute_H_row(i):
        H_row = np.zeros(L)
        deg_i = D_matrix[i, i]
        for j in range(L):
            deg_j = D_matrix[j, j]
            for k in range(L):
                if eig_val[k] != min(eig_val):
                    t_i = (eig_vec[i, k] * eig_vec[i, k]) / deg_i
                    t_j = (eig_vec[j, k] * eig_vec[j, k]) / deg_j
                    t_ij = (eig_vec[i, k] * eig_vec[j, k]) / np.sqrt(deg_i * deg_j)
                    H_row[j] += d * (1.0 / eig_val[k]) * (t_j - t_ij)
        return H_row

    # Use joblib for parallelization
    with Parallel(n_jobs=-1, backend="loky") as parallel:
        H_rows = parallel(delayed(compute_H_row)(i) for i in range(L))

    # Combine rows into the final matrix
    H = np.array(H_rows)
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:.2f} seconds")
    return H


from joblib import Parallel, delayed
import numpy as np

def hitting_matrix_p2(correlation_matrix):
    start_time = time.perf_counter()
    correlation_matrix = np.array(abs(correlation_matrix))  # Ensure absolute values
    np.fill_diagonal(correlation_matrix, 0)  # Set diagonal to 0

    L = correlation_matrix.shape[0]
    A_matrix = correlation_matrix.copy()

    # Degree matrix
    row_sums = A_matrix.sum(axis=1) # instead of d_matrix loop we sum columns without the loop
    d_max = row_sums.max()

    # Ensure graph connectivity
    for j in range(L):
      if np.max(A_matrix[j,:]) < .05:
          A_matrix[j,j] = d_max - row_sums[j]

    row_sums = A_matrix.sum(axis=1)  # Recalculate after adjustment
    D_inv = np.diag(1.0 / row_sums)
    D_sqrt = np.diag(np.sqrt(row_sums))
    D_sqrt_inv = np.diag(1.0 / np.sqrt(row_sums))

    # Transition probability matrix and Graph Laplacian
    p_matrix = D_inv @ A_matrix
    eye_P = np.eye(L) - p_matrix
    G_Lap_n = D_sqrt @ eye_P @ D_sqrt_inv

    # Eigen decomposition
    eig_val, eig_vec = np.linalg.eigh(G_Lap_n)

    # Precompute reusable quantities
    eig_val_nonzero = eig_val[eig_val > eig_val.min()]
    eig_vec_squared = eig_vec ** 2
    d_total = row_sums.sum()

    def compute_H_row(i):
        H_row = np.zeros(L)
        deg_i = row_sums[i]
        for j in range(L):
            deg_j = row_sums[j]
            t_ij = (
                eig_vec_squared[i, eig_val > eig_val.min()] / deg_i
                - eig_vec[i, eig_val > eig_val.min()]
                * eig_vec[j, eig_val > eig_val.min()]
                / np.sqrt(deg_i * deg_j)
            )
            H_row[j] = np.sum(d_total * t_ij / eig_val_nonzero)
        return H_row

    # Parallelize computation of rows
    with Parallel(n_jobs=-1, backend="loky") as parallel:
        H = np.array(parallel(delayed(compute_H_row)(i) for i in range(L)))
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:.2f} seconds")
    return H




#*#
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=-1):
  start_time = time.perf_counter()
  print("parallel \n", hitting_matrix_p(cor_mat_ddc[1]))
  print("parallel 2 \n", hitting_matrix_p2(cor_mat_ddc[1]))
  print("og \n", hitting_matrix(cor_mat_ddc[1]))
  end_time = time.perf_counter()
  print(f"\n Complete time: {end_time - start_time:.2f} seconds")



