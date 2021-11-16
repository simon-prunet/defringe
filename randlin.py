
try:
  import cupy as cp
  GPU = True
except:
  print("cupy not installed. This probably means there is no GPU on this host.")
  print("Using numpy instead")
  import numpy as cp
  GPU = False

def gpu_random_svd(X_gpu,k,s,q=0):
  '''
  Uses cupy, and randomized SVD computation with power iterations
  Computes approximate SVD of X_gpu (which is of size m x n)
  k: target rank (assumed known. For unknown rank implement adaptive method)
  s: oversampling. range finder will be of size m x l = m x (k+s)
  q: number of power iterations, useful in case of flat spectrum
  '''
  if (GPU and not isinstance(X_gpu,cp.ndarray)):
    print("Input array must be a cupy.ndarray")
    return
  l=k+s
  n = X_gpu.shape[1]
  if (l>n) : l=n

  # Generate random sampling matrix O, uniform distribution between -1 and 1
  O_gpu = cp.random.uniform(low=-1.0,high=1.0,size=(n,l)).astype(X_gpu.dtype)
  # Build sample matrix Y = X.O
  # Y approximates the range of X
  Y_gpu = cp.dot(X_gpu,O_gpu)
  Q_gpu, R_gpu = cp.linalg.qr(Y_gpu,'reduced')
  # Renormalized power iterations
  for i in range(1,q+1):
    Yt_gpu = cp.dot(X_gpu.T,Q_gpu)
    Qt_gpu, Rt_gpu = cp.linalg.qr(Yt_gpu,'reduced')
    Y_gpu = cp.dot(X_gpu,Qt_gpu)
    Q_gpu, R_gpu = cp.linalg.qr(Y_gpu,'reduced')

  B_gpu = cp.dot(Q_gpu.T,X_gpu)

  M_gpu, D_gpu, VT_gpu = cp.linalg.svd(B_gpu,full_matrices=False)
  # Come back to Fortran ordering for consistency

  U_gpu = cp.dot(Q_gpu,M_gpu)
  return (U_gpu,D_gpu,VT_gpu)

def eig_svd(X):
  '''
  Computes SVD using eigendecomposition of X^T.X
  Assumes X is tall
  '''

  if (GPU and not isinstance(X,cp.ndarray)):
    print ("Input array must be a cupy.ndarray")
    return
  B = cp.dot(X.T,X)
  d,V = cp.linalg.eig(B)
  sqrtd = cp.sqrt(d)
  S = cp.diag(sqrtd)
  Sm1 = cp.diag(1./sqrtd)
  U = cp.dot(X,cp.dot(V,Sm1))
  return (U,S,V.T)



