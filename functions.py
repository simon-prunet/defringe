import numpy as nm
try:
  import cupy as cp
  GPU = True
except:
  print("cupy not installed. This probably means there is no GPU on this host.")
  print("Using numpy instead")
  import numpy as cp
  GPU = False
from images import data
import randlin

def fringes(images,masks,tol=1e-10,rank=None,unshrinking=True,sigmas=None,lfac=1.0,random=None):
  ''' Starts from a collection of images and masks, 
      computes the target regularization parameter lambda~\sqrt(npix)\sigma
      using a robust estimation of the image noise level, and 
      makes a low-rank fit to the images in the following way:
      1) Computes the SOFT-INPUTE solution with warm start (boosting lambda first)
      2) For a given rank, fixing the fringe modes obtained above, recompute the low-rank coefficients
         with a linear regression (ML solution). This step is important to debias the result. '''


  # Check if inputs are of the right kind
  if (GPU):
    if (not isinstance(images,cp.ndarray) or not isinstance(masks,cp.ndarray)):
      print("Input arrays must be cupy.ndarray")
      return
  # Compute target lambda:
  npix=images.shape[0]
  nobs=images.shape[1]
  if (sigmas is None):
    sigmas= cp.asarray([ robust_sigma((images*masks)[:,i]) for i in range(nobs) ])
  else:
    sigmas = cp.asarray(sigmas)
  scaled_images = images / sigmas[None,:]
  print ('sigmas:')
  print (sigmas)

  lamb = nm.sqrt(npix) *nm.sqrt(cp.sum(masks)/(npix*nobs*1.))*lfac
  print ('lambda:',lamb)
  print ('Computing low-rank solution, with regularization lambda = %.2f'%lamb)
  # Warm start, to speed up convergence. This is the SOFT-INPUTE STEP
  print ('First iterations with lambda*10 (warm start)...')
  Z = svd_iterate(scaled_images,masks,lamb*10.,tol=tol,random=random)
  print ('Second iterations with target lambda...')
  Z = svd_iterate(scaled_images,masks,lamb,Zold=Z,tol=tol,random=random) 
  # Now un-shrink via ML solution on current singular vectors, for a given rank
  U,D,VT = cp.linalg.svd(Z,full_matrices=False)

  print ('Singular values:')
  print (D[:10])
  #print ('Weights:')
  #print (VT[:10,:].T)
  if (unshrinking):
    print ('Unshrinking...')
    ##Z = unshrink(scaled_images,masks,Z,rank=rank)
    Z = regress_lsingular(scaled_images,masks,U,rank=rank)
    if (random is None):
      U,D,VT = cp.linalg.svd(Z,full_matrices=False)
    else:
      U,D,VT = randlin.gpu_random_svd(Z,*random)
    print ('Singular values after unshrinking:')
    print (D[:10])
    #print ('Weights after unshrinking:')
    #print VT[:rank,:].T
  return Z * sigmas[None,:]

def svd_iterate(X,masks,lam=1.,Zold=None,tol=1e-5,trunc=None,hard=False,rank=None,verbose=False,random=None):

  # Solve the nuclear norm regularized problem with 
  # partially observed data matrix X
  # X, masks, Zold are (npix,nobs) matrices
  # i.e. images have been flatten along columns
  # This is the "SOFT-INPUTE" algorithm of Mazumeder & Hastie

  npix, nobs = X.shape
  if (Zold is None):
    Zold = cp.zeros((npix,nobs))
  
  frac_err=1e30
  while (frac_err > tol):
    if (not hard):
      Znew = soft_svd(masks*X + (1.0-masks)*Zold,lam,trunc=trunc,random=random)
    else:
      if (rank is None):
        rank=3
      Znew = hard_svd(masks*X + (1.0-masks)*Zold,rank,random=random)
    frac_err = frob2(Znew-Zold)/frob2(Znew)
    print ("fractional error = %g"%frac_err)
    if (verbose):
      # Computes chi2 and nuclear norm terms, and print them
      chi2 = 0.5 * frob2(masks*(Znew-X)) / (npix*nobs)
      if (hard):
        print("chi2 = %f"%chi2)
      else:
        nuke = lam * nuclear(Znew) / (npix*nobs)
        print("chi2 = %f, nuclear norm = %f, sum = %f"%(chi2,nuke,chi2+nuke))

    Zold = Znew

  return Znew

def soft_svd(X, lam, trunc=None, random=(3,0,1)):
    
  # Solves the nuclear norm regularized prob
  # argmin 1/2 ||X-Z||^2 + lam ||Z||*
  # (first norm is Frobenius, second is nuclear norm)
  # Solution is soft-thresholded SVD, ie replace singular
  # values of X (d1,...,dr) by ((d1-lam)+,...,(dr-lam)+)
  # where (t)+ = max(t,0)
  # See Mazumder, Hastie, Tibshirani 2012
  # In case random keyword is not None, computes randomized SVD, 
  # inputs are assumed to be equal
  # to (k,s,q), where k is the target rank, s the oversampling, q the number of iterations.

  if random is None:
    U,D,VT = cp.linalg.svd(X,full_matrices=False)
  else:
    U,D,VT = randlin.gpu_random_svd(X,*random)
  rankmax = D.size
  vlam = cp.ones(rankmax) * lam
  if (trunc is not None):
    vlam[0:trunc]=0.
  DD = cp.fmax(D-vlam,cp.zeros(rankmax))

  return cp.dot(U, cp.dot(cp.diag(DD),VT))


def hard_svd(X, rank, random=(3,0,1)):
  # SVD truncation
  # If random is not None, inputs are respectively
  # target rank of randomized SVD, oversampling and 
  # number of power iterations.
  if (random is None):
    U,D,VT=cp.linalg.svd(X,full_matrices=False)
  else:
    U,D,VT=randlin.gpu_random_svd(X,*random)
  D[rank:] = 0.
  return cp.dot(U, cp.dot(cp.diag(D),VT))

def frob2(X):
  # Square of Frobenius norm
  return cp.sum(X**2)

def L1(X):
  # Sum of absolute values of elements
  return cp.sum(cp.abs(X))

def nuclear(X,trunc=None):
  # Nuclear norm, sum of singular values

  s = cp.linalg.svd(X,compute_uv=False,full_matrices=False)
  if (trunc):
    return s[trunc:].sum()
  else:
    return s.sum()

def regress_lsingular(images, masks, U, niter = 1, rank=None):
  npix, nobs = images.shape
  if (rank is None):
    rank=3
  UU = U[:,:rank]
  X = images * masks
  Znew = cp.zeros( images.shape, dtype=nm.float32 )

  for i in range(nobs):
    B = UU * cp.reshape( masks[:, i], (npix,1) )
    if niter > 1:
      eps = cp.median( nm.abs(X[:,i])/100.)
      W = cp.ones((npix,1), dtype=nm.float32)
      for j in range( niter ):
        res = images[:,i] - Znew[:, i]
        W = 1./(1.0+(res/eps)**2)**0.5
        BTB = cp.dot(B.T, B*nm.reshape(W, (npix,1)) )
        RHS = cp.dot(B.T, W*X[:, i] )
        coeff = cp.dot( cp.linalg.inv(BTB),RHS )
        Znew[:, i] = cp.dot(UU, coeff)
    else:
      BTB = cp.dot( B.T, B )
      RHS = cp.dot( B.T, X[:, i] )
      coeff = cp.dot( cp.linalg.inv(BTB),RHS )
      Znew[:, i] = cp.dot(UU, coeff)
  return (Znew)


