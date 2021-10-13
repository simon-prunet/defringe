from astropy.io import fits as pf
import numpy as nm
try:
  import cupy as cp
  GPU = True
except:
  print("cupy not installed. This probably means there is no GPU on this host.")
  print("Using numpy instead")
  import numpy as cp
  GPU = False


class data (object):

  '''
  This class will be used to read images, create masks, etc.
  and form the data matrix to be processed for fringe removal
  '''
  ODOMETER_FILE = 'odom_z_14Am01'
  ROOTDIR = '/data101/prunet/fringe-data/'
  CHIPMASK_FILE = 'masks/2003A.mask.0.36.02.fits'
  # THESE ARE SPECIFIC TO THE MEGACAM CHIPS, SHOULD BE ADAPTED TO YOUR OWN NEEDS
  CCDNUM = 13
  OVERSCAN_X_MIN=32
  OVERSCAN_X_MAX=2080
  OVERSCAN_Y_MIN=0
  OVERSCAN_Y_MAX=4612
  WINSMALL = 500


  def __init__(self,odometer_file=ODOMETER_FILE,rootdir=ROOTDIR,chipmask_file=CHIPMASK_FILE,ccdnum=CCDNUM,small=True):

    # First create list of image names. We will assume here that the images are in 'split' format,
    # i.e. one separate FITS file per CCD.
    self.names = self.create_image_names(rootdir+odometer_file,rootdir+'nodefringe',[ccdnum])
    # Now read the images, and store them as column vectors of the output matrix. Thus, each column of the matrix
    # contains all pixels of a given image.
    self.images = self.read_images(self.names,small=small)
    # Now do the same for images corrected with a robust regression on a single median fringe template
    self.elxnames = self.create_image_names(rootdir+odometer_file,rootdir+'elixir-defringe',[ccdnum])
    elximages = self.read_images(self.elxnames,small=small)
    self.elximages = cp.asarray(elximages)
    # Remove median per image of elximages, for plotting purposes
    elximage_medians = nm.median(self.elximages,axis=0)
    self.elximage_medians = cp.asarray(elximage_medians)
    self.elximages -= self.elximage_medians[None,:]
    # Now do the same for the original images
    self.image_medians = cp.median(self.images,axis=0)
    self.images -= self.image_medians[None,:]
    # Now read the corresponding mask for dead/hot pixels, etc.
    chipmask = self.read_chipmask(rootdir+chipmask_file,ccdnum,small=small)
    self.chipmask = cp.asarray(chipmask)
    return

  def create_image_names(self,odomfile,rootdir,ccdlist,mef=False):
    """
      Store filenames of all the images in a list
      Input: 
        odomfile - filename indices of all the images taken dring a run
        rootdir  - where the .fits stored
        ccdlist  - 0 to 35
        mef(optional) - multi-extension fit file

      Output: A list with all the filenames 
      Example: 
        names=firstpass.create_image_names('../odomlist_z_14Am01',
                                            '../z_14Am01_nodefringe',
                                            [0])
    """
    odometers=nm.loadtxt(odomfile,dtype=nm.unicode_)
    ccdlist=["%02d"%int(i) for i in ccdlist]
    image_names=[]
    for odom in odometers:
      if (mef):
        image_names += [rootdir+'/'+odom+'p.fits']
      else:
        image_names += [rootdir+'/'+odom+'p/'+odom+'p'+ccd+'.fits' for ccd in ccdlist]
    return (image_names)

  def read_images(self,image_names,cut_overscan=True,small=False,mef=False,ccd=0):
    """
      Read data from magacam images. Take data from each image as a column vector, 
      and store the column vectors in a large matrix.

      cut_overscan: get data without the border details of the image
      small: only get data of a small window size (500x500) from the original image
      ccd: 0-35 are valid numbers 
    """  
    images=[]
    for name in image_names:
      if (mef):
        image = pf.getdata(name,ccd+1)
      else:
        image = pf.getdata(name) #Split mode, image in primary hdu
      image = image.T
      if (cut_overscan):
        image = image[self.OVERSCAN_X_MIN:self.OVERSCAN_X_MAX,self.OVERSCAN_Y_MIN:self.OVERSCAN_Y_MAX]
      if (small):
        image = image[:self.WINSMALL,:self.WINSMALL]
        images.append(cp.reshape(image,(self.WINSMALL*self.WINSMALL,)))
      else:
        images.append(cp.reshape(image,((self.OVERSCAN_X_MAX-self.OVERSCAN_X_MIN)*(self.OVERSCAN_Y_MAX-self.OVERSCAN_Y_MIN),)))
    return cp.transpose(cp.asarray(images))

  def read_chipmask(self,MEF,chipnum,cut_overscan=True,small=False):
    """
      Return a mask (as column vector) for specified CCD. Input is a full FP pixel mask as a MEF file. 
    """

    f=pf.open(MEF)
    chipmask = cp.asarray(f[chipnum+1].data,dtype=float)
    chipmask = chipmask.T * 1.0
    f.close()
    if (cut_overscan):
      chipmask = chipmask[self.OVERSCAN_X_MIN:self.OVERSCAN_X_MAX,self.OVERSCAN_Y_MIN:self.OVERSCAN_Y_MAX]
    if (small):
      chipmask = chipmask[:self.WINSMALL,:self.WINSMALL]
      return cp.reshape(chipmask,(self.WINSMALL*self.WINSMALL,))
    else:
      return cp.reshape(chipmask,((self.OVERSCAN_X_MAX-self.OVERSCAN_X_MIN)*(self.OVERSCAN_Y_MAX-self.OVERSCAN_Y_MIN),))
    return(chipmask)

  def create_masks(self,images,chipmask,kappa=2.0,tol=1e-10,include_inputmask=True,robust=False):
    """
      Apply mask_sources to all images to create the overall mask
      Assumes images is (npix,nobs), and chipmask is (npix,) in chipmask_mode
      or (npix,nobs) in imagemask_mode
    """    
    ndim_mask=chipmask.ndim
    imagemask_mode=False
    if (ndim_mask > 1):
      imagemask_mode=True

    npix=images.shape[0]
    nobs=images.shape[1]
    masks = cp.ndarray((npix,nobs),dtype=nm.float64)
    for i in range(nobs):
      if (imagemask_mode):
        masks[:,i] = self.mask_sources(images[:,i],chipmask[:,i],kappa,tol,include_inputmask,robust)
      else:
        masks[:,i] = self.mask_sources(images[:,i],chipmask,kappa,tol,include_inputmask,robust)
    return masks  

  
  def mask_sources(self,image,mask,kappa=2.0,tol=1e-10,include_inputmask=True,robust=True):

    # Iterative kappa sigma-clipping to flag bright sources
    # Outputs a new mask based on the input mask

    msk = cp.ones(cp.shape(mask),dtype=nm.float64)
    img = image.copy()
    img *= mask

    if (robust is not True):
      rms = img.std()
      frac_err = 1e30
      while (frac_err > tol):
        rms_old = rms
        clip = cp.where( (img-cp.median(img))/rms_old > kappa)
        msk[clip] = 0.0
        img *= msk
        rms = img.std()
        frac_err = cp.abs(rms-rms_old)/rms_old
        print ("fractional rms variation = %.3f"%frac_err)
    else:
      rms = self.robust_sigma(img)
      clip = cp.where( (img-cp.median(img))/rms > kappa)
      msk[clip] = 0.0
      img *= msk

    if (include_inputmask):
      return msk*mask
    else:
      return msk

  def robust_sigma(self,X):
    # Computes estimate of rms via MAD
    return 1.4826 * cp.median(cp.abs(X-cp.median(X)))
