"""Produce synthetic data based on real CFHT MegaCam images."""

import numpy as np
from astropy.io import fits

def generate_subimages(im, sub_im_size=(75,75), stride=(10, 10),
                       syms=True, axial_sym=True, diag_sym=True, rots=True):
    """Generate small, possibly overlapping subimages from one larger image.

    Parameters
    ----------
    im : 2d `np.ndarray`
        The initial image from which subimages are to be extracted.
    sub_im_size : Sequence, optional
        The desired subimage dimensions. Defaults to (75, 75).
    stride : Sequence, optional
        Desired "stride." If smaller than ``sub_im_size``, subimages will
        overlap. Defaults to (10, 10).
    syms : bool, optional
        Whether symmetries should be applied to each subimage to increase the
        number of datapoints (ie, a form of "data augmentation"). Defaults to
        True
    axial_sym : bool, optional
        Whether vertical and horizontal axial symmetries should be applied.
        Ignored if ``syms`` is False.
    diag_sym : bool, optional
        Whether diagonal and antidiagonal axial symmetries should be applied.
        Ignored if ``syms`` is False.
    rots : bool, optional
        Whether 90, 180, and -90 degrees rotations should be applied.
        Ignored if ``syms`` is False.

    Returns
    -------
    synth_data : `np.ndarray`
        Array of shape ``(n_subimages, sub_im_size[0], sub_im_size[1])``
        containing unique subimages.
    """
    sub_ims = []
    # leave up to a single stride unused
    orig = np.random.choice(stride[0]), np.random.choice(stride[1])
    # compute how many subimages fit
    nrows, ncols = np.array(im.shape - np.array(sub_im_size)) // np.array(stride)
    # extract them
    sub_orig = np.array(orig) - np.array(stride)
    for i in range(nrows):
        sub_orig[0] += stride[0]
        sub_orig[1] = orig[1] - stride[1]
        for j in range(ncols):
            sub_orig[1] +=  stride[1]
            sub_im = im[sub_orig[0]:sub_orig[0] + sub_im_size[0], sub_orig[1]:sub_orig[1] + sub_im_size[1]]
            sub_ims.append(sub_im)
    print(f"Number of subimages before symmetries:\t{len(sub_ims)}")
    synth_data = sub_ims.copy()
    if syms:
        for sub_im in sub_ims:
            synth_data.extend(apply_syms(sub_im, axial_sym, diag_sym, rots))
    synth_data = np.array(synth_data)
    print(f"Datacube size after symmetries:\t{synth_data.shape}")
    return synth_data

def apply_syms(im, axial_sym=True, diag_sym=True, rot=True):
    """Apply set of specified symmetries to input image.

    Parameters
    ----------
    im : 2d `np.ndarray`
         The image symmetries should be applied to.
    axial_sym : bool, optional
        Whether vertical and horizontal axial symmetries should be applied.
    diag_sym : bool, optional
        Whether diagonal and antidiagonal axial symmetries should be applied.
    rots : bool, optional
        Whether 90, 180, and -90 degrees rotations should be applied.
    """
    all_ims = []
    if axial_sym:
        all_ims.extend([np.copy(im)[::-1], np.copy(im)[:,::-1]])
    if diag_sym:
        all_ims.extend([np.copy(im).T, np.copy(im)[::-1, ::-1].T])
    if rot:
        all_ims.extend([np.rot90(np.copy(im)), np.copy(im)[::-1, ::-1], np.rot90(np.copy(im)[::-1, ::-1])])
    return all_ims

def simple_radius(mean_rad=5, sig_rad=5, min_rad=5):
    """Draw a left-truncated Gaussian realization.

    Parameters
    ----------
    mean_rad : float, optional
        Mean of the distribution. Defaults to 5.
    sig_rad : float
        Standard deviation of the distribution. Defaults to 5.
    min_rad : float
        Truncation value. All draws lower than this value will be ignored.
        Defaults to 5.
    """
    rad = 0
    while rad < min_rad:
        rad = mean_rad + np.random.randn()*sig_rad
    return rad

def simple_mask(im, max_sources=8, radii_distrib=simple_radius, **kwargs):
    """Quick and dirty function to draw detection masks on synthetic data.

    All masks are circular, with radii drawn from a callable (then converted to
    int) in units of pixels.

    Parameters
    ----------
    im : 2d `np.ndarray`
         The image a mask should be generated for.
    max_sources : int, optional
         Maximum number of sources per image. The number will be drawn from a
         uniform distribution between 1 and this value. Defaults to 8.
    radii_distrib : callable, optional
         Function to draw a single radius realization. Defaults to
         ``simple_radius``.
    All extra keyword arguments are passed on to ``radii_distrib``.
    """
    nsource = np.random.choice(range(1, max_sources))
    mask = np.zeros(im.shape)
    im_x, im_y = im.shape
    for _ in range(nsource):
        # draw a position and a radius
        center = np.random.choice(im_x), np.random.choice(im_y)
        radius = int(radii_distrib(**kwargs))
        # iterate over pixels in a square centered on the position; if it falls within the radius
        # AND within the image, mask it
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                pix_x, pix_y = center[0] + i, center[1] + j
                if pix_x>=0 and pix_y>=0 and pix_x<im_x and pix_y<im_y and\
                        np.sqrt((pix_x - center[0])**2 + (pix_y - center[1])**2) <= radius:
                    mask[pix_x, pix_y] = 1
    return mask

def draw_coefs(mean0=3e4, sig0=0.5e4, mean1=3.5e3, sig1=0.2e3):
    """Draw a pair of coefficients following Gaussian distributions.

    The default values are chosen to roughly reproduce the distribution
    of coefficients measured in Figure 4 of Prunet, 2021.
    """
    return mean0 + np.random.randn()*sig0, mean1+np.random.randn()*sig1

def generate_synth_data(path_to_modes='./', seed=11,
                        base_image_mean=11):
    np.random.seed(seed)
    # load modes for the full image
    mode0 = fits.open(path_to_modes + 'mode0.fits')[0].data
    mode1 = fits.open(path_to_modes + 'mode1.fits')[0].data
    # generate sub images for each
    subims_mode0 = generate_subimages(mode0)
    subims_mode1 = generate_subimages(mode1)
    # generate "ground truth" images
    base_datacube_shape = subims_mode0.shape
    base_images = base_image_mean + np.random.randn(*base_datacube_shape)
    # and coefficients for each mode
    coefs = [draw_coefs() for _ in range(base_images.shape[0])]
    # recombine to create observed data
    fringed_images = np.array([im + c1*m1 + c2*m2 for im, m1, m2, (c1, c2) in zip(
                               base_images, subims_mode0, subims_mode1, coefs)])
    # generate masks
    masks = np.array([simple_mask(im) for im in fringed_images])
    return base_images, fringed_images, masks
