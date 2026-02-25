""" B0 displacement correction 

> Koch KM, Rothman DL, de Graaf RA: 
  Optimization of static magnetic field homogeneity in the human and animal brain in vivo. 
  Prog Nucl Magn Reson Spectrosc 2009; 54:69–96.

"""
import numpy as np
from mutools.utils import interpolate
from scipy import ndimage


def displacement_correction(
    images, deltaB0, BW, *, mask=None, fe_axis=0, pe_axis=1, dmax=5
):
    """compute and apply displacement correction to multiple images"""
    matrix = np.shape(images[0])
    dpmap, dfmap = displacement_map(
        deltaB0, BW, matrix, mask=mask, fe_axis=fe_axis, pe_axis=pe_axis, dmax=dmax
    )
    corr = apply_displacement_correction(
        images, dpmap, dfmap, fe_axis=fe_axis, pe_axis=pe_axis
    )
    return corr, dpmap, dfmap


def displacement_map(
    deltaB0, BW, matrix, *, mask=None, sign=1, fe_axis=0, pe_axis=1, dmax=5
):
    """Compute displacement map from deltaB0 map

    shape = (Nfe, Npe)
    df = gamma * deltaB0 / BW * Nfe
    dp = gamma * deltaB0 / BW * Nfe * Npe

    Args
        deltaB0: delta B0 map [rad/ms]
        BW: pixel bandwidth [Hz/pixel]
        matrix: DTI matrix (shape)

    Return
        dpmap: phase-encoding direction displacement map
        dfmap: frequency-encoding direction displacement map (often negligible)
    """
    # geometrical info
    if mask is not None:
        # remove background pixels
        deltaB0 = deltaB0 * (mask > 0)

    # shape of output map
    if matrix is None:
        matrix = deltaB0.shape

    # convert BW (rad.kHz/pixel)
    rBW = BW * 2 * np.pi * 1e-3

    # displacement maps
    df = sign * deltaB0 / rBW
    dp = sign * deltaB0 / rBW * matrix[pe_axis]

    # keep displacement below dmax
    dfmap = np.clip(df, -dmax, dmax)
    dpmap = np.clip(dp, -dmax, dmax)

    # filter coordinates
    filter_size = dmax + 2
    dfmap = deltaB0 * 0 + ndimage.uniform_filter(dfmap, filter_size, mode="nearest")
    dpmap = deltaB0 * 0 + ndimage.uniform_filter(dpmap, filter_size, mode="nearest")

    return dpmap, dfmap


def apply_displacement_correction(images, dpmap, dfmap=None, *, fe_axis=0, pe_axis=1):
    """apply displacement correction to multiple images"""

    # get pixel coordinates of dti images
    grid = np.asarray(np.indices(images[0].shape), dtype=float)

    # apply displacement maps
    opts = {"order": 1, "mode": "constant", "cval": np.nan}
    dpmap = interpolate.interpolate_like(images[0], dpmap, **opts)
    grid[pe_axis] += dpmap

    if dfmap is not None:
        dfmap = interpolate.interpolate_like(images[0], dfmap, **opts)
        grid[fe_axis] += dfmap

    # remove nans in images
    nanmasks = [~np.isnan(im) for im in images]
    images = [np.nan_to_num(im) for im in images]

    opts = {"order": 3, "mode": "constant", "cval": np.nan}
    corr = [
        im * 0 + ndimage.map_coordinates(im.astype(float), grid, **opts)
        for im in images
    ]
    nanmasks = [
        ndimage.map_coordinates(mask.astype(float), grid, **opts) > 0.5
        for mask in nanmasks
    ]
    for im, mask in zip(corr, nanmasks):
        im[im < 0] = 0
        im[~mask] = np.nan
    return corr
