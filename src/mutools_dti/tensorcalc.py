""" DTI tensor calculation

Following:

> Kingsley PB
  Introduction to diffusion tensor imaging mathematics: Part III. 
  Tensor calculation, noise, simulations, and optimization. 
  Concepts in Magnetic Resonance Part A 2006; 28A:155–179.

"""

import csv
import numpy as np
from scipy import optimize

NAX = np.newaxis


def save_stats(filename, values):
    """save values to csv file"""
    fieldnames = []
    for row in values:
        fieldnames.extend([field for field in row if not field in fieldnames])

    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(row)


def inv_bmatrix(bmatrix):
    """ get b-value and gradient direction from b-matrix """
    bmat = [[bmatrix[0], bmatrix[1], bmatrix[2]],
            [bmatrix[1], bmatrix[3], bmatrix[4]], 
            [bmatrix[2], bmatrix[4], bmatrix[5]]]
    ev, evec = np.linalg.eig(bmat)
    order = np.argsort(ev)
    ev, evec = ev[order], evec[:, order]
    bvalue = ev[-1]
    bvec = evec[:, -1]
    
    return bvalue, bvec

#def mean_bmatrix(bmatrices):
#    return np.mean(np.asarray(bmatrices).astype(float), axis=0)


def rearrange_bmatrix(bmatrix, order=[0, 3, 5, 1, 2, 4]):
    """eg. from [xx, xy, xz, yy, yz, zz] (Siemens) to [xx, yy, zz, xy, xz, yz]"""
    return [bmatrix[i] for i in order]


def dti_stats(maps, roi, labels=None, stats=["MEAN", "STD", "MED", "MIN", "MAX"]):
    """Compute ROI statistics"""

    labelset = np.unique(roi[roi > 0])
    if labels is None:
        labels = {label: f"label_{label:02d}" for label in labelset}

    res = []
    for label in labelset:
        mask = roi == label
        res.append({"LABEL": labels[label]})
        for name, vol in maps.items():
            values = vol[~np.isnan(vol) & mask]
            for stat in stats:
                if stat == "MEAN":
                    value = np.mean(values)
                elif stat == "STD":
                    value = np.std(values)
                elif stat == "MED":
                    value = np.median(values)
                elif stat == "MIN":
                    value = np.min(values)
                elif stat == "MAX":
                    value = np.max(values)
                else:
                    raise ValueError(f"Unknown statistics: {stat}")
                res[-1].update({f"{name}_{stat}": value})
    return res


def dti_metrics(D, mask, clip_values=True, scale_diff=True):
    """ compute DTI metrics on roi 

    Args:
        D: (6xNxM...) diffusion tensor in row format [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
        roi: (NxM...) ROI/mask
    """
    mask = mask > 0
    Drow = np.asarray(D)[mask]
    nvalue = len(Drow)

    # reshape D into a matrix
    D = np.zeros((nvalue, 3, 3))
    indices = tuple(zip(*[(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]))
    D[(slice(None),) + indices] = Drow[:, :6]

    # compute eigenvalues (from largest to smallest)
    l = np.linalg.eigvalsh(D, UPLO="U")[:, ::-1]
    invalid = np.any(l < 0, axis=1)
    l[invalid] = np.nan

    # rotational invariants
    I1 = np.sum(l, axis=1) # trace
    I2 = np.sum(l * l[:, [1, 2, 0]], axis=1)
    #I3 = np.product(l, axis=1) # determinant
    I4 = I1**2 - 2*I2 # trace(D:D)

    MD = I1 / 3 # `A` or `ADC`
    #Dsurf = (I2 / 3)**0.5 # `J`: 
    #Dvol = I3**(1 / 3) # `G`
    # Dmag = (I4 / 3)**0.5 
    # Dan = 6 * MD**2 - 2 * I2
    # K = I2 / I1
    # H = 3 * I3 / I2

    # Diffusion anisotropy indices
    I4[I4 == 0] = 1e-5    #avoid zero-division
    FA = (1 - I2 / I4)**0.5 # Fractional Anisotropy
    #sRA = (1 - 3 * I2 / I1**2) * 0.5 # Scale Relative Anisotropy
    #VF = 1 - 27 * I3 / I1**3 # Volume Fraction
    # UAsurf = 1 - Dsurf / MD
    # UAvol = 1 - Dvol / MD
    # UAvs = 1 - Dvol / Dsurf
    # LI = (FA + FA**2) / 2 # Lattice Index
    e1 = l[:, 0] # Axial diffusivity = first eigenvalue
    e2 = l[:,1]  # second eigenvalue
    e3 = l[:,2]  #third eigenvalue
    Drad = (l[:, 1] + l[:, 2]) / 2 # radial diffusivity

    # nan values
    invalid = fillvolume(~invalid, mask)

    if clip_values:
        FA[FA > 1] = 1
        MD[MD > 0.003] = 0.003
        e1[e1 > 0.003] = 0.003
        e2[e2 > 0.003] = 0.003
        e3[e3 > 0.003] = 0.003
        Drad[Drad > 0.003] = 0.003
    if scale_diff:
        MD = MD * 1000
        e1 = e1 * 1000
        e2 = e2 * 1000
        e3 = e3 * 1000
        Drad = Drad * 1000

    return {
        'MD': fillvolume(MD, mask),
        'FA': fillvolume(FA, mask),
        #'sRA': fillvolume(sRA, mask),
        'e1': fillvolume(e1, mask),
        'e2': fillvolume(e2, mask), 
        'e3': fillvolume(e3, mask),
        'Drad': fillvolume(Drad, mask),
    }, invalid




#def tensor_calc(S0, S, b, mask=None, sigma=None, nonlin=True):
# =============================================================================
# def tensor_calc(S, b, mask=None, sigma=None, nonlin=True):
#     """ Calculate DTI tensor with the "B-matrix approach"
#     
#     Args
#         S0: NxMx... non-diffusion weighted volume
#         S: sequence of diffusion weighted volumes
#         b: sequence of B-matrices (in row format: [xx, yy, zz, xy, xz, yz])
#         mask: subset of pixels where to compute D
#         sigma: noise standard deviation (if using linear weighted least-squares)
# 
#     Return
#         D: NxMx6 diffusion tensor (in row format: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz])
#     """
#     #S0 = np.asarray(S0)
#     #shape = S0.shape
#     S = np.asarray(S)
#     bmat = np.asarray(b)
#     ndir, *shape_ = S.shape
#     ndir_, nb = bmat.shape
#     assert shape == tuple(shape_)
#     assert ndir == ndir_
#     assert nb == 6
# 
#     if mask is None:
#         mask = True
#     mask = (mask > 0) & np.all(S > 0, axis=0) & (S0 > 0)
#     S0 = S0[mask]
#     S = S[:, mask]
#     b2mat = np.c_[bmat[:, :3], 2 * bmat[:, 3:]]
# 
#     # "B matrix" approach
#     x = np.concatenate([np.log(S0[NAX]), np.log(S)], axis=0)
#     B = np.block([[np.zeros((1,6)), np.ones((1,1))],[-bmat[:, :3], -2*bmat[:, 3:], np.ones((ndir, 1))]])
# 
#     # remove nans in X
#     x = np.nan_to_num(x)
# 
#     
#     # least-square or direct solution
#     if ndir == 6:
#         # direct solution
#         alpha = np.linalg.solve(B, x)
#         D = alpha[:6]
#         
#     elif sigma is None:
#         # linear least-square 
#         alpha, *_ = np.linalg.lstsq(B, x, rcond=None)
#         D = alpha[:6]
# 
#     elif sigma is not None:
#         # weighted linear least-square 
#         noisevar = sigma[mask]**2
#         W = np.concatenate([S0[NAX], S], axis=0).T**2 / noisevar[:, NAX]
#         BB = B.T @ (W[..., NAX] * B)
#         # invBB = np.linalg.inv(BB)
#         invBB = np.linalg.pinv(BB)
#         alpha = np.einsum('...ij,...kj,...k->...i', invBB, W[..., NAX]*B, x.T).T
#         D = alpha[:6]
# 
#     if ndir > 6 and nonlin:
#         # non-linear least-square
#         nS = np.clip(S/S0[NAX], 0, None)
#         
#         def cost(D): # cost function
#             D = np.reshape(D, (6, -1))
#             # cost = np.sum((nS - np.exp(-b2mat @ D))**2)
#             cost = np.sum((nS - np.exp(-np.clip(b2mat @ D, 0, None)))**2)
#             return cost
# 
#         def jac(D):
#             D = np.reshape(D, (6, -1))
#             # R = np.exp(-b2mat @ D)
#             R = np.exp(-np.clip(b2mat @ D, 0, None))
#             # print('gradient')
#             return (2 * b2mat.T @ (R * (nS - R))).ravel()
# 
#         # minimize cost function
#         res = optimize.minimize(cost, np.ravel(D), jac=jac, options={'disp': True}, method='cg')
#         D = np.reshape(res.x, (6, -1))
# 
#     # L2 residuals
#     resid = np.sqrt((S/S0[NAX] - np.exp(-b2mat @ D))**2).sum(axis=0)
#     
#     # return
#     D = np.stack([fillvolume(D[i], mask) for i in range(6)], axis=-1)
#     resid = fillvolume(resid, mask)
# 
#     return D, resid
# =============================================================================

def tensor_calc(S, b, mask=None, return_lsq=False, method='NLLS'):
    """ Calculate DTI tensor with the "B-matrix approach"

    Args
        S: sequence of diffusion weighted volumes
        b: sequence of B-matrices (in row format: [xx, yy, zz, xy, xz, yz])
        mask: subset of pixels where to compute D

    Return
        S0: NxM estimate of S0
        D: NxMx6 diffusion tensor (in row format: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz])
        resid: residual of diffusion tensor
    """
    S = np.asarray(S)
    bmat = np.asarray(b)
    ndir = S.shape[0]
    assert ndir, 6 == bmat.shape

    if mask is None:
        mask = True
    mask = (mask > 0) & np.all(S > 0, axis=0)
    S = S[:, mask]
    
    #LSQ as initial guess, B-matrix approach
    x = np.log(S)
    x = np.nan_to_num(x) #remove nans in x

    B = np.block([[-bmat[:, :3], -2 * bmat[:, 3:], np.ones((ndir, 1))]])
    alpha, *_ = np.linalg.lstsq(B, x, rcond=None)

    # estimated D and S0
    D_lsq = alpha[:6]
    S0_lsq = np.exp(alpha[-1])

    #NLSQ fit
    b2mat = np.c_[bmat[:, :3], 2 * bmat[:, 3:]]

    def cost(arr, S):
        nobs = S.shape[1]
        arr = np.reshape(arr, (-1, nobs))
        S0, D = arr[-1], arr[:6]
        cost = 1 / nobs * np.sum((S - S0 * np.exp(-b2mat @ D))**2)
        return cost
    
    def jac(arr, S):
        nobs = S.shape[1]
        arr = np.reshape(arr, (-1, nobs))
        S0, D = arr[-1], arr[:6]
        R = np.exp(-b2mat @ D)
        # derivative w/r to S0
        dS0 = -2 * np.sum(R * (S - S0*R), axis=0)
        # derivative w/r to D
        dD = 2 * b2mat.T @ (S0 * R * (S - S0 * R))
        grad = 1/nobs * np.concatenate([dD, dS0[NAX]], axis=0).ravel()
        return grad
    
    # minimize cost function
    init = np.concatenate([D_lsq, S0_lsq[NAX]]).flatten()   #SR 11.08.2023 flatten x0
    res = optimize.minimize(cost, init, jac=jac, tol=1e-6, options={'disp': True}, method='cg', args=(S,))
    sol = res.x.reshape(7, -1)
    S0, D = sol[-1], sol[:6]

    # L2 residuals
    resid = np.sum((S - S0 * np.exp(-b2mat @ D))**2, axis=0)**0.5
    resid /= np.sum(S ** 2, axis=0) ** 0.5

    # return
    S0 = fillvolume(S0, mask)
    D = np.stack([fillvolume(D[i], mask) for i in range(6)], axis=-1)
    resid = fillvolume(resid, mask)

    if return_lsq:
        S0_lsq = fillvolume(S0_lsq, mask)
        D_lsq = np.stack([fillvolume(D_lsq[i], mask) for i in range(6)], axis=-1)
        return S0_lsq, D_lsq, S0, D, resid
    else:
        return S0, D, resid


def fillvolume(values, mask):
    arr = np.zeros(mask.shape, dtype=values.dtype)
    arr[mask] = values
    return arr
