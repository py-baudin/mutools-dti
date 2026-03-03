""" RPBM fitting

Following:

> Novikov DS, Vieremans E, Jensen JH, Helpern JA
  Random walks with barriers
  Nature Physics 7, 508-514, 2011
> Fieremans E, Lemberskiy G, Veraart J, Sigmund EE, Gyftopoulos S, Novikov DS
  In vivo measurement of membrane permeability and myofiber size in human muscle
  using time-dependent diffusion tensor imaging and the random permeable barrier model
  NMR in Biomedicine, 2016

Code adapted from Matlab code:
https://github.com/NYU-DiffusionMRI/RPBM/tree/master
"""

import numpy as np
from scipy.integrate import quad


class RPBM_Dictionary:
    def __init__(self, gridsize, ndict, difftimes, model):
        """ RPBM Dictionary
        Generate a dictionary with RD signal values over time for RPBM processing.
        The code is based on the original matlab implementation.
        tau and zeta values are generated randomly for ndict dictionaries to ensure randomization.
        Input:
            gridsize    number of tau and zeta values per dictionary
            ndict       number of individual dictionaries
            difftimes   Diffusion times
            model       Dt model
        Output:
            self.tau    tau values for each dictionary (size gridsize x ndict)
            self.zeta   zeta values fo reach dictionary (size gridsize x ndict)
            self.ndict  number of dictionaries
            self.RD_signal  RD signal over time (diffusiontimes)
        """

        # Generate random tau and zeta values for ndict dictionaries with n=gridsize values
        self.tau = np.random.rand(gridsize,ndict) * 1000
        self.zeta = np.random.rand(gridsize,ndict) * 5
        self.ndict = ndict

        self.RD_signal = np.zeros([len(difftimes), gridsize, ndict])

        for g in range(0, gridsize):
            for j in range(0, ndict):
                for ind, ti in enumerate(difftimes):
                    self.RD_signal[ind,g,j] = model(ti/self.tau[g,j], self.zeta[g,j])


def match_rpbm_robust(RD, Dfix, dictionary, SNR=30):
    # Scale dictionary per voxel
    _RD_signal = dictionary.RD_signal * Dfix

    # Add noise
    sigma = 3/SNR
    noise = np.random.randn(*_RD_signal.shape) * sigma
    _RD_signal = _RD_signal + noise

    # Iteratively estimate best tau and zeta
    tau = []
    zeta = []
    for n in range(0, dictionary.ndict):
        diff = _RD_signal[...,n] - RD[:, np.newaxis]
        sse = np.sum(diff * diff, axis=0)
        idx = np.unravel_index(np.argmin(sse), sse.shape)
        _tau = dictionary.tau[idx, n]
        _zeta = dictionary.zeta[idx, n]

        tau.append(_tau)
        zeta.append(_zeta)

    tau_final = np.median(tau)
    zeta_final = np.median(zeta)
    return tau_final, zeta_final, tau, zeta


def rpbm_process(X, C):
    # Convert inputs to NumPy arrays for vectorized operations
    X = np.array(X)
    C = np.array(C)

    D0, tau, zeta = X

    # Compute RPBM parameters
    L = np.sqrt(D0 * tau)
    a = 2 * L / zeta
    kappa = D0 / (2 * L)
    SV = zeta * 2 / L
    TD = a**2 / (2 * D0)
    TR = a / (2 * kappa)
    a_corr = 6.29 / SV

    # Store RPBM parameters
    RPBM = {
        'D0': D0,
        'tau': tau,
        'zeta': zeta,
        'tortuosity': 1 + zeta,
        'L': L,
        'a': a,
        'a_corr': a_corr,
        'kappa': kappa,
        'SV': SV,
        'TD': TD,
        'TR': TR
    }

    # Error bounds
    D0_err = C[0][0] if C[0][0] == C[0][1] else (C[0][1] - D0) / 2
    tau_err = (C[1][1] - tau) / 2
    zeta_err = (C[2][1] - zeta) / 2

    # Compute uncertainties using relative error propagation
    rel_D0 = D0_err / D0
    rel_tau = tau_err / tau
    rel_zeta = zeta_err / zeta

    L_err = 0.5 * np.sqrt(rel_D0**2 + rel_tau**2) * L
    a_err = 2 * np.sqrt((L_err / L)**2 + rel_zeta**2)
    kappa_err = 0.5 * np.sqrt((L_err / L)**2 + rel_D0**2)
    SV_err = 0.5 * np.sqrt((L_err / L)**2 + rel_zeta**2)
    a_corr_err = 6.29 / SV_err
    TD_err = 0.5 * np.sqrt((2 * a_err / a)**2 + rel_D0**2)
    TR_err = 0.5 * np.sqrt((a_err / a)**2 + (kappa_err / kappa)**2)

    uRPBM = {
        'D0': D0_err,
        'tau': tau_err,
        'zeta': zeta_err,
        'tortuosity': zeta_err,
        'L': L_err,
        'a': a_err,
        'a_corr': a_corr_err,
        'kappa': kappa_err,
        'SV': SV_err,
        'TD': TD_err,
        'TR': TR_err
    }

    return RPBM, uRPBM


def rpbm_calc_dt(t, zeta):
    lambda_ = 1 + zeta
    sqrt_lambda = np.sqrt(lambda_)
    A = 2 * (sqrt_lambda - 1) / lambda_ ** 2
    B = 2 * (sqrt_lambda - 1) * (sqrt_lambda - 2) / lambda_ ** 3
    C = (8 * (sqrt_lambda - 1) ** 2 - zeta * sqrt_lambda) / lambda_ ** 4

    def integrand(y):
        sqrt_y = np.sqrt(y)
        i_sqrt_y = 1j * sqrt_y
        denominator = (i_sqrt_y - 1) ** 2
        sqrt_inner = np.sqrt(1 + zeta / denominator)
        complex_term = 1 / (1 + zeta + 2j * sqrt_y * (i_sqrt_y - 1) * (1 - sqrt_inner))
        term3 = np.imag(complex_term)
        return (A * sqrt_y + C * y ** (3 / 2) + term3) * np.exp(-t * y) / y ** 2

    integral, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-10, epsrel=1e-10)

    dt = (integral / (np.pi * t)
          + 1 / lambda_
          + 2 * A / np.sqrt(np.pi * t)
          + B / t
          - C * t ** (-3 / 2) / np.sqrt(np.pi))

    return dt
