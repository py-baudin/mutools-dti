""" SNR calculation

Based on: 
    Dietrich, Raya, Reeder, Reiser, Schoenberg
    Measurement of Signal-to-Noise Ratios in MRI Images: Influence of Multichannel Coils, Parallel Imaging, and Reconstruction Filters
    JMRI 2007
    Appendix A: Definition of SNR and Methods of SNR Determination

"""
import itertools
import numpy as np
import scipy.ndimage as ndi

def compute_snr(signal, noise, method = 'std', rician_cor = True, filter_noise = False, kernelsize = 2, filter_SNR = False):
    """ Returns the SNR estimate
        Input: signal
               mean or standard deviation of the noise (specified in 'method')
    """
    #options: filter noise, radius noise kernel, filter SNR image
    
    match method:
        case 'mean':
            #correction for Rician noise
            if filter_noise:
                noise = ndi.gaussian_filter(noise, kernelsize/2)
            if rician_cor:
                noise = noise / np.sqrt(np.pi/2)
        case 'std':
            if rician_cor:
                noise = noise / np.sqrt(np.pi/2)
    
    snr = np.abs(signal) / noise
    if filter_SNR:
        snr = ndi.gaussian_filter(snr, 1/2)
    return snr

def noise_estim(noise_images, mask=None, radius=2, axes=(0,1)):
    """ estimate noise standard deviation from noise images """
    data = np.asarray(noise_images)
    is_complex = np.iscomplexobj(data)

    nrep, *shape = data.shape
    ndim = len(shape)

    if axes is None:
        axes = list(range(ndim))
    else:
        axes = list(axes)

    # patch
    var = np.zeros(shape)
    indices = list(itertools.product(*[np.arange(-radius, radius + 1) if i in axes else [0] for i in range(ndim)]))
    patches = np.nan * np.ones(shape + [len(indices)], dtype=data.dtype)
    for irep in range(nrep):
        for i, index in enumerate(indices):
            center = tuple([slice(None if j>=0 else -j, None if j<=0 else -j) for j in index])
            slices = tuple([slice(None if j<=0 else j, None if j>=0 else j) for j in index])
            patches[center + (i,)] = data[(irep,) + slices]
        if is_complex:
            var += 0.5 * np.nanvar(patches.real, axis=-1) + 0.5 * np.nanvar(patches.imag, axis=-1)
        else:
            var += np.nanvar(patches, axis=-1) * 1.5267**2
        
    var /= nrep
    std_noise = 0 * noise_images[0].real + np.sqrt(var)

    return std_noise
