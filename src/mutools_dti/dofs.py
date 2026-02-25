""" Dixon olephinic fat suppression for diffusion weigthed images

Burakiewicz J, Hooijmans MT, Webb AG, Verschuuren JJGM, Niks EH, Kan HE:
Improved olefinic fat suppression in skeletal muscle DTI using a magnitude-based dixon method: DTI Dixon Olefinic Fat Suppression.
Magn Reson Med 2018; 79:152–159.


"""
import numpy as np
from scipy import ndimage
from mutools.fatwater import dualecho, fwqpbo, fatmodel, utils


make_ffmap = utils.make_ffmap


def foreground_mask(volumes, threshold=0.6):
    """make"""
    meanvol = np.mean(np.abs(volumes), axis=0)
    vrange = np.unique(meanvol)
    mask = np.zeros_like(volumes[0], dtype=bool)
    mask[:] = meanvol > np.percentile(vrange, threshold * 100)
    struct = ndimage.generate_binary_structure(2, 1)[..., np.newaxis]
    mask[:] = ndimage.binary_fill_holes(mask, structure=struct)
    mask[:] = ndimage.binary_dilation(mask, structure=struct, iterations=2)
    return mask


def dofs_b0(echo_times, volumes, mask=None, B0=3, method='fwqpbo'):
    """compute fwmap and wmap from non-diffusion weighted images"""
    model = fatmodel.FatOlefinic()

    if method == 'fwqpbo':
        opts = {
            "field_strength": B0,
            "fat_model": model,
            "pixel_spacing": volumes[0].spacing,
            "nR2": 500,
            "r2_max": 5e2,
            "nB0": 1000,
            "return_arguments": ["wmap", "fmap", "b0map", "r2star", 'resids'], #"prediction"],
        }
        # wmap, fmap, b0map, r2star, pred = fwqpbo.reconstruct(
        wmap, fmap, b0map, r2star, resids = fwqpbo.reconstruct(
            echo_times, volumes, mask=mask, **opts
        )
    else:
        raise NotImplementedError(method)
    

    # Filter R2* map to prevent potential artifacts
    r2star_unfiltered = r2star.copy()
    r2star_filtered = ndimage.median_filter(r2star, 3)
    r2star[:] = r2star_filtered

    #Filter R2* map to prevent potential artifacts 
    r2star_unfiltered = r2star.copy()
    r2star_filtered = ndimage.median_filter(r2star, 3)
    r2star[:] = r2star_filtered

    return {
        "wmap": wmap,
        "mask": mask,
        "fmap": fmap,
        "b0map": b0map,
        "r2star_unfiltered": r2star_unfiltered,
        "r2star": r2star,
        # "prediction": pred,
        'resids': resids,
    }


def dofs(echo_times, volumes, resb0, bvalue, B0=3, **kwargs):
    """compute fwmap and wmap from magnitude-only diffusion weighted images"""

    wmap0 = resb0["wmap"]
    fmap0 = resb0["fmap"]
    b0map = resb0["b0map"]
    r2star = resb0["r2star"]
    mask = resb0["mask"] > 0

    spacing = np.array(getattr(volumes[0], "spacing", [1] * volumes[0].ndim))

    spacing = np.array(getattr(volumes[0], 'spacing', [1]*volumes[0].ndim))

    # find echo pairs
    indices = list(range(len(echo_times)))
    pairs = kwargs.get("pairs", list(zip(indices[0::2], indices[1::2])))

    # Fat/water chemical displacement
    apply_b0_correction = kwargs.get("apply_b0_correction", True)

    # diffusion adjusted ffmap
    adc_water = kwargs.get("adc_water", 1.5 * 1e-3)  # mm2/s
    adc_fat = kwargs.get("adc_fat", 1 * 1e-5)  # mm2/s
    F = fmap0 * np.exp(-bvalue * adc_fat)
    W = wmap0 * np.exp(-bvalue * adc_water)
    ffmap0 = F * (F > 1e-8) / np.maximum(F + W, 1e-8)

    # fat model
    if apply_b0_correction:
        # pixel bandwidth
        BW = 1e-3 * kwargs.get("bandwidth", 1680)
        # shift direction
        pe_axis = kwargs.get("pe_axis", 1)
        nline = volumes[0].shape[pe_axis]
        # Olefinic fat shift in kHz
        freqshift = fatmodel.cshift_to_frequency(fatmodel.FatOlefinic.cshifts[0], B0=B0)
        # Olefinic fat shift in pixels
        shift_dist = -freqshift / BW * nline / 2

        # get shifted deltaB0 for olefinic peak
        coords = np.asarray(np.indices(b0map.shape)).astype(float)
        coords[pe_axis] += shift_dist
        b0map = np.nan_to_num(b0map.copy(), 0)
        b0map = ndimage.median_filter(b0map, 3)
        deltab0 = b0map - ndimage.map_coordinates(
            b0map, coords, order=3, mode="nearest"
        )
        model = FatOlefinicCorrected(deltab0[mask])
    else:
        model = fatmodel.FatOlefinic()

    # get water and fat magnitudes
    opts = {"model": model, "unwrap": False, "B0": B0}

    wmap = np.zeros_like(wmap0)
    fmap = np.zeros_like(fmap0)
    nvalue = 0

    for i1, i2 in pairs:
        # adjust volumes magnitudes with R2* prior
        S1 = volumes[i1] * np.exp(1e-3 * r2star * echo_times[i1])
        S2 = volumes[i2] * np.exp(1e-3 * r2star * echo_times[i2])

        # dual echo reconstruct
        echos = echo_times[i1], echo_times[i2]
        A, B = dualecho.dual_echo(echos, [S1, S2], mask=mask, **opts)
        B = np.maximum(B, 0)  # fix negative values in B

        # solve W and F
        mask01 = (ffmap0 < 0.1) & mask
        C1 = A > B
        C2 = ~C1
        wmap[mask01 & C1] += A[mask01 & C1]
        fmap[mask01 & C1] += B[mask01 & C1]
        wmap[mask01 & C2] += B[mask01 & C2]
        fmap[mask01 & C2] += A[mask01 & C2]

        mask09 = (ffmap0 > 0.9) & mask
        wmap[mask09 & C2] += A[mask09 & C2]
        fmap[mask09 & C2] += B[mask09 & C2]
        wmap[mask09 & C1] += B[mask09 & C1]
        fmap[mask09 & C1] += A[mask09 & C1]

        mask10 = ~(mask01 | mask09) & mask
        D1 = np.abs(W - A) < np.abs(W - B)
        D2 = ~D1
        wmap[mask10 & D1] += A[mask10 & D1]
        fmap[mask10 & D1] += B[mask10 & D1]
        wmap[mask10 & D2] += B[mask10 & D2]
        fmap[mask10 & D2] += A[mask10 & D2]
        nvalue += 1

    wmap /= nvalue
    fmap /= nvalue
    wmap[~mask] = np.nan
    fmap[~mask] = np.nan
    return wmap, fmap


class FatOlefinicCorrected(fatmodel.FatOlefinic):
    """Fat model for olefinic peak with chemical displacement correction"""

    def __init__(self, b0map):
        self.b0map = np.asarray(b0map)

    def __call__(self, echo_times, B0=3):
        signal = super().__call__(echo_times, B0=B0)
        # correct with B0
        deltab0 = self.b0map[np.newaxis]
        naxes = (slice(None),) + (np.newaxis,) * self.b0map.ndim
        echos = np.asarray(echo_times)
        signal = signal[naxes] * np.exp(1j * deltab0 * echos[naxes])
        return signal
