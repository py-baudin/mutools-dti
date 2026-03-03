""" DWI parsers and basic machines """
import machines as ma

from mutools.toolbox.common.dicom import dicom_loader, dicom_filter
from mutools.toolbox.common.handlers import default_handler
from mutools.toolbox.common.labels import auto_labels


# dicom filter for dwi
dicom_filter_dwi = ma.machine(dicom_filter, output="stack_dwi")


@ma.machine()
@ma.input("stack", "stack_dwi")
@ma.output("dwi", handler=default_handler)
@ma.parameter("which", default="STE", help="Select data types to load")
def dti_parse(stack, which):
    from mutools_dti import readers

    # parse options
    which = which.split(";")
    opts = {
        "parse_ste": "STE" in which,
        'parse_se': 'SE' in which,
        'parse_noise': 'noise' in which,
    }

    # parse dwi data
    volumes, metadata = readers.parse_dicom_dwi(stack["stack"], **opts)
    return {"volumes": volumes, "info": {"metadata": metadata}}


prog_parse_dti = [dicom_loader, dicom_filter_dwi, dti_parse]


@ma.machine()
@ma.input('data', 'dwi', handler=default_handler)
@ma.parameter('coeff', default=1.3)
@ma.output('dti_denoised', handler=default_handler)
def dti_denoise(data, coeff):
    """ denoise data with LPCA """
    import numpy as np
    from mutools.utils import imageutils
    from mutools.noise import lpca
    from mutools_dti import utils

    db = utils.NanoDB(data['info']['metadata'])
    volumes = data['volumes']

    images = {}
    denoised = {}
    formatter = utils.Formatter("dwi_ste_tm{mixtime:03.0f}_ro{readout:02.0f}")

    # denoise for all mixtimes and readouts
    for (mixtime, readout), group in db.groupby("mixtime", "readout"):

        title = formatter(mixtime, readout)
        print(f"Denoising: {title}")
        if readout == 1:
            bv = group.unique("bvalue")
            print(f"TM{mixtime} b-values: {bv}")

        bmin = min(group.unique("bvalue"))
        first = group(group.bvalue == bmin).single()
        others= list(group(group.bvalue != bmin))
        vols = [volumes[name] for name in [first] + others]

        # foreground mask
        mask = lpca.foreground_mask(vols[0])

        # estimate local variance and SNR
        _sigma2, _, _snr_b400 = lpca.noise_estimation(vols[1:])
        sigma2 = 0 * np.abs(vols[0]).copy()
        sigma2[:] = _sigma2
        snr_b400 = 0 * np.abs(vols[0]).copy()
        snr_b400[:] = _snr_b400
        snr = np.abs(volumes[first]) / np.sqrt(sigma2)

        # denoise volumes
        vols_dn = lpca.lpca_denoising(vols, sigma2, coeff=float(coeff))
        denoised.update({name: vol for name, vol in zip([first] + others, vols_dn)})

        # store mask, sigma2 and snr
        denoised[title + "_mask"] = mask
        denoised[title + "_sigma2"] = sigma2
        denoised[title + "_mSNR_b0"] = snr
        denoised[title + "_mSNR_b400"] = snr_b400

        # add overview
        imgs_wn = imageutils.volumes_overview(vols, axis=2, n=4)
        imgs_dn = imageutils.volumes_overview(vols_dn, axis=2, n=4)
        images.update(
            {
                title + "-overview-original.png": imgs_wn,
                title + "-overview-denoised.png": imgs_dn,
            }
        )

    return {
        "volumes": denoised,
        "info": {"metadata": db},
        "images": images,
    }

# Elastix configuration
ELASTIX = {
    "PyramidGaussian": {"maxlevel": 5},
    "InterpolatorLinear": {"mode": "nearest"},
    # "Transform": "TransformAffine",
    # "TransformAffine": {"axis": 2},
    "Transform": "TransformRigid",
    "TransformRigid": {"axis": 2},
    # "Metric": "MetricMutualInformation",
    "Metric": "MetricMeanSquares",
}


@ma.machine()
@ma.input("data", "dti_denoised", handler=default_handler)
@ma.input("satmask", "dwi", handler=default_handler)
@ma.output("dti_registered", handler=default_handler)
def dti_register(data, satmask):
    """register data with Elastix"""
    import numpy as np
    from mutools.registration import elastix
    from mutools.utils import arrayutils, imageutils
    from mutools_dti import utils, plot

    db = utils.NanoDB(data["info"]["metadata"])
    readouts = db.unique("readout")
    volumes = data["volumes"]

    satvol = satmask["volumes"]
    satvol = {k: v for k, v in satvol.items() if k.startswith("saturation_mask")}

    images = {}
    figures = {}
    transforms = {}
    registered = {}
    formatter = utils.Formatter("dwi_ste_tm{mixtime:03.0f}")

    # register options
    opts = dict(return_transform=True, reset=True, config=ELASTIX)

    # register onto first readout image with TM = 100ms
    for mixtime, group in db.groupby("mixtime"):
        title = formatter(mixtime)
        if mixtime == 100:
            # set reference volume to: b=0 volume / first readout of first TM 
            bmin = min(group.unique("bvalue"))
            ref_name = group(group.bvalue == bmin)(group.readout == readouts[0]).first()
            mov_names = sorted(set(group) - {ref_name})
            print(f"Registering to: {title} (ref: {ref_name})")
            ref = volumes[ref_name]
        else:
            # all other volumes are moving volumes
            mov_names = sorted(set(group))
        # register
        print(f"Registering TM={mixtime}")
        mov = [volumes[name] for name in mov_names]
        reg, trans = elastix.register(ref, mov, **opts)

        # store
        if mixtime == 100:
            registered.update({ref_name: ref})
        registered.update({name: vol for name, vol in zip(mov_names, reg)})
        transforms.update(
            {name: transform.serialize() for name, transform in zip(mov_names, trans)}
        )

        # overwiew
        cbmov = [arrayutils.make_checkboard(ref, vol, size=10) for vol in mov]
        img_nreg = imageutils.volumes_overview(cbmov, axis=2, n=4)
        cbreg = [arrayutils.make_checkboard(ref, vol, size=10) for vol in reg]
        img_wreg = imageutils.volumes_overview(cbreg, axis=2, n=4)
        images.update(
            {
                title + "-overview-original.png": img_nreg,
                title + "-overview-registered.png": img_wreg,
            }
        )

        # profiles
        fig_nreg = plot.plot_profiles(
            [ref] + mov, title=title + " original", locs=[0.25, 0.5, 0.75]
        )
        fig_wreg = plot.plot_profiles(
            [ref] + reg, title=title + " registered", locs=[0.25, 0.5, 0.75]
        )
        figures.update(
            {
                title + "-profiles-original.png": fig_nreg,
                title + "-profiles-registered.png": fig_wreg,
            }
        )

        # transform parameters
        # tmp = np.asarray(
        #     [np.asarray(transforms[key]["Parameters"]) for key in transforms.keys()]
        # )

    # transform saturation masks
    if satvol:
        refmask = "tm100_ro01_idir01"
        masknames = [
            n.replace("saturation_mask", "dwi_ste") for n in list(satvol.keys())
        ]
        ref_name_m = [n for n in list(satvol.keys()) if refmask in n]
        mask_transforms = {
            vol: tf for vol, tf in transforms.items() if vol in masknames
        }
        mov_names_m = [
            n.replace("dwi_ste", "saturation_mask") for n in mask_transforms.keys()
        ]

        if ref_name_m:
            ref_m = [satvol[name] for name in ref_name_m]
        mov_m = [satvol[name] for name in mov_names_m]
        mask_t = elastix.transform(list(mask_transforms.values()), mov_m)
        if ref_name_m:
            empty = (0 * ref_m[0].copy()).astype(bool)
        else:
            tmp = satvol[mov_names_m[0]]
            empty = (0 * tmp[0].copy()).astype(bool)
        mask_t = [empty + (np.asarray(mask) > 0.5) for mask in mask_t]

        # store
        if ref_name_m:
            registered.update({name: vol for name, vol in zip(ref_name_m, ref_m)})
        registered.update({name: vol for name, vol in zip(mov_names_m, mask_t)})
    
    
    return {
        "volumes": registered,
        "info": {"metadata": db, "transform": transforms},
        "images": images,
        "figures": figures,
    }


@ma.machine()
@ma.input('data', 'dti_denoised', handler=default_handler)
@ma.input('satmask', 'dwi', handler=default_handler)
@ma.input('transform', 'dti_registered', handler=default_handler)
@ma.output('dti_snr_masked', handler=default_handler)
def dti_regmaskSNR(data, satmask, transform):
    import re
    import numpy as np
    from mutools import io
    from mutools.registration import elastix
    from mutools_dti import utils

    # register SNR_b0
    db = utils.NanoDB(data['info']['metadata'])
    snr = data['volumes']
    snr = {k: v for k, v in snr.items() if '_mSNR' in k}

    snr_b0 = {k: v for k, v in snr.items() if '_mSNR_b0' in k}

    transforms = utils.NanoDB(transform['info']['transform'])
    # transform SNR_b0
    ref = 'tm100_ro01_mSNR_b0'
    ref_name = [n for n in list(snr_b0.keys()) if ref in n]
    snr_transforms = {vol: tf for vol, tf in transforms.items() if vol.replace('idir01', 'mSNR_b0') in snr_b0.keys()}
    mov_names = sorted(set(snr_b0.keys()) - set(ref_name))
    ref_snr = [snr_b0[name] for name in ref_name]
    mov_snr = [snr_b0[name] for name in mov_names]
    snr_t = elastix.transform(list(snr_transforms.values()), mov_snr)
    # store
    snr.update({name: vol for name, vol in zip(ref_name, ref_snr)})
    snr.update({name: vol for name, vol in zip(mov_names, snr_t)})

    # apply saturation masks (Dicom saturated values) to SNR maps 
    satvol = satmask['volumes']
    satvol = {k: v for k, v in satvol.items() if k.startswith('saturation_mask')}
    if satvol:
        for s in snr.keys():
            vol = io.Volume(snr[s])    # transform Image issue
            maskname = s.replace('dwi_ste', 'saturation_mask')
            maskname = re.sub('mSNR_.*', 'idir01', maskname)
            satmask = satvol.get(maskname, np.zeros_like(vol)) > 0
            vol[satmask] = np.nan
            snr[s] = vol

    # Average SNR_b0 over readouts 
    snr_b0 = {k: v for k, v in snr.items() if '_mSNR_b0' in k}
    tms = db.unique("mixtime")
    for tm in tms:
        tmp = {k: v for k, v in snr_b0.items() if 'tm' + str(tm) in k}
        _vols = [io.Volume(tmp[name]) for name in tmp.keys()]
        vols = np.abs(_vols).astype(float)
        _msnr = np.nanmean(vols, axis=0)
        msnr = 0 * np.abs(_vols[0]).copy()
        msnr[:] = _msnr
        snr.update({'mSNR_b0_tm' + str(tm): msnr})

    return{
        'volumes': snr,
        'info': {'metadata': db},
    }


@ma.machine()
@ma.input("data", "dti_denoised", handler=default_handler)
@ma.output("roi_dti", handler=default_handler)
@ma.parameter("labels", ma.Path(), default=None, help="Label file (default: auto)")
@ma.parameter("no_labels", is_flag=True, help="Do not require a label file.")
def RoiDTI(data, labels, no_labels):
    """Create empty ROI for segmentation from denoised DTI-b0 data."""
    import numpy as np
    from mutools.io.roilabels import read_labels

    volumes = data["volumes"]
    
    # select 1st b=0 image (first readout, TM = 100ms)
    key = 'dwi_ste_tm100_ro01_idir01'
    vol = np.abs(volumes[key])
    roi = (vol * 0).astype('uint8')

    # select label file
    index = ma.get_context().indices['roi_dti']
    if no_labels:
        labels = None
    elif labels:
        labels = read_labels(labels)
    else:
        labels = auto_labels(index, required=True)

    return {'volumes': {"vol": vol, "roi": roi}, "labels": {'labels': labels}}


# TEMP
ECHO_TIMES = [1.11, 3.33, 5.55, 7.76, 9.98, 12.19] # ms
# ECHO_TIMES = [1.1, 3.3, 5.5, 7.8, 10.0, 12.2]  # ms
FIELD_STRENGTH = 2.89
DOFS_PAIRS = [(0, 4), (2, 5), (1, 3)]

@ma.machine()
@ma.input('data', 'dti_registered', handler=default_handler)
#@ma.input('data', 'dti_denoised', handler=default_handler)
@ma.output('dti_dofs', handler=default_handler)
#@ma.parameter('averaging', default=None)
@ma.parameter('method', ma.Choice(['fwqpbo']), default='fwqpbo') # todo: add alternative method
def dti_dofs(data, method):
    """Dixon olefinic fat suppression"""
    import numpy as np
    from scipy import ndimage
    from mutools_dti import utils, dofs

    db = utils.NanoDB(data["info"]["metadata"])
    db = db(db.type == "STE")
    volumes = data["volumes"]
    satvol = {k: v for k, v in volumes.items() if k.startswith("saturation_mask")}

    formatter1 = utils.Formatter("dwi_ste_tm{mixtime:03.0f}")
    formatter2 = utils.Formatter(
        "dwi_ste_tm{mixtime:03.0f}_pair{readoutpair:1d}_idir{bidx:02d}"
    )
    formatter3 = utils.Formatter("dwi_ste_tm{mixtime:03.0f}_idir{bidx:02d}")
    formatter4 = utils.Formatter(
        "saturation_mask_tm{mixtime:03.0f}_ro{readout:02d}_idir{bidx:02d}"
    )

    # dofsopts = {'pairs': DOFS_PAIRS, 'B0': FIELD_STRENGTH}

    fatsuppressed = {}
    mixtimes = db.unique("mixtime")
    for mixtime, group in db.groupby("mixtime"):
        title = formatter1(mixtime)
        print(f"Fat/water separation: {title}")
        if mixtime == min(mixtimes):
            # first solve for B = 0
            bmin = min(group.unique("bvalue"))
            bminfiles = list(group(group.bvalue == bmin))
            bminvols = [volumes[file] for file in bminfiles]

            # mask
            mask = dofs.foreground_mask(bminvols)
            # fw separation for (nominal) b=0
            # exclude/Minimize effect of muscle twitches
            bmin_median = np.median(np.abs(np.stack(bminvols, axis=-1)), axis=-1)
            for vol in bminvols:
                vol[np.abs(vol) < 0.65 * bmin_median] = np.nan
            nanmask = np.isnan(np.abs(np.stack(bminvols)))
            fatsuppressed.update({"nanmask": np.moveaxis(nanmask, 0, -1)})
            b0res = dofs.dofs_b0(ECHO_TIMES, bminvols, mask=mask, B0=FIELD_STRENGTH, method=method)
            ffmap = dofs.make_ffmap(b0res["wmap"], b0res["fmap"], mask)

            name = formatter1(mixtime=mixtime)
            fatsuppressed.update(
                {
                    name + "_b0_mask": mask,
                    name + "_b0_ffmap": ffmap,
                    name + "_b0_wmap": b0res["wmap"],
                    name + "_b0_fmap": b0res["fmap"],
                    name + "_b0_B0map": b0res["b0map"],
                    name + "_b0_r2star": b0res["r2star"],
                    name + "_b0_resids": b0res["resids"],
                }
            )

        # fw separation for all b-values using pre-calculated T2* and B0 map from first mixing time
        for idx, subgroup in group.groupby("idir"):
            files = list(subgroup)
            vols = [volumes[file] for file in files]
            bval = subgroup.unique("bvalue")[0]

            # dofs pair-wise
            _wmaps = []
            for pairi in list(range(len(DOFS_PAIRS))):
                pair = []
                pair.append(DOFS_PAIRS[pairi])
                opts = {
                    "pairs": pair,
                    "B0": FIELD_STRENGTH,
                    "apply_b0_correction": False,
                }

                wmap, fmap = dofs.dofs(ECHO_TIMES, vols, b0res, bval, **opts)
                ffmap = dofs.make_ffmap(wmap, fmap, mask)
                # clipping to avoid binning issues
                wmap[wmap>5e4] = 0
                fmap[fmap>5e4] = 0

                _wmaps.append(wmap)

                name = formatter2(mixtime=mixtime, readoutpair=pairi+1, bidx=idx)
                fatsuppressed.update(
                    {
                        name + "_wmap": wmap,
                        name + "_fmap": fmap,
                        name + "_ffmap": ffmap,
                    }
                )

            # AVERAGING OF DOFS PAIRS
            # The mean and a weighted mean are calculated. A weighted mean might reduce signal drops from muscle twitches.
            # Saturated voxels (from Dicom saturation) will be excluded by applying the saturation masks before averaging.
            for pairi in list(range(len(DOFS_PAIRS))):
                ro = DOFS_PAIRS[pairi]
                key1 = formatter4(mixtime=mixtime, readout=ro[0]+1, bidx=idx)
                key2 = formatter4(mixtime=mixtime, readout=ro[1]+1, bidx=idx)
                satmask = (satvol.get(key1, np.zeros_like(wmap)) > 0) | (
                        satvol.get(key2, np.zeros_like(wmap)) > 0
                )

                mask = satmask == False
                _wmaps[pairi] = _wmaps[pairi] * mask


            name2 = formatter3(mixtime=mixtime, bidx=idx)
            spacing = np.array(wmap.spacing)
            wmap_avg_m = 0 * wmap.copy()
            wmap_avg_wm = 0 * wmap.copy()
            _wmaps = np.array(_wmaps)
            _wmaps[_wmaps == 0] = np.nan
            wmask = np.ma.masked_array(_wmaps, np.isnan(_wmaps))

            # STANDARD MEAN
            wmap_avg_m[:] = np.ma.mean(wmask, axis=0)
            fatsuppressed.update({name2 + "_wmap_mean": wmap_avg_m})

            # WEIGHTED MEAN
            # estimate weights, ignoring nans and zeros
            weights = []
            norm = np.percentile(_wmaps, 99, axis=0)
            for p in list(range(len(DOFS_PAIRS))):
                _weight = 0 * wmap.copy()
                tmp = np.squeeze(_wmaps[p, ::]) / norm

                v = tmp.copy()
                v[np.isnan(v)] = 0
                vv = ndimage.gaussian_filter(v, sigma=3/spacing)
                w = 0 * tmp + 1
                w[np.isnan(tmp)] = 0
                ww = ndimage.gaussian_filter(w, sigma=3/spacing)
                ww[ww == 0] = np.nan

                _weight[:] = vv/ww

                fatsuppressed.update({name2 + "_wmap_weights_" + str(p+1): _weight})
                weights.append(_weight)
            _weights = np.array(weights)
            _weights[_weights == 0] = 1e-5
            #weighted averaging (similar to Fuehres et al, MRM 2023)
            wmap_avg_wm[:] = np.ma.average(
                wmask, axis=0, weights=np.array(_weights)**4
            )

            fatsuppressed.update({name2 + '_wmap_weighted_mean': wmap_avg_wm})

    return {'volumes': fatsuppressed, 'info': {'metadata': db}}


#BANDWIDTH = 1680 # tmp

# =============================================================================
# @ma.machine()
# @ma.input('data', 'dti_dofs', handler=default_handler)
# @ma.input('dixon', 'dixon3pt', type='fwdata')
# @ma.output('dti_b0corr', handler=default_handler)
# def dti_b0corr(data, dixon):
#     """ B0 correction """
#     bandwidth = BANDWIDTH #db.unique('pixel_bandwidth')[0]
#     
#     volumes = data['volumes']
#     deltaB0 = dixon['b0map']
# 
#     wmapfiles = [file for file in volumes if 'wmap' in file]
#     wmapvols = [volumes[file] for file in wmapfiles]
# 
#     # B0 displacement correction
#     corr, dpmap, dfmap = b0corr.displacement_correction(wmapvols, deltaB0, bandwidth)
#     b0corrected = dict(zip(wmapfiles, corr))
#     b0corrected.update({
#         'dpmap': dpmap,
#         'dfmap': dfmap,
#     })
# 
#     # store
#     return {'volumes': b0corrected}
# =============================================================================

@ma.machine()
#@ma.input('b0corr', 'dti_b0corr', handler=default_handler)
@ma.input("data", "dti_dofs", handler=default_handler)
@ma.output("dti_fit", handler=default_handler)
@ma.parameter("avg", default="_wmap_weighted_mean")
@ma.parameter("testing", default="False")
def dti_tensorcalc(data, avg="_wmap_weighted_mean", testing="False"):
    """ Estimate diffusion tensors"""
    from mutools import io
    from mutools_dti import utils, tensorcalc


    db = utils.NanoDB(data["info"]["metadata"])
    wmaps = data["volumes"]
    wmaps = {k: v for k, v in wmaps.items() if k.endswith(avg)}

    formatter1 = utils.Formatter("dwi_ste_tm{mixtime:03.0f}_idir{direction:02d}")
    formatter2 = utils.Formatter("dwi_ste_tm{mixtime:03.0f}")

    tensors = {}
    for mixtime, group in db(db.type == "STE").groupby("mixtime"):
        title = formatter2(mixtime=mixtime)
        print(f"Compute tensor for: {title}")

        # load volumes
        idirs = group.unique("idir")
        
        files = [formatter1(mixtime=mixtime, direction=idir) + avg for idir in idirs]
        vols = [wmaps[file] for file in files]
        #bmatrix the same for all readouts
        group = group(group.readout==1)
        bmat = [group[k]["bmatrix"] for k in group.keys()]
        bmat = [tensorcalc.rearrange_bmatrix(bmatrix) for bmatrix in bmat]
        
        #exclude zero voxels (from saturation)
        b0file = [b0f for b0f in files if "idir01" in b0f]
        b0 = wmaps[b0file[0]]
        mask = b0 > 0

        if testing == "False":
            S0fit, tensor, resids = tensorcalc.tensor_calc(vols, bmat, mask=mask)
            diffmaps, invalid = tensorcalc.dti_metrics(tensor, mask)

        elif testing == "True":
            S0_lsq, tensor_lsq, S0fit, tensor, resids = tensorcalc.tensor_calc(
                vols, bmat, mask=mask, return_lsq=True
            )
            diffmaps_lsq, invalid_lsq = tensorcalc.dti_metrics(tensor_lsq, mask)
            diffmaps, invalid = tensorcalc.dti_metrics(tensor, mask)
            tensors.update(
                {
                    title + "_S0_lsq": io.Volume(S0_lsq, **vols[0].metadata),
                    title + "_MD_lsq": io.Volume(diffmaps_lsq["MD"], **vols[0].metadata),
                    title + "_FA_lsq": io.Volume(diffmaps_lsq["FA"], **vols[0].metadata),
                    title + "_e1_lsq": io.Volume(diffmaps_lsq["e1"], **vols[0].metadata),
                    title + "_e2_lsq": io.Volume(diffmaps_lsq["e2"], **vols[0].metadata),
                    title + "_e3_lsq": io.Volume(diffmaps_lsq["e3"], **vols[0].metadata),
               }
            )

        tensors.update(
            {
                title + "_mask": io.Volume(mask, **vols[0].metadata),
                title + "_S0_fit": io.Volume(S0fit, **vols[0].metadata),
                title + "_tensor": tensor,
                title + "_resids": io.Volume(resids, **vols[0].metadata),
                title + "_MD": io.Volume(diffmaps["MD"], **vols[0].metadata),
                title + "_e1": io.Volume(diffmaps["e1"], **vols[0].metadata),
                title + "_e2": io.Volume(diffmaps["e2"], **vols[0].metadata),
                title + "_e3": io.Volume(diffmaps["e3"], **vols[0].metadata),
                title + "_RD": io.Volume(diffmaps["Drad"], **vols[0].metadata),
                title + "_FA": io.Volume(diffmaps["FA"], **vols[0].metadata),
                title + "_invalid_voxels": invalid,
            }
        )
        
    return {
        "volumes": {**tensors},
        "info": {"metadata": db},
    }


@ma.machine()
@ma.input("data", "dti_fit", handler=default_handler)
@ma.input("mask", "dti_denoised", handler=default_handler)
@ma.output("dti_rpbm", handler=default_handler)
@ma.parameter("difftimes", default=[116.3,216.3,316.3,416.3])
@ma.parameter("QC", default=False)
@ma.parameter("fit_method", default='dictionary')
# Options: 'dictionary', 'lsq'
@ma.parameter("RPBM_dict", default='')
def dti_rpbm(data, mask, difftimes=[116.3,216.3,316.3,416.3], QC=False, fit_method='dictionary', RPBM_dict=''):
    """ Estimate RPBM parameters """
    import numpy as np
    from scipy import ndimage, stats
    from scipy.optimize import curve_fit
    from mutools_dti import utils, rpbm
    from mutools import io

    # Difftimes list may be handled as str...
    if isinstance(difftimes, str):
        difftimes = difftimes.strip("[]")
        difftimes = [float(x) for x in difftimes.split(",")]

    if fit_method == 'dictionary':
        # Load or generate dictionary
        if not RPBM_dict:
            print(f"No RPBM dictionary provided. Generate dictionary.")
            rpbmdict = rpbm.RPBM_Dictionary(1000, 500, difftimes, rpbm.rpbm_calc_dt)
        else:
            try:
                _rpbmdict = np.load(RPBM_dict)
                rpbmdict = rpbm.RPBM_Dictionary.__new__(rpbm.RPBM_Dictionary)
                rpbmdict.__dict__.update({k: _rpbmdict[k] for k in _rpbmdict.files})

                print(f"Dictionary found and loaded: {RPBM_dict}")

            except ValueError:
                _rpbmdict = np.load(RPBM_dict, allow_pickle=True)
                rpbmdict = _rpbmdict['arr_0'].item()

                print(f"Dictionary found and loaded: {RPBM_dict}")

            except FileNotFoundError:
                print(f"Provided RPBM dictionary file does not exist!")

    # Load and prepare mask
    mask = mask["volumes"]
    mask = mask['dwi_ste_tm100_ro01_mask']
    mask = ndimage.binary_fill_holes(mask, axes=(0,1))

    # Load and prepare data
    db = utils.NanoDB(data["info"]["metadata"])

    vols = data["volumes"]
    ADs = {k: v for k, v in vols.items() if k.endswith("_e1")}
    RDs = {k: v for k, v in vols.items() if k.endswith("_RD")}

    volADs = [ADs[f] for f in ADs]
    volRDs = [RDs[f] for f in RDs]

    AD = np.stack(volADs, axis=-1)
    RD = np.stack(volRDs, axis=-1)

    # Fix AD to AD(TM>100ms)
    Dfix = np.nanmean(AD[...,1::], axis=-1)
    uDfix = np.nanstd(AD[...,1::], axis=-1, ddof=1)

    # QC step: Create mask
    finite_mask = np.all(np.isfinite(RD), axis=-1) & np.isfinite(Dfix)

    if (QC==True) | (QC=='True') | (QC==1):
        print(f"Use finite mask and enforce decreasing RD as quality control")
        # RDmask = np.all(np.diff(RD, axis=-1) < 0, axis=-1)    # Too strict
        RDmed = np.nanmedian(RD, axis=-1)
        eps = 0.01 * RDmed[... ,np.newaxis]
        eps_end = 0.005 * RDmed
        dRD = np.diff(RD, axis=-1)
        viol = np.sum(dRD > eps, axis=-1)
        max_viol = 1
        ddRD = RD[...,0] >= (RD[..., -1] - eps_end)
        RDmask = ddRD & (viol <= max_viol) & (RDmed > 0)

        valid_mask = finite_mask & RDmask
    else:
        print(f"Use finite mask and denoising mask")
        valid_mask = finite_mask & mask

    # Flatten arrays
    N = np.prod(volRDs[0].shape)
    TMs = len(difftimes)
    RD_flat = RD.reshape(N, TMs)
    Dfix_flat = Dfix.reshape(N)
    uDfix_flat = uDfix.reshape(N)
    valid_idx = np.where(valid_mask.reshape(N))[0]

    # Allocate outputs
    acorr = np.full(N, np.nan, dtype=np.float32)
    kappa = np.full(N, np.nan, dtype=np.float32)
    SV    = np.full(N, np.nan, dtype=np.float32)
    tau   = np.full(N, np.nan, dtype=np.float32)
    zeta  = np.full(N, np.nan, dtype=np.float32)
    tortuosity = np.full(N, np.nan, dtype=np.float32)
    TD    = np.full(N, np.nan, dtype=np.float32)
    TR    = np.full(N, np.nan, dtype=np.float32)
    uacorr = np.full(N, np.nan, dtype=np.float32)
    ukappa = np.full(N, np.nan, dtype=np.float32)
    uSV    = np.full(N, np.nan, dtype=np.float32)
    utau   = np.full(N, np.nan, dtype=np.float32)
    uzeta  = np.full(N, np.nan, dtype=np.float32)
    utortuosity = np.full(N, np.nan, dtype=np.float32)
    uTD    = np.full(N, np.nan, dtype=np.float32)
    uTR    = np.full(N, np.nan, dtype=np.float32)


    if fit_method == 'lsq':
        def fit_func(t, tau, zeta, Dfix_i):
            return Dfix_i * np.array([rpbm.rpbm_calc_dt(ti / tau, zeta) for ti in t])

    for idx in valid_idx:
        iRD = RD_flat[idx,:]
        iDfix = Dfix_flat[idx]
        iuDfix = uDfix_flat[idx]

        if fit_method == 'lsq':
            params, pcov = curve_fit(
                lambda t, tau, zeta: fit_func(t, tau, zeta, iDfix),
                difftimes,
                iRD,
                p0=[300, 2],    # tau[ms], zeta
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev = 10000,
            )

            tau_fit, zeta_fit = params

            # confidence intervals for processing (95%)
            alpha = 0.05
            n = len(iRD)
            dof = max(0, n - 2)
            tval = stats.t.ppf(1.0 - alpha / 2., dof) if dof > 0 else 1.96
            conf_int = tval * np.sqrt(np.diag(pcov))

            confidence = [[iuDfix, iuDfix],
                          [tau_fit - conf_int[0], tau_fit + conf_int[0]],
                          [zeta_fit - conf_int[1], zeta_fit + conf_int[1]]]

        elif fit_method == 'dictionary':
            tau_fit, zeta_fit, tau_list, zeta_list = rpbm.match_rpbm_robust(iRD, iDfix, rpbmdict)

            # Confidence intervals either 95% (2.5/97.5 percentiles) or 1sigma (16/84 percentiles)
            tau_l = np.percentile(tau_list, 2.5)
            tau_h = np.percentile(tau_list, 97.5)
            zeta_l = np.percentile(zeta_list, 2.5)
            zeta_h = np.percentile(zeta_list, 97.5)

            confidence = [[iuDfix, iuDfix],
                          [tau_l, tau_h],
                          [zeta_l, zeta_h]]

        else:
            raise Exception("Unknown fit method given as input.")

        rpbm_params, urpbm_params = rpbm.rpbm_process(
            [iDfix, tau_fit, zeta_fit],
            confidence
        )

        acorr[idx] = rpbm_params.get("a_corr", np.nan)
        kappa[idx] = rpbm_params.get("kappa", np.nan)
        SV[idx] = rpbm_params.get("SV", np.nan)
        tau[idx] = rpbm_params.get("tau", np.nan)
        zeta[idx] = rpbm_params.get("zeta", np.nan)
        tortuosity[idx] = rpbm_params.get("tortuosity", np.nan)
        TD[idx] = rpbm_params.get("TD", np.nan)
        TR[idx] = rpbm_params.get("TR", np.nan)

        uacorr[idx] = urpbm_params.get("a_corr", np.nan)
        ukappa[idx] = urpbm_params.get("kappa", np.nan)
        uSV[idx] = urpbm_params.get("SV", np.nan)
        utau[idx] = urpbm_params.get("tau", np.nan)
        uzeta[idx] = urpbm_params.get("zeta", np.nan)
        utortuosity[idx] = urpbm_params.get("tortuosity", np.nan)
        uTD[idx] = urpbm_params.get("TD", np.nan)
        uTR[idx] = urpbm_params.get("TR", np.nan)

    # Reshape back
    acorr = acorr.reshape(volRDs[0].shape)
    kappa = kappa.reshape(volRDs[0].shape)
    SV = SV.reshape(volRDs[0].shape)
    tau = tau.reshape(volRDs[0].shape)
    zeta = zeta.reshape(volRDs[0].shape)
    tortuosity = tortuosity.reshape(volRDs[0].shape)
    TD = TD.reshape(volRDs[0].shape)
    TR = TR.reshape(volRDs[0].shape)

    uacorr = uacorr.reshape(volRDs[0].shape)
    ukappa = ukappa.reshape(volRDs[0].shape)
    uSV = uSV.reshape(volRDs[0].shape)
    utau = utau.reshape(volRDs[0].shape)
    uzeta = uzeta.reshape(volRDs[0].shape)
    utortuosity = utortuosity.reshape(volRDs[0].shape)
    uTD = uTD.reshape(volRDs[0].shape)
    uTR = uTR.reshape(volRDs[0].shape)

    # Clip parameter maps to realistic values (exclude outliers)
    acorr[acorr>500] = np.nan
    kappa[kappa>1] = np.nan
    SV[SV>1] = np.nan
    tau[tau>10000] = np.nan
    zeta[zeta>10] = np.nan
    tortuosity[tortuosity>20] = np.nan
    TD[TD>100000] = np.nan
    TR[TR>100000] = np.nan

    # Use metadata of first RD
    meta = getattr(volRDs[0], "meta", {})

    return {
        "volumes":{
            "RPBM_acorr": io.Volume(acorr, **meta),
            "RPBM_kappa": io.Volume(kappa, **meta),
            "RPBM_SV": io.Volume(SV, **meta),
            "RPBM_tau": io.Volume(tau, **meta),
            "RPBM_zeta": io.Volume(zeta, **meta),
            "RPBM_tortuosity": io.Volume(tortuosity, **meta),
            "RPBM_Td": io.Volume(TD, **meta),
            "RPBM_Tr": io.Volume(TR, **meta),
            "RPBM_uacorr": io.Volume(uacorr, **meta),
            "RPBM_ukappa": io.Volume(ukappa, **meta),
            "RPBM_uSV": io.Volume(uSV, **meta),
            "RPBM_utau": io.Volume(utau, **meta),
            "RPBM_uzeta": io.Volume(uzeta, **meta),
            "RPBM_utortuosity": io.Volume(utortuosity, **meta),
            "RPBM_uTd": io.Volume(uTD, **meta),
            "RPBM_uTr": io.Volume(uTR, **meta),
            "denoised_mask": io.Volume(mask, **meta),
            "finite_mask": io.Volume(finite_mask, **meta),
            "valid_mask": io.Volume(valid_mask, **meta)
        },
        "info": {"metadata": db},
    }


# @ma.machine()
# @ma.input('dti', handler=default_handler)
# @ma.input('roi', type='roi', variable=True)
# @ma.output('dti_indices', handler=default_handler)
# def dti_indices(dti, roi):
#     """ Compute diffusion indices"""
#     db = utils.NanoDB(dti['info']['metadata'])
#     volumes = dti['volumes']
#
#     # load ROI
#     labels = roi['labels']
#     roi = roi['roi']
#
#     formatter = utils.Formatter('dwi_ste_tm{mixtime:03.0f}')
#     results = {}
#     maps = {}
#     for mixtime, group in db.groupby('mixtime'):
#         title = formatter(mixtime=mixtime)
#         print(f"Compute Diffusion indices for {title}")
#
#         tensor = volumes[title + '_tensor']
#         mask = volumes[title + '_mask']
#         diffmaps, invalid = tensorcalc.dti_metrics(tensor, mask)
#
#         maps.update({f'{title}_{name}': diffmaps[name] for name in diffmaps})
#         maps.update({f'{title}_mask': mask})
#
#         # add SNR maps
#         snr0 = volumes[title + '_snr0']
#         snr400 = volumes[title + '_snr400']
#
#         # interpolate roi
#         if not roi.shape == mask.shape:
#             roi = interpolate.interpolate_like(mask, roi, method='nearest')
#
#         # compute statistics in ROIs
#         stats = tensorcalc.dti_stats(maps, roi * invalid, labels=labels)
#         results[title + '_stats'] = stats
#
#     return {
#         'volumes': {**maps, 'roi': roi},
#         'tables': results,
#         'labels': {'labels': labels}
#     }


