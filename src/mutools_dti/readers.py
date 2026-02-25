import numpy as np
import re

from dicomstack import DICOM
from mutools import io

from . import utils
from . import tensorcalc

def parse_dicom_dwi(stack, parse_ste=True, parse_se=False, parse_noise=False):
    """ parse DICOM DWI data"""

    volumes, metadata = {}, {}
    if parse_ste:
        # stimulated echo
        vol, meta = parse_dicom_ste_aim(stack)
        volumes.update(vol)
        metadata.update(meta)

    # if parse_se:
    #     # spin-echo
    #     vol, meta = parse_dicom_se_aim(stack)
    #     volumes.update(vol)
    #     metadata.update(meta)

    # if parse_noise:
    #     vol, meta = parse_dicom_noise_aim(stack)
    #     volumes.update(vol)
    #     metadata.update(meta)

    return volumes, metadata


def parse_dicom_ste_aim(stack, formatter=None):
    """parse DICOM DWI-STE data"""

    if formatter is None:
        formatter = utils.Formatter('dwi_ste_tm{mixtime}_ro{no_readout:02d}_idir{bindex:02d}')
        formatter2 = utils.Formatter('saturation_mask_tm{mixtime}_ro{no_readout:02d}_idir{bindex:02d}')

    # select data by prefix
    stack = stack(DICOM.SeriesDescription.startswith("aim_ep2d_diff_TM"))
    # unique series descriptions
    descriptions = stack.unique("SeriesDescription")

    volumes = {}
    metadata = {}

    print("Parsing STE data:")
    for descr in descriptions:
        # parse description
        print(f"\t{descr}")
        tmp = re.match(r"(\w+)_ep2d_diff_(TM\d+).*$", descr)
        site, tm = tmp.groups()
        mixtime = int(tm[2:])

        # select stack
        _stack = stack(DICOM.SeriesDescription == descr)
        # get other attributes
        series_numbers = _stack.unique("SeriesNumber")
        # iterate over sequence names
        names = _stack.unique('SequenceName')
        
        # get number of gradient directions        
        ndir = len(_stack.unique('[DiffusionGradientDirection]'))

        # iterate over DTI directions, b-values, readout times (sequences)
        for seq in names:
            # get readout number
            match = re.search(r'(#\d+)', seq)
            scan = match.groups()
            scan = int(scan[0][1:])
            readout = int(np.floor((scan - 1) / ndir) + 1)
            idir = scan % ndir if scan % ndir != 0 else ndir
            
            volume = 1
            for series in series_numbers:
                query = (DICOM.SeriesNumber == series) & (DICOM.SequenceName == seq)
                __stack = _stack(query)
                volpart = __stack.as_volume()
                if __stack.has_field("RescaleSlope"):
                    # phase
                    volpart = np.exp(1j * np.pi * volpart / 4096.0)
                volume = volume * volpart
                
            # diffdir = list(__stack.single("[DiffusionGradientDirection]"))
            # bvalue = __stack.single('[Bvalue]')

            # create b-matrix from elements, extract b-value and direction via eigenvalue decomposition
            bmatrix = __stack.single('[Bmatrix]')
            bvalue, diffdir = tensorcalc.inv_bmatrix(bmatrix)
            

            volume = io.Volume(volume)
            volname = formatter(mixtime, readout, idir)

            # misc acquisition info
            pixel_bandwidth = __stack.single("PixelBandwidth")
            # bandwidth_per_pixel_phase_encode = __stack.single('[BandwidthPerPixelPhaseEncode]')
            phase_encoding_direction = __stack.single("InPlanePhaseEncodingDirection")
            
            #saturation mask
            if np.max(np.abs(volume)) > 4094.5:
                satmask = np.abs(volume) > 4094.5
                satvox = np.sum(satmask[:])
                maskname = formatter2(mixtime, readout, idir)
                volumes[maskname] = satmask
            else:
                satvox = 0
            
            volumes[volname] = volume
            metadata[volname] = {
                "type": "STE",
                "mixtime": mixtime,
                "readout": readout,
                "idir": idir, # direction index
                "bmatrix": bmatrix,
                "bvalue": bvalue,
                "num_directions": ndir,
                "diffusion_direction": diffdir,
                "scan_index": scan,
                'pixel_bandwidth': pixel_bandwidth,
                'phase_encoding_direction': phase_encoding_direction,
                'dicom saturated voxels': satvox,
                "DICOM": {"SeriesDescription": descr, "SeriesNumber": series_numbers, "SequenceName": seq},
            }

    return volumes, metadata



# =============================================================================
# def parse_dicom_se_aim(stack, formatter=None):
#     """ parse DICOM DWI-SE data """
# 
#     if formatter is None:
#         formatter = utils.Formatter('dwi_se_b{bvalue}_ro{readout:03.0f}_dir{direction:02d}')
# 
#     # select data by prefix
#     stack = stack(DICOM.SeriesDescription.startswith("aim_dwi_se"))
# 
#     # unique series descriptions
#     descriptions = stack.unique("SeriesDescription")
# 
#     volumes = {}
#     metadata = {}
#     
#     print('Parsing SE data:')
#     for descr in descriptions:
#         # parse description
#         print(f"\t{descr}")
# 
#         match = re.match(r"(\w+)_dwi_se_(b\d+)_(ro\d+)$", descr)
#         site, bvalue, readout = match.groups()
# 
#         bvalue = int(bvalue[1:])
#         readout = float(readout[2:])
# 
#         # select stack
#         _stack = stack(DICOM.SeriesDescription == descr)
# 
#         # get other attributes
#         series_numbers = _stack.unique("SeriesNumber")
#         # extract bmatrices from first series
#         bmatrices = _stack(DICOM.SeriesNumber == series_numbers[0]).unique("[Bmatrix]")
#         # bvalue = _stack(DICOM.SeriesNumber == series_numbers[0]).single("[Bvalue]")
# 
#         # iterate over DTI directions
#         for idir in range(len(bmatrices) or 1):
#             volume = 1
#             bmatrix = bmatrices[idir] if bmatrices else None
#             for series in series_numbers:
#                 query = (DICOM.SeriesNumber == series) & (DICOM["[Bmatrix]"] == bmatrix)
#                 __stack = _stack(query)
#                 volpart = __stack.as_volume()
#                 if __stack.has_field("RescaleSlope"):
#                     # phase
#                     volpart = np.exp(1j * np.pi * volpart / 4096.0)
#                 volume = volume * volpart
# 
#             diffdir = __stack.single("[DiffusionGradientDirection]", default=None)
#             if diffdir:
#                 bmatrix = list(bmatrix)
#                 diffdir = list(diffdir)
# 
#             # gradient direction
#             volume = io.Volume(volume)
#             volname = formatter(bvalue, readout, idir)
# 
#             # misc acquisition info
#             pixel_bandwidth = __stack.single("PixelBandwidth")
#             # bandwidth_per_pixel_phase_encode = __stack.single('[BandwidthPerPixelPhaseEncode]')
#             phase_encoding_direction = __stack.single("InPlanePhaseEncodingDirection")
# 
#             volumes[volname] = volume
#             metadata[volname] = {
#                 "type": 'SE',
#                 "readout": readout,
#                 "bvalue": bvalue,
#                 "bmatrix": bmatrix,
#                 "diffdir": diffdir,
#                 # misc
#                 "site": site,
#                 'pixel_bandwidth': pixel_bandwidth,
#                 'phase_encoding_direction': phase_encoding_direction,
#                 "DICOM": {"SeriesDescription": descr, "SeriesNumber": series_numbers},
#             }
#         
#     return volumes, metadata
# 
# 
# 
# def parse_dicom_noise_aim(stack, formatter=None):
#     """ parse DICOM DWI-STE data """
# 
#     if formatter is None:
#         formatter = utils.Formatter('dwi_ste_noise_{index:02d}')
# 
#      # select DWI STE data only
#     _stack = stack(DICOM.SeriesDescription.contains('noise'))
# 
#     series_numbers = _stack.unique("SeriesNumber")
# 
#     volumes = {}
#     metadata = {}
#     print('Parsing noise data:')
#     for i, sn in enumerate(series_numbers):
#         descr = _stack.single(DICOM.SeriesDescription)
#         print(f"\t{descr}")
# 
#         index = i//2
#         volname = formatter(index)
#         volume = volumes.get(volname, 1)
# 
#         __stack = _stack(DICOM.SeriesNumber == sn)
#         volpart = __stack.as_volume()
#         if __stack.has_field("RescaleSlope"):
#             # is phase
#             volpart = np.exp(1j * np.pi * volpart / 4096.0)
#         volume = volume * volpart
# 
#         # store volume
#         volumes[volname] = io.Volume(volume)
#         
#         # metadata
#         metadata[volname] = {
#             "type": 'noise',
#             # misc
#             "DICOM": {"SeriesDescription": descr, "SeriesNumber": sn},
#         }
#     
#     return volumes, metadata
# 
# =============================================================================
