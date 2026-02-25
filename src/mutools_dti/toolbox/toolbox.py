# prevent matplotlib threading issues
import matplotlib

matplotlib.use("Agg")


import machines as ma
from . import dti, ivim


toolbox = ma.Toolbox('Mutools-DTI')

# Diffusion
group = "Diffusion"
toolbox.add_program("dti-parse", dti.prog_parse_dti, group=group)
toolbox.add_program('dti-denoise', dti.dti_denoise, group=group)
toolbox.add_program('dti-register', dti.dti_register, group=group)
toolbox.add_program('dti-regmaskSNR', dti.dti_regmaskSNR, group=group)
toolbox.add_program('dti-dofs', dti.dti_dofs, group=group)
toolbox.add_program('dti-roi', dti.RoiDTI, group=group)
toolbox.add_program('dti-tensor', dti.dti_tensorcalc, group=group)

toolbox.add_program(
    "simple-ivim",
    ivim.prog_simple_ivim,
    group="IVIM",
    help="Simple IVIM method from 2 Siemens ADC maps.",
)

toolbox.description = "Mutools for DWI/DTI data"
toolbox.name = "MuTools-DIT"
toolbox.meta["DEFAULT_CONFIG"] = "mutools.yml"
