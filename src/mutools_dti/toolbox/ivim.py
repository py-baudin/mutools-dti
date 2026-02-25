import machines as ma
from mutools.toolbox.common import dicom, handlers


@ma.machine()
@ma.input("stack")
@ma.output("ivim", handler=handlers.universal_handler)
def ivim(stack):
    from mutools_dti import ivim

    # parse dicom stack
    info, volumes = ivim.reader(stack["stack"])

    # compute perfusion map
    perf, pfrac, mask = ivim.simple_ivim(info["bvalues"], volumes)

    return {
        "info": {"info": info},
        "volumes": {
            "pmap": perf,
            "pfrac": pfrac,
            "adc1": volumes[0],
            "adc2": volumes[1],
            "mask": mask,
        },
    }


prog_simple_ivim = [dicom.dicom_loader, ivim]
