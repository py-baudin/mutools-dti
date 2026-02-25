import numpy as np
from matplotlib import pyplot as plt


def plot_profiles(images, locs=[0.5], axis=1, title=""):
    shape = images[0].shape
    indices = [
        tuple(int(shape[i] * loc) if i != axis else ... for i in range(len(shape)))
        for loc in locs
    ]
    fig, axes = plt.subplots(
        nrows=len(locs), ncols=2, sharex=True, sharey=True, figsize=(10, 8)
    )
    for i in range(len(locs)):
        plt.sca(axes[i, 0])
        loc = indices[i]
        plt.title(f"Profile location: {loc}")
        profiles = np.stack([np.abs(np.asarray(im)[loc]) for im in images], axis=1)
        plt.imshow(profiles, aspect="auto", interpolation="nearest")
        plt.ylabel(f"Axis {axis + 1}")
        if i == len(locs) - 1:
            plt.xlabel("image index")

        plt.sca(axes[i, 1])
        loc = indices[i]
        plt.title(f"Difference from first image")
        diff = profiles / profiles[:, 0:1] - 1
        # breakpoint()
        plt.imshow(diff, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar()
        plt.ylabel(f"Axis {axis + 1}")
        if i == len(locs) - 1:
            plt.xlabel("image index")
    plt.suptitle(title)
    plt.tight_layout()
    return fig
