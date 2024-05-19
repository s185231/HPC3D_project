import numpy as np
import matplotlib.pyplot as plt

def extract_samples(mask, pixels_per_mm, mm_per_grid=5, discard_threshold=0.5):
    """
    Extracts a grid of samples from a mask where each sample has a meaningful amount of data.

    Args:
        mask: A 3D numpy array representing the mask.
        pxels_per_mm: The number of pixels per mm in the volume.
        mm_per_grid: The physical size of each grid element.
        discard_threshold: The threshold for discarding samples based on the mask.

    Returns:
        A list of dicts that map the indices of the extracted samples
    """
    samples = []
    for i in range(0, mask.shape[0], int(mm_per_grid * pixels_per_mm)):
        for j in range(0, mask.shape[1], int(mm_per_grid * pixels_per_mm)):
            for k in range(0, mask.shape[2], int(mm_per_grid * pixels_per_mm)):
                if np.mean(mask[i:i + int(mm_per_grid * pixels_per_mm), j:j + int(mm_per_grid * pixels_per_mm), k:k + int(mm_per_grid * pixels_per_mm)]) > discard_threshold:
                    samples.append({'i': (i, i + int(mm_per_grid * pixels_per_mm)),
                                    'j': (j, j + int(mm_per_grid * pixels_per_mm)),
                                    'k': (k, k + int(mm_per_grid * pixels_per_mm))})
    return samples

def plot_samples(mask, grid_dict, view=(0, 0), linewidth=3, alpha=1, save_path=None):
    """
    Plots all the samples extracted from extract_samples as a 3D plot as cubes

    mask: The shape of the volume
    grid_dict: A dict containing the indices of the volume to extract as given in extract_samples
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask_shape = mask.shape

    # plot the samples
    for sample in grid_dict:
        i, i_mark = sample['i']
        j, j_mark = sample['j']
        k, k_mark = sample['k']

        ax.plot([i, i_mark], [j, j], [k, k], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i_mark], [j_mark, j_mark], [k, k], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i_mark], [j, j], [k_mark, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i_mark], [j_mark, j_mark], [k_mark, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i], [j, j_mark], [k, k], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i_mark, i_mark], [j, j_mark], [k, k], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i], [j, j_mark], [k_mark, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i_mark, i_mark], [j, j_mark], [k_mark, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i], [j, j], [k, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i_mark, i_mark], [j, j], [k, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i, i], [j_mark, j_mark], [k, k_mark], c='r', linewidth=linewidth, alpha=alpha)
        ax.plot([i_mark, i_mark], [j_mark, j_mark], [k, k_mark], c='r', linewidth=linewidth, alpha=alpha)

    ax.set_xlim(0, mask_shape[0])
    ax.set_ylim(0, mask_shape[1])
    ax.set_zlim(0, mask_shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(*view)
    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
        axis._axinfo['tick']['inward_factor'] = 0.0
        axis._axinfo['tick']['outward_factor'] = 0.0
        axis.set_pane_color((0.95, 0.95, 0.95))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if save_path:
        plt.savefig(save_path+'grid_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_grid(volume, grid_dict):
    """
    Extracts a sliced volume based on the grid_dict.
    
    volume: A 3D numpy array representing the volume.
    grid_dict: A dict containing the indices of the volume to extract as given in extract_samples
    
    """
    i, i_mark = grid_dict['i']
    j, j_mark = grid_dict['j']
    k, k_mark = grid_dict['k']

    return volume[i:i_mark, j:j_mark, k:k_mark]
    