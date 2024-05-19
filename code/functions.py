import numpy as np
import matplotlib.pyplot as plt
import time
import localthickness as lt

################
## STATISTICS ##
################
def get_thickness(bin_sample, mask_sample, pixel_size):
    """
    Calculates the local thickness of a binary sample.

    Args:
        bin_sample: A 3D numpy array representing the binary sample.
        mask_sample: A 3D numpy array representing the mask of the sample.
        pixel_size: The physical size of each pixel.

    Returns:
        A 3D numpy array representing the local thickness of the sample.
    """

    # get the local thickness radius
    local_thickness = lt.local_thickness(bin_sample, mask=mask_sample)
    # get the diameter
    thickness_true = 2*local_thickness
    # convert to physical size
    thickness_true *= pixel_size

    return thickness_true

def get_statistics(bin_sample, mask_sample, pixel_size):
    """
    Calculates the statistics of a binary sample.

    Args:
        bin_sample: A 3D numpy array representing the binary sample.
        mask_sample: A 3D numpy array representing the mask of the sample.
        pixel_size: The physical size of each pixel.

    Returns:
        A dict containing the statistics of the sample.
    """

    # get the stone/total ratio
    ratio = np.sum(bin_sample)/np.sum(mask_sample)

    # get the stone local thickness and time it
    start_time_stone = time.time()
    thickness_stone = get_thickness(bin_sample, mask_sample, pixel_size)
    end_time_stone = time.time()

    # get the bubbles local thickness and time it   
    start_time_bubbles = time.time()
    thickness_bubbles = get_thickness(1-bin_sample, mask_sample, pixel_size)
    end_time_bubbles = time.time()

    # get the time it took to calculate the thickness
    time_stone = end_time_stone - start_time_stone
    time_bubbles = end_time_bubbles - start_time_bubbles

    # get the mean thickness
    mean_thickness_stone = np.mean(thickness_stone[thickness_stone > 0])
    mean_thickness_bubbles = np.mean(thickness_bubbles[thickness_bubbles > 0])

    # get the median thickness
    median_thickness_stone = np.median(thickness_stone[thickness_stone > 0])
    median_thickness_bubbles = np.median(thickness_bubbles[thickness_bubbles > 0])

    # get the max thickness
    max_thickness_stone = np.max(thickness_stone)
    max_thickness_bubbles = np.max(thickness_bubbles)

    # get the thickness variance
    var_thickness_stone = np.var(thickness_stone[thickness_stone > 0])
    var_thickness_bubbles = np.var(thickness_bubbles[thickness_bubbles > 0])

    # create a dictionary with the results
    dict = {'ratio': ratio, 'mean_thickness_stone': mean_thickness_stone, 'mean_thickness_bubbles': mean_thickness_bubbles, 'median_thickness_stone': median_thickness_stone, 'median_thickness_bubbles': median_thickness_bubbles, 'max_thickness_stone': max_thickness_stone, 'max_thickness_bubbles': max_thickness_bubbles, 'var_thickness_stone': var_thickness_stone, 'var_thickness_bubbles': var_thickness_bubbles, 'time_stone': time_stone, 'time_bubbles': time_bubbles}

    return dict

############################
## SAMPLE GRID EXTRACTION ## 
############################
def extract_samples(mask, pixels_per_mm, mm_per_grid=5, discard_threshold=0.5):
    """
    Extracts a grid of samples from a mask where each sample has a meaningful amount of data.
    Each sample is defined by a minimum point (i,j,k) and a maximum point (i_mark, j_mark, k_mark) in the volume.
    From these points, every other corner of the cube can be calculated.

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

def plot_samples(mask, grid_dict, view=(0, 0), linewidth=3, alpha=1, save_path=None):
    """
    Plots all the samples extracted from extract_samples as a 3D plot as cubes

    mask: The shape of the volume
    grid_dict: A dict containing the indices of the volume to extract as given in extract_samples
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask_shape = mask.shape

    # plot every sample
    for sample in grid_dict:
        # Extract the minimum point (i,j,k) and the maximum point (i_mark, j_mark, k_mark) of the sample
        i, i_mark = sample['i']
        j, j_mark = sample['j']
        k, k_mark = sample['k']

        # Plot every edge of the cube
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

    # Plot adjustments
    ax.set_xlim(0, mask_shape[0])
    ax.set_ylim(0, mask_shape[1])
    ax.set_zlim(0, mask_shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(*view)
    ax.axis('equal')
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

def plot_local_thickness(vol, thickness1, thickness2=None, view="sagittal", slice=0, figsize=(15,10)):
    '''
    INPUT:
    vol: Original volume
    thickness1: local thickness of vol
    thickness2: Different local thickness of vol, can be omitted if only one is needed
    slice: Slice to view
    view: Which view to view. Options are (sagittal, coronal, axial)

    OUTPUT
    no output, but plots the local thickness.
    '''

    num_plots=3 if thickness2 else 2

    #----------------Sagittal slice----------------
    fig, ax = plt.subplots(1,num_plots, figsize=figsize)
    match view.lower():
        case "sagittal":
            ax[0].imshow(np.squeeze(vol[slice,:,:]), cmap='gray')
            ax[0].set_title('Original')

            ax[1].imshow(np.squeeze(thickness1[slice,:,:]), cmap='hot')
            ax[1].set_title('Local thickness')
            if thickness2:
                ax[2].imshow(np.squeeze(thickness2[slice,:,:]), cmap='hot')
                ax[2].set_title('Local thickness')

        case "coronal":
            ax[0].imshow(np.squeeze(vol[:,slice,:]), cmap='gray')
            ax[0].set_title('Original slice')

            ax[1].imshow(np.squeeze(thickness1[:,slice,:]), cmap='hot')
            ax[1].set_title('Local thickness')

            if thickness2:
                ax[2].imshow(np.squeeze(thickness2[:,slice,:]), cmap='hot')
                ax[2].set_title('Local thickness')

        case "axial":
            ax[0].imshow(np.squeeze(vol[:,:,slice]), cmap='gray')
            ax[0].set_title('Original slice')

            ax[1].imshow(np.squeeze(thickness1[:,:,slice]), cmap='hot')
            ax[1].set_title('Local thickness')

            if thickness2:
                ax[2].imshow(np.squeeze(thickness2[:,:,slice]), cmap='hot')
                ax[2].set_title('Local thickness')


    plt.show()


    