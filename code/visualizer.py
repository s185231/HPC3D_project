import matplotlib.pyplot as plt
import numpy as np

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

    num_plots=3 if thickness2 else num_plots=2

    #----------------Sagittal slice----------------
    fig, ax = plt.subplots(1,num_plots,figsize=figsize)
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