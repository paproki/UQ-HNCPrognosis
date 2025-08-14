"""
Script to visualize a 3D medical image segmentation mask by overlaying it onto the original image.
The process is offline rendering, meaning that no image will be displayed.

Necessary packages:
- nibabel
- numpy
- matplotlib

Install the packages using:
pip install nibabel numpy matplotlib
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from scipy.ndimage import binary_erosion, binary_dilation


def overlay_slices(mri_image, label_image, opacity=0.8, num_slices=5, slice_orientation='axial', contour_only =  False):
    """
    Overlay the label image onto the MRI image.

    Parameters:
    mri_image (numpy.ndarray): The MRI image data.
    label_image (numpy.ndarray): The label image data.
    opacity (float): Opacity of the label overlaid onto the image.
    num_slices (int): Number of slices to overlay.
    slice_orientation (str): Orientation of the slices ('axial', 'coronal', 'sagittal').
    
    Returns:
    list: List of overlay images.
    """
    slices = []
    
    if slice_orientation == 'axial':
        slice_indices = np.where(np.any(label_image, axis=(0, 1)))[0]
    elif slice_orientation == 'coronal':
        slice_indices = np.where(np.any(label_image, axis=(0, 2)))[0]
    elif slice_orientation == 'sagittal':
        slice_indices = np.where(np.any(label_image, axis=(1, 2)))[0]
    else:
        raise ValueError("Invalid slice orientation. Choose from 'axial', 'coronal', or 'sagittal'.")
    
    selected_slices = np.linspace(0, len(slice_indices) - 1, num_slices, dtype=int)
    
    # Detect unique labels in the label image
    unique_labels = np.unique(label_image[label_image > 0])  # Exclude the background (assumed to be 0)

    # Generate a colormap for the labels
    
    colormap = plt.get_cmap('tab20', len(unique_labels))  # Use HSV colormap to generate distinct colors
    label_colors = {label: colormap(i)[:3] for i, label in enumerate(unique_labels)}  # Map labels to RGB

    for idx in selected_slices:
        slice_idx = slice_indices[idx]
        
        if slice_orientation == 'axial':
            mri_slice = mri_image[:, :, slice_idx]
            label_slice = label_image[:, :, slice_idx]
        elif slice_orientation == 'coronal':
            mri_slice = mri_image[:, slice_idx, :]
            label_slice = label_image[:, slice_idx, :]
        elif slice_orientation == 'sagittal':
            mri_slice = mri_image[slice_idx, :, :]
            label_slice = label_image[slice_idx, :, :]
        
        # Normalize the MRI slice to [0, 1]
        mri_slice = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice))
        overlay = np.stack((mri_slice, mri_slice, mri_slice), axis=-1)

        for label, color in label_colors.items():
            if contour_only:
                # Create a binary mask for the current label
                binary_label = (label_slice == label).astype(np.float32)
                
                # Extract the contour of the binary label mask
                contour = binary_label - binary_erosion(binary_label, structure=np.ones((3, 3))).astype(np.float32)
                contour[contour < 0] = 0  # Ensure no negative values
                
                # Overlay the contour with the specified color
                overlay[contour > 0] = (1 - opacity) * overlay[contour > 0] + opacity * np.array(color)
            else:
                # Overlay the entire label region with the specified color
                overlay[label_slice == label] = (1 - opacity) * overlay[label_slice == label] + opacity * np.array(color)
        


        # if contour_only:
        #     # Extract the contour of the label mask
        #     print("HHHEEEERRREEEEEEE")
        #     contour = label_slice - binary_erosion(label_slice, structure=np.ones((10, 10))).astype(np.float32)
        #     contour[contour < 0] = 0  # Ensure no negative values
        #     overlay[contour > 0] = (1 - opacity) * overlay[contour > 0] + opacity * np.array([1, 0, 0])
        # else:
        #     # Overlay the entire label mask
        #     overlay[label_slice > 0] = (1 - opacity) * overlay[label_slice > 0] + opacity * np.array([1, 0, 0])
        

        #overlay[label_slice > 0] = (1 - opacity) * overlay[label_slice > 0] + opacity * np.array([1, 0, 0])
        
        slices.append(overlay)
    
    return slices

def save_overlay_grid(slices, output_path, grid_shape):
    """
    Save the overlay images as a single PNG file organized in a grid.

    Parameters:
    slices (list): List of overlay images.
    output_path (str): Path to save the output PNG file.
    grid_shape (tuple): Shape of the grid (rows, columns).
    """
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    for i, ax in enumerate(axes.flat):
        if i < len(slices):
            ax.imshow(slices[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to handle command line arguments and process the images.
    """
    parser = argparse.ArgumentParser(description='Visualize a 3D medical image segmentation mask by overlaying it onto the original image.')
    parser.add_argument('mri_path', type=str, help='Path to the input MRI NIfTI image.')
    parser.add_argument('label_path', type=str, help='Path to the input label NIfTI image.')
    parser.add_argument('output_path', type=str, help='Path to save the output PNG file.')
    parser.add_argument('--opacity', type=float, default=0.8, help='Opacity of the label overlaid onto the image (default: 0.8).')
    parser.add_argument('--num_slices', type=int, default=5, help='Number of slices to overlay (default: 5).')
    parser.add_argument('--grid_shape', type=str, default='1x5', help='Grid shape for the output image (default: 1x5).')
    parser.add_argument('--slice_orientation', type=str, default='axial', choices=['axial', 'coronal', 'sagittal'], help='Orientation of the slices (default: axial).')
    parser.add_argument('--wireframe', action='store_true', help='Overlay only the contour of the label mask (default: False).')

    args = parser.parse_args()
    
    # Load the MRI and label images
    mri_image = nib.load(args.mri_path).get_fdata()
    label_image = nib.load(args.label_path).get_fdata()
    
    # Parse the grid shape
    grid_shape = tuple(map(int, args.grid_shape.split('x')))
    
    print(f'Input: {args.mri_path}')
    print(f'Mask: {args.label_path}')
    print(f'Output: {args.output_path}')
    print(f'Grid: {args.grid_shape}')
    print(f'Num Slices: {args.num_slices}')
    print(f'Orientation: {args.slice_orientation}')
    print(f'Opacity: {args.opacity}')
    print(f'Wireframe: {args.wireframe}')

    # Generate the overlay slices
    slices = overlay_slices(mri_image, label_image, opacity=args.opacity, num_slices=args.num_slices, slice_orientation=args.slice_orientation, contour_only=args.wireframe)
    
    # Save the overlay slices as a single PNG file
    save_overlay_grid(slices, args.output_path, grid_shape)

if __name__ == "__main__":
    main()
