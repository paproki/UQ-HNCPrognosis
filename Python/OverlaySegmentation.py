#!/usr/bin/env python3
"""
Advanced Medical Image Overlay Visualization Tool

This script creates high-quality overlay visualizations of medical images with segmentation masks.
Supports multiple orientations, flexible slice selection, opacity control, and wireframe mode.

Author: UQ Medical Imaging Team
Dependencies: nibabel, numpy, matplotlib, scipy
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
from scipy import ndimage
from pathlib import Path


def get_slice_indices(image_shape, orientation, num_slices, label_data=None):
    """
    Get slice indices for visualization based on orientation and label presence.
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the 3D image
    orientation : str
        Slice orientation ('axial', 'coronal', 'sagittal')
    num_slices : int
        Number of slices to extract
    label_data : numpy.ndarray, optional
        Label data to find slices with content
        
    Returns:
    --------
    list: Slice indices to visualize
    """
    if orientation == 'axial':
        axis_dims = (0, 1)
        slice_axis = 2
    elif orientation == 'coronal':
        axis_dims = (0, 2)
        slice_axis = 1
    elif orientation == 'sagittal':
        axis_dims = (1, 2)
        slice_axis = 0
    else:
        raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'")
    
    total_slices = image_shape[slice_axis]
    
    if label_data is not None:
        # Find slices that contain label data
        if orientation == 'axial':
            slice_has_label = np.any(label_data, axis=axis_dims)
        elif orientation == 'coronal':
            slice_has_label = np.any(label_data, axis=axis_dims)
        elif orientation == 'sagittal':
            slice_has_label = np.any(label_data, axis=axis_dims)
        
        label_indices = np.where(slice_has_label)[0]
        
        if len(label_indices) > 0:
            # Select evenly spaced slices from those containing labels
            if len(label_indices) >= num_slices:
                indices = np.linspace(0, len(label_indices) - 1, num_slices, dtype=int)
                return label_indices[indices]
            else:
                # If fewer label slices than requested, use all of them
                return label_indices
    
    # Fallback: evenly spaced slices across the entire volume
    return np.linspace(0, total_slices - 1, num_slices, dtype=int)


def apply_orientation_fix(data, affine, orientation):
    """
    Apply orientation corrections to match radiological viewing conventions.
    
    Parameters:
    -----------
    data : numpy.ndarray
        3D image data
    affine : numpy.ndarray
        4x4 affine transformation matrix
    orientation : str
        Slice orientation
        
    Returns:
    --------
    numpy.ndarray: Oriented data
    """
    # Get orientation codes from affine matrix
    ornt = nib.orientations.io_orientation(affine)
    
    # Apply standard radiological orientation
    # RAS+ (Right-Anterior-Superior) is the standard
    data_oriented = nib.orientations.apply_orientation(data, ornt)
    
    # Additional flips for proper display orientation per view
    if orientation == 'axial':
        # For axial: flip left-right for radiological convention
        data_oriented = np.flip(data_oriented, axis=0)
    elif orientation == 'coronal':
        # For coronal: flip anterior-posterior and superior-inferior
        data_oriented = np.flip(data_oriented, axis=1)
        data_oriented = np.flip(data_oriented, axis=2)
    elif orientation == 'sagittal':
        # For sagittal: flip anterior-posterior 
        data_oriented = np.flip(data_oriented, axis=1)
        data_oriented = np.flip(data_oriented, axis=2)
    
    return data_oriented


def extract_slice(data, orientation, slice_idx):
    """Extract a 2D slice from 3D data based on orientation."""
    if orientation == 'axial':
        slice_2d = data[:, :, slice_idx]
        # Transpose for proper anatomical display (anterior at top)
        return np.rot90(slice_2d, k=1)
    elif orientation == 'coronal':
        slice_2d = data[:, slice_idx, :]
        # Transpose for proper anatomical display
        return np.rot90(slice_2d, k=1)
    elif orientation == 'sagittal':
        slice_2d = data[slice_idx, :, :]
        # Transpose for proper anatomical display
        return np.rot90(slice_2d, k=1)


def create_contour_mask(binary_mask, contour_width=1):
    """
    Create contour/wireframe mask from binary mask.
    
    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask (0s and 1s)
    contour_width : int
        Width of the contour in pixels
        
    Returns:
    --------
    numpy.ndarray: Contour mask
    """
    # Create structuring element for morphological operations
    struct = ndimage.generate_binary_structure(2, 1)
    
    # Dilate and erode to create contour
    dilated = ndimage.binary_dilation(binary_mask, structure=struct, iterations=contour_width)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct, iterations=contour_width)
    
    # Contour is the difference between dilated and eroded
    contour = dilated & ~eroded
    
    return contour.astype(np.float32)


def normalize_image(image, percentile_range=(1, 99)):
    """
    Normalize image intensity using percentile range for better visualization.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    percentile_range : tuple
        Lower and upper percentiles for normalization
        
    Returns:
    --------
    numpy.ndarray: Normalized image [0, 1]
    """
    p_low, p_high = np.percentile(image, percentile_range)
    image_norm = np.clip(image, p_low, p_high)
    image_norm = (image_norm - p_low) / (p_high - p_low + 1e-8)
    return image_norm


def get_label_colors(unique_labels, colormap='tab20'):
    """
    Generate distinct colors for each label.
    
    Parameters:
    -----------
    unique_labels : array-like
        Unique label values
    colormap : str
        Matplotlib colormap name
        
    Returns:
    --------
    dict: Mapping from label to RGB color
    """
    if len(unique_labels) == 0:
        return {}
    
    # Use different colormaps based on number of labels
    if len(unique_labels) <= 10:
        cmap = plt.get_cmap('tab10')
    elif len(unique_labels) <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('hsv')
    
    colors = {}
    for i, label in enumerate(unique_labels):
        if len(unique_labels) <= 20:
            colors[label] = cmap(i)[:3]  # RGB only
        else:
            colors[label] = cmap(i / len(unique_labels))[:3]
    
    return colors


def create_overlay_slice(image_slice, label_slice, opacity=0.7, wireframe=False, 
                        contour_width=1, label_colors=None):
    """
    Create overlay visualization for a single slice.
    
    Parameters:
    -----------
    image_slice : numpy.ndarray
        2D image slice
    label_slice : numpy.ndarray
        2D label slice
    opacity : float
        Opacity of overlay (0.0 to 1.0)
    wireframe : bool
        Whether to show only contours
    contour_width : int
        Width of contours in pixels
    label_colors : dict
        Mapping from label values to colors
        
    Returns:
    --------
    numpy.ndarray: RGB overlay image
    """
    # Normalize image slice
    image_norm = normalize_image(image_slice)
    
    # Create RGB base image
    overlay = np.stack([image_norm, image_norm, image_norm], axis=-1)
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return overlay
    
    # Generate colors if not provided
    if label_colors is None:
        label_colors = get_label_colors(unique_labels)
    
    # Apply overlays for each label
    for label in unique_labels:
        if label not in label_colors:
            continue
            
        binary_mask = (label_slice == label)
        color = np.array(label_colors[label][:3])  # Ensure RGB only
        
        if wireframe:
            # Create contour mask
            contour_mask = create_contour_mask(binary_mask, contour_width)
            mask_indices = contour_mask > 0
        else:
            # Use filled regions
            mask_indices = binary_mask
        
        if np.any(mask_indices):
            # Apply overlay with specified opacity
            overlay[mask_indices] = (1 - opacity) * overlay[mask_indices] + opacity * color
    
    return np.clip(overlay, 0, 1)


def create_overlay_visualization(image_path, label_path, output_path, orientation='axial',
                               num_slices=5, opacity=0.7, wireframe=False, contour_width=1,
                               grid_cols=None, figsize_per_slice=(4, 4), dpi=150):
    """
    Create comprehensive overlay visualization.
    
    Parameters:
    -----------
    image_path : str
        Path to input image (NIfTI format)
    label_path : str
        Path to label image (NIfTI format)
    output_path : str
        Path for output image
    orientation : str
        Slice orientation ('axial', 'coronal', 'sagittal')
    num_slices : int
        Number of slices to visualize
    opacity : float
        Overlay opacity (0.0 to 1.0)
    wireframe : bool
        Show contours only
    contour_width : int
        Contour width in pixels
    grid_cols : int, optional
        Number of columns in grid (auto-calculated if None)
    figsize_per_slice : tuple
        Size of each subplot in inches
    dpi : int
        Output resolution
    """
    
    # Load images
    print(f"Loading images...")
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    
    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()
    
    print(f"Original image shape: {image_data.shape}")
    print(f"Original label shape: {label_data.shape}")
    
    # Check dimensions match
    if image_data.shape != label_data.shape:
        raise ValueError(f"Image and label dimensions don't match: {image_data.shape} vs {label_data.shape}")
    
    # Apply orientation corrections to match radiological viewing conventions
    print(f"Applying orientation corrections for {orientation} view...")
    image_data = apply_orientation_fix(image_data, image_nii.affine, orientation)
    label_data = apply_orientation_fix(label_data, label_nii.affine, orientation)
    
    print(f"Oriented image shape: {image_data.shape}")
    print(f"Oriented label shape: {label_data.shape}")
    
    # Get slice indices
    slice_indices = get_slice_indices(image_data.shape, orientation, num_slices, label_data)
    actual_num_slices = len(slice_indices)
    
    print(f"Selected {actual_num_slices} slices: {slice_indices}")
    
    # Get unique labels for consistent coloring
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels > 0]
    label_colors = get_label_colors(unique_labels)
    
    print(f"Found {len(unique_labels)} labels: {unique_labels}")
    
    # Calculate grid layout
    if grid_cols is None:
        grid_cols = min(5, actual_num_slices)  # Max 5 columns
    grid_rows = int(np.ceil(actual_num_slices / grid_cols))
    
    # Create figure
    fig_width = grid_cols * figsize_per_slice[0]
    fig_height = grid_rows * figsize_per_slice[1]
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Handle single subplot case
    if actual_num_slices == 1:
        axes = [axes]
    elif grid_rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    # Create overlays for each slice
    for i, slice_idx in enumerate(slice_indices):
        # Extract slices
        image_slice = extract_slice(image_data, orientation, slice_idx)
        label_slice = extract_slice(label_data, orientation, slice_idx)
        
        # Create overlay
        overlay = create_overlay_slice(image_slice, label_slice, opacity, wireframe, 
                                     contour_width, label_colors)
        
        # Display slice
        ax = axes[i]
        im = ax.imshow(overlay, interpolation='nearest')
        ax.set_title(f'{orientation.capitalize()} slice {slice_idx}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(actual_num_slices, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    mode_str = "Wireframe" if wireframe else "Filled"
    fig.suptitle(f'Medical Image Overlay - {orientation.capitalize()} ({mode_str}, opacity={opacity})', 
                 fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Overlay visualization saved to: {output_path}")
    
    # Print summary
    print("\nVisualization Summary:")
    print(f"  Orientation: {orientation}")
    print(f"  Slices shown: {actual_num_slices}")
    print(f"  Mode: {'Wireframe' if wireframe else 'Filled overlay'}")
    print(f"  Opacity: {opacity}")
    print(f"  Labels found: {len(unique_labels)}")
    if wireframe:
        print(f"  Contour width: {contour_width} pixels")


def main():
    parser = argparse.ArgumentParser(
        description='Create high-quality overlay visualizations of medical images with segmentation masks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic overlay with 5 axial slices
  python OverlaySegmentation.py image.nii.gz labels.nii.gz output.png
  
  # Coronal wireframe with custom opacity
  python OverlaySegmentation.py image.nii.gz labels.nii.gz output.png \\
    --orientation coronal --wireframe --opacity 0.8 --num_slices 8
  
  # High-resolution sagittal overlay
  python OverlaySegmentation.py image.nii.gz labels.nii.gz output.png \\
    --orientation sagittal --num_slices 10 --dpi 300 --contour_width 2
        """
    )
    
    # Positional arguments
    parser.add_argument('image_path', 
                       help='Path to input medical image (NIfTI format)')
    parser.add_argument('label_path', 
                       help='Path to segmentation labels (NIfTI format)')
    parser.add_argument('output_path', 
                       help='Path for output visualization image (PNG/PDF/SVG)')
    
    # Optional arguments
    parser.add_argument('--orientation', 
                       choices=['axial', 'coronal', 'sagittal'], 
                       default='axial',
                       help='Slice orientation (default: axial)')
    
    parser.add_argument('--num_slices', 
                       type=int, 
                       default=5,
                       help='Number of slices to visualize (default: 5)')
    
    parser.add_argument('--opacity', 
                       type=float, 
                       default=0.7,
                       help='Overlay opacity from 0.0 to 1.0 (default: 0.7)')
    
    parser.add_argument('--wireframe', 
                       action='store_true',
                       help='Show only contours/wireframe instead of filled regions')
    
    parser.add_argument('--contour_width', 
                       type=int, 
                       default=1,
                       help='Width of contours in pixels (default: 1)')
    
    parser.add_argument('--grid_cols', 
                       type=int,
                       help='Number of columns in output grid (auto-calculated if not specified)')
    
    parser.add_argument('--dpi', 
                       type=int, 
                       default=150,
                       help='Output resolution in DPI (default: 150)')
    
    parser.add_argument('--figsize_per_slice', 
                       type=float, 
                       nargs=2, 
                       default=[4, 4],
                       help='Size of each subplot in inches [width height] (default: 4 4)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not os.path.exists(args.label_path):
        raise FileNotFoundError(f"Label file not found: {args.label_path}")
    
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("Opacity must be between 0.0 and 1.0")
    
    if args.num_slices < 1:
        raise ValueError("Number of slices must be at least 1")
    
    # Create visualization
    create_overlay_visualization(
        image_path=args.image_path,
        label_path=args.label_path,
        output_path=args.output_path,
        orientation=args.orientation,
        num_slices=args.num_slices,
        opacity=args.opacity,
        wireframe=args.wireframe,
        contour_width=args.contour_width,
        grid_cols=args.grid_cols,
        figsize_per_slice=tuple(args.figsize_per_slice),
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()