#!/usr/bin/env python

import sys
import itk
import argparse
import numpy as np

def extract_and_overlay_slices(input_image, label_map, output_image, opacity, num_slices, orientation):
    # Define pixel types and dimensions
    PixelType = itk.ctype("unsigned char")
    LabelType = itk.ctype("unsigned char")
    ULLabelType = itk.ctype("unsigned long")
    Dimension = 3

    # Define image types
    ImageType = itk.Image[PixelType, Dimension]
    LabelImageType = itk.Image[LabelType, Dimension]
    RGBImageType = itk.Image[itk.RGBPixel[PixelType], 2]

    # Read input image and label map
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(input_image)
    reader.Update()

    label_reader = itk.ImageFileReader[LabelImageType].New()
    label_reader.SetFileName(label_map)
    label_reader.Update()

    # Convert label map to label map type
    LabelObjectType = itk.StatisticsLabelObject[ULLabelType, Dimension]
    LabelMapType = itk.LabelMap[LabelObjectType]

    converter = itk.LabelImageToLabelMapFilter[LabelImageType, LabelMapType].New()
    converter.SetInput(label_reader.GetOutput())
    converter.Update()

    # Create overlay filter
    overlay_filter = itk.LabelMapOverlayImageFilter[LabelMapType, ImageType, RGBImageType].New()
    overlay_filter.SetInput(converter.GetOutput())
    overlay_filter.SetFeatureImage(reader.GetOutput())
    overlay_filter.SetOpacity(opacity)

    # Extract slices based on orientation
    input_array = itk.array_from_image(reader.GetOutput())
    label_array = itk.array_from_image(label_reader.GetOutput())

    if orientation == "axial":
        slice_axis = 2
    elif orientation == "coronal":
        slice_axis = 1
    elif orientation == "sagittal":
        slice_axis = 0
    else:
        raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")

    # Calculate slice indices
    total_slices = input_array.shape[slice_axis]
    slice_indices = np.linspace(0, total_slices - 1, num=num_slices, dtype=int)

    # Extract and overlay slices
    overlay_slices = []
    for idx in slice_indices:
        if orientation == "axial":
            input_slice = input_array[:, :, idx]
            label_slice = label_array[:, :, idx]
        elif orientation == "coronal":
            input_slice = input_array[:, idx, :]
            label_slice = label_array[:, idx, :]
        elif orientation == "sagittal":
            input_slice = input_array[idx, :, :]
            label_slice = label_array[idx, :, :]

        # Create 2D images from slices
        input_slice_image = itk.image_from_array(input_slice)
        label_slice_image = itk.image_from_array(label_slice)

        # Convert label slice to label map
        label_slice_converter = itk.LabelImageToLabelMapFilter[LabelImageType, LabelMapType].New()
        label_slice_converter.SetInput(label_slice_image)
        label_slice_converter.Update()

        # Overlay slices
        overlay_filter_slice = itk.LabelMapOverlayImageFilter[LabelMapType, ImageType, RGBImageType].New()
        overlay_filter_slice.SetInput(label_slice_converter.GetOutput())
        overlay_filter_slice.SetFeatureImage(input_slice_image)
        overlay_filter_slice.SetOpacity(opacity)
        overlay_filter_slice.Update()

        overlay_slices.append(itk.array_from_image(overlay_filter_slice.GetOutput()))

    # Concatenate slices horizontally
    concatenated_slices = np.concatenate(overlay_slices, axis=1)

    # Save concatenated slices as a single 2D RGB image
    concatenated_image = itk.image_from_array(concatenated_slices.astype(np.uint8))
    itk.imwrite(concatenated_image, output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay 3D Label Map on Top of a 3D MRI Image and Save Concatenated 2D Slices.")
    parser.add_argument("input_image", help="Input 3D MRI image file.")
    parser.add_argument("label_map", help="Input 3D label map file.")
    parser.add_argument("output_image", help="Output 2D RGB image file.")
    parser.add_argument("--opacity", type=float, default=0.5, help="Opacity of the overlay (0.0 to 1.0).")
    parser.add_argument("--num_slices", type=int, default=5, help="Number of slices to overlay and concatenate.")
    parser.add_argument("--orientation", choices=["axial", "coronal", "sagittal"], default="axial",
                        help="Slice orientation: axial, coronal, or sagittal.")

    args = parser.parse_args()

    extract_and_overlay_slices(
        args.input_image,
        args.label_map,
        args.output_image,
        args.opacity,
        args.num_slices,
        args.orientation
    )
