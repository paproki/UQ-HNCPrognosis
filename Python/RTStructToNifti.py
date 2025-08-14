import os
import argparse
import pydicom
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def read_reference_image(reference_path):
    """
    Read reference image (NIfTI or DICOM series)
    
    Parameters:
    -----------
    reference_path : str
        Path to reference image or DICOM folder
    
    Returns:
    --------
    dict: Reference image details
    """
    # Check if it's a directory (likely DICOM series)
    if os.path.isdir(reference_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_path)
        reader.SetFileNames(dicom_names)
        ct_image = reader.Execute()
        
        # Read DICOM files for slice positions
        dicom_files = [pydicom.dcmread(f) for f in dicom_names]
        slice_positions = [float(dcm.ImagePositionPatient[2]) for dcm in dicom_files]
        
        return {
            'image': ct_image,
            'origin': ct_image.GetOrigin(),
            'spacing': ct_image.GetSpacing(),
            'direction': ct_image.GetDirection(),
            'size': ct_image.GetSize(),
            'slice_positions': sorted(slice_positions),
            'dicom_files': dicom_files,
            'is_dicom': True
        }
    
    # Assume it's a NIfTI file
    try:
        nifti_image = sitk.ReadImage(reference_path)
        
        # Print out detailed image information for debugging
        print("NIfTI Image Details:")
        print(f"Origin: {nifti_image.GetOrigin()}")
        print(f"Spacing: {nifti_image.GetSpacing()}")
        print(f"Direction: {nifti_image.GetDirection()}")
        print(f"Size: {nifti_image.GetSize()}")
        
        # Compute slice positions based on image orientation
        # This is a more robust way to handle NIfTI slice positions
        direction = np.array(nifti_image.GetDirection()).reshape(3, 3)
        slice_axis = np.argmax(np.abs(direction[:, 2]))
        
        # Compute slice positions along the primary slice axis
        slice_positions = [
            nifti_image.GetOrigin()[slice_axis] + 
            i * nifti_image.GetSpacing()[slice_axis] * direction[slice_axis, 2]
            for i in range(nifti_image.GetSize()[slice_axis])
        ]
        
        return {
            'image': nifti_image,
            'origin': nifti_image.GetOrigin(),
            'spacing': nifti_image.GetSpacing(),
            'direction': nifti_image.GetDirection(),
            'size': nifti_image.GetSize(),
            'slice_positions': slice_positions,
            'is_dicom': False,
            'slice_axis': slice_axis
        }
    except Exception as e:
        raise ValueError(f"Could not read reference image: {e}")

def read_input_structures_from_config(config_file):
    """
    Read a list of input structures

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    list of strings: List of strings containing the structure name
    """
    input_structs = []
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                parts = line.split()
                if len(parts) > 1:
                    raise ValueError(f"Invalid line in config file: '{line}'. Expected format: '<struct>'.")
                structure = parts[0]
                input_structs.append(structure)
    return input_structs

def transform_point(point, reference_details):
    """
    Transform point from DICOM patient coordinate system to image index
    
    Parameters:
    -----------
    point : list or numpy array
        3D point in patient coordinate system
    reference_image : SimpleITK.Image
        Reference image to use for transformation
    
    Returns:
    --------
    numpy array: Transformed point in image index space
    """
    # Convert point to numpy array 
    point = np.array(point)
    
    # Get image properties
    origin = np.array(reference_details['origin'])
    spacing = np.array(reference_details['spacing'])
    direction = np.array(reference_details['direction']).reshape(3, 3)
    
    # If NIfTI, apply more complex transformation
    if not reference_details['is_dicom']:
        # Transform point using direction matrix
        # First, translate point relative to origin
        translated_point = point - origin
        
        # Apply inverse direction matrix
        # This handles different orientations
        transformed_point = np.linalg.inv(direction).dot(translated_point)
        
        # Convert to image index space
        image_point = transformed_point / spacing
        
        return image_point
    
    # For DICOM, use simpler transformation
    transformed_point = (point - origin) / spacing
    
    return transformed_point

def convert_rtstruct_to_label(rtstruct_path, reference_path, output_path, structlist = "", debug=False):
    """
    Convert RTStruct DICOM to a precise label image matching the CT image
    """
    # Read CT series details
    ref_details = read_reference_image(reference_path)
    ref_image = ref_details['image']
    print(ref_details['origin'])
    print(ref_details['spacing'])
    print(ref_details['direction'])
    print(ref_details['slice_positions'])
    
    # Read the RTStruct DICOM file
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    # Initialize label image with zeros
    label_image = sitk.Image(ref_details['size'], sitk.sitkUInt8)
    label_image.SetOrigin(ref_details['origin'])
    label_image.SetSpacing(ref_details['spacing'])
    label_image.SetDirection(ref_details['direction'])
    
    # Convert label image to numpy for manipulation
    label_array = sitk.GetArrayFromImage(label_image)

    input_structs = []
    if structlist != None:
        input_structs = read_input_structures_from_config(structlist)
    

    print(input_structs)
    # Process each ROI in the RTStruct
    roi_labels = {}
    current_label = 1  # Start labels from 1 to distinguish from background
    
    for roi_sequence in rtstruct.StructureSetROISequence:
        roi_number = roi_sequence.ROINumber
        roi_name = roi_sequence.ROIName.lower()

        if len(input_structs) == 0 or any(roi_name in x  for x in input_structs):
            # Find corresponding contour sequence
            contour_sequence = None
            for seq in rtstruct.ROIContourSequence:
                if seq.ReferencedROINumber == roi_number:
                    contour_sequence = seq
                    break
            
            if not contour_sequence or not hasattr(contour_sequence, 'ContourSequence'):
                print(f"Skipping ROI: {roi_name} (No contour data)")
                continue
            
            
            # Assign a unique label to this ROI
            roi_labels[roi_name] = current_label
            print(f"Processing ROI: {roi_name}")
            
            # Process each contour for this ROI
            slice_contours = {}
            for contour in contour_sequence.ContourSequence:
                # Extract contour data
                if not hasattr(contour, 'ContourData'):
                    continue
                
                contour_data = contour.ContourData
                
                # Convert 1D list to 3D points
                points = np.array(contour_data).reshape(-1, 3)
                
                # Group points by slice
                for point in points:
                    slice_z = point[2]
                    if slice_z not in slice_contours:
                        slice_contours[slice_z] = []
                    slice_contours[slice_z].append(point[:2])
            
            # Rasterize contours for each slice
            for slice_z, poly_points in slice_contours.items():
                # Find the corresponding slice index
                try:
                    slice_index = np.abs(np.array(ref_details['slice_positions']) - slice_z).argmin()
                    
                    # Transform points to image space
                    transformed_points = [
                        transform_point([p[0], p[1], slice_z], ref_details)[:2] 
                        for p in poly_points
                    ]
                    
                    # Create mask for this slice
                    mask = np.zeros(ref_details['size'][:2], dtype=np.uint8)
                    
                    # Use skimage for polygon filling
                    from skimage.draw import polygon
                    rr, cc = polygon(
                        np.array([p[1] for p in transformed_points]),
                        np.array([p[0] for p in transformed_points]),
                        shape=mask.shape
                    )
                    mask[rr, cc] = 1
                    
                    # Update label array
                    label_array[slice_index][mask == 1] = current_label
                    
                except Exception as e:
                    print(f"Error processing slice at z={slice_z}: {e}")
            
            # Increment label for next ROI
            current_label += 1
    
    # Convert back to SimpleITK image
    output_label_image = sitk.GetImageFromArray(label_array)
    output_label_image.SetOrigin(ref_details['origin'])
    output_label_image.SetSpacing(ref_details['spacing'])
    output_label_image.SetDirection(ref_details['direction'])
    
    # Save the label image
    sitk.WriteImage(output_label_image, output_path)
    
    # Debugging: Visualize alignment
    if debug:
        # Convert CT and Label to numpy for visualization
        ct_array = sitk.GetArrayFromImage(ref_image)
        label_array = sitk.GetArrayFromImage(output_label_image)
        
        # Create overlay visualization
        plt.figure(figsize=(15, 5))
        
        # Original CT Slice
        plt.subplot(131)
        plt.title('CT Slice')
        plt.imshow(ct_array[ct_array.shape[0]//2], cmap='gray')
        
        # Label Mask
        plt.subplot(132)
        plt.title('Label Mask')
        plt.imshow(label_array[label_array.shape[0]//2], cmap='jet', alpha=0.5)
        
        # Overlay
        plt.subplot(133)
        plt.title('CT with Label Overlay')
        plt.imshow(ct_array[ct_array.shape[0]//2], cmap='gray')
        plt.imshow(label_array[label_array.shape[0]//2], cmap='jet', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('debug_alignment.png')
        print("Debug visualization saved to debug_alignment.png")
    
    # Print ROI labels for reference
    print("ROI Labels:")
    for roi, label in roi_labels.items():
        print(f"{roi}: {label}")
    
    return output_label_image

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert RTStruct to Label Image')
    parser.add_argument('--rtstruct', 
                        required=True, 
                        help='Path to the RTStruct DICOM file')
    parser.add_argument('--reference', 
                        required=True, 
                        help='Path to the folder containing CT DICOM series')
    parser.add_argument('--structlist', 
                        required=False, 
                        help='Input file with a list of structure labels')
    parser.add_argument('--output', 
                        required=True, 
                        help='Output path for the label image')
    parser.add_argument('--debug', 
                        action='store_true',
                        help='Enable debug visualization')
    
    # Parse arguments
    args = parser.parse_args()

    # Convert RTStruct to label image
    label_image = convert_rtstruct_to_label(
        args.rtstruct, 
        args.reference, 
        args.output,
        args.structlist,
        debug=args.debug
    )
    
    print(f"Label image saved to {args.output}")

if __name__ == "__main__":
    main()