"""
    This file is a utility that can be used to reconstruct label images from RTStruct files used
    in radiationtherapy.

    TODO: Clean up a make functions more generalised and less focused on only Head and Neck Cancer

    Right now the cleanest and most generalisable version would be (convert_rtstruct_to_label_2)

"""

import os
import argparse
import pydicom
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.draw import polygon, polygon_perimeter
from scipy import ndimage
import json

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
    # Previous implementation remains the same
    # (Copy the entire function from the original script)
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
        
        # Compute slice positions based on image orientation
        direction = np.array(nifti_image.GetDirection()).reshape(3, 3)
        slice_axis = np.argmax(np.abs(direction[:, 2]))
        
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


def load_regions_from_json(json_file):
    """
    Loads region mappings from a JSON file into a Python dictionary.

    Args:
        json_file (str): Path to the JSON file containing region mappings.

    Returns:
        dict: A dictionary where keys are region names and values are scalar values,
              or None if an error occurs.
    """
    try:
        with open(json_file, 'r') as f:
            regions_dict = json.load(f)
        return regions_dict
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def transform_point_precise(point, reference_details, subpixel_precision=False):
    """
    Enhanced coordinate transformation with configurable precision
    
    Parameters:
    -----------
    point : numpy array
        3D point in original coordinate system
    reference_details : dict
        Image reference details
    subpixel_precision : bool
        Enable subpixel-level precision handling
    
    Returns:
    --------
    numpy array: Transformed point with controlled precision
    """
    # Convert point to numpy array 
    point = np.array(point, dtype=np.float64)  # Use double precision
    
    # Get image properties
    origin = np.array(reference_details['origin'], dtype=np.float64)
    spacing = np.array(reference_details['spacing'], dtype=np.float64)
    direction = np.array(reference_details['direction'], dtype=np.float64).reshape(3, 3)
    
    # If NIfTI, apply more complex transformation
    if not reference_details['is_dicom']:
        # Translate point relative to origin
        translated_point = point - origin
        
        # Apply inverse direction matrix
        transformed_point = np.linalg.inv(direction).dot(translated_point)
        
        # Convert to image index space
        image_point = transformed_point / spacing
    else:
        # For DICOM, use simpler transformation
        transformed_point = (point - origin) / spacing
        image_point = transformed_point
    
    # Optional: Round to nearest pixel or allow subpixel precision
    if not subpixel_precision:
        image_point = np.round(image_point).astype(np.int32)
    
    return image_point

def rasterize_contour(poly_points, image_size, method='skimage'):
    """
    Multiple methods for contour rasterization
    
    Parameters:
    -----------
    poly_points : list of points
        Transformed polygon points
    image_size : tuple
        Size of the image
    method : str
        Rasterization method
    
    Returns:
    --------
    numpy array: Binary mask
    """
    mask = np.zeros(image_size[:2], dtype=np.uint8)
    
    if method == 'skimage':
        # Standard polygon fill
        rr, cc = polygon(
            np.array([p[1] for p in poly_points]),
            np.array([p[0] for p in poly_points]),
            shape=mask.shape
        )
        mask[rr, cc] = 1
    
    elif method == 'skimage_with_border':
        # Fill with contour border preservation
        rr, cc = polygon(
            np.array([p[1] for p in poly_points]),
            np.array([p[0] for p in poly_points]),
            shape=mask.shape
        )
        mask[rr, cc] = 1
        
        # Add contour border
        border_rr, border_cc = polygon_perimeter(
            np.array([p[1] for p in poly_points]),
            np.array([p[0] for p in poly_points]),
            shape=mask.shape
        )
        mask[border_rr, border_cc] = 2
    
    elif method == 'scipy_distance':
        # Use distance transform for smoother boundaries
        points = np.array(poly_points)
        dist_mask = np.zeros(image_size[:2], dtype=np.float64)
        dist_mask[points[:, 1].astype(int), points[:, 0].astype(int)] = 1
        
        # Distance transform
        dist = ndimage.distance_transform_edt(1 - dist_mask)
        mask = (dist <= 0.5).astype(np.uint8)
    
    return mask

def is_substring_of_key(dictionary, substring):
    """
    Checks if a string is a substring of any key in a dictionary.

    Args:
        dictionary: The dictionary to search.
        substring: The string to check.

    Returns:
        True if the string is a substring of at least one key, False otherwise.
    """
    for key in dictionary:
        if substring in key:
            return True  # Found a match!
    return False  # No match found

def convert_rtstruct_to_label(rtstruct_path, reference_path, output_path, 
                               structlist=None, 
                               precision_method='standard', 
                               rasterization_method='skimage',
                               subpixel_precision=False):
    """
    Convert RTStruct DICOM to a precise label image matching the CT image

    The function takes the RTStruct path, reference
    DCM folder path and output filename. 
    --> In this version the function takes as input a structure file (json) with
    a structure as follows "name": label_value
    {
        "gtvp": 1,
        "gtvp2": 2,
        "gtvp3": 3,
        "gtvn": 4
    }

    If the regions passed is None tje prorgam will reconstruct all the structures
    and assign increasing labels from 1 ++ (Untested)
    
    Additional parameters for improved mask generation:
    precision_method : str
        Method of coordinate transformation ('standard' or 'precise')
    rasterization_method : str
        Method of contour rasterization ('skimage', 'skimage_with_border', 'scipy_distance')
    subpixel_precision : bool
        Enable subpixel-level precision handling
    """
    # Read CT series details
    ref_details = read_reference_image(reference_path)
    
    # Read the RTStruct DICOM file
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    # Initialize label image with zeros
    label_image = sitk.Image(ref_details['size'], sitk.sitkUInt8)
    label_image.SetOrigin(ref_details['origin'])
    label_image.SetSpacing(ref_details['spacing'])
    label_image.SetDirection(ref_details['direction'])
    
    # Convert label image to numpy for manipulation
    label_array = sitk.GetArrayFromImage(label_image)

    regions = []
    if structlist is not None:
        # Example usage:
        regions = load_regions_from_json(structlist)

    # Process each ROI in the RTStruct
    roi_labels = {}
    current_label = 1  # Start labels from 1 to distinguish from background
    
    for roi_sequence in rtstruct.StructureSetROISequence:
        roi_number = roi_sequence.ROINumber
        roi_name = roi_sequence.ROIName.lower()
        print(f'ROI Name: {roi_name}')

        #For now there are too many gtvn so assign single label
        if "gtvn" in roi_name:
            roi_name = "gtvn"
        
        #Different dataset uses gtv instead of gtvp
        if "gtv" in roi_name:
            roi_name = "gtvp"

        if len(regions) == 0 or any(roi_name in x for x in regions.keys()):
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
            if len(regions) > 0:
                current_label = regions[roi_name]
            print(f"Processing ROI: {roi_name} with label {current_label}")
            
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
                try:
                    # Find the corresponding slice index
                    slice_index = np.abs(np.array(ref_details['slice_positions']) - slice_z).argmin()
                    
                    # Transform points based on selected method
                    if precision_method == 'precise':
                        transformed_points = [
                            transform_point_precise([p[0], p[1], slice_z], 
                                                    ref_details, 
                                                    subpixel_precision)[:2] 
                            for p in poly_points
                        ]
                    else:
                        # Use original transformation method
                        transformed_points = [
                            transform_point_precise([p[0], p[1], slice_z], ref_details)[:2] 
                            for p in poly_points
                        ]
                    
                    # Rasterize contour using selected method
                    mask = rasterize_contour(
                        transformed_points, 
                        ref_details['size'], 
                        method=rasterization_method
                    )
                    
                    # Update label array
                    label_array[slice_index][mask == 1] = current_label
                    
                except Exception as e:
                    print(f"Error processing slice at z={slice_z}: {e}")
            
            # Increment label for next ROI
            #current_label += 1
    
    # Convert back to SimpleITK image
    output_label_image = sitk.GetImageFromArray(label_array)
    output_label_image.SetOrigin(ref_details['origin'])
    output_label_image.SetSpacing(ref_details['spacing'])
    output_label_image.SetDirection(ref_details['direction'])
    
    # Save the label image
    sitk.WriteImage(output_label_image, output_path)
    
    return output_label_image


def get_label(lookup_dict, name):
    """
    Look through the dictionnary of label names and labels to find 
    'name'. If name is found in either primary tumour list of labels or 
    lymph nodes list of labels, it will return the type of structure (
    between GTVPrimary or LymphNodes and the associated label)
    """
    if lookup_dict['GTVPrimary']['name'] == name:
        return 'GTVPrimary', lookup_dict['GTVPrimary']['label']
    elif lookup_dict['LymphNodes']['name'] == name:
        return 'LymphNodes', lookup_dict['LymphNodes']['label']
    
    return None, 0

def convert_rtstruct_to_label_2(rtstruct_path, reference_path, output_path, 
                               regions=None, 
                               precision_method='standard', 
                               rasterization_method='skimage',
                               subpixel_precision=False):
    """
    Convert RTStruct DICOM to a precise label image matching the CT image

    The function takes the RTStruct path, reference
    DCM folder path and output filename. 
    --> In this version the function takes as input a dictionnary for the
      patient as follows
    {
        'GTVPrimary': {'name': 'GTV', 'label': 1}, 
        'LymphNodes': {'name': None, 'label': 4}
    }

    If the regions passed is None tje prorgam will reconstruct all the structures
    and assign increasing labels from 1 ++ (Untested)
    
    Additional parameters for improved mask generation:
    precision_method : str
        Method of coordinate transformation ('standard' or 'precise')
    rasterization_method : str
        Method of contour rasterization ('skimage', 'skimage_with_border', 'scipy_distance')
    subpixel_precision : bool
        Enable subpixel-level precision handling
    """
    # Read CT series details
    ref_details = read_reference_image(reference_path)
    
    # Read the RTStruct DICOM file
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    # Initialize label image with zeros
    label_image = sitk.Image(ref_details['size'], sitk.sitkUInt8)
    label_image.SetOrigin(ref_details['origin'])
    label_image.SetSpacing(ref_details['spacing'])
    label_image.SetDirection(ref_details['direction'])
    
    # Convert label image to numpy for manipulation
    label_array = sitk.GetArrayFromImage(label_image)
    print(regions)
    # Process each ROI in the RTStruct
    roi_labels = {}
    current_label = 1  # Start labels from 1 to distinguish from background

    #  Go through ROIs and retrieve name
    #      Get label from dict field for the ROI name
    #      If not 0 then do stuff, otherwise skip
    for roi_sequence in rtstruct.StructureSetROISequence:
        roi_number = roi_sequence.ROINumber
        roi_name = roi_sequence.ROIName
        print(f'ROI Name: {roi_name}')

        label_type, label = None, 0
        if regions is not None:
            label_type, label = get_label(regions, roi_name)

        if regions is None or (label_type is not None and label > 0):
            
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
            if label > 0:
                current_label = label

            print(f"Processing ROI: {roi_name} with label {current_label}")
            
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
                try:
                    # Find the corresponding slice index
                    slice_index = np.abs(np.array(ref_details['slice_positions']) - slice_z).argmin()
                    
                    # Transform points based on selected method
                    if precision_method == 'precise':
                        transformed_points = [
                            transform_point_precise([p[0], p[1], slice_z], 
                                                    ref_details, 
                                                    subpixel_precision)[:2] 
                            for p in poly_points
                        ]
                    else:
                        # Use original transformation method
                        transformed_points = [
                            transform_point_precise([p[0], p[1], slice_z], ref_details)[:2] 
                            for p in poly_points
                        ]
                    
                    # Rasterize contour using selected method
                    mask = rasterize_contour(
                        transformed_points, 
                        ref_details['size'], 
                        method=rasterization_method
                    )
                    
                    # Update label array
                    label_array[slice_index][mask == 1] = current_label
                    
                except Exception as e:
                    print(f"Error processing slice at z={slice_z}: {e}")
            
            # Increment label for next ROI if regions not specific as json dict field
            if label == 0:
                current_label += 1
    
    # Convert back to SimpleITK image
    output_label_image = sitk.GetImageFromArray(label_array)
    output_label_image.SetOrigin(ref_details['origin'])
    output_label_image.SetSpacing(ref_details['spacing'])
    output_label_image.SetDirection(ref_details['direction'])
    
    # Save the label image
    sitk.WriteImage(output_label_image, output_path)
    
    return output_label_image

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert RTStruct to Label Image with Improved Precision')
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
    
    # New parameters for improved mask generation
    parser.add_argument('--precision-method', 
                        choices=['standard', 'precise'], 
                        default='standard',
                        help='Method of coordinate transformation')
    parser.add_argument('--rasterization-method', 
                        choices=['skimage', 'skimage_with_border', 'scipy_distance'], 
                        default='skimage',
                        help='Method of contour rasterization')
    parser.add_argument('--subpixel-precision', 
                        action='store_true',
                        help='Enable subpixel-level precision handling')
    
    # Parse arguments
    args = parser.parse_args()

    # Convert RTStruct to label image with new options
    label_image = convert_rtstruct_to_label(
        args.rtstruct, 
        args.reference, 
        args.output,
        args.structlist,
        precision_method=args.precision_method,
        rasterization_method=args.rasterization_method,
        subpixel_precision=args.subpixel_precision
    )
    
    print(f"Label image saved to {args.output}")

if __name__ == "__main__":
    main()