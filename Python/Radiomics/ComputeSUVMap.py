import numpy as np
import nibabel as nib
import SimpleITK as sitk
import argparse
import pydicom
import datetime


def compute_suv_map(dicom_directory, output_filename):
    """
    Computes the SUV map of a raw input PET volume from a DICOM series and saves it as an image.
    
    Parameters:
    - dicom_directory: Path to the directory containing the DICOM series.
    - output_filename: Path to save the output SUV map image (nifti).
    
    Returns:
    - suv_map: 3D NumPy array converted to SUVs (standard uptake values).
    """
    
    # Create an ImageSeriesReader object
    reader = sitk.ImageSeriesReader()
    
    # Get the series IDs from the directory
    series_ids = reader.GetGDCMSeriesIDs(dicom_directory)
    
    if not series_ids:
        raise ValueError("No DICOM series found in the specified directory.")
    
    # Get the file names for the first series
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_directory, series_ids[0])
    
    # Set the file names for the reader
    reader.SetFileNames(series_file_names)
    
    # Read the DICOM series into a SimpleITK image
    pet_image = reader.Execute()
    
    # Convert to NumPy array
    raw_pet = sitk.GetArrayFromImage(pet_image)  # Shape: [slices, height, width]
    
    # Extract necessary parameters from DICOM header for SUV calculation
    dicom_data = pydicom.dcmread(series_file_names[0])  # Read first file to extract metadata

    # Accessing required metadata from Radiopharmaceutical Information Sequence
    if hasattr(dicom_data, 'RadiopharmaceuticalInformationSequence') and len(dicom_data.RadiopharmaceuticalInformationSequence) > 0:
        radiopharmaceutical_info = dicom_data.RadiopharmaceuticalInformationSequence[0]
        
        patient_weight = float(dicom_data.PatientWeight)*1000  # in kg
        
        print('Found dcm/radiotherapy data:')
        print(f'AcquisitionTime: {dicom_data.AcquisitionTime} - length {len(dicom_data.AcquisitionTime)}')
        print(f'RadiopharmaceuticalStartTime: {radiopharmaceutical_info.RadiopharmaceuticalStartTime} - length {len(radiopharmaceutical_info.RadiopharmaceuticalStartTime)}')
        print(f'RadionuclideHalfLife: {radiopharmaceutical_info.RadionuclideHalfLife}')
        print(f'RadionuclideTotalDose: {radiopharmaceutical_info.RadionuclideTotalDose}')

        #Extract date time based on dicom tags. Some of the values go down to float precision, need to account for that
        scantime, injection_time = None, None
        if len(dicom_data.AcquisitionTime) > 6:
            scantime = datetime.datetime.strptime(dicom_data.AcquisitionTime,'%H%M%S.%f')
        else:
            scantime = datetime.datetime.strptime(dicom_data.AcquisitionTime,'%H%M%S')

        if len(radiopharmaceutical_info.RadiopharmaceuticalStartTime) > 6:
            injection_time  = datetime.datetime.strptime(radiopharmaceutical_info.RadiopharmaceuticalStartTime,'%H%M%S.%f')  # Radiopharmaceutical Start Time
        else:
            injection_time  = datetime.datetime.strptime(radiopharmaceutical_info.RadiopharmaceuticalStartTime,'%H%M%S')  # Radiopharmaceutical Start Time

        half_life = float(radiopharmaceutical_info.RadionuclideHalfLife)  # Radiopharmaceutical Half Life
        total_dose = float(radiopharmaceutical_info.RadionuclideTotalDose)  # Radiopharmaceutical Dose
        
        # Convert time to seconds
        #start_time_seconds = int(start_time[:2]) * 3600 + int(start_time[2:4]) * 60 + int(start_time[4:6])
        
        # Calculate decay factor
        decay_factor = np.exp(-np.log(2) * ((scantime-injection_time).seconds / half_life))
        
        # Calculate decayed dose during procedure
        decayed_dose = total_dose * decay_factor
        
        # Compute SUV map
        suv_map = (raw_pet * patient_weight) / (decayed_dose)  # Convert to SUVs
        
        # Convert numpy array to SimpleITK image
        suv_map_image = sitk.GetImageFromArray(suv_map)
        
        # Copy original image information to SUV map image
        suv_map_image.CopyInformation(pet_image)
        
        # Write the SUV map image to disk
        sitk.WriteImage(suv_map_image, output_filename)
        
        return suv_map
    
    else:
        raise ValueError("Radiopharmaceutical Information Sequence is missing or empty.")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert PET Image into sub map')
    parser.add_argument('--input_pet', 
                        required=True, 
                        help='Input PET dicom folder')
    parser.add_argument('--output', 
                        required=True, 
                        help='Output path')
        
    # Parse arguments
    args = parser.parse_args()

        # Convert RTStruct to label image with new options
    compute_suv_map(args.input_pet, args.output)


if __name__ == "__main__":
    main()

