"""
    This file is used to convert the dicom folders and RTStructs from the Quebec dataset
    It is made to process one patient folder at a time and will reconstruct PET-CT images
    and associated RTStructs.

    This works with the original CT images where the delineations were made and the Co-registered
    structures.

    The main function process_patient_folder
"""

import os
import pydicom
import json
import argparse
from RTStructToNifti_advanced import convert_rtstruct_to_label_2
from Radiomics.ComputeSUVMap import compute_suv_map
import subprocess

def create_lookup_dict(input_file):
    """
    Create a lookup dictionnary from the jason file passed
    as parameter.
    input: the json file
    return: the lookup dictionnary
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    lookup_dict = {}
    for patient in data['data']:
        patient_id = patient['ID']
        lookup_dict[patient_id] = {
            'GTVPrimary': {
                'name': patient['GTVPrimary']['name'],
                'label': patient['GTVPrimary']['label']
            },
            'LymphNodes': {
                'name': patient['LymphNodes']['name'],
                'label': patient['LymphNodes']['label']
            }
        }
    return lookup_dict

def is_rtstruct(file_path):
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        return dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'
    except:
        return False

def get_modality(file_path):
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        return dcm.Modality
    except:
        return "Unknown"

def get_series_descr(file_path):
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        return dcm.SeriesDescription
    except:
        return "Unknown"

def find_associated_dicom_folder(rtstruct_file, base_path):
    """
    Function used to find the dicom series associated with a particular RTStruct
    It also works with co-registered series when the CT scan that was used to 
    create the radiotherapy structures was on a different machine than the PET 
    scan (i.e., not on a PET-CT).
    Parameter: rtstruct file in question
    Base path: the path fo the patient
    """
    rtstruct = pydicom.dcmread(rtstruct_file)
    ref_sop_instance_uid = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.dcm'):
                dcm_path = os.path.join(root, file)
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                if dcm.SeriesInstanceUID == ref_sop_instance_uid:
                    return os.path.dirname(dcm_path), get_modality(dcm_path), ''.join(filter(str.isalnum, get_series_descr(dcm_path)))
    
    return None, "Unknown"

def process_patient_folder(base_path, output_dir, structlist, ID, dodicoms = True, dosuv = True, dortstruct = True):
    """
    This function process a DCM folder for a given patient. It takes the base
    path of the patient folder as input, the output directory as well as the 
    json file containing the names associated with tumours/nodes and the labels
    I have assigned with them. It also requires the patient ID. 
    It takes optional parameters which allows to choose to perform the dicom, suv 
    or rtstruct conversion

    This script will find the RTStruct files and automatically find the series
    they are associated with. 

    It uses RTStructToNifti_advanced.convert_rtstruct_to_label_2 to convert the RStructs
    It uses Radiomics.compute_suv_map to normalise PET signal
    It uses milxDICOMApp to convert the DCM images


    The config files containing the structure names and labels should be formatted as follows

    {
        "data": [
            {
                "ID": "HN-CHUS-005",
                "GTVPrimary": {
                    "name": "GTV",
                    "label": 1
                },
                "LymphNodes": {
                    "name": "GTV1,GTV2,GTV3,GTV4",
                    "label": 4
                }
            },
            {
                "ID": "HN-CHUS-003",
                "GTVPrimary": {
                    "name": "GTV",
                    "label": 1
                },
                "LymphNodes": {
                    "name": null,
                    "label": 4
                }
            }
            ...
            ...
    }

    """
    id_lookup = create_lookup_dict(structlist)  # Create the lookup table ONCE

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_rtstruct(file_path):
                associated_folder, modality, series_desc = find_associated_dicom_folder(file_path, base_path)
                # print(f"RTStruct: {file_path}")
                # print(f"Associated DICOM folder: {associated_folder}")
                # print(f"Image Type: {modality}")
                # print(f"Series Descriptiond: {series_desc}")
                # print(f'Regions available: \n{id_lookup[ID]}\n')

                #milxDICOMApp -c -p $OUTPUT_DIR/${ID}/${ID}_IMG_ '$DICOM'
                base_dcm_filename = output_dir + "/" + ID + "_" + modality + "_"
                if dodicoms == True:
                    print('Converting DICOM to nifti')
                    cmdrun = subprocess.run(["milxDICOMApp", "-c", "-p", base_dcm_filename, associated_folder])
                    print(f"DCM cvrt: {' '.join(cmdrun.args)}")

                
                if dosuv == True and modality == "PT":
                    print('Normalising PET scan (calculating suv map)')
                    output_suv_map = output_dir + "/" + ID + "_" + modality + "_"  + series_desc + "_suvmap.nii.gz"
                    compute_suv_map(associated_folder, output_suv_map)

                if dortstruct == True:
                    print('Converting RTStruct file to label image')
                    output_label = output_dir + "/" + ID + "_" + modality + "_"  + series_desc + "_RTStruct.nii.gz"
                    convert_rtstruct_to_label_2(
                        file_path, 
                        associated_folder, 
                        output_label,
                        id_lookup[ID],
                        precision_method="precise"
                    )                    

# Usage
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Find and convert RTStruct to Label Images for a single patient')
    parser.add_argument('--input_dir', 
                        required=True, 
                        help='Path to the RTStruct DICOM file')
    parser.add_argument('--output_dir', 
                        required=True, 
                        help='Path to the folder containing CT DICOM series')
    parser.add_argument('--structlist', 
                        required=False, 
                        help='Input file with a list of structure labels')
    parser.add_argument('--id', 
                        required=False, 
                        help='Id of the subject')
    parser.add_argument('--dodicoms', 
                        action='store_true',
                        help='Enable dicom convertion')
    parser.add_argument('--dosuv', 
                        action='store_true',
                        help='Enable suv map creation')
    parser.add_argument('--dortstruct', 
                        action='store_true',
                        help='Enable RTStruct conversion to label map')
    
    
    # Parse arguments
    args = parser.parse_args()
    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'RTStruct config: {args.structlist}')
    print(f'Patient ID: {args.id}')
    print(f'Do Dicom Conversion?: {args.dodicoms}')
    print(f'Do compute SUV map ?: {args.dosuv}')
    print(f'Do RT Struct conversion ?: {args.dortstruct}')

    # Find RTStructs and use conversion function
    process_patient_folder(args.input_dir, args.output_dir, args.structlist, args.id, args.dodicoms, args.dosuv, args.dortstruct)


if __name__ == "__main__":
    main()