import os
import pydicom

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

def process_dicom_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_rtstruct(file_path):
                associated_folder, modality, series_desk = find_associated_dicom_folder(file_path, base_path)
                
                print(f"RTStruct: {file_path}")
                print(f"Associated DICOM folder: {associated_folder}")
                print(f"Image Type: {modality}\n")
                print(f"Image Type2: {series_desk}\n")

# Usage
base_folder = "/media/uqapapro/TRANSCEND/Data/HNC/Head-Neck-PET-CT/manifest-VpKfQUDr2642018792281691204/Head-Neck-PET-CT/"
process_dicom_folder(base_folder)