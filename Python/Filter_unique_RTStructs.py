import json
import pydicom
import SimpleITK as sitk
import argparse
import os


def extract_roi_names(input, output_file):
    """
    Extracts unique ROI names from RTStruct files listed in a JSON file
    and saves them to a text file.

    Args:
        json_file (str): Path to the JSON file containing RTStruct file paths.
        output_file (str): Path to the output text file to save unique ROI names.
    """

    roi_names = set()  # Use a set to store unique ROI names
    i = 0
    for root, dirs, files in os.walk(input):
        dobreak=False
        for file in files:
            file_path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                #Check if it is a RTStruct file
                if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
                    print(f'Opening {file_path}')
                    if hasattr(dcm, 'StructureSetROISequence'):  # Check if the sequence exists
                        for roi_item in dcm.StructureSetROISequence:
                            roi_name = roi_item.ROIName.lower()
                            roi_names.add(roi_name)
                    else:
                        print(f"Warning: StructureSetROISequence not found in {dcm}")
                    i = i + 1
                    print(f"i is {i}")
                    if i > 100:
                        dobreak = True
            except:
                continue
        if dobreak:
            print(f'Breaking')
            break

    # Write the unique ROI names to the output text file
    with open(output_file, 'w') as outfile:
        for name in sorted(roi_names):  # Sort the ROI names alphabetically
            outfile.write(name + '\n')

    print(f"Unique ROI names saved to {output_file}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Load RTStruct data and check for ROIs and save unique ones to file')
    parser.add_argument('--input_dir', 
                        required=True, 
                        help='Input folder')
    parser.add_argument('--output', 
                        required=True, 
                        help='Output text file')
    
    # Parse arguments
    args = parser.parse_args()

    # Convert RTStruct to label image with new options
    extract_roi_names(args.input_dir, args.output)
    #process_json(args.input)

if __name__ == "__main__":
    main()