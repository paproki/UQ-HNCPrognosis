import os
import argparse
from radiomics import featureextractor  # Import the feature extractor

def ExtractRadiomics(input_ct, input_mask, output = None):
    """
    This is a test function to extract radiomics features
    from images
    """

    # Set parameters for feature extraction
    params = {
        'binWidth': 25,
        'sigma': [1, 2, 3],
        'verbose': True,
    }

    # Instantiate the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    # Execute feature extraction
    result = extractor.execute(input_ct, input_mask)

    # Print extracted features
    for key, val in result.items():
        print(f"{key}: {val}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert RTStruct to Label Image with Improved Precision')
    parser.add_argument('--input_ct', 
                        required=True, 
                        help='Path to the RTStruct DICOM file')
    parser.add_argument('--input_mask', 
                        required=True, 
                        help='Path to the folder containing CT DICOM series')
    # parser.add_argument('--output', 
    #                     required=True, 
    #                     help='Output path for the label image')
        
    # Parse arguments
    args = parser.parse_args()

    # Convert RTStruct to label image with new options
    label_image = ExtractRadiomics(
        args.input_ct, 
        args.input_mask, 
    )

if __name__ == "__main__":
    main()