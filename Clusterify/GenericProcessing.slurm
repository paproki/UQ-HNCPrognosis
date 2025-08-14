#!/bin/bash --login
#SBATCH --job-name=HNC_Processing                     # Job name
#SBATCH --nodes=1                                     # Number of nodes
#SBATCH --ntasks-per-node=1                           # Number of tasks per node
#SBATCH --cpus-per-task=12                            # Number of CPU cores per task
#SBATCH --mem=200G                                    # Memory per node (200 GB)
#SBATCH --time=24:00:00                               # Time limit (24 hours)
#SBATCH --gres=gpu:l40s:1                             # Request 1 GPU (adjust based on availability)
#SBATCH --output=hnc_processing_%j.out                # Output log file (%j is job ID)
#SBATCH --error=hnc_processing_%j.err                 # Error log file
#SBATCH --account=a_lena_neuro                        # Adjust to your account
#SBATCH --partition=gpu_cuda                          # Partition name (adjust to your HPC's GPU partition)
#SBATCH --qos=gpu

# Load required modules
module load miniconda3                                # Load conda
module load cuda                                      # Load CUDA for GPU support (adjust version if required)
source $EBROOTMINICONDA3/etc/profile.d/conda.sh

# Activate your conda environment for medical imaging processing
# Replace 'DeepLearningBase' with your environment name containing:
# - SimpleITK, nibabel, pydicom for medical imaging
# - PyTorch, MONAI for deep learning (if needed)
# - pandas, numpy for data processing
conda activate DeepLearningBase

# Print environment info for debugging
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Example usage patterns - uncomment and modify as needed:

# 1. Convert RTStruct files to NIfTI format
# echo "Converting RTStruct files to NIfTI..."
# python Python/RTStructToNifti.py --rtstruct /path/to/rtstruct.dcm --reference /path/to/ct/dicom/folder --output /path/to/labels.nii.gz

# 2. Process multiple patients with RTStruct conversion
# echo "Processing multiple patients..."
# python Python/FindAndConvertRTStructs.py --input_dir /path/to/patient/folders --output_dir /path/to/results --structlist Configs/HNC_Quebec_regions_tumor.json --id PatientID --dosuv --dortstruct

# 3. Extract radiomics features
# echo "Extracting radiomics features..."
# python Python/Radiomics/TestRadiomics.py --input_ct /path/to/image.nii.gz --input_mask /path/to/mask.nii.gz

# 4. Compute SUV maps for PET data
# echo "Computing SUV maps..."
# python Python/Radiomics/ComputeSUVMap.py --input_pet /path/to/pet/dicom/folder --output /path/to/suv_map.nii.gz

# 5. Create overlay visualizations
# echo "Creating overlay visualizations..."
# python Python/OverlaySegmentation.py /path/to/image.nii.gz /path/to/segmentation.nii.gz /path/to/overlay.png --orientation axial --opacity 0.8 --num_slices 5

# 6. Create wireframe visualizations
# echo "Creating wireframe contours..."
# python Python/OverlaySegmentation.py /path/to/image.nii.gz /path/to/segmentation.nii.gz /path/to/wireframe.png --wireframe --contour_width 2 --orientation sagittal

# Add your processing commands here:
echo "Starting Head & Neck Cancer processing pipeline..."

# Your custom processing pipeline goes here
# Example:
# python your_analysis_script.py --config config.json

echo "Processing completed successfully!"

# Deactivate conda environment
conda deactivate

echo "Job finished at: $(date)"