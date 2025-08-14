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
# python Python/RTStructToNifti.py --input_dir /path/to/dicom/data --output_dir /path/to/output --reference_image /path/to/reference.nii.gz

# 2. Process multiple patients with RTStruct conversion
# echo "Processing multiple patients..."
# python Python/FindAndConvertRTStructs.py --input_dir /path/to/patient/folders --output_dir /path/to/results --structlist /path/to/roi_config.json

# 3. Extract radiomics features
# echo "Extracting radiomics features..."
# python Python/Radiomics/TestRadiomics.py --image /path/to/image.nii.gz --mask /path/to/mask.nii.gz --output /path/to/features.csv

# 4. Compute SUV maps for PET data
# echo "Computing SUV maps..."
# python Python/Radiomics/ComputeSUVMap.py --input_dir /path/to/pet/dicom --output /path/to/suv_map.nii.gz

# 5. Create overlay visualizations
# echo "Creating overlay visualizations..."
# python Python/OverlaySegmentation.py --image /path/to/image.nii.gz --mask /path/to/segmentation.nii.gz --output /path/to/overlay.png

# Add your processing commands here:
echo "Starting Head & Neck Cancer processing pipeline..."

# Your custom processing pipeline goes here
# Example:
# python your_analysis_script.py --config config.json

echo "Processing completed successfully!"

# Deactivate conda environment
conda deactivate

echo "Job finished at: $(date)"