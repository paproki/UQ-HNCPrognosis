# UQ-HNCPrognosis

A focused toolkit for Head & Neck Cancer (HNC) radiation therapy structure analysis and radiomics feature extraction for prognosis research.

## Overview

This repository contains specialized tools for processing radiation therapy structures and extracting prognostic features from Head & Neck Cancer imaging data. The tools are designed for research workflows involving DICOM RT-Struct conversion, radiomics analysis, and visualization.

## Repository Structure

```
UQ-HNCPrognosis/
├── Python/                          # Python analysis tools
│   ├── Radiomics/                   # Radiomics feature extraction
│   │   ├── ComputeSUVMap.py        # SUV map calculation for PET
│   │   └── TestRadiomics.py        # Radiomics feature extraction
│   ├── RTStructToNifti.py          # Standard RTStruct conversion
│   ├── RTStructToNifti_advanced.py # Advanced RTStruct processing
│   ├── RTStructToNifti_RTTools.py  # RTTools-based conversion
│   ├── FindAndConvertRTStructs.py  # Batch RTStruct processing
│   ├── FindRTStructFilesAndReferenceDCM.py  # RTStruct file discovery
│   ├── Filter_unique_RTStructs.py  # ROI standardization
│   ├── OverlaySegmentation.py      # Visualization overlays
│   └── OverlaySegmentation_deepseek.py  # Enhanced visualization
├── Configs/                         # Configuration files
│   ├── HNC_Quebec_regions_tumor.json    # Quebec dataset ROI config
│   └── HNC_RADCURE_regions_tumor_nodes.json  # RADCURE dataset config
├── Clusterify/                      # HPC processing templates
│   └── GenericProcessing.slurm     # SLURM job template
└── README.md                        # This file
```

## Quick Start

### Environment Setup

This toolkit requires a conda environment with medical imaging and machine learning libraries:

```bash
# Create environment with required packages
conda create -n hnc-analysis python=3.9
conda activate hnc-analysis

# Install core medical imaging libraries
conda install -c conda-forge simpleitk nibabel pydicom
conda install pandas numpy matplotlib scikit-image

# For radiomics (choose one):
pip install pyradiomics              # Standard radiomics
# OR
pip install radiomics                # Alternative implementation

# For advanced processing (optional)
conda install -c pytorch pytorch torchvision
pip install monai                    # Medical AI framework
```

### Basic Usage Examples

#### 1. Convert RTStruct to NIfTI Labels

```bash
# Basic conversion
python Python/RTStructToNifti.py \
    --rtstruct /path/to/rtstruct.dcm \
    --reference /path/to/ct/dicom/folder \
    --output /path/to/output_labels.nii.gz

# Advanced conversion with precision options
python Python/RTStructToNifti_advanced.py \
    --rtstruct /path/to/rtstruct.dcm \
    --reference /path/to/ct/dicom/folder \
    --output /path/to/output_labels.nii.gz \
    --structlist Configs/HNC_Quebec_regions_tumor.json \
    --precision-method precise \
    --subpixel-precision
```

#### 2. Batch Process Multiple Patients

```bash
# Process entire dataset with RTStruct conversion and SUV maps
python Python/FindAndConvertRTStructs.py \
    --input_dir /path/to/patient/folders \
    --output_dir /path/to/results \
    --structlist Configs/HNC_RADCURE_regions_tumor_nodes.json \
    --id Patient001 \
    --dosuv --dortstruct
```

#### 3. Extract Radiomics Features

```bash
# Compute SUV maps for PET
python Python/Radiomics/ComputeSUVMap.py \
    --input_pet /path/to/pet/dicom/folder \
    --output /path/to/suv_map.nii.gz

# Extract radiomics features
python Python/Radiomics/TestRadiomics.py \
    --input_ct /path/to/ct_image.nii.gz \
    --input_mask /path/to/tumor_mask.nii.gz
```

#### 4. Create Visualizations

```bash
# Generate overlay images (positional arguments)
python Python/OverlaySegmentation.py \
    /path/to/ct_image.nii.gz \
    /path/to/segmentation.nii.gz \
    /path/to/overlay.png \
    --opacity 0.8 \
    --num_slices 5 \
    --slice_orientation axial \
    --wireframe

# Alternative enhanced overlay tool
python Python/OverlaySegmentation_deepseek.py \
    /path/to/ct_image.nii.gz \
    /path/to/segmentation.nii.gz \
    /path/to/overlay.png \
    --opacity 0.5 \
    --num_slices 6 \
    --orientation coronal
```

#### 5. Additional Utilities

```bash
# Extract unique ROI names from RTStruct files
python Python/Filter_unique_RTStructs.py \
    --input_dir /path/to/rtstruct/files \
    --output /path/to/unique_rois.txt
```

## HPC Processing

### SLURM Job Submission

The repository includes a generic SLURM template for cluster computing:

```bash
# Edit the template for your specific needs
nano Clusterify/GenericProcessing.slurm

# Submit job
sbatch Clusterify/GenericProcessing.slurm
```

### Template Customization

Modify the SLURM script to:
- Adjust resource requirements (CPU, memory, GPU)
- Change conda environment name
- Add your specific processing pipeline
- Update account and partition settings

## Configuration Files

### ROI Mapping

Configuration files define how anatomical structures are mapped to label values:

- **`HNC_Quebec_regions_tumor.json`**: Quebec multi-center study ROI mapping
- **`HNC_RADCURE_regions_tumor_nodes.json`**: RADCURE dataset primary tumor and lymph node mapping

Example structure:
```json
{
    "data": [
        {
            "ID": "HN-HGJ-001",
            "GTVPrimary": {"name": "GTV", "label": 1},
            "LymphNodes": {"name": "GTV-N", "label": 4}
        }
    ]
}
```

## Tool Descriptions

### RTStruct Processing Tools

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `RTStructToNifti.py` | Basic RTStruct conversion | Standard DICOM→NIfTI conversion |
| `RTStructToNifti_advanced.py` | Enhanced conversion | Batch processing, validation |
| `RTStructToNifti_RTTools.py` | RTTools integration | Alternative conversion method |
| `FindAndConvertRTStructs.py` | Comprehensive processing | Patient-level batch processing |
| `Filter_unique_RTStructs.py` | ROI standardization | Extract unique structure names |

### Radiomics & Analysis

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `ComputeSUVMap.py` | PET SUV calculation | Decay correction, weight normalization |
| `TestRadiomics.py` | Feature extraction | Shape, texture, first-order features |

### Visualization Tools

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `OverlaySegmentation.py` | Create overlays | Multi-slice visualization |
| `OverlaySegmentation_deepseek.py` | Enhanced overlays | Advanced visualization options |

## Typical Workflow

1. **Data Organization**: Organize DICOM files by patient
2. **RTStruct Conversion**: Convert radiation therapy structures to NIfTI labels
3. **Quality Control**: Verify conversions and anatomical accuracy
4. **Radiomics Extraction**: Extract quantitative imaging features
5. **Statistical Analysis**: Correlate features with clinical outcomes
6. **Visualization**: Create publication-ready figures

## Research Applications

This toolkit supports research in:
- **Radiation Therapy Planning**: Automated structure segmentation
- **Prognostic Modeling**: Radiomics-based outcome prediction
- **Multi-Center Studies**: Standardized processing across institutions
- **Treatment Response**: Longitudinal analysis of therapy response

## Important Notes

### Data Requirements
- DICOM RT-Struct files with properly named anatomical structures
- Co-registered CT/PET imaging data
- Patient metadata for SUV calculations

### Quality Assurance
- Always verify RTStruct conversions visually
- Check coordinate system consistency between images and structures
- Validate radiomics features against known benchmarks

### Clinical Considerations
- Ensure compliance with data privacy regulations
- Validate results before clinical application
- Consider inter-observer variability in manual contouring

## Citation

If you use these tools in your research, please cite:
```
[Your publication/preprint information here]
```

## Support

For questions or issues:
1. Check the individual script documentation (`python script.py --help`)
2. Review the original UQUtility repository for additional context
3. Contact [your contact information]

## License

[Specify your license here]