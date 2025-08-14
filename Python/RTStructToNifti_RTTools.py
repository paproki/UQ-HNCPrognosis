from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

import os 
import zipfile

# array manipulation and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# medical image manipulation 
import SimpleITK as sitk

def display_slices(image, mask, skip=1):
    """
    Displays a series of slices in z-direction that contains the segmented regions of interest.
    Ensures all contours are displayed in consistent and different colors.
        Parameters:
            image (array-like): Numpy array of image.
            mask (array-like): Numpy array of mask.
            skip (int): Only print every nth slice, i.e. if 3 only print every 3rd slice, default 1.
        Returns:
            None (series of in-line plots).
    """

    slice_locations = np.unique(np.where(mask != 0)[0]) # get indexes for where there is a contour present 
    slice_start = slice_locations[0] # first slice of contour 
    slice_end = slice_locations[len(slice_locations)-1] # last slice of contour
    
    counter = 1
    
    for img_arr, contour_arr in zip(image[slice_start:slice_end+1], mask[slice_start:slice_end+1]): # plot the slices with contours overlayed ontop
        if counter % skip == 0: # if current slice is divisible by desired skip amount 
            masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)
            plt.imshow(img_arr, cmap='gray', interpolation='none')
            plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.5, vmin = 1, vmax = np.amax(mask)) # vmax is set as total number of contours so same colors can be displayed for each slice
            plt.show()
        counter += 1

DICOM_path = '/media/uqapapro/TRANSCEND/Data/HNC/Head-Neck-Radiomics-HN1/manifest-1568995398587/HEAD-NECK-RADIOMICS-HN1/HN1057'
print(DICOM_path)


#time
Dicom_reader = DicomReaderWriter(description='Examples', arg_max=True)
print('Estimated 30 seconds, depending on number of cores present in your computer')
Dicom_reader.walk_through_folders(DICOM_path) # need to define in order to use all_roi method

all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present, and print them

Dicom_reader.where_is_ROI(ROIName='gtvn_[iil2a]')



#Contour_Names = ['tumor', 'lymph_nodes'] 
#associations = [ROIAssociationClass('lymph_nodes',['gtvn_[r2a]', 'gtvn_[ir2b]', 'gtvn_[r3]', 'gtvn_[il2a]', 'gtvn_[iil2a]', 'gtvn_[ir1b]', 'gtvn_[iir1b]', 'gtvn_[iiir1b]', 'gtvn_[iir2b]', 'gtvn_[l1b]', 'gtvn_[l3]', 'gtvn_[iiir2b]', 'gtvn_[ivr2b]', 'gtvn_[vr2b]', 'gtvn_[l1a]']),
#               ROIAssociationClass('tumor', ['gtvp'])]
Contour_Names = ['tumor'] 
associations = [ROIAssociationClass('tumor', ['gtv', 'gtv-1', 'gtv-2'])]

Dicom_reader.set_contour_names_and_associations(contour_names=Contour_Names, associations=associations)
indexes = Dicom_reader.which_indexes_have_all_rois()

for idx in indexes:

    pt_indx = idx
    Dicom_reader.set_index(pt_indx)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    Dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index

    image = Dicom_reader.ArrayDicom # image array
    mask = Dicom_reader.mask # mask array
    dicom_sitk_handle = Dicom_reader.dicom_handle # SimpleITK image handle
    mask_sitk_handle = Dicom_reader.annotation_handle # SimpleITK mask handle

    #n_slices_skip = 4
    #display_slices(image, mask, skip = n_slices_skip) # visualize that our segmentations were succesfully convereted

    nifti_path = os.path.join('/home/uqapapro/debug-vrac/', 'Trial') # nifti subfolder
    if not os.path.exists(nifti_path):
        os.makedirs(nifti_path)
    dicom_sitk_handle = Dicom_reader.dicom_handle # SimpleITK image handle
    mask_sitk_handle = Dicom_reader.annotation_handle # SimpleITK mask handle
    sitk.WriteImage(dicom_sitk_handle, os.path.join(nifti_path, str(idx) + '_image.nii.gz'))
    sitk.WriteImage(mask_sitk_handle, os.path.join(nifti_path, str(idx) + '_mask.nii.gz'))