import os
import nibabel as nib
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_slices(file_path, output_dir, key):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()
    if key == 'data':
        data = (data / data.max(axis=(0,1,2))).transpose(1,3,0,2)
    elif key == 'brain_mask':
        data = data.transpose(1,0,2)
    # Create directory if it does not exist
    create_dir(output_dir)

    # Split into slices along the first dimension and save each slice
    for i in range(data.shape[0]):
        slice_data = data[i, :, :]
        slice_img = nib.Nifti1Image(slice_data, np.eye(4))
        slice_filename = os.path.join(output_dir, f'slice_{i:04d}.nii.gz')
        nib.save(slice_img, slice_filename)

# Source directory
src_dir = '../../../../../data/hcp'

# Iterate over each ID directory
for id_dir in os.listdir(src_dir):
    print(f'Processing ID {id_dir}')
    id_path = os.path.join(src_dir, id_dir)
    if os.path.isdir(id_path):
        diffusion_path = os.path.join(id_path, 'Diffusion')
        data_file = os.path.join(diffusion_path, 'data.nii.gz')
        brain_mask_file = os.path.join(diffusion_path, 'nodif_brain_mask.nii.gz')
        
        if os.path.exists(data_file) and os.path.exists(brain_mask_file):
            # Create output directories
            data_output_dir = os.path.join(diffusion_path, 'data')
            brain_mask_output_dir = os.path.join(diffusion_path, 'brain_mask')
            
            create_dir(data_output_dir)
            create_dir(brain_mask_output_dir)
            
            # Save slices
            save_slices(data_file, data_output_dir, key='data')
            save_slices(brain_mask_file, brain_mask_output_dir, key='brain_mask')