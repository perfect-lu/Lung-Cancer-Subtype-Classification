import os
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage

def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize_ct(volume):
    min_val = -1000
    max_val = 400
    volume[volume < min_val] = min_val
    volume[volume > max_val] = max_val
    volume = (volume - min_val) / (max_val - min_val)
    volume = volume.astype("float32")
    return volume

def normalize_pet(volume):
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / std
    volume = volume.astype("float32")
    return volume

def resize_volume(img_data):
    target_shape = (128, 128, 128)
    padded_data = np.zeros(target_shape)

    x_offset = (target_shape[0] - img_data.shape[0]) // 2
    y_offset = (target_shape[1] - img_data.shape[1]) // 2
    z_offset = (target_shape[2] - img_data.shape[2]) // 2

    x_slice_src = slice(max(0, -x_offset), min(img_data.shape[0], target_shape[0] - max(0, x_offset)))
    y_slice_src = slice(max(0, -y_offset), min(img_data.shape[1], target_shape[1] - max(0, y_offset)))
    z_slice_src = slice(max(0, -z_offset), min(img_data.shape[2], target_shape[2] - max(0, z_offset)))

    x_slice_dst = slice(max(0, x_offset), max(0, x_offset) + (x_slice_src.stop - x_slice_src.start))
    y_slice_dst = slice(max(0, y_offset), max(0, y_offset) + (y_slice_src.stop - y_slice_src.start))
    z_slice_dst = slice(max(0, z_offset), max(0, z_offset) + (z_slice_src.stop - z_slice_src.start))

    padded_data[x_slice_dst, y_slice_dst, z_slice_dst] = img_data[x_slice_src, y_slice_src, z_slice_src]

    return padded_data

def save_nifti_file(volume, original_path, target_dir="MMdata1"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_name = os.path.basename(original_path)
    new_path = os.path.join(target_dir, file_name)

    original_img = nib.load(original_path)
    affine = original_img.affine
    new_img = nib.Nifti1Image(volume, affine)
    nib.save(new_img, new_path)

def process_scan(path, modality):
    volume = read_nifti_file(path)
    if modality == "CT":
        volume = normalize_ct(volume)
    elif modality == "PET":
        volume = normalize_pet(volume)
    volume = resize_volume(volume)
    return volume

def process_and_save_scan_paths(source_dir='MMdata', target_dir="MMdata1"):
    modalities = ['CT', 'PET','CTPET']
    subfolders = ['lin', 'xian', 'xiao']

    for modality in modalities:
        for subfolder in subfolders:
            source_folder = os.path.join(source_dir, "test", modality, subfolder)
            target_folder = os.path.join(target_dir, "test", modality, subfolder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            for file_name in os.listdir(source_folder):
                path = os.path.join(source_folder, file_name)
                volume = process_scan(path, modality)
                save_nifti_file(volume, path, target_folder)

if __name__ == "__main__":
    process_and_save_scan_paths()
