###########单PET##################

# import os
# import torch
# import numpy as np
# import nibabel as nib
# from scipy import ndimage
# torch.__version__
# # nii.gz格式是医学图像常用的压缩格式，python中可用nibabel来读取保存。

# def read_nifti_file(filepath):
#     """读取和加载数据"""
#     # Read file
#     scan = nib.load(filepath)
#     # Get raw data
#     scan = scan.get_fdata()
#     return scan

# # def normalize(volume):#CT
# #     """数据归一化"""
# #     min = -1000
# #     max = 400
# #     volume[volume < min] = min
# #     volume[volume > max] = max
# #     volume = (volume - min) / (max - min)
# #     volume = volume.astype("float32")
# #     return volume

# def normalize(volume):#PET
#     """Normalize the data using Z-score normalization."""
#     mean = volume.mean()
#     std = volume.std()
#     volume = (volume - mean) / std
#     volume = volume.astype("float32")
#     return volume

# def resize_volume(img_data):
#     # Create an empty array with the target shape, filled with zeros
#     target_shape = (128, 128, 128)
#     padded_data = np.zeros(target_shape)

#     # Calculate the cropping or padding indices
#     x_offset = (target_shape[0] - img_data.shape[0]) // 2
#     y_offset = (target_shape[1] - img_data.shape[1]) // 2
#     z_offset = (target_shape[2] - img_data.shape[2]) // 2

#     # Calculate the slices for inserting the image data into the padded array
#     x_slice_src = slice(max(0, -x_offset), min(img_data.shape[0], target_shape[0] - max(0, x_offset)))
#     y_slice_src = slice(max(0, -y_offset), min(img_data.shape[1], target_shape[1] - max(0, y_offset)))
#     z_slice_src = slice(max(0, -z_offset), min(img_data.shape[2], target_shape[2] - max(0, z_offset)))

#     x_slice_dst = slice(max(0, x_offset), max(0, x_offset) + (x_slice_src.stop - x_slice_src.start))
#     y_slice_dst = slice(max(0, y_offset), max(0, y_offset) + (y_slice_src.stop - y_slice_src.start))
#     z_slice_dst = slice(max(0, z_offset), max(0, z_offset) + (z_slice_src.stop - z_slice_src.start))

#     # Insert the image data into the padded array
#     padded_data[x_slice_dst, y_slice_dst, z_slice_dst] = img_data[x_slice_src, y_slice_src, z_slice_src]

#     return padded_data

# def save_nifti_file(volume, original_path, target_dir="MMdata_1"):
#     """Save processed volume as a NIFTI file in the specified directory."""
#     # Ensure target directory exists
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     # Generate new file path
#     file_name = os.path.basename(original_path)
#     new_path = os.path.join(target_dir, file_name)

#     # Load the original NIfTI file to get the affine matrix
#     original_img = nib.load(original_path)
#     affine = original_img.affine
#     # Convert numpy array back to NIFTI
#     new_img = nib.Nifti1Image(volume, affine)

#     # Save NIFTI file
#     nib.save(new_img, new_path)

# def process_scan(path):
#     """读取和调整数据大小"""
#     # 读取扫描文件
#     volume = read_nifti_file(path)
#     # 归一化
#     volume = normalize(volume)
#     # 调整宽度、高度和深度
#     volume = resize_volume(volume)
#     return volume

# def load_scans(scan_paths):
#     """Load and process scans from given paths."""
#     scans = [process_scan(path) for path in scan_paths]
#     return np.array(scans)
# def process_and_save_scan_paths(source_dir='MMdata', target_dir="MMdata1"):#处理+保存
#     for folder in ["train/ai_pet_suv", "train/hjy_pet_suv", "train/no_pet_suv"]:
#         source_folder = os.path.join(source_dir, folder)
#         for file_name in os.listdir(source_folder):
#             path = os.path.join(source_folder, file_name)
#             volume = read_nifti_file(path)
#             volume = normalize(volume)
#             volume = resize_volume(volume)
#             save_nifti_file(volume, path, os.path.join(target_dir, folder))
# # def process_and_save_scan_paths(source_dir='MMdata', target_dir="MMdata_1"):
# #     subfolders = ['ai_pet_suv', 'hjy_pet_suv', 'no_pet_suv']
# #     for folder in ["train, "test"]:
# #         for subfolder in subfolders:
# #             source_folder = os.path.join(source_dir, folder,subfolder)
# #             target_folder = os.path.join(target_dir, folder,subfolder)
# #             for file_name in os.listdir(source_folder):
# #                 path = os.path.join(source_folder,file_name)
# #                 volume = read_nifti_file(path)
# #                 volume = normalize(volume)
# #                 volume = resize_volume(volume)
# #                 save_nifti_file(volume, path, os.path.join(target_dir,file_name))
# if __name__ == "__main__":
#     process_and_save_scan_paths()

##########PET和CT数据###########################
import os
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage

def read_nifti_file(filepath):
    """读取和加载数据"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize_ct(volume):
    """CT数据的最大最小归一化"""
    min_val = -1000
    max_val = 400
    volume[volume < min_val] = min_val
    volume[volume > max_val] = max_val
    volume = (volume - min_val) / (max_val - min_val)
    volume = volume.astype("float32")
    return volume

def normalize_pet(volume):
    """PET数据的Z-score归一化"""
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / std
    volume = volume.astype("float32")
    return volume

def resize_volume(img_data):
    """调整图像大小到128x128x128"""
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

# def resize_volume(img_data, target_shape=(128, 128, 128)):
#     """
#     调整输入图像体积大小到目标的 target_shape，并使用线性插值缩放。
#     如果输入图像的尺寸比目标尺寸小，则对其进行插值放大；
#     如果输入图像的尺寸比目标尺寸大，则对其进行插值缩小。
    
#     Args:
#         img_data (numpy array): 输入的3D图像体积（NIfTI格式数据）
#         target_shape (tuple): 目标体积大小，默认值为 (128, 128, 128)
    
#     Returns:
#         numpy array: 调整大小后的图像数据
#     """
#     # 获取原始图像的形状
#     original_shape = img_data.shape
    
#     # 计算每个维度的缩放因子
#     factors = [t / s for t, s in zip(target_shape, original_shape)]
    
#     # 使用线性插值进行缩放
#     resized_data = ndimage.zoom(img_data, zoom=factors, order=1)
    
#     # 检查缩放后的形状，并将其适配到目标形状
#     resized_shape = resized_data.shape
#     if resized_shape != target_shape:
#         # 如果缩放后的形状不完全等于目标形状，进行零填充或裁剪
#         padded_data = np.zeros(target_shape, dtype=np.float32)
        
#         # 计算中心对齐的偏移量
#         x_offset = (target_shape[0] - resized_shape[0]) // 2
#         y_offset = (target_shape[1] - resized_shape[1]) // 2
#         z_offset = (target_shape[2] - resized_shape[2]) // 2
        
#         # 计算有效的源切片
#         x_slice_src = slice(max(0, -x_offset), min(resized_shape[0], target_shape[0] - max(0, x_offset)))
#         y_slice_src = slice(max(0, -y_offset), min(resized_shape[1], target_shape[1] - max(0, y_offset)))
#         z_slice_src = slice(max(0, -z_offset), min(resized_shape[2], target_shape[2] - max(0, z_offset)))

#         # 计算目标切片
#         x_slice_dst = slice(max(0, x_offset), max(0, x_offset) + (x_slice_src.stop - x_slice_src.start))
#         y_slice_dst = slice(max(0, y_offset), max(0, y_offset) + (y_slice_src.stop - y_slice_src.start))
#         z_slice_dst = slice(max(0, z_offset), max(0, z_offset) + (z_slice_src.stop - z_slice_src.start))

#         # 将插值后的数据填充到目标大小的数组中
#         padded_data[x_slice_dst, y_slice_dst, z_slice_dst] = resized_data[x_slice_src, y_slice_src, z_slice_src]

#         return padded_data
    
    return resized_data

def save_nifti_file(volume, original_path, target_dir="MMdata1"):
    """保存处理后的NIfTI文件"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_name = os.path.basename(original_path)
    new_path = os.path.join(target_dir, file_name)

    original_img = nib.load(original_path)
    affine = original_img.affine
    new_img = nib.Nifti1Image(volume, affine)
    nib.save(new_img, new_path)

def process_scan(path, modality):
    """根据模态选择不同的归一化处理"""
    volume = read_nifti_file(path)
    if modality == "CT":
        volume = normalize_ct(volume)
    elif modality == "PET":
        volume = normalize_pet(volume)
    volume = resize_volume(volume)
    return volume

def process_and_save_scan_paths(source_dir='MMdata', target_dir="MMdata1"):
    """处理并保存CT和PET扫描数据"""
    modalities = ['CT', 'PET']
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
