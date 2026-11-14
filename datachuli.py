# 这些文件以NIFTI格式提供，扩展名为.nii。要阅读扫描，我们使用`nibabel`包。您可以通过`pip install nibabel`安装包。 CT扫描以Hounsfield单位（HU）存储原始体素强度。在这个数据集中，它们的范围从-1024到2000以上。400以上的骨骼具有不同的放射性强度，因此这被用作更高的界限。通常使用-1000到400之间的阈值来规范CT扫描。
# 要处理数据，我们执行以下操作：
# - 我们首先将体积旋转90度，因此方向是固定的。
# - 我们将HU值缩放为介于0和1之间。
# - 我们调整宽度，高度和深度的大小。
# 在这里，我们定义了几个辅助函数来处理数据。在构建训练和验证数据集时将使用这些功能。
#

import os
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage
torch.__version__
# nii.gz格式是医学图像常用的压缩格式，python中可用nibabel来读取保存。

def read_nifti_file(filepath):
    """读取和加载数据"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

# def normalize(volume):#CT
#     """数据归一化"""
#     min = -1000
#     max = 400
#     volume[volume < min] = min
#     volume[volume > max] = max
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

def normalize(volume):#PET
    """Normalize the data using Z-score normalization."""
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / std
    volume = volume.astype("float32")
    return volume


# def resize_volume(img):
#     """跨 z 轴调整大小"""
#     # 设置所需的深度
#     desired_depth = 128
#     desired_width = 128
#     desired_height = 128
#     # 获取当前深度
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # 计算深度因子
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # 旋转
#     # img = ndimage.rotate(img, 90, reshape=False)
#     # 跨z轴调整大侠
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img

def resize_volume(img_data):
    # Create an empty array with the target shape, filled with zeros
    target_shape = (128, 128, 128)
    padded_data = np.zeros(target_shape)

    # Calculate the cropping or padding indices
    x_offset = (target_shape[0] - img_data.shape[0]) // 2
    y_offset = (target_shape[1] - img_data.shape[1]) // 2
    z_offset = (target_shape[2] - img_data.shape[2]) // 2

    # Calculate the slices for inserting the image data into the padded array
    x_slice_src = slice(max(0, -x_offset), min(img_data.shape[0], target_shape[0] - max(0, x_offset)))
    y_slice_src = slice(max(0, -y_offset), min(img_data.shape[1], target_shape[1] - max(0, y_offset)))
    z_slice_src = slice(max(0, -z_offset), min(img_data.shape[2], target_shape[2] - max(0, z_offset)))

    x_slice_dst = slice(max(0, x_offset), max(0, x_offset) + (x_slice_src.stop - x_slice_src.start))
    y_slice_dst = slice(max(0, y_offset), max(0, y_offset) + (y_slice_src.stop - y_slice_src.start))
    z_slice_dst = slice(max(0, z_offset), max(0, z_offset) + (z_slice_src.stop - z_slice_src.start))

    # Insert the image data into the padded array
    padded_data[x_slice_dst, y_slice_dst, z_slice_dst] = img_data[x_slice_src, y_slice_src, z_slice_src]

    return padded_data

def save_nifti_file(volume, original_path, target_dir="MosMedData_1"):
    """Save processed volume as a NIFTI file in the specified directory."""
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Generate new file path
    file_name = os.path.basename(original_path)
    new_path = os.path.join(target_dir, file_name)

    # Load the original NIfTI file to get the affine matrix
    original_img = nib.load(original_path)
    affine = original_img.affine
    # Convert numpy array back to NIFTI
    new_img = nib.Nifti1Image(volume, affine)

    # Save NIFTI file
    nib.save(new_img, new_path)


def process_scan(path):
    """读取和调整数据大小"""
    # 读取扫描文件
    volume = read_nifti_file(path)
    # 归一化
    volume = normalize(volume)
    # 调整宽度、高度和深度
    volume = resize_volume(volume)
    return volume

def load_scans(scan_paths):
    """Load and process scans from given paths."""
    scans = [process_scan(path) for path in scan_paths]
    return np.array(scans)
def process_and_save_scan_paths(source_dir='MosMedData', target_dir="MosMedData_1"):#处理+保存
    for folder in ["AI_PET_SUV", "HE_PET_SUV"]:
        source_folder = os.path.join(source_dir, folder)
        for file_name in os.listdir(source_folder):
            path = os.path.join(source_folder, file_name)
            volume = read_nifti_file(path)
            volume = normalize(volume)
            volume = resize_volume(volume)
            save_nifti_file(volume, path, os.path.join(target_dir, folder))



if __name__ == "__main__":
    process_and_save_scan_paths()
