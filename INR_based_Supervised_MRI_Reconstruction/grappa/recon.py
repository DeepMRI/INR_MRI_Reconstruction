import numpy as np
import os
from pygrappa import grappa


def get_filepath(main_path):
    # read all file path in main_path
    file_path = []
    filename = []
    for root, dirs, files in os.walk(main_path):
        if len(files) != 0:
            for i in files:
                file_path.append(os.path.join(root, i))
                filename.append(i)

    return file_path, filename


if __name__ == '__main__':

    down_scale = 8
    center_ratio = 0.08
    dataset = 'brain'

    save_path = 'E:\\dataset\\fastMRI_{}\\Reconstruction\\restored\\grappa\\scale={}'.format(dataset, down_scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path, filename = get_filepath('E:\\dataset\\fastMRI_{}\\Reconstruction\\raw_data\\test'.format(dataset))

    for i in range(len(file_path)):

        kspace = np.load(file_path[i])['fullysampled_kspace']       # [coil_num, height, width]
        # ACS
        idx_lower = int((1 - center_ratio) * kspace.shape[-1] / 2)
        idx_upper = int((1 + center_ratio) * kspace.shape[-1] / 2)
        ACS = kspace[..., idx_lower:idx_upper].copy()
        # undersampled kspace
        undersampled_kspace = np.zeros_like(kspace)
        undersampled_kspace[..., ::down_scale] = kspace[..., ::down_scale].copy()
        undersampled_kspace[..., idx_lower:idx_upper] = kspace[..., idx_lower:idx_upper]

        # grappa reconstruction
        recon_kspace = grappa(undersampled_kspace, ACS, coil_axis=0)

        np.savez(os.path.join(save_path, filename[i]), recon_kspace=recon_kspace)