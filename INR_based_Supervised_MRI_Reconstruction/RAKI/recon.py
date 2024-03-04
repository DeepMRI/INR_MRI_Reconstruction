import numpy as np
import os
from RAKI import RAKI
from rRAKI import rRAKI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    # parameters
    params = {
        'kernel_x_1': 5,
        'kernel_y_1': 2,

        'kernel_x_2': 1,
        'kernel_y_2': 1,

        'kernel_last_x': 3,
        'kernel_last_y': 2,

        'layer1_channels': 32,
        'layer2_channels': 8,
                                    # rRAKI     RAKI
        'MaxIteration': 2000,       # 2000      1000
        'LearningRate': 1e-3,       # 1e-3      3e-3

        'center_ratio': 0.08,
        'down_scale': 4,

        'load_weight': False,
        'save_weight': True,

        'dataset': 'brain',
        'method': 'rRAKI'
              }


    weight_path = os.path.join('weight', params['method'] + '-' + params['dataset'], 'scale={}'.format(params['down_scale']))
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    save_path = 'E:\\dataset\\fastMRI_{}\\Reconstruction\\restored\\{}\\scale={}'.format(params['dataset'], params['method'], params['down_scale'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path, filename = get_filepath('E:\\dataset\\fastMRI_{}\\Reconstruction\\raw_data\\test'.format(params['dataset']))

    for i in range(len(file_path)):

        params['weight_path'] = os.path.join(weight_path, filename[i])

        # data
        kspace = np.load(file_path[i])['fullysampled_kspace']               # [coil_num, height, width]
        kspace = np.transpose(kspace, [1, 2, 0])                            # [height, width, coil_num]
        # ACS
        idx_lower = int((1 - params['center_ratio']) * kspace.shape[-2] / 2)
        idx_upper = int((1 + params['center_ratio']) * kspace.shape[-2] / 2)
        ACS = kspace[:, idx_lower:idx_upper, :].copy()
        ACS = np.concatenate([ACS.real, ACS.imag], axis=-1)[None, ...]      # [1, height, ACS center num, coil_num * 2]
        # undersampled kspace
        undersampled_kspace = np.zeros_like(kspace)
        undersampled_kspace[:, ::params['down_scale'], :] = kspace[:, ::params['down_scale'], :].copy()
        undersampled_kspace[:, idx_lower:idx_upper, :] = kspace[:, idx_lower:idx_upper, :].copy()
        undersampled_kspace = np.concatenate([undersampled_kspace.real, undersampled_kspace.imag], axis=-1)[None, ...]      # [1, height, width, coil_num * 2]

        # RAKI reconstruction
        if params['method'] == 'RAKI':
            recon_kspace = RAKI(params, undersampled_kspace, ACS)               # [height, width, coil_num]
        else:
            recon_kspace = rRAKI(params, undersampled_kspace, ACS)

        recon_kspace = np.transpose(recon_kspace, [2, 0, 1])                # [coil_num, height, width]

        np.savez(os.path.join(save_path, filename[i]), recon_kspace=recon_kspace)
