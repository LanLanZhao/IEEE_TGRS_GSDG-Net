import os

import hdf5storage
import numpy as np


class LoadingData:
    """Load hyperspectral datasets from MATLAB files."""

    def __init__(self, base_path='dataset'):
        self.g_truth = None
        self.data_cube = None
        self.base_path = base_path
        self.data_info = {
            'indian': ('indian_pines_corrected.mat', 'indian_pines_gt.mat', 'indian_pines_corrected', 'indian_pines_gt'),
            'paviau': ('PaviaU.mat', 'PaviaU_gt.mat', 'paviaU', 'paviaU_gt'),
            'ksc': ('KSC_corrected.mat', 'KSC_gt.mat', 'KSC', 'KSC_gt'),
            'sali': ('Salinas_corrected.mat', 'Salinas_gt.mat', 'salinas_corrected', 'salinas_gt'),
            'botswana': ('Botswana.mat', 'Botswana_gt.mat', 'Botswana', 'Botswana_gt'),
            'houston': ('Houston13.mat', 'Houston13_gt.mat', 'Houston', 'Houston_gt'),
            'hanchuan': ('WHU_Hi_HanChuan.mat', 'WHU_Hi_HanChuan_gt.mat', 'WHU_Hi_HanChuan', 'WHU_Hi_HanChuan_gt'),
            'honghu': ('WHU_Hi_HongHu.mat', 'WHU_Hi_HongHu_gt.mat', 'WHU_Hi_HongHu', 'WHU_Hi_HongHu_gt'),
            'longkou': ('WHU_Hi_LongKou.mat', 'WHU_Hi_LongKou_gt.mat', 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt'),
            'Houston2018': ('_houston2018.mat', '_houston2018_gt.mat', 'houston2018', 'houston2018_gt'),
            'paviac': ('PaviaC.mat', 'PaviaC_gt.mat', 'pavia', 'pavia_gt'),
            'SZUR1': ('SZUTreeHSI_R1.mat', 'SZUTreeHSI_R1_gt.mat', 'hyperspectral_data_98bands', 'ndvi_label'),
            'SZUR2': ('SZUTreeHSI_R2.mat', 'SZUTreeHSI_R2_gt.mat', 'hyperspectral_data_98bands', 'ndvi_label'),
            'UP': ('Utopia.mat', 'Utopia_gt.mat', 'Utopia', 'Utopia_gt'),
            'HC': ('holden.mat', 'holden_gt.mat', 'holden', 'holden_gt'),
            'NF': ('NiliFossae.mat', 'NiliFossae_gt.mat', 'NiliFossae', 'NiliFossae_gt'),
            'loukia': ('Loukia.mat', 'Loukia_gt.mat', 'ori_data', 'map'),
            'dioni': ('Dioni.mat', 'Dioni_gt_out68.mat', 'ori_data', 'map'),
            'tea': ('tea.mat', 'tea_gt.mat', 'tea', 'tea_gt'),
            'xuzhou': ('xuzhou.mat', 'xuzhou_gt.mat', 'xuzhou', 'xuzhou_gt'),
            'chi': ('Chikusei.mat', 'Chikusei_gt.mat', 'chikusei', 'GT'),
            'trento': ('Trento.mat', 'Trento_gt.mat', 'HSI_Trento', 'GT_Trento'),
            'kansas': ('kansas.mat', 'kansas_gt.mat', 'data', 'data'),
            'ln01': ('LN01_HHSI.mat', 'LN01_GT.mat', 'Out', 'cdata'),
            'ln02': ('LN02_HSI.mat', 'LN02_GT.mat', 'HSI', 'data'),
            'muufl': ('Muufl.mat', 'Muufl_gt.mat', 'HSI', 'gt'),
            'qingyu': ('QUH-Qingyun.mat', 'QUH-Qingyun_GT.mat', 'Chengqu', 'ChengquGT'),
        }

    def Loading(self, name='indian', num_components=None):
        if name not in self.data_info:
            raise ValueError(f"Invalid dataset name: {name}")

        data_file, gt_file, data_key, gt_key = self.data_info[name]
        data_path = os.path.join(self.base_path, data_file)
        gt_path = os.path.join(self.base_path, gt_file)
        data_dict = hdf5storage.loadmat(data_path)
        gt_dict = hdf5storage.loadmat(gt_path)

        self.data_cube = data_dict[data_key]
        self.g_truth = gt_dict[gt_key]
        height, width, bands = self.data_cube.shape

        print(f"Loaded data from {data_path}")
        print(f"Loaded labels from {gt_path}")
        print(f"Data shape: {self.data_cube.shape}; label shape: {self.g_truth.shape}")

        if name in {'hanchuan', 'honghu', 'longkou'}:
            self.data_cube = self.data_cube.astype(np.int64)
        if name == 'tea':
            self.g_truth = self.g_truth.reshape(height, width)

        class_count = len(np.unique(self.g_truth)) - 1
        return self.data_cube, self.g_truth, height, width, bands, class_count
