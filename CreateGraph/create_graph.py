import numpy as np


def get_label(gt_reshape, train_index, val_index, test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_reshape[val_index[i]]

    return train_samples_gt, test_samples_gt, val_samples_gt


def label_to_one_hot(data_gt, class_num):
    height, width = data_gt.shape
    one_hot_label = []
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_num, dtype=np.int64)
            if data_gt[i, j] != 0:
                temp[int(data_gt[i, j]) - 1] = 1
            one_hot_label.append(temp)
    return np.reshape(one_hot_label, [height * width, class_num])


def get_label_mask(train_samples_gt, test_samples_gt, val_samples_gt, data_gt, class_num):
    height, width = data_gt.shape
    temp_ones = np.ones([class_num])

    train_label_mask = np.zeros([height * width, class_num])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones

    test_label_mask = np.zeros([height * width, class_num])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones

    val_label_mask = np.zeros([height * width, class_num])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones

    return train_label_mask, test_label_mask, val_label_mask
