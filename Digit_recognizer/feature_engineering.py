#! /bin/env/python3

import numpy as np
import pandas as pd

def filling_missing_values(all_data, miss_data):
    """
    Filling up the missing values of digits images
    Rule: filling the missing values of one pixel with an average of its 8 surrounding pixels.
    So the missing pixel is at the center of nine block box(2D). If the missing pixel is at
    the boundary, we consider only the top/bottom half of the box. If the missing pixel is at
    the corner, we consider only the corner of the box

    :param all_data: all input data without labels; The shape is (num_test + num_train, 28, 28)
    :type all_data: numpy array
    :param miss_matrix: True/False matrix of missing value, shape is same as all_data
    :type miss_matrix: numpy array
    :return:
    """


    num_data = all_data.shape[0]
    nine_block_box = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    num_points = nine_block_box.shape[0]
    for i in range(num_data):
        img_matrix = all_data[i]
        miss_matrix = miss_data[i]
        for j, k in zip(img_matrix[0], img_matrix[1]):
            if miss_matrix[j][k] == True:
                refer = np.repeat(np.array([j, k]), num_points)
                points = nine_block_box + refer
                filtered_points = list(filter(lambda x: np.all(x >= 0), points))
                all_data[i][j][k] = np.mean(
                    [img_matrix[s][t] for s, t in zip(filtered_points[:, 0], filtered_points[:, 1])])

    return all_data






