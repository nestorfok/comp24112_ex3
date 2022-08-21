#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alessio Sarullo

YOU SHOULD NOT MODIFY THIS CODE.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat


def _check_dim(dim, h, w):
    if dim != h * w:
        raise ValueError(f'Dimensions do not match: h={h}, w={w} ' +
                         f'data length={dim}.')


def load_data():
    dataset_mat = loadmat('ORLfacedata.mat', squeeze_me=True)
    data = dataset_mat['data']
    labels = dataset_mat['labels']

    # Data is saved column-first. Needs to be transposed.
    data = data.reshape(400, 32, 32).transpose(0, 2, 1).reshape(400, 1024)
    return data, labels


def show_single_face(img, h=32, w=32):
    show_faces(np.atleast_2d(img), num_per_row=1, h=h, w=w)


def show_faces(data, num_per_row=10, h=32, w=32):
    _check_dim(data.shape[1], h, w)

    num_imgs = data.shape[0]
    num_img_in_last_row = num_imgs % num_per_row
    if num_img_in_last_row > 0:
        num_empty_imgs = num_per_row - num_img_in_last_row
        zero_imgs = np.zeros([num_empty_imgs, data.shape[1]], dtype=data.dtype)
        ext_data = np.concatenate([data, zero_imgs], axis=0)
    else:
        ext_data = data
    assert ext_data.shape[0] % num_per_row == 0

    num_rows = ext_data.shape[0] // num_per_row
    img_grid = ext_data.reshape(num_rows, num_per_row, h, w)
    img_grid = img_grid.transpose([0, 2, 1, 3])
    img_grid = img_grid.reshape(num_rows * h, num_per_row * w)
    plt.figure()
    plt.imshow(img_grid, cmap='gray')
    plt.axis('off')


def partition_data(labels, num_per_class):
    examples_per_class = {}
    for i, l in enumerate(labels):
        examples_per_class.setdefault(l, []).append(i)

    num_ex_smallest_class = min([len(x) for x in examples_per_class.values()])
    if num_per_class > num_ex_smallest_class:
        raise ValueError(f'The smallest class only has ' +
                         f'{num_ex_smallest_class} examples ' +
                         f'({num_per_class} required).')

    train_inds, test_inds = set(), set()
    for cl, examples in examples_per_class.items():
        perm_cl_examples = np.random.permutation(examples)
        train_inds |= set(perm_cl_examples[:num_per_class].tolist())
        test_inds |= set(perm_cl_examples[num_per_class:].tolist())
    assert len(train_inds) == len(examples_per_class) * num_per_class
    assert sorted(train_inds | test_inds) == list(range(labels.shape[0]))

    test_inds = np.array(sorted(test_inds))
    train_inds = np.array(sorted(train_inds))
    return train_inds, test_inds


def split_left_right(data, h=32, w=32):
    _check_dim(data.shape[1], h, w)
    data = data.reshape(-1, h, w)
    left = data[:, :, :w // 2].reshape(data.shape[0], -1)
    right = data[:, :, w // 2:].reshape(data.shape[0], -1)
    return left, right

def join_left_right(left, right, h=32, w=16):
    _check_dim(left.shape[1], h, w)
    _check_dim(right.shape[1], h, w)
    joined = np.concatenate((left.reshape(left.shape[0], h, w),
                             right.reshape(right.shape[0], h, w)),
                            axis=-1).reshape(left.shape[0], -1)
    return joined


def show_split_faces(data, num_per_row=10, h=32, w=16):
    show_faces(data, num_per_row=num_per_row, h=h, w=w)
