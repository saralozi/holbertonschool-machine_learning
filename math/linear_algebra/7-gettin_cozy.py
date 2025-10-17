#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""


def cat_matrices2D(arr1, arr2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    result = []
    if axis == 1:
        if len(arr1) == len(arr2):
            for i in range(len(arr1)):
                result.append(arr1[i] + arr2[i])
            return result
        else:
            return None
    else:
        if len(arr1[0]) == len(arr2[0]):
            result = arr1 + arr2
            return result
        else:
            return None
