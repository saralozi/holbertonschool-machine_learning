#!/usr/bin/env python3
"""A script that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    arr3 = []

    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr3.append(arr1[i] + arr2[i])
        return arr3
    else:
        return None
