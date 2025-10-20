#!/usr/bin/env python3
"""A script that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two 2D matrices element-wise"""
    if (len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])):
        mat3 = []

        for i in range(len(mat1)):
            row = []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            mat3.append(row)
        return mat3
    else:
        return None
