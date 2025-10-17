#!/usr/bin/env python3
"""A script that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    mat3 = []

    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat2[0])):
                element = 0
                for k in range(len(mat2)):
                    element += mat1[i][k] * mat2[k][j]
                row.append(element)
            mat3.append(row)
        return mat3
    else:
        return None
