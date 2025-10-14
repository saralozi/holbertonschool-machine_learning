#!/usr/bin/env python3

def matrix_transpose(matrix):

    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0] * rows for i in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed