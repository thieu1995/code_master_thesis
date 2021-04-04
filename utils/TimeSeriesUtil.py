#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 18:31, 17/02/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array


# 1.A Univariate 2D Model (Like: MLP, FLNN, SOM, ...)
def univariate_split_sequence_2d(sequence:array, n_steps:int) -> (array, array):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# 1.B Univariate 3D Model (Like: RNN, LSTM, GRU,...)

# 2. Multivariate 2D Model
## 2.1 Multiple Input Series
## 2.2 Multiple Parallel Series

### 2.1 Multiple Input Series
def multivariate_split_input_series_2d(sequences: array, n_steps: int) -> (array, array):
    ## sequences is the 2d dataset with the last column is y.
    ## Multiple Inputs and Single Output
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]  # Take the last column as y
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


### 2.2 Multiple Parallel Series
def multivariate_split_parallel_series_2d(sequences: array, n_steps: int) -> (array, array):
    ## sequences is the 2d dataset, using all column as features and labels is the next rows of previous n_steps.
    ## Multiple Inputs and Multiple Outputs
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]     # Take the last n_steps row as y
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)





