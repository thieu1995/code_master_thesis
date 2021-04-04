#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:41, 16/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import keras.backend as K


def mse(y_true, y_pred):
    return K.mean(K.pow(y_true - y_pred, 2))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))
