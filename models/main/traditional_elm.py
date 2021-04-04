#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import add, matmul, dot
from numpy.random import uniform
from numpy.linalg import pinv
from models.root.traditional.root_elm import RootElm


class Elm(RootElm):
    """
        Amazing tutorial: https://www.kaggle.com/robertbm/extreme-learning-machine-example
    """

    def __init__(self, root_base_paras=None, root_elm_paras=None):
        super().__init__(root_base_paras, root_elm_paras)

    def training(self):
        """
        1. Random weights between input and hidden layer
        2. Calculate output of hidden layer
        3. Calculate weights between hidden and output layer based on matrix multiplication
        """
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        w1 = uniform(size=[self.input_size, self.size_hidden])
        b = uniform(size=[1, self.size_hidden])
        H = self.activation_func(add(matmul(self.X_train, w1), b))
        w2 = dot(pinv(H), self.y_train)
        self.model = {"w1": w1, "b": b, "w2": w2}

    def forecasting(self, X, y):
        hidd = self.activation_func(add(matmul(X, self.model["w1"]), self.model["b"]))
        y_pred = matmul(hidd, self.model["w2"])
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(y, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, y, y_pred
