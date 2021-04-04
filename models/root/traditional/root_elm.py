#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 23:27, 28/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
from time import time
from config import Config
import utils.MathUtil as my_math


class RootElm(RootBase):
    def __init__(self, root_base_paras=None, paras=None):
        super().__init__(root_base_paras)
        self.activation_name = paras["activation"]
        if paras["size_hidden"] is None:
            self.size_hidden = 2 * self.size_input ** 2 + 1  # 2n^2 + 1
        else:
            self.size_hidden = paras["size_hidden"]
        self.obj = None
        self.activation_func = getattr(my_math, self.activation_name)
        self.filename = f"{self.size_hidden}-{self.activation_name}-{self.obj}"

    def processing(self):
        self.time_total = time()
        self.preprocessing_dataset()
        self.time_train = time()
        self.training()
        self.time_train = round(time() - self.time_train, 4)
        self.time_predict = time()
        train_y_true_unscaled, train_y_pred_unscaled, train_y_true_scaled, train_y_pred_scaled = self.forecasting(self.X_train, self.y_train)
        test_y_true_unscaled, test_y_pred_unscaled, test_y_true_scaled, test_y_pred_scaled = self.forecasting(self.X_test, self.y_test)
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_total = round(time() - self.time_total, 3)
        results = {
            Config.Y_TRAIN_TRUE_SCALED: train_y_true_scaled,
            Config.Y_TRAIN_TRUE_UNSCALED: train_y_true_unscaled,
            Config.Y_TRAIN_PRED_SCALED: train_y_pred_scaled,
            Config.Y_TRAIN_PRED_UNSCALED: train_y_pred_unscaled,
            Config.Y_TEST_TRUE_SCALED: test_y_true_scaled,
            Config.Y_TEST_TRUE_UNSCALED: test_y_true_unscaled,
            Config.Y_TEST_PRED_SCALED: test_y_pred_scaled,
            Config.Y_TEST_PRED_UNSCALED: test_y_pred_unscaled
        }
        self.save_results(results, self.loss_train)
