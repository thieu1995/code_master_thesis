#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 23:34, 10/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
import utils.MathUtil as my_math
from config import Config
from time import time


class RootFlnn(RootBase):
    def __init__(self, root_base_paras=None, paras=None):
        super().__init__(root_base_paras)
        self.activation_name = paras["activation"]
        self.activation_func = getattr(my_math, self.activation_name)
        self.activation_backward_func = getattr(my_math, "derivative_" + self.activation_name)
        self.expand_name = paras["expand"]
        self.expand_func = getattr(my_math, "expand_" + self.expand_name)

    def processing(self):
        self.time_total = time()
        self.preprocessing_dataset(self.expand_func)
        self.time_train = time()
        self.training()
        self.time_train = round(time() - self.time_train, 4)
        self.time_predict = time()
        train_y_true_unscaled, train_y_pred_unscaled, train_y_true_scaled, train_y_pred_scaled = self.forecasting(self.X_train, self.y_train)
        test_y_true_unscaled, test_y_pred_unscaled, test_y_true_scaled, test_y_pred_scaled = self.forecasting(self.X_test, self.y_test)
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_total = round(time() - self.time_total, 4)
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
