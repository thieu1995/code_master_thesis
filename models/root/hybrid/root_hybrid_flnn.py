#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:16, 26/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from permetrics.regression import Metrics
from models.root.root_base import RootBase
import utils.MathUtil as my_math
from config import Config
from time import time
from numpy import reshape, add, matmul


class RootHybridFlnn(RootBase):
    def __init__(self, root_base_paras=None, root_hybrid_flnn=None):
        super().__init__(root_base_paras)

        self.activation_name = root_hybrid_flnn["activation"]
        self.activation_func = getattr(my_math, self.activation_name)
        self.expand_name = root_hybrid_flnn["expand"]
        self.expand_func = getattr(my_math, "expand_" + self.expand_name)

        self.lb = root_hybrid_flnn["lb"]
        self.ub = root_hybrid_flnn["ub"]
        self.obj = Config.METRICS_TRAINING
        self.filename = f"{self.obj}-{self.activation_name}-{self.expand_name}"

    def __settings__(self):
        self.size_input, self.size_output = self.X_train.shape[1], self.y_train.shape[1]
        self.size_w = self.size_input * self.size_output
        self.size_b = self.size_output
        self.problem_size = self.size_w + self.size_b
        self.lb = self.lb * self.problem_size
        self.ub = self.ub * self.problem_size

    def __decode_model__(self, solution):
        w = reshape(solution[:self.size_w], (self.size_input, self.size_output))
        b = reshape(solution[self.size_w:], (-1, self.size_output))
        self.model = {"w": w, "b": b}

    def forecasting(self, data_X=None, data_y=None):
        y_pred = self.activation_func(add(matmul(data_X, self.model["w"]), self.model["b"]))
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(data_y, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, data_y, y_pred

    # Evaluates the objective function
    def objective_function(self, solution=None):
        w = reshape(solution[:self.size_w], (self.size_input, self.size_output))
        b = reshape(solution[self.size_w:], (-1, self.size_output))
        y_pred = self.activation_func(add(matmul(self.X_train, w), b))
        obj = Metrics(self.y_train.flatten(), y_pred.flatten())
        return obj.get_metric_by_name(self.obj.upper(), {"decimal": 8})[self.obj.upper()]


    def training(self):  # Depend the child class of this class. They will implement their training function
        pass

    def processing(self):
        self.time_total = time()
        self.preprocessing_dataset(self.expand_func)
        self.time_train = time()
        self.__settings__()
        self.training()
        self.__decode_model__(self.solution)
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
