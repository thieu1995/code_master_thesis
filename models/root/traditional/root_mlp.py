#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:44, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
from time import time
from config import Config


class RootMlp(RootBase):
	"""
		This is the root of all Multi-layer Perceptron-based models like FFNN, ANN, MLNN,...
	"""

	def __init__(self, root_base_paras=None):
		super().__init__(root_base_paras)

	def forecasting(self, X, y):
		# Evaluate models on the test set
		y_pred = self.model.predict(X)
		y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
		y_true_unscaled = self.time_series._inverse_scaling__(y, scale_type=self.scaling)
		return y_true_unscaled, y_pred_unscaled, y, y_pred

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
