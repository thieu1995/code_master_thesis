#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:08, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
from numpy import reshape
from config import Config
from time import time


class RootRnn(RootBase):
	"""
		This is the root of all RNN-based models like RNNs, LSTMs, GRUs,...
	"""
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		super().__init__(root_base_paras)
		self.list_layers = root_rnn_paras["list_layers"]
		self.n_layers = len(self.list_layers)
		self.epoch = root_rnn_paras["epoch"]
		self.batch_size = root_rnn_paras["batch_size"]
		self.optimizer = root_rnn_paras["optimizer"]
		self.lr = root_rnn_paras["learning_rate"]
		self.obj = Config.METRICS_TRAINING
		self.valid_split = root_rnn_paras["valid_split"]
		self.filename = f"{self.epoch}-{self.batch_size}-{self.n_layers}-{self.optimizer}-{self.lr}-{self.obj}-{self.valid_split}"

	def forecasting(self, X, y):
		y_pred = self.model.predict(X)
		y = reshape(y, y_pred.shape)
		y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
		y_true_unscaled = self.time_series._inverse_scaling__(y, scale_type=self.scaling)
		return y_true_unscaled, y_pred_unscaled, y, y_pred

	def processing(self):
		self.time_system = time()
		self.preprocessing_dataset()
		self.time_total_train = time()
		self.training()
		self.time_total_train = round(time() - self.time_total_train, 4)
		self.time_epoch = round(self.time_total_train / self.epoch, 4)
		self.time_predict = time()

		train_y_true_unscaled, train_y_pred_unscaled, train_y_true_scaled, train_y_pred_scaled = self.forecasting(self.X_train, self.y_train)
		test_y_true_unscaled, test_y_pred_unscaled, test_y_true_scaled, test_y_pred_scaled = self.forecasting(self.X_test, self.y_test)
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
		self.time_predict = round(time() - self.time_predict, 8)
		self.time_system = round(time() - self.time_system, 4)
		self.save_results(results, self.loss_train)
