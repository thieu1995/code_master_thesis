#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:48, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from models.root.traditional.root_mlp import RootMlp
from config import Config


class Mlp(RootMlp):
	def __init__(self, root_base_paras=None, root_mlp_paras=None):
		super().__init__(root_base_paras)
		self.list_layers = root_mlp_paras["list_layers"]
		self.n_layers = len(self.list_layers)
		self.epoch = root_mlp_paras["epoch"]
		self.batch_size = root_mlp_paras["batch_size"]
		self.optimizer = root_mlp_paras["optimizer"]
		self.lr = root_mlp_paras["learning_rate"]
		self.obj = Config.METRICS_TRAINING
		self.valid_split = root_mlp_paras["valid_split"]
		self.filename = f"{self.epoch}-{self.batch_size}-{self.n_layers}-{self.optimizer}-{self.lr}-{self.obj}-{self.valid_split}"

	def training(self):
		self.model = Sequential()
		for idx, layer in enumerate(self.list_layers):
			if layer["n_nodes"] is None:
				self.list_layers[idx]["n_nodes"] = (self.n_layers - idx) * self.size_input + 1
			if idx == 0:
				self.model.add(Dense(layer["n_nodes"], input_dim=self.size_input, activation=layer["activation"]))
			else:
				self.model.add(Dense(layer["n_nodes"], activation=layer["activation"]))
			if idx != (self.n_layers - 1):
				self.model.add(Dropout(layer["dropout"]))
		# Configure the model and start training
		self.opt = getattr(optimizers, self.optimizer)(learning_rate=self.lr)
		self.model.compile(optimizer=self.opt, loss=self.obj)

		# self.model.compile(optimizer=self.optimizer, loss=getattr(MathUtil, self.obj))
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size,
		                    verbose=self.verbose, validation_split=self.valid_split)
		self.loss_train = ml.history["loss"]
