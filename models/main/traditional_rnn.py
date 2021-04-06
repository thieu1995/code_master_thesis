#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:20, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras import optimizers
from models.root.traditional.root_rnn import RootRnn


class Rnn(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		super().__init__(root_base_paras, root_rnn_paras)

	def training(self):
		self.model = Sequential()
		for idx, layer in enumerate(self.list_layers):
			if layer["n_nodes"] is None:
				self.list_layers[idx]["n_nodes"] = (self.n_layers - idx) * self.size_input + 1
			if idx == 0:
				self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"], input_shape=(self.X_train.shape[1], 1)))    # First hidden layer
			elif idx == (self.n_layers - 1):
				self.model.add(Dense(units=1, activation=layer["activation"]))          # Output layer
			else:
				self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"]))        # Hidden layer
			if idx != (self.n_layers - 1):
				self.model.add(Dropout(layer["dropout"]))

		# Configure the model and start training
		self.opt = getattr(optimizers, self.optimizer)(learning_rate=self.lr)
		self.model.compile(optimizer=self.opt, loss=self.obj)

		# self.model.compile(optimizer=self.optimizer, loss=getattr(MathUtil, self.obj))
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size,
		                    verbose=self.verbose, validation_split=self.valid_split)
		self.loss_train = ml.history["loss"]


class Lstm(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		super().__init__(root_base_paras, root_rnn_paras)

	def training(self):
		self.model = Sequential()
		for idx, layer in enumerate(self.list_layers):
			if layer["n_nodes"] is None:
				self.list_layers[idx]["n_nodes"] = (self.n_layers - idx) * self.size_input + 1
			if idx == 0:
				if self.n_layers >= 3:
					self.model.add(LSTM(units=layer["n_nodes"], return_sequences=True, activation=layer["activation"], input_shape=(None, 1)))
				else:
					self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"], input_shape=(None, 1)))  # First hidden layer
			elif idx == (self.n_layers - 1):
				self.model.add(Dense(units=1, activation=layer["activation"]))  # Output layer
			else:
				self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"]))  # Hidden layer
			if idx != (self.n_layers - 1):
				self.model.add(Dropout(layer["dropout"]))

		# Configure the model and start training
		self.opt = getattr(optimizers, self.optimizer)(learning_rate=self.lr)
		self.model.compile(optimizer=self.opt, loss=self.obj)

		# self.model.compile(optimizer=self.optimizer, loss=getattr(MathUtil, self.obj))
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size,
		                    verbose=self.verbose, validation_split=self.valid_split)
		self.loss_train = ml.history["loss"]


class Gru(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		super().__init__(root_base_paras, root_rnn_paras)

	def training(self):
		self.model = Sequential()
		for idx, layer in enumerate(self.list_layers):
			if layer["n_nodes"] is None:
				self.list_layers[idx]["n_nodes"] = (self.n_layers - idx) * self.size_input + 1
			if idx == 0:
				if self.n_layers >= 3:
					self.model.add(GRU(units=layer["n_nodes"], return_sequences=True, activation=layer["activation"], input_shape=(self.X_train.shape[1], 1)))
				else:
					self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"], input_shape=(self.X_train.shape[1], 1)))  # First hidden layer
			elif idx == (self.n_layers - 1):
				self.model.add(Dense(units=1, activation=layer["activation"]))  # Output layer
			else:
				self.model.add(LSTM(units=layer["n_nodes"], activation=layer["activation"]))  # Hidden layer
			if idx != (self.n_layers - 1):
				self.model.add(Dropout(layer["dropout"]))

		# Configure the model and start training
		self.opt = getattr(optimizers, self.optimizer)(learning_rate=self.lr)
		self.model.compile(optimizer=self.opt, loss=self.obj)

		# self.model.compile(optimizer=self.optimizer, loss=getattr(MathUtil, self.obj))
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size,
		                    verbose=self.verbose, validation_split=self.valid_split)
		self.loss_train = ml.history["loss"]


class Cnn(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None, cnn_paras=None):
		super().__init__(root_base_paras, root_rnn_paras)
		self.filter_size = cnn_paras["filter_size"]
		self.kernel_size = cnn_paras["kernel_size"]
		self.pool_size = cnn_paras["pool_size"]
		self.activation = cnn_paras["activation"]
		self.filename = f"{self.filename}-{self.filter_size}-{self.kernel_size}-{self.pool_size}-{self.activation}"

	def training(self):
		#  The CNN 1-HL architecture
		self.model = Sequential()
		self.model.add(Conv1D(filters=self.filter_size, kernel_size=self.kernel_size, activation=self.activation, input_shape=(self.X_train.shape[1], 1)))
		self.model.add(MaxPooling1D(pool_size=self.pool_size))
		self.model.add(Flatten())

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
