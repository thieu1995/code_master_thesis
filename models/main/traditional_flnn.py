#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 23:40, 10/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.traditional.root_flnn import RootFlnn
from utils.PreprocessingUtil import MiniBatch
from sklearn.metrics import mean_squared_error
from numpy import add, dot, matmul, zeros, sum
from numpy.random import uniform
from config import Config


class Flnn(RootFlnn):
    def __init__(self, root_base_paras=None, root_flnn_paras=None):
        super().__init__(root_base_paras, root_flnn_paras)
        self.epoch = root_flnn_paras["epoch"]
        self.batch_size = root_flnn_paras["batch_size"]
        self.lr = root_flnn_paras["learning_rate"]
        self.beta = root_flnn_paras["beta"]
        self.obj = Config.METRICS_TRAINING
        self.filename = f"{self.epoch}-{self.batch_size}-{self.lr}-{self.beta}-{self.activation_name}-{self.expand_name}-{self.obj}"

    def training(self):
        input_size, output_size = self.X_train.shape[1], self.y_train.shape[1]
        ## init hyper and momentum parameters
        w, b = uniform(0, 1, (input_size, output_size)), zeros((1, output_size))
        vdw, vdb = zeros((input_size, output_size)), zeros((1, output_size))

        seed = 0
        for epoch in range(self.epoch):
            seed += 1
            mini_batches = MiniBatch(self.X_train, self.y_train, self.batch_size).random_mini_batches(seed)

            total_error = 0
            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch
                X_batch, y_batch = X_batch.T, y_batch.T
                m = X_batch.shape[0]

                # Feed Forward
                output = self.activation_func(add(matmul(X_batch, w), b))

                total_error += mean_squared_error(output, y_batch)

                # Backpropagation
                dout = output - y_batch
                dz = dout * self.activation_backward_func(output)

                db = 1. / m * sum(dz, axis=0, keepdims=True)
                dw = 1. / m * matmul(X_batch.T, dz)

                vdw = self.beta * vdw + (1 - self.beta) * dw
                vdb = self.beta * vdb + (1 - self.beta) * db

                # Update weights
                w -= self.lr * vdw
                b -= self.lr * vdb
            self.loss_train.append(total_error / len(mini_batches))
            if self.verbose:
                print("> Epoch: {}, Best error: {}".format(epoch+1, total_error / len(mini_batches)))
        self.model = {"w": w, "b": b}

    def forecasting(self, X, y):
        y_pred = self.activation_func(dot(X, self.model["w"]) + self.model["b"])
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(y, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, y, y_pred
