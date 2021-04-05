#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:28, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# ------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
import utils.MathUtil as my_math
from utils.GraphUtil import draw_predict_line_with_error
from utils.IOUtil import save_results_to_csv, save_to_csv_dict, save_to_csv
from keras.models import Sequential
from keras.layers import Dense
from permetrics.regression import Metrics
import pickle
from time import time
from config import Config


class RootSsnnBase(RootBase):
    """
        Baseline model for all kind of Self-Structure Neural Network combine with gradient descent training algorithm
    """
    def __init__(self, root_base_paras=None, root_ssnn_paras=None):
        super().__init__(root_base_paras)
        self.epoch = root_ssnn_paras["epoch"]
        self.batch_size = root_ssnn_paras["batch_size"]
        self.activations = root_ssnn_paras["activations"]
        self.activation1_func = getattr(my_math, self.activations[0])
        self.optimizer = root_ssnn_paras["optimizer"]
        self.obj = Config.METRICS_TRAINING
        self.filename = f"{self.epoch}-{self.batch_size}-{'_'.join(self.activations)}-{self.obj}-{self.optimizer}"
        self.n_clusters, self.clustering, self.cluster_score, self.time_cluster = None, None, None, None
        self.S_train, self.S_test = None, None

    def clustering_process(self):
        """
            0. immune + mutation
            1. immune
            2. mean shift
            3. dbscan
            4. kmeans
            5. kmeans++
            6. exp_max (Expectation–Maximization)

            7. immune + kmeans++
            8. immune + exp_max
        """
        pass

    def transforming_process(self):
        self.S_train = self.clustering._transforming__(self.activation1_func, self.X_train)
        self.S_test = self.clustering._transforming__(self.activation1_func, self.X_test)
        self.input_size, self.output_size = self.S_train.shape[1], self.y_train.shape[1]

    def training(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.output_size, input_dim=self.input_size, activation=self.activations[1]))
        self.model.compile(loss=self.obj, optimizer=self.optimizer)
        ml = self.model.fit(self.S_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.verbose)
        self.loss_train = ml.history["loss"]

    def forecasting(self, X, y):
        y_pred = self.model.predict(X)
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(y, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, y, y_pred

    def save_results(self, results: dict, loss_train=None):
        pathsave_model = f"{self.pathsave}/{Config.RESULTS_FOLDER_MODEL}"
        pathsave_visual = f"{self.pathsave}/{Config.RESULTS_FOLDER_VISUALIZE}"

        # For this project only
        # train_y_pred_unscaled[train_y_pred_unscaled < 0] = 0
        # test_y_pred_unscaled[test_y_pred_unscaled < 0] = 0

        RM1 = Metrics(results[Config.Y_TRAIN_TRUE_UNSCALED].flatten(), results[Config.Y_TRAIN_PRED_UNSCALED].flatten())
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_TESTING)
        RM2 = Metrics(results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten())
        mm2 = RM2.get_metrics_by_list_names(Config.METRICS_TESTING)

        item = {'model_name': self.filename, 'time_train': self.time_train, 'time_predict': self.time_predict, 'time_total': self.time_total,
                'n_clusters': self.n_clusters, 'silhouette': self.cluster_score[0], 'calinski': self.cluster_score[1], 'davies': self.cluster_score[2]}
        for metric_name, value in mm1.items():
            item[metric_name + "_train"] = value
        for metric_name, value in mm2.items():
            item[metric_name + "_test"] = value
        filename_metrics = f"{Config.FILENAME_METRICS}-{self.filename}"
        save_results_to_csv(item, filename_metrics, pathsave_model)

        ## Save prediction results of training set and testing set to csv file
        data = {key: results[key] for key in Config.HEADER_TRAIN_CSV}
        filename_pred_train = f"{Config.FILENAME_PRED_TRAIN}-{self.filename}"
        save_to_csv_dict(data, filename_pred_train, pathsave_model)

        data = {key: results[key] for key in Config.HEADER_TEST_CSV}
        filename_pred_test = f"{Config.FILENAME_PRED_TEST}-{self.filename}"
        save_to_csv_dict(data, filename_pred_test, pathsave_model)

        ## Save loss train to csv file
        if self.obj is not None:
            data = [list(range(1, len(loss_train) + 1)), loss_train]
            header = ["Epoch", self.obj]
            filename_loss = f"{Config.FILENAME_LOSS_TRAIN}-{self.filename}"
            save_to_csv(data, header, filename_loss, pathsave_model)

        ## Save models
        if Config.SAVE_MODEL:
            if self.model_name in Config.MODEL_KERAS:
                self.model.save(f'{pathsave_model}/{Config.FILENAME_MODEL}-{self.filename}.h5')
            else:
                model_filename = open(f'{pathsave_model}/{Config.FILENAME_MODEL}-{self.filename}.pkl', 'wb')
                pickle.dump(self, model_filename)
                model_filename.close()
        ## Visualization
        if self.visualize:
            draw_predict_line_with_error([results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten()],
                                         [item["MAE_test"], item["RMSE_test"]], self.filename, pathsave_visual, Config.VISUALIZE_TYPES)
        if self.verbose:
            print(f'Predict DONE - RMSE: {item["RMSE_test"]:.5f}, MAE: {item["MAE_test"]:.5f}')

    def processing(self):
        self.time_total = time()
        self.preprocessing_dataset()
        self.time_train = time()
        self.time_clustering = time()
        self.clustering_process()
        self.time_clustering = round(time() - self.time_clustering, 4)
        self.transforming_process()
        self.training()
        self.time_train = round(time() - self.time_train, 4)
        self.time_predict = time()
        train_y_true_unscaled, train_y_pred_unscaled, train_y_true_scaled, train_y_pred_scaled = self.forecasting(self.S_train, self.y_train)
        test_y_true_unscaled, test_y_pred_unscaled, test_y_true_scaled, test_y_pred_scaled = self.forecasting(self.S_test, self.y_test)
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

