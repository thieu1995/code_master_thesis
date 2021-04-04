#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:20, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from utils.PreprocessingUtil import TimeSeries
from utils.IOUtil import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.GraphUtil import draw_predict_line_with_error
from permetrics.regression import Metrics
import pickle
from config import Config


class RootBase:
    """ This is the root of all networks """

    def __init__(self, root_base_paras=None):
        self.data_original = root_base_paras["data_original"]
        self.train_split = root_base_paras["train_split"]
        self.data_windows = root_base_paras["data_windows"]
        self.scaling = root_base_paras["scaling"]
        self.feature_size = root_base_paras["feature_size"]
        self.network_type = root_base_paras["network_type"]
        self.size_input = len(self.data_windows) * self.feature_size
        self.visualize = root_base_paras["visualize"]
        self.verbose = root_base_paras["verbose"]
        self.model_name = root_base_paras["model_name"]
        self.pathsave = root_base_paras["pathsave"]

        self.filename, self.obj = None, None
        self.model, self.solution, self.loss_train= None, None, []
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.time_train, self.time_predict, self.time_total = None, None, None
        self.time_series = None

    def preprocessing_dataset(self, *args):
        ## standardize the dataset using the mean and standard deviation of the training data.
        self.time_series = TimeSeries(data=self.data_original, train_split=self.train_split)
        data_new = self.time_series._scaling__(self.scaling)

        if len(args) != 0 and isinstance(args[0], object):
            self.X_test, self.y_test = self.time_series._make_train_test_with_expanded__(data_new, self.data_windows, self.time_series.train_split,
                                                                                None, self.network_type, args[0])
            self.X_train, self.y_train = self.time_series._make_train_test_with_expanded__(data_new, self.data_windows, 0, self.time_series.train_split,
                                                                                  self.network_type, args[0])
        else:
            self.X_test, self.y_test = self.time_series._make_train_test_data__(data_new, self.data_windows, self.time_series.train_split, None, self.network_type)
            self.X_train, self.y_train = self.time_series._make_train_test_data__(data_new, self.data_windows, 0, self.time_series.train_split, self.network_type)

    def save_results(self, results:dict, loss_train=None):
        pathsave_model = f"{self.pathsave}/{Config.RESULTS_FOLDER_MODEL}"
        pathsave_visual = f"{self.pathsave}/{Config.RESULTS_FOLDER_VISUALIZE}"

        # For this project only
        # train_y_pred_unscaled[train_y_pred_unscaled < 0] = 0
        # test_y_pred_unscaled[test_y_pred_unscaled < 0] = 0

        RM1 = Metrics(results[Config.Y_TRAIN_TRUE_UNSCALED].flatten(), results[Config.Y_TRAIN_PRED_UNSCALED].flatten())
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_TESTING)
        RM2 = Metrics(results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten())
        mm2 = RM2.get_metrics_by_list_names(Config.METRICS_TESTING)

        item = {'model_name': self.filename, 'time_train': self.time_train, 'time_predict': self.time_predict, 'time_total': self.time_total}
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

        data = {key: results[key] for key in Config.HEADER_TRAIN_CSV}
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
            draw_predict_line_with_error([ results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten() ],
                                         [item["MAE_test"], item["RMSE_test"]], self.filename, pathsave_visual, Config.VISUALIZE_TYPES)
        if self.verbose:
            print(f'Predict DONE - RMSE: {item["RMSE_test"]:.5f}, MAE: {item["MAE_test"]:.5f}')

    def forecasting(self, X, y):
        pass

    def training(self):
        pass

    def processing(self):
        pass
