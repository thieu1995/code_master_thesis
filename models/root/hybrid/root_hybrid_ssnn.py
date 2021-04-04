#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:52, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pickle
from time import time
from numpy import reshape, add, matmul
from models.root.root_base import RootBase
from utils.ClusterUtil.algorithm.dbscan import DbscanSklearn
from utils.ClusterUtil.algorithm.expectation_maximization import GaussianMixtureSklearn
from utils.ClusterUtil.algorithm.immune import ImmuneInspiration, SomInspiration
from utils.ClusterUtil.algorithm.kmeans import BaseKmeans, KMeansPlusPlus
from utils.ClusterUtil.algorithm.mean_shift import MeanShiftSklearn
from permetrics.regression import Metrics
from config import Config
import utils.MathUtil as my_math
from utils.GraphUtil import draw_predict_line_with_error
from utils.IOUtil import save_results_to_csv, save_to_csv_dict, save_to_csv


class RootHybridSsnnBase(RootBase):
    """
        This is root of all hybrid models which include Self-Structure Neural Network and Optimization Algorithms.
        (No more gradient descent here)
    """
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None):
        super().__init__(root_base_paras)
        self.lb = root_hybrid_ssnn_paras["lb"]
        self.ub = root_hybrid_ssnn_paras["ub"]
        self.activations = root_hybrid_ssnn_paras["activations"]
        self.activation1_func = getattr(my_math, self.activations[0])
        self.activation2_func = getattr(my_math, self.activations[1])
        self.obj = Config.METRICS_TRAINING
        self.filename = f"{self.obj}-{'_'.join(self.activations)}"

        self.n_clusters, self.clustering, self.cluster_score, self.time_cluster = None, None, None, None
        self.S_train, self.S_test = None, None

    def clustering_process(self):
        pass

    def transforming_process(self):
        self.S_train = self.clustering._transforming__(self.activation1_func, self.X_train)
        self.S_test = self.clustering._transforming__(self.activation1_func, self.X_test)
        self.input_size, self.output_size = self.S_train.shape[1], self.y_train.shape[1]

        ## This variables below shoudn't come from here, but for simplicity I put it here.
        self.w_size = self.input_size * self.output_size
        self.b_size = self.output_size
        self.problem_size = self.w_size + self.b_size
        self.lb = self.lb * self.problem_size
        self.ub = self.ub * self.problem_size

    def forecasting(self, X, y):
        y_pred = self.activation2_func(add(matmul(X, self.model["w"]), self.model["b"]))
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
        self.decode_solution(self.solution)
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

    ## Helper functions
    def decode_solution(self, solution):
        w = reshape(solution[:self.w_size], (self.input_size, self.output_size))
        b = reshape(solution[self.w_size:], (-1, self.output_size))
        self.model = {"w": w, "b": b}
        return {"w": w, "b": b}

    # Evaluates the objective function
    def objective_function(self, solution=None):
        w = reshape(solution[:self.w_size], (self.input_size, self.output_size))
        b = reshape(solution[self.w_size:], (-1, self.output_size))
        y_pred = self.activation2_func(add(matmul(self.S_train, w), b))
        obj = Metrics(self.y_train.flatten(), y_pred.flatten())
        return obj.get_metric_by_name(self.obj.upper(), {"decimal": 8})[self.obj.upper()]


class RootHybridImmuneSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None,
                 stimulation_level=0.15, positive_number=0.15, distance_level=0.15, max_cluster=1000, mutation_id=0):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.stimulation_level = stimulation_level
        self.positive_number = positive_number
        self.distance_level = distance_level
        self.max_cluster = max_cluster
        self.mutation_id = mutation_id
        self.filename = f"{self.filename}-{stimulation_level}-{positive_number}-{distance_level}-{max_cluster}-{mutation_id}"

    def clustering_process(self):
        self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                            distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridSoniaSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.stimulation_level = sonia_paras["stimulation_level"]       # 0.15
        self.positive_number = sonia_paras["positive_number"]           # 0.15
        self.distance_level = sonia_paras["distance_level"]             # 0.15
        self.max_cluster = sonia_paras["max_cluster"]                   # 1000
        self.mutation_id = sonia_paras["mutation_id"]                   # 0
        self.filename = f"{self.filename}-{self.stimulation_level}-{self.positive_number}-{self.distance_level}-{self.max_cluster}-{self.mutation_id}"

    def clustering_process(self):
        self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                            distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = \
            self.clustering._immune_mutation__(self.X_train, self.list_clusters, self.distance_level, self.mutation_id)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)



class RootHybridKmeanSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = BaseKmeans(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, type=0)


class RootHybridMeanShiftSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, bandwidth=0.15):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.bandwidth = bandwidth
        self.filename = f"{self.filename}-{bandwidth}"

    def clustering_process(self):
        self.clustering = MeanShiftSklearn(self.bandwidth)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridDbscanSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, eps=0.15, min_samples=3):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.eps = eps
        self.min_samples = min_samples
        self.filename = f"{self.filename}-{eps}-{min_samples}"

    def clustering_process(self):
        self.clustering = DbscanSklearn(eps=self.eps, min_samples=self.min_samples)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridKmeanDoublePlusSsnn(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = KMeansPlusPlus(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridGaussianSsnn(RootHybridSsnnBase):
    """
    Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
    """

    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = GaussianMixtureSklearn(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridImmuneKmeanSsnn(RootHybridSsnnBase):
    """
        Immune + Kmeans++ (No need Kmeans)
    """

    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, immune_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.stimulation_level = immune_paras["stimulation_level"]
        self.positive_number = immune_paras["positive_number"]
        self.distance_level = immune_paras["distance_level"]
        self.max_cluster = immune_paras["max_cluster"]
        self.mutation_id = immune_paras["mutation_id"]
        self.filename = f"{self.filename}-{self.stimulation_level}-{self.positive_number}-{self.distance_level}-{self.max_cluster}-{self.mutation_id}"

    def clustering_process(self):
        clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                       distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
        self.clustering = KMeansPlusPlus(t0)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class RootHybridImmuneGaussSsnn(RootHybridSsnnBase):
    """
       Immune + Expectation–Maximization
    """

    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, immune_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras)
        self.stimulation_level = immune_paras["stimulation_level"]
        self.positive_number = immune_paras["positive_number"]
        self.distance_level = immune_paras["distance_level"]
        self.max_cluster = immune_paras["max_cluster"]
        self.mutation_id = immune_paras["mutation_id"]
        self.filename = f"{self.filename}-{self.stimulation_level}-{self.positive_number}-{self.distance_level}-{self.max_cluster}-{self.mutation_id}"

    def clustering_process(self):
        clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                       distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
        self.clustering = GaussianMixtureSklearn(t0)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

