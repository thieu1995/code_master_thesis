#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:42, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.traditional.root_ssnn import RootSsnnBase
from utils.ClusterUtil.algorithm.immune import ImmuneInspiration
from utils.ClusterUtil.algorithm.kmeans import BaseKmeans, KMeansPlusPlus
from utils.ClusterUtil.algorithm.mean_shift import MeanShiftSklearn
from utils.ClusterUtil.algorithm.expectation_maximization import GaussianMixtureSklearn
from utils.ClusterUtil.algorithm.dbscan import DbscanSklearn


class Sonia(RootSsnnBase):
    """
        Traditional Self-Organized Network Inspired by Immune Algorithm (SONIA)
            (Self-Organizing Neural Network)
    """
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, stimulation_level=0.15, positive_number=0.15,
                 distance_level=0.15, max_cluster=1000, mutation_id=0):
        super().__init__(root_base_paras, root_ssnn_paras)
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
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = \
            self.clustering._immune_mutation__(self.X_train, self.list_clusters, self.distance_level, self.mutation_id)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class ImmuneSsnn(RootSsnnBase):
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, stimulation_level=0.15, positive_number=0.15,
                 distance_level=0.15, max_cluster=1000, mutation_id=0):
        super().__init__(root_base_paras, root_ssnn_paras)
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


class KmeanSsnn(RootSsnnBase):
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = BaseKmeans(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, type=0)


class MeanShiftSsnn(RootSsnnBase):
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, bandwidth=0.15):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.bandwidth = bandwidth
        self.filename = f"{self.filename}-{bandwidth}"

    def clustering_process(self):
        self.clustering = MeanShiftSklearn(self.bandwidth)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class DbscanSsnn(RootSsnnBase):
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, eps=0.15, min_samples=3):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.eps = eps
        self.min_samples = min_samples
        self.filename = f"{self.filename}-{eps}-{min_samples}"

    def clustering_process(self):
        self.clustering = DbscanSklearn(eps=self.eps, min_samples=self.min_samples)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class KmeanDoublePlusSsnn(RootSsnnBase):
    def __init__(self, root_base_paras=None, root_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = KMeansPlusPlus(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class GaussianSsnn(RootSsnnBase):
    """
    Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
    """

    def __init__(self, root_base_paras=None, root_ssnn_paras=None, n_clusters=10):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.n_clusters = n_clusters
        self.filename = f"{self.filename}-{n_clusters}"

    def clustering_process(self):
        self.clustering = GaussianMixtureSklearn(self.n_clusters)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class ImmuneKmeanSsnn(RootSsnnBase):
    """
        Immune + Kmeans++ (No need Kmeans)
    """

    def __init__(self, root_base_paras=None, root_ssnn_paras=None, stimulation_level=0.15, positive_number=0.15,
                 distance_level=0.15, max_cluster=1000, mutation_id=0):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.stimulation_level = stimulation_level
        self.positive_number = positive_number
        self.distance_level = distance_level
        self.max_cluster = max_cluster
        self.mutation_id = mutation_id
        self.filename = f"{self.filename}-{stimulation_level}-{positive_number}-{distance_level}-{max_cluster}-{mutation_id}"

    def clustering_process(self):
        clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                       distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
        self.clustering = KMeansPlusPlus(t0)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


class ImmuneGaussSsnn(RootSsnnBase):
    """
       Immune + Expectation–Maximization
    """

    def __init__(self, root_base_paras=None, root_ssnn_paras=None, stimulation_level=0.15, positive_number=0.15,
                 distance_level=0.15, max_cluster=1000, mutation_id=0):
        super().__init__(root_base_paras, root_ssnn_paras)
        self.stimulation_level = stimulation_level
        self.positive_number = positive_number
        self.distance_level = distance_level
        self.max_cluster = max_cluster
        self.mutation_id = mutation_id
        self.filename = f"{self.filename}-{stimulation_level}-{positive_number}-{distance_level}-{max_cluster}-{mutation_id}"

    def clustering_process(self):
        clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                       distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
        t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
        self.clustering = GaussianMixtureSklearn(t0)
        self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
        self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


