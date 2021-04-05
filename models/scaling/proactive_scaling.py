#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:51, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.base_strategy import BaseStrategy
from numpy import ceil, zeros, array, reshape, max, sum
from utils.IOUtil import load_csv


class SLABasedOnResources(BaseStrategy):
    """
        Allocate VMs based on resource metric usage
    """

    def __init__(self, max_vms=10, scaling_factor=2.0, adaptation_len=3, VM_capacity=None, resources=None):
        super().__init__()
        self.max_vms = max_vms
        self.scaling_factor = scaling_factor
        self.adap_len = adaptation_len
        self.VM_capacity = VM_capacity
        if VM_capacity is None:
            self.VM_capacity = (0.25, 0.03)
        self.resources = resources
        if resources is None:
            self.resources = ('CPU', 'RAM')

    def __allocate_VM(self, res_consump, idx):
        capa = self.VM_capacity[idx]
        return ceil(res_consump / capa)

    def __allocate_VMs(self, resources):
        return ceil(resources / self.VM_capacity)

    def allocate_VM(self, resource_actual=None, resource_predict=None, id_metric=None):
        allocated = zeros(len(resource_actual))
        allocated[:self.adap_len] = resource_actual[:self.adap_len]
        for idx in range(self.adap_len, len(resource_actual)):
            allocated[idx] = self.scaling_factor * resource_predict[idx] + (1.0 / self.adap_len) * \
                             sum([max(0, (resource_actual[i] - resource_predict[i])) for i in range(idx - self.adap_len, idx)])
        return self.__allocate_VM(array(allocated), id_metric)

    def calculate_VMs_allocated(self, resources_actual=None, resources_predict=None):
        number_of_VMs = []
        for index_met in range(len(self.resources)):
            temp = self.allocate_VM(resources_actual[:, index_met], resources_predict[:, index_met], index_met)
            number_of_VMs.append(temp)
        number_of_VMs = reshape(array(number_of_VMs), (-1, len(self.resources)))
        return reshape(max(number_of_VMs, axis=1), (-1, 1))

    def sla_violate(self, allocated_VMs=None, used_VMs=None):
        total_time = len(allocated_VMs)
        max_VMs_used = max(used_VMs, axis=1)
        number_of_violate = max_VMs_used - allocated_VMs
        return float(len(number_of_violate[number_of_violate >= 0])) / total_time


class SLABasedOnVms(BaseStrategy):
    """
        Allocate VMs based on Vms usage
    """
    def __init__(self, max_vms=100, scaling_factor=2.0, adaptation_len=3, VM_capacity=None, resources=None, pathfiles=None):
        super().__init__()
        self.max_vms = max_vms
        self.scaling_factor = scaling_factor
        self.adap_len = adaptation_len
        self.VM_capacity = VM_capacity
        if VM_capacity is None:
            self.VM_capacity = (0.25, 0.03)
        self.resources = resources
        if resources is None:
            self.resources = ('CPU', 'RAM')
        self.pathfiles = pathfiles

    def calculate_VMs_allocated(self, resource_actuals=None, resource_predicts=None):
        vms_actuals = ceil(resource_actuals / self.VM_capacity)
        vms_predicts = ceil(resource_predicts / self.VM_capacity)

        vms_actual = reshape(max(vms_actuals, axis=1), (-1, 1))
        vms_predict = reshape(max(vms_predicts, axis=1), (-1, 1))

        vms_allocated = zeros(shape=vms_actual.shape)
        vms_allocated[:self.adap_len] = vms_actual[:self.adap_len]

        for idx in range(self.adap_len, len(vms_actual)):
            vms_allocated[idx] = self.scaling_factor * vms_predict[idx] + (1.0 / self.adap_len) * \
                                 sum([max(0, (vms_actual[i] - vms_predict[i])) for i in range(idx - self.adap_len, idx)])
        return reshape(ceil(vms_allocated), (-1, 1))

    def sla_violate(self, list_resource_files=None):
        resource_actuals = []
        resource_predicts = []
        for resource_file in list_resource_files:
            res_loaded = load_csv(resource_file)
            res_true, res_pred = res_loaded[:, 0], res_loaded[:, 1]
            resource_actuals.append(res_true)
            resource_predicts.append(res_pred)
        resource_actuals = array(resource_actuals).T
        resource_predicts = array(resource_predicts).T
        number_of_VMs = self.calculate_VMs_allocated(resource_actuals=resource_actuals, resource_predicts=resource_predicts)

        resource_allocated = number_of_VMs * self.VM_capacity
        violated_list_boolean = (resource_actuals >= resource_allocated).any(axis=1)
        return sum(violated_list_boolean) / len(violated_list_boolean), (resource_allocated, number_of_VMs)

    def get_predicted_and_allocated_vms(self, list_resource_files=None):
        resource_actuals = []
        resource_predicts = []
        for resource_file in list_resource_files:
            res_loaded = load_csv(resource_file)
            res_true, res_pred = res_loaded[:, 0], res_loaded[:, 1]
            resource_actuals.append(res_true)
            resource_predicts.append(res_pred)
        resource_actuals = array(resource_actuals).T
        resource_predicts = array(resource_predicts).T

        vms_actuals = ceil(resource_actuals / self.VM_capacity)
        vms_predicts = ceil(resource_predicts / self.VM_capacity)

        vms_actual = reshape(max(vms_actuals, axis=1), (-1, 1))
        vms_predict = reshape(max(vms_predicts, axis=1), (-1, 1))

        vms_allocated = zeros(shape=vms_actual.shape)
        vms_allocated[:self.adap_len] = vms_actual[:self.adap_len]
        sla = zeros(shape=vms_actual.shape)

        for idx in range(self.adap_len, len(vms_actual)):
            vms_allocated[idx] = self.scaling_factor * vms_predict[idx] + (1.0 / self.adap_len) * sum(
                [max(0, (vms_actual[i] - vms_predict[i])) for i in range(idx - self.adap_len, idx)])
            sla[idx] = vms_allocated[idx] - self.scaling_factor * vms_predict[idx]
        vms_allocated = reshape(ceil(vms_allocated), (-1, 1))
        sla = reshape(ceil(sla), (-1, 1))
        return vms_predict, vms_actual, vms_allocated, sla