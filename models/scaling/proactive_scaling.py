#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:51, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.strategy_base import BaseStrategy
from numpy import ceil, zeros, array, reshape, max, concatenate, sum
from utils.IOUtil import load_csv


class SLABasedOnResources(BaseStrategy):
    """
        Allocate VMs based on resource metric usage
    """

    def __init__(self, max_vms=10, scaling_coefficient=2.0, adaptation_len=3, capacity_VM=(0.25, 0.03), metrics=('CPU', 'RAM')):
        self.max_vms = max_vms
        self.sla_coef = scaling_coefficient
        self.adap_len = adaptation_len
        self.capacity_VM = capacity_VM
        self.metrics = metrics

    def __allocate_VM(self, res_consump, idx):
        capa = self.capacity_VM[idx]
        return ceil(res_consump / capa)

    def __allocate_VMs(self, resources):
        return ceil(resources / self.capacity_VM)

    def allocate_VM(self, resource_actual=None, resource_predict=None, id_metric=None):
        allocated = zeros(len(resource_actual))
        allocated[:self.adap_len] = resource_actual[:self.adap_len]
        for idx in range(self.adap_len, len(resource_actual)):
            allocated[idx] = self.sla_coef * resource_predict[idx] + (1.0 / self.adap_len) * \
                             sum([max(0, (resource_actual[i] - resource_predict[i])) for i in range(idx - self.adap_len, idx)])
        return self.__allocate_VM(array(allocated), id_metric)

    def allocate_VMs(self, resources_actual=None, resources_predict=None):
        number_of_VMs = []
        for index_met in range(len(self.metrics)):
            temp = self.allocate_VM(resources_actual[:, index_met], resources_predict[:, index_met], index_met)
            number_of_VMs.append(temp)
        number_of_VMs = reshape(array(number_of_VMs), (-1, len(self.metrics)))
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

    def __init__(self, max_vms=100, scaling_coefficient=2.0, adaptation_len=3, capacity_VM=(0.25, 0.03),
                 metrics=('CPU', 'RAM'), pathfiles=None):
        self.max_vms = max_vms
        self.sla_coef = scaling_coefficient
        self.adap_len = adaptation_len
        self.capacity_VM = capacity_VM
        self.metrics = metrics

    def __convert_resources_to_Vms(self, res_consump):
        return ceil(res_consump / self.capacity_VM)

    def allocate_VMs(self, resources_actuals=None, resources_predicts=None):
        vms_actuals = self.__convert_resources_to_Vms(resources_actuals)
        vms_predicts = self.__convert_resources_to_Vms(resources_predicts)

        vms_actual = reshape(max(vms_actuals, axis=1), (-1, 1))
        vms_predict = reshape(max(vms_predicts, axis=1), (-1, 1))

        vms_allocated = zeros(shape=vms_actual.shape)
        vms_allocated[:self.adap_len] = vms_actual[:self.adap_len]

        for idx in range(self.adap_len, len(vms_actual)):
            vms_allocated[idx] = self.sla_coef * vms_predict[idx] + (1.0 / self.adap_len) * \
                                 sum([max(0, (vms_actual[i] - vms_predict[i])) for i in range(idx - self.adap_len, idx)])
        return reshape(ceil(vms_allocated), (-1, 1))

    def sla_violate(self, cpu_file, ram_file):
        cpu_loaded = load_csv(cpu_file)
        ram_loaded = load_csv(ram_file)
        cpu_actual, cpu_predict = cpu_loaded[:, 0:1], cpu_loaded[:, 1:2]
        ram_actual, ram_predict = ram_loaded[:, 0:1], ram_loaded[:, 1:2]

        resources_actuals = concatenate((cpu_actual, ram_actual), axis=1)
        resources_predicts = concatenate((cpu_predict, ram_predict), axis=1)

        number_of_VMs = self.allocate_VMs(resources_actuals=resources_actuals, resources_predicts=resources_predicts)

        cpu_alloc = number_of_VMs * self.capacity_VM[0]
        ram_alloc = number_of_VMs * self.capacity_VM[1]

        c = array((cpu_actual >= cpu_alloc))
        d = array((ram_actual >= ram_alloc))
        z = concatenate((c, d), axis=1)
        e = array([(x or y) for x, y in z])
        return float(len(e[e == True])) * 100 / len(e), (cpu_alloc, ram_alloc, number_of_VMs)

    def get_predicted_and_allocated_vms(self, cpu_file, ram_file):
        cpu_loaded = load_csv(cpu_file)
        ram_loaded = load_csv(ram_file)
        cpu_actual, cpu_predict = cpu_loaded[:, 0:1], cpu_loaded[:, 1:2]
        ram_actual, ram_predict = ram_loaded[:, 0:1], ram_loaded[:, 1:2]

        resources_actuals = concatenate((cpu_actual, ram_actual), axis=1)
        resources_predicts = concatenate((cpu_predict, ram_predict), axis=1)

        vms_actuals = self.__convert_resources_to_Vms(resources_actuals)
        vms_predicts = self.__convert_resources_to_Vms(resources_predicts)

        vms_actual = reshape(max(vms_actuals, axis=1), (-1, 1))
        vms_predict = reshape(max(vms_predicts, axis=1), (-1, 1))

        vms_allocated = zeros(shape=vms_actual.shape)
        vms_allocated[:self.adap_len] = vms_actual[:self.adap_len]
        sla = zeros(shape=vms_actual.shape)

        for idx in range(self.adap_len, len(vms_actual)):
            vms_allocated[idx] = self.sla_coef * vms_predict[idx] + (1.0 / self.adap_len) * sum(
                [max(0, (vms_actual[i] - vms_predict[i])) for i in range(idx - self.adap_len, idx)])
            sla[idx] = vms_allocated[idx] - self.sla_coef * vms_predict[idx]
        vms_allocated = reshape(ceil(vms_allocated), (-1, 1))
        sla = reshape(ceil(sla), (-1, 1))
        return vms_predict, vms_actual, vms_allocated, sla