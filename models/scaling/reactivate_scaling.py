#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:21, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.strategy_base import BaseStrategy
from numpy import array, ceil, reshape, max


class OnDemandScaling(BaseStrategy):
    """
         Allocate VMs based on resource metric usage
    """
    def __init__(self, capacity_VM=(0.25, 0.03), metrics=('CPU', 'RAM')):
        self.capacity_VM = array(capacity_VM)
        self.metrics = metrics

    def __allocate_VM(self, resource, idx):
        """
        :param resources_usage: numpy array
        :return:
        """
        capa = self.capacity_VM[idx]
        return ceil(resource / capa)

    def __allocate_VMs(self, resources):
        return ceil(resources / self.capacity_VM)

    def allocate_VMs(self, resources_usage=None):
        number_of_VMs = self.__allocate_VMs(resources_usage)
        return reshape(max(number_of_VMs, axis=1), (-1, 1))

    def allocate_VMs_by_idx(self, resources_usage=None):
        if resources_usage is not array:
            resources_usage = array(resources_usage)
        number_of_VMs = []
        for idx in range(len(self.metrics)):
            number_of_VMs.append(self.__allocate_VM(resources_usage[:, idx], idx))
        return reshape(max(number_of_VMs, axis=1), (-1, 1))

