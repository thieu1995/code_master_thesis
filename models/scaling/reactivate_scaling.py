#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:21, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.base_strategy import BaseStrategy
from numpy import array, ceil, reshape, max


class DemandScalingBasedOnResources(BaseStrategy):
    """
         Allocate VMs based on resource metric usage
    """
    def __init__(self, VM_capacity=(0.25, 0.03), resources=('CPU', 'RAM')):
        super().__init__()
        self.VM_capacity = VM_capacity
        if VM_capacity is None:
            self.VM_capacity = (0.25, 0.03)
        self.resources = resources
        if resources is None:
            self.resources = ('CPU', 'RAM')

    def __allocate_VM(self, resource, idx):
        """
        :param resources_usage: numpy array
        :return:
        """
        capa = self.VM_capacity[idx]
        return ceil(resource / capa)

    def calculate_VMs_allocated(self, resources_usage=None):
        number_of_VMs = ceil(resources_usage / self.VM_capacity)
        return reshape(max(number_of_VMs, axis=1), (-1, 1))

    def allocate_VMs_by_idx(self, resources_usage=None):
        if resources_usage is not array:
            resources_usage = array(resources_usage)
        number_of_VMs = []
        for idx in range(len(self.resources)):
            number_of_VMs.append(self.__allocate_VM(resources_usage[:, idx], idx))
        return reshape(max(number_of_VMs, axis=1), (-1, 1))

