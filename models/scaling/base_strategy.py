#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:49, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import concatenate


class BaseStrategy:

    def __init__(self):
        pass

    def basic_allocate_VMs(self, resource_used):
        pass

    def sla_violation(self, actual_used=None, allocated=None):
        delta = actual_used - allocated
        violation_count = len(delta[delta > 0])
        return float(violation_count) / len(actual_used)

    def __util_level__(self, resource_used=None, resource_allocated=None):
        temp = 1 if resource_allocated == 0 else resource_allocated
        return float(resource_used) / temp

    def __calculate_adi__(self, util_level=None, bound=None):
        lv = 0
        if util_level <= bound[0]:
            lv = bound[0] - util_level
        elif util_level >= bound[1]:
            lv = util_level - bound[1]
        return lv

    def adi_qos_calculation(self, bound=(0.5, 0.8), resource_used=None, resource_allocated=None):
        time_used = concatenate((resource_used, resource_allocated), axis=1)
        adi_list = []
        for w, m in time_used:
            util_level = self.__util_level__(w, m)
            adi = self.__calculate_adi__(util_level, bound)
            adi_list.append(adi)
        return sum(adi_list)
