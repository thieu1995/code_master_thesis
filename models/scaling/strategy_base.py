#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:49, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

class BaseStrategy(object):
    def basic_allocate_VMs(self, resource_used):
        pass

    def sla_violation(self, actual_used=None, allocated=None):
        delta = actual_used - allocated
        violation_count = len(delta[delta > 0])
        return float(violation_count) / len(actual_used)