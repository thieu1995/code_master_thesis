#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:19, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.reactivate_scaling import OnDemandScaling as BrokerScaling
from utils.IOUtil import load_csv, save_number_of_vms
from numpy import array, concatenate
from config import Config


def get_real_vms_usages(cpu, ram, broker):
    cpu_loaded = load_csv(cpu, cols=[2, 3])
    ram_loaded = load_csv(ram, cols=[2, 3])

    resources_actual = concatenate((cpu_loaded[:, 0:1], ram_loaded[:, 0:1]), axis=1)
    number_of_VMs = array(broker.allocate_VMs(resources_usage=resources_actual))
    save_number_of_vms(number_of_VMs, "vms_real_used_CPU_RAM.csv")


broker = BrokerScaling()

cpu_file = f"{Config.DATA_RESULTS}/gg_cpu/0.8/1/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.15-0.01-0.5-1000-0-1000-50"
ram_file = f"{Config.DATA_RESULTS}/gg_ram/0.8/1/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.15-0.01-0.5-1000-0-1000-50"
get_real_vms_usages(cpu_file, ram_file, broker)
