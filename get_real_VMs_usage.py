#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:19, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.scaling.reactivate_scaling import DemandScalingBasedOnResources as BrokerScaling
from utils.IOUtil import load_csv, save_number_of_vms
from numpy import array, concatenate
from config import Config


slide_dict = {
    "slide1": [
        f"{Config.DATA_RESULTS}/gg_cpu/0.8/1/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.15-0.01-0.5-1000-0-1000-50",
        f"{Config.DATA_RESULTS}/gg_ram/0.8/1/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.15-0.01-0.5-1000-0-1000-50"
    ],
    "slide2": [
        f"{Config.DATA_RESULTS}/gg_cpu/0.8/12/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.21-0.01-0.5-1000-0-1000-50",
        f"{Config.DATA_RESULTS}/gg_ram/0.8/12/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.21-0.01-0.5-1000-0-1000-50"
    ],
    "slide3": [
        f"{Config.DATA_RESULTS}/gg_cpu/0.8/123/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.26-0.01-0.5-1000-0-1000-50",
        f"{Config.DATA_RESULTS}/gg_ram/0.8/123/trial-0/AEO-SSNN/model/pred_test-mse-elu_elu-0.26-0.01-0.5-1000-0-1000-50"
    ]
}

broker = BrokerScaling()
for key, slide in slide_dict.items():

    cpu_loaded = load_csv(slide[0], cols=[2, 3])
    ram_loaded = load_csv(slide[1], cols=[2, 3])

    resources_actual = concatenate((cpu_loaded[:, 0:1], ram_loaded[:, 0:1]), axis=1)
    number_of_VMs = array(broker.calculate_VMs_allocated(resources_usage=resources_actual))
    save_number_of_vms(number_of_VMs, f"{Config.DATA_RESULTS}/{key}-{Config.FILE_VMS_REAL_USAGE}")
