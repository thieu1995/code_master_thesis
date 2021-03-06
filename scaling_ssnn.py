#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:44, 05/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.scaling.proactive_scaling import SLABasedOnVms as BrokerScaling
from utils.IOUtil import load_csv, save_scaling_results_to_csv, load_number_of_vms
from config import Config, ModelConfig
from numpy import array, sum, round, sqrt


def calculate_adi_sla(cpu_path, ram_path, filename, real_VMs_used):
    cpu_file = f"{cpu_path}/{filename}"
    ram_file = f"{ram_path}/{filename}"
    violated_arrays = []
    adi_arrays = []
    for s_coff in s_coffs:

        violated_arr = []
        adi_arr = []
        for L_adap in L_adaps:
            broker = BrokerScaling(scaling_factor=s_coff, adaptation_len=L_adap)
            neural_net = broker.sla_violate(list_resource_files=[cpu_file, ram_file], list_cols=[[2, 3], [2, 3]])
            adi = broker.adi_qos_calculation(bound=(0.6, 0.8), VMs_used=real_VMs_used, VMs_allocated=neural_net[1][-1])
            violated_arr.append(round(neural_net[0], 3))
            adi_arr.append(round(adi, 3))

        violated_arrays.append(violated_arr)
        adi_arrays.append(adi_arr)

    violated_path_file = f"{cpu_path}/{Config.FILE_SLA_VIOLATE}-{filename}"
    adi_path_file = f"{cpu_path}/{Config.FILE_QOS_VIOLATE}-{filename}"

    save_scaling_results_to_csv(violated_arrays, violated_path_file)
    save_scaling_results_to_csv(adi_arrays, adi_path_file)


s_coffs = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
L_adaps = [1, 2, 3, 6, 12, 24]

for model in ModelConfig.MHA_SSNN_MODELS:

    for idx_windows, windows in enumerate(Config.DATASET_WINDOWS[0]):
        real_VMs_used = load_number_of_vms(f"{Config.DATA_RESULTS}/slide{len(windows)}-{Config.FILE_VMS_REAL_USAGE}")
        for idx_train_rate, train_rate in enumerate(Config.TRAIN_SPLITS):
            for n_trial in range(Config.N_TRIALS):

                # Create a combination of hybrid SONIA parameters
                for sp in list(ParameterGrid(ModelConfig.HYBRID_SONIA_PARAS)):
                    sp["stimulation_level"] = round(sqrt(len(windows)) * sp["stimulation_level"], 2)
                    filename1 = f"{Config.METRICS_TRAINING}-{'_'.join(sp['activations'])}-{sp['stimulation_level']}-{sp['positive_number']}-{sp['distance_level']}-{sp['max_cluster']}-{sp['mutation_id']}"
                    # Create combination of algorithm parameters
                    for mha_paras in list(ParameterGrid(model["param_grid"])):
                        filename2 = '-'.join([str(mha_paras[k]) for k in model["param_grid"].keys()])
                        filename = f"{Config.FILENAME_PRED_TEST}-{filename1}-{filename2}"
                        cpu_path = f"{Config.DATA_RESULTS}/gg_cpu/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{model['name']}/{Config.FILENAME_MODEL}"
                        ram_path = f"{Config.DATA_RESULTS}/gg_ram/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{model['name']}/{Config.FILENAME_MODEL}"
                        calculate_adi_sla(cpu_path, ram_path, filename, real_VMs_used)
