#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:35, 10/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from math import sqrt
from config import Config, ModelConfig
from models.main import hybrid_sonia
from utils.IOUtil import load_csv
import multiprocessing
from time import time


def setting_and_running(my_model):
    print(f"Start training model: {my_model['name']}")
    for idx_file, data_filename in enumerate(Config.DATASET_NAMES):
        data_windows = Config.DATASET_WINDOWS[idx_file]
        data_columns = Config.DATASET_COLUMNS[idx_file]
        data_original = load_csv(f"{Config.DATA_INPUT}{data_filename}", cols=data_columns)
        for idx_windows, windows in enumerate(data_windows):
            for idx_train_rate, train_rate in enumerate(Config.TRAIN_SPLITS):
                for n_trial in range(Config.N_TRIALS):
                    # Create a combination of hybrid SONIA parameters
                    for sonia_paras in list(ParameterGrid(ModelConfig.HYBRID_SONIA_PARAS)):
                        # Create combination of algorithm parameters
                        for mha_paras in list(ParameterGrid(my_model["param_grid"])):
                            root_base_paras = {
                                "data_original": data_original,
                                "train_split": train_rate,  # should use the same in all test
                                "data_windows": windows,  # same
                                "scaling": Config.SCALING_METHOD,  # minmax or std
                                "feature_size": len(data_columns),  # same, usually : 1
                                "network_type": Config.NETWORK_2D,  # RNN-based: 3D, others: 2D
                                "visualize": Config.VISUALIZE,
                                "verbose": Config.VERBOSE,
                                "model_name": my_model['name'],
                                "pathsave": f"{Config.DATA_RESULTS}/{data_filename}/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{my_model['name']}"
                            }
                            root_hybrid_ssnn_paras = {
                                "lb": ModelConfig.LB,
                                "ub": ModelConfig.UB,
                                "activations": sonia_paras["activations"]
                            }
                            sonia_paras["stimulation_level"] = round(sqrt(len(windows) * len(data_columns)) * sonia_paras["stimulation_level"], 2)
                            # sonia_paras["distance_level"] = sqrt(len(windows) * len(data_columns)) * sonia_paras["stimulation_level"]
                            md = getattr(hybrid_sonia, my_model["class"])(root_base_paras, root_hybrid_ssnn_paras, sonia_paras, mha_paras)
                            md.processing()


models = [
    {"name": "GA-SSNN", "class": "GaSonia", "param_grid": getattr(ModelConfig, "ga_paras")},
    {"name": "DE-SSNN", "class": "DeSonia", "param_grid": getattr(ModelConfig, "de_paras")},
    {"name": "CRO-SSNN", "class": "CroSonia", "param_grid": getattr(ModelConfig, "cro_paras")},

    {"name": "PSO-SSNN", "class": "PsoSonia", "param_grid": getattr(ModelConfig, "pso_paras")},
    {"name": "WOA-SSNN", "class": "WoaSonia", "param_grid": getattr(ModelConfig, "woa_paras")},
    {"name": "HHO-SSNN", "class": "HhoSonia", "param_grid": getattr(ModelConfig, "hho_paras")},

    {"name": "MVO-SSNN", "class": "MvoSonia", "param_grid": getattr(ModelConfig, "mvo_paras")},
    {"name": "TWO-SSNN", "class": "TwoSonia", "param_grid": getattr(ModelConfig, "two_paras")},
    {"name": "EO-SSNN", "class": "EoSonia", "param_grid": getattr(ModelConfig, "eo_paras")},

    {"name": "TLO-SSNN", "class": "TloSonia", "param_grid": getattr(ModelConfig, "tlo_paras")},
    {"name": "QSA-SSNN", "class": "QsaSonia", "param_grid": getattr(ModelConfig, "qsa_paras")},

    {"name": "IWO-SSNN", "class": "IwoSonia", "param_grid": getattr(ModelConfig, "iwo_paras")},
    {"name": "SMA-SSNN", "class": "SmaSonia", "param_grid": getattr(ModelConfig, "sma_paras")},

    {"name": "AEO-SSNN", "class": "AeoSonia", "param_grid": getattr(ModelConfig, "aeo_paras")},
    {"name": "SCA-SSNN", "class": "ScaSonia", "param_grid": getattr(ModelConfig, "sca_paras")},
    {"name": "HS-SSNN", "class": "HsSonia", "param_grid": getattr(ModelConfig, "hs_paras")},

]

if __name__ == '__main__':
    starttime = time()
    processes = []
    for my_md in models:
        p = multiprocessing.Process(target=setting_and_running, args=(my_md,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))
