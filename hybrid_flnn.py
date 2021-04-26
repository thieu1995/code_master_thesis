#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:52, 26/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from config import Config, ModelConfig
from models.main import hybrid_flnn
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
                    for flnn_paras in list(ParameterGrid(ModelConfig.HYBRID_FLNN_PARAS)):
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
                            md = getattr(hybrid_flnn, my_model["class"])(root_base_paras, flnn_paras, mha_paras)
                            md.processing()


if __name__ == '__main__':
    starttime = time()
    processes = []
    for my_md in ModelConfig.MHA_FLNN_MODELS:
        p = multiprocessing.Process(target=setting_and_running, args=(my_md,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))
