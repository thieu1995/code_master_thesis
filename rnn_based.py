#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:30, 10/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.main import traditional_rnn
from utils.IOUtil import load_csv
from config import Config, ModelConfig
import multiprocessing
from time import time
import os
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['GOTO_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)  # matrix multiplication and reductions
tf.config.threading.set_inter_op_parallelism_threads(2)  # number of threads used by independent non-blocking operations


def setting_and_running(my_model):
    print(f"Start running: {my_model['name']}")
    for idx_file, data_filename in enumerate(Config.DATASET_NAMES):
        data_windows = Config.DATASET_WINDOWS[idx_file]
        data_columns = Config.DATASET_COLUMNS[idx_file]
        data_original = load_csv(f"{Config.DATA_INPUT}{data_filename}", cols=data_columns)
        for idx_windows, windows in enumerate(data_windows):
            for idx_train_rate, train_rate in enumerate(Config.TRAIN_SPLITS):
                for n_trial in range(Config.N_TRIALS):
                    # Create combination of params.
                    for item in list(ParameterGrid(my_model["param_grid"])):
                        root_base_paras = {
                            "data_original": data_original,
                            "train_split": train_rate,  # should use the same in all test
                            "data_windows": windows,  # same
                            "scaling": Config.SCALING_METHOD,  # minmax or std
                            "feature_size": len(data_columns),  # same, usually : 1
                            "network_type": Config.NETWORK_3D,  # RNN-based: 3D, others: 2D
                            "visualize": Config.VISUALIZE,
                            "verbose": Config.VERBOSE,
                            "model_name": my_model['name'],
                            "pathsave": f"{Config.DATA_RESULTS}/{data_filename}/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{my_model['name']}"
                        }
                        md = getattr(traditional_rnn, my_model["class"])(root_base_paras, item)
                        md.processing()

models = [
    {"name": "rnn", "class": "Rnn", "param_grid": getattr(ModelConfig, "rnn")},
    {"name": "lstm", "class": "Lstm", "param_grid": getattr(ModelConfig, "lstm")},
    {"name": "gru", "class": "Gru","param_grid": getattr(ModelConfig, "gru")}
]

if __name__ == '__main__':
    starttime = time()
    processes = []
    for idx_md, my_md in enumerate(models):
        p = multiprocessing.Process(target=setting_and_running, args=(my_md,))
        processes.append(p)
        p.start()
        # Pin created processes in a round-robin                                # 0%8 = 0 --> core_id: 0, pid: rnn
        os.system("taskset -p -c %d %d" % ((idx_md % os.cpu_count()), p.pid))   # 1 % 8 = 1 --> core_id: 1, pid: lstm

    for process in processes:
        process.join()
    print('That took: {} seconds'.format(time() - starttime))
