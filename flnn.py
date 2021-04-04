#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:02, 11/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.main.traditional_flnn import Flnn
from utils.IOUtil import load_csv
from time import time
from config import Config, ModelConfig, get_model_name

# name of the models ==> such as: rnn1hl.csv
model_name = get_model_name(__file__)

def train_model(item):
    root_base_paras = {
        "data_original": data_original,
        "train_split": train_rate,  # should use the same in all test
        "data_windows": windows,  # same
        "scaling": Config.SCALING_METHOD,  # minmax or std
        "feature_size": len(data_columns),  # same, usually : 1
        "network_type": Config.NETWORK_2D,  # RNN-based: 3D, others: 2D
        "visualize": Config.VISUALIZE,
        "verbose": Config.VERBOSE,
        "model_name": model_name,
        "pathsave": f"{Config.DATA_RESULTS}/{data_filename}/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{model_name}"
    }
    md = Flnn(root_base_paras=root_base_paras, root_flnn_paras=item)
    md.processing()


start_time = time()
for idx_file, data_filename in enumerate(Config.DATASET_NAMES):
    data_windows = Config.DATASET_WINDOWS[idx_file]
    data_columns = Config.DATASET_COLUMNS[idx_file]
    data_original = load_csv(f"{Config.DATA_INPUT}{data_filename}", cols=data_columns)
    for idx_windows, windows in enumerate(data_windows):
        for idx_train_rate, train_rate in enumerate(Config.TRAIN_SPLITS):
            for n_trial in range(Config.N_TRIALS):
                # Create combination of params.
                for item in list(ParameterGrid(ModelConfig.flnn)):
                    train_model(item)
end_time = time() - start_time
print("Taken: {} seconds".format(end_time))


