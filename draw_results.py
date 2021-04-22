#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:21, 13/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pandas import read_csv
from sklearn.model_selection import ParameterGrid
from numpy import sqrt
from config import Config, ModelConfig
from utils.GraphUtil import draw_true_predict

DATASET_NAMES = ["CPU", "RAM"]

def draw_loss_and_metrics(models):
    for idx_file, data_filename in enumerate(Config.DATASET_NAMES):

        data_windows = Config.DATASET_WINDOWS[idx_file]
        data_columns = Config.DATASET_COLUMNS[idx_file]
        for idx_windows, windows in enumerate(data_windows):
            for idx_train_rate, train_rate in enumerate(Config.TRAIN_SPLITS):
                for n_trial in range(Config.N_TRIALS):
                    for model in models:
                        # Create a combination of hybrid SONIA parameters
                        for sp in list(ParameterGrid(ModelConfig.HYBRID_SONIA_PARAS)):
                            sp["stimulation_level"] = round(sqrt(len(windows) * len(data_columns)) * sp["stimulation_level"], 2)
                            # Create combination of algorithm parameters
                            path_general = f"{Config.DATA_RESULTS}/{data_filename}/{train_rate}/{''.join([str(x) for x in windows])}/trial-{n_trial}/{model['name']}"
                            path_model = f"{path_general}/{Config.RESULTS_FOLDER_MODEL}"
                            filename1 = f"{Config.METRICS_TRAINING}-{'_'.join(sp['activations'])}-{sp['stimulation_level']}-{sp['positive_number']}-{sp['distance_level']}-{sp['max_cluster']}-{sp['mutation_id']}"
                            # Create combination of algorithm parameters
                            for mha_paras in list(ParameterGrid(model["param_grid"])):
                                filename2 = '-'.join([str(mha_paras[k]) for k in model["param_grid"].keys()])

                                # Load metrics
                                filepath = f"{path_model}/pred_test-{filename1}-{filename2}.csv"
                                df = read_csv(filepath, usecols=["y_test_true_unscaled", "y_test_pred_unscaled"])
                                values = df.values
                                lines = [values[:750, 0], values[:750, 1]]
                                draw_true_predict(lines, model["name"], ["Observed", 'Predicted'],
                                                  coordinate_titles=["Timestamp (5 minutes)", DATASET_NAMES[idx_file]],
                                                  filename=f"750_true_predict-{filename1}-{filename2}",
                                                  pathsave=f"{path_general}/{Config.RESULTS_FOLDER_VISUALIZE}/",
                                                  exts=[".png", ".pdf"])

draw_loss_and_metrics(ModelConfig.MHA_SSNN_MODELS)

