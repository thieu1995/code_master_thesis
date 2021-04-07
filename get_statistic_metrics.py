#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:56, 06/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pathlib import Path
from pandas import read_csv, DataFrame
from sklearn.model_selection import ParameterGrid
from numpy import savetxt, array, sqrt
from config import Config, ModelConfig
from utils.IOUtil import save_to_csv_dict


def save_fast_to_csv(list_results, list_paths, columns):
    for idx, results in enumerate(list_results):
        df = DataFrame(results, columns=columns)
        df.to_csv(list_paths[idx], index=False)
    return True


def read_results_from_files(models):


    for idx_file, data_filename in enumerate(Config.DATASET_NAMES):
        matrix_results = []

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
                                filepath = f"{path_model}/metrics-{filename1}-{filename2}.csv"
                                df = read_csv(filepath, usecols=Config.FILE_METRIC_CSV_HEADER)
                                values = df.values.tolist()[0]
                                results = [''.join([str(x) for x in windows]), train_rate, n_trial, model['name']] + values
                                matrix_results.append(array(results))

        matrix_results = array(matrix_results)
        matrix_dict = {}
        for idx, key in enumerate(Config.FILE_METRIC_CSV_HEADER_FULL):
            matrix_dict[key] = matrix_results[:, idx]
        ## Save final file to csv
        save_to_csv_dict(matrix_dict, 'statistics_final', f"{Config.DATA_RESULTS}/{data_filename}")
        # savetxt(f"{Config.DATA_RESULTS}/statistics_final.csv", matrix_results, delimiter=",")
        df = read_csv(f"{Config.DATA_RESULTS}/{data_filename}/statistics_final.csv", usecols=Config.FILE_METRIC_CSV_HEADER_FULL)
        print(df)


df_results = read_results_from_files(ModelConfig.MHA_SSNN_MODELS)
# print(df_results.info())

#
# ## Read the final csv file and calculate min,max,mean,std,cv. for each: test_size | m_rule | obj | model | paras | trial 1 -> n
#
# for test_size in Config.TEST_SIZE:
#     for m_rule in Config.M_RULES:
#         for obj in Config.OBJ_FUNCS:
#             pathsave = f"{Config.DATA_RESULTS}/{test_size}-{m_rule}/{obj}/{Config.FILENAME_STATISTICS}"
#             Path(pathsave).mkdir(parents=True, exist_ok=True)
#             min_results, mean_results, max_results, std_results, cv_results = [], [], [], [], []
#             for model in MhaConfig.models:
#                 parameters_grid = list(ParameterGrid(model["param_grid"]))
#                 keys = model["param_grid"].keys()
#                 for mha_paras in parameters_grid:
#                     model_name = "".join([f"-{mha_paras[key]}" for key in keys])
#                     model_name = model_name[1:]
#                     df_result = df_results[(df_results["test_size"] == test_size) & (df_results["m_rule"] == m_rule) &
#                                            (df_results["obj"] == obj) & (df_results["model"] == model["name"]) &
#                                            (df_results["model_name"] == model_name)][Config.FILE_METRIC_CSV_HEADER_CALCULATE]
#
#                     t1 = df_result.min(axis=0).to_numpy()
#                     t2 = df_result.mean(axis=0).to_numpy()
#                     t3 = df_result.max(axis=0).to_numpy()
#                     t4 = df_result.std(axis=0).to_numpy()
#                     t5 = t4 / t2
#
#                     t1 = [test_size, m_rule, obj, model["name"], model_name] + t1.tolist()
#                     t2 = [test_size, m_rule, obj, model["name"], model_name] + t2.tolist()
#                     t3 = [test_size, m_rule, obj, model["name"], model_name] + t3.tolist()
#                     t4 = [test_size, m_rule, obj, model["name"], model_name] + t4.tolist()
#                     t5 = [test_size, m_rule, obj, model["name"], model_name] + t5.tolist()
#
#                     min_results.append(t1)
#                     mean_results.append(t2)
#                     max_results.append(t3)
#                     std_results.append(t4)
#                     cv_results.append(t5)
#             save_fast_to_csv([min_results, mean_results, max_results, std_results, cv_results],
#                              [f"{pathsave}/{Config.FILE_MIN}", f"{pathsave}/{Config.FILE_MEAN}",
#                               f"{pathsave}/{Config.FILE_MAX}", f"{pathsave}/{Config.FILE_STD}", f"{pathsave}/{Config.FILE_CV}"],
#                              columns=Config.FILE_METRIC_HEADER_STATISTICS)
#
#
#
#
# from sklearn.model_selection import ParameterGrid
# from math import sqrt
# from config import Config, ModelConfig
# from models.main import hybrid_sonia
# from utils.IOUtil import load_csv
# import multiprocessing
# from time import time
#
