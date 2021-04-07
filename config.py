#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:38, 16/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from os.path import abspath, dirname, splitext, basename, realpath

basedir = abspath(dirname(__file__))


def get_model_name(pathfile):
    # name of the models ==> such as: rnn1hl.csv --> name: rnn1hl
    return str(splitext(basename(realpath(pathfile)))[0])


class Config:
    DATA_DIRECTORY = f'{basedir}/dataset'
    DATA_INPUT = f'{DATA_DIRECTORY}/input_data/'
    DATA_RESULTS = f'{DATA_DIRECTORY}/results'
    RESULTS_FOLDER_VISUALIZE = "visual"
    RESULTS_FOLDER_MODEL = "model"

    DATASET_NAMES = ["gg_cpu", "gg_ram"]  # ["it_eu_5m", "it_uk_5m", "worldcup98_5m"]
    DATASET_COLUMNS = [[1], [2]]
    DATASET_WINDOWS = [
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ],
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ],
    ]


    FILENAME_LOSS_TRAIN = "loss_train"
    FILENAME_PRED_TRAIN = "pred_train"
    FILENAME_PRED_TEST = "pred_test"
    FILENAME_METRICS = "resources"
    FILENAME_STATISTICS = "statistics"
    FILENAME_MODEL = "model"
    FILE_MIN = "min.csv"
    FILE_MEAN = "mean.csv"
    FILE_MAX = "max.csv"
    FILE_STD = "std.csv"
    FILE_CV = "cv.csv"
    FILE_VMS_REAL_USAGE = "VMS_real.csv"
    FILE_SLA_VIOLATE = "sla_violate"
    FILE_QOS_VIOLATE = "qos_adi"

    VISUALIZE_TYPES = [".png", ".pdf"]

    METRICS_TRAINING = "mse"
    METRICS_TESTING = ["MAE", "RMSE", "MAPE", "EVS", "R2", "NSE", "R", "WI", "CI"]

    Y_TRAIN_TRUE_SCALED = "y_train_true_scaled"
    Y_TRAIN_TRUE_UNSCALED = "y_train_true_unscaled"
    Y_TRAIN_PRED_SCALED = "y_train_pred_scaled"
    Y_TRAIN_PRED_UNSCALED = "y_train_pred_unscaled"
    Y_TEST_TRUE_SCALED = "y_test_true_scaled"
    Y_TEST_TRUE_UNSCALED = "y_test_true_unscaled"
    Y_TEST_PRED_SCALED = "y_test_pred_scaled"
    Y_TEST_PRED_UNSCALED  = "y_test_pred_unscaled"

    HEADER_TRAIN_CSV = ["y_train_true_scaled", "y_train_pred_scaled", "y_train_true_unscaled", "y_train_pred_unscaled"]
    HEADER_TEST_CSV = ["y_test_true_scaled", "y_test_pred_scaled", "y_test_true_unscaled", "y_test_pred_unscaled"]

    FILE_METRIC_CSV_HEADER = ["model_name", "time_train", "time_predict", "time_total", "n_clusters", "silhouette", "calinski",
              "davies", "MAE_train", "RMSE_train", "MAPE_train", "EVS_train",
              "R2_train", "NSE_train", "R_train", "WI_train", "CI_train",
              "MAE_test", "RMSE_test", "MAPE_test", "EVS_test", "R2_test", "NSE_test", "R_test", "WI_test", "CI_test"]
    FILE_METRIC_CSV_HEADER_FULL = ["windows", "train_rate", "trial", "model",
        "model_name", "time_train", "time_predict", "time_total", "n_clusters", "silhouette", "calinski",
        "davies", "MAE_train", "RMSE_train", "MAPE_train", "EVS_train",
        "R2_train", "NSE_train", "R_train", "WI_train", "CI_train",
        "MAE_test", "RMSE_test", "MAPE_test", "EVS_test", "R2_test", "NSE_test", "R_test", "WI_test", "CI_test"
    ]

    N_TRIALS = 5
    TRAIN_SPLITS = [0.8]
    SCALING_METHOD = "minmax"  # std, minmax, loge, kurtosis, kurtosis_std, boxcox, boxcox_std
    VISUALIZE = True
    NETWORK_2D = "2D"
    NETWORK_3D = "3D"
    VERBOSE = 0  # 0: nothing, 1 : full detail, 2: short version
    SAVE_MODEL = False

    MODEL_KERAS = ["mlp", "rnn", "cnn", "gru", "lstm"]

    LIST_COLOURS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                   "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                   "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                   "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                   "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                   "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                   "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                   "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
                   "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                   "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                   "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                   "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                   "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]


class ModelConfig:
    EPOCHS = [1000]
    BATCH_SIZES = [64]
    POP_SIZES = [50]
    OPTIMIZERS = ["SGD", "Adam"]  ### SGD, Adam, Adagrad, Adadelta, RMSprop, Adamax, Nadam

    LB = [-1]
    UB = [1]

    #### : ELM
    elm = {
        "size_hidden": [100, None],
        "activation": ['elu']
    }
    ####: FLNN
    flnn = {
        "expand": ["chebyshev"],  # chebyshev, legendre, laguerre, powerseries, trigonometric
        "activation": ["elu"],
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "learning_rate": [0.025],
        "beta": [0.90]
    }
    ####: MLP
    mlp = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "optimizer": OPTIMIZERS,
        "learning_rate": [0.01],
        "valid_split": [0.3],
        "list_layers": [
            [
                {"n_nodes": 15, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
            [
                {"n_nodes": 10, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 5, "activation": "relu", "dropout": 0.2},  # Second hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
        ]
    }
    ####: RNN
    rnn = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "optimizer": OPTIMIZERS,
        "learning_rate": [0.01],
        "valid_split": [0.3],
        "list_layers": [
            [
                {"n_nodes": 15, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
            [
                {"n_nodes": 10, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 5, "activation": "relu", "dropout": 0.2},  # Second hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
        ]
    }
    ####: LSTM
    lstm = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "optimizer": OPTIMIZERS,
        "learning_rate": [0.01],
        "valid_split": [0.3],
        "list_layers": [
            [
                {"n_nodes": 15, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
            [
                {"n_nodes": 10, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 5, "activation": "relu", "dropout": 0.2},  # Second hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
        ]
    }
    ####: GRU
    gru = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "optimizer": OPTIMIZERS,
        "learning_rate": [0.01],
        "valid_split": [0.3],
        "list_layers": [
            [
                {"n_nodes": 15, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
            [
                {"n_nodes": 10, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 5, "activation": "relu", "dropout": 0.2},  # Second hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
        ]
    }
    ####: CNN
    cnn = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "optimizer": OPTIMIZERS,
        "learning_rate": [0.01],
        "valid_split": [0.3],
        "list_layers": [
            [
                {"n_nodes": 15, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
            [
                {"n_nodes": 10, "activation": "relu", "dropout": 0.2},  # First hidden layer
                {"n_nodes": 5, "activation": "relu", "dropout": 0.2},  # Second hidden layer
                {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
            ],
        ],
        "filter_size": [64, ],
        "kernel_size": [2, ],
        "pool_size": [2, ],
        "activation": ["relu"]
    }
    ####: SONIA
    sonia = {
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZES,
        "activations": [("elu", "elu"), ],
        "optimizer": OPTIMIZERS,

        # "stimulation_level": [0.1, 0.15, 0.2, 0.25, 0.3],
        # "positive_number": [0.01, 0.02, 0.05],
        # "distance_level": [0.2, 0.3, 0.4, 0.5],
        # "max_cluster": [1000, ],
        # "mutation_id": [0, ],
        "stimulation_level": [0.15, 0.25],
        "positive_number": [0.01],
        "distance_level": [0.5],
        "max_cluster": [1000, ],
        "mutation_id": [0, ],
    }

    #### Hybrid SONIA

    HYBRID_SONIA_PARAS = {
        "activations": [("elu", "elu"), ],

        # "stimulation_level": [0.1, 0.15, 0.2, 0.25, 0.3],
        # "positive_number": [0.01, 0.02, 0.05 ],
        # "distance_level": [0.2, 0.3, 0.4, 0.5],
        # "max_cluster": [1000, ],
        # "mutation_id": [0, ],
        "stimulation_level": [0.15, 0.25],
        "positive_number": [0.01],
        "distance_level": [0.5],
        "max_cluster": [1000, ],
        "mutation_id": [0, ],
    }

    ## Evolutionary-based group
    ga_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "pc": [0.85],
        "pm": [0.025]
    }
    de_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "wf": [0.85],
        "cr": [0.8],
    }
    cro_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "po": [0.4],
        "Fb": [0.9],
        "Fa": [0.1],
        "Fd": [0.1],
        "Pd": [0.1],
        "G": [(0.02, 0.2)],
        "GCR": [0.1],
        # "k": [3],

    }
    ## Swarm-based group
    pso_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "c1": [1.2],
        "c2": [1.2],
        "w_min": [0.4],
        "w_max": [0.9],
    }
    woa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    hho_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }

    abc_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "couple_bees": [(16, 4)]
    }
    gwo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    ssa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "ST": [0.8],
        "PD": [0.2],
        "SD": [0.1],
    }
    mfo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    alo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    goa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "c_minmax": [(0.00004, 1)]
    }
    bbo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "p_m": [0.01],
        "elites": [2]
    }
    salpso_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    do_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    fa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "m": [50],  # Same as pop_size
        "a": [0.04],
        "b": [0.8],
        "A_": [40],
        "m_": [5],
    }
    beesa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "site_ratio": [(0.5, 0.4), ],  # (selected_site_ratio, elite_site_ratio)
        "site_bee_ratio": [(0.1, 2), ],  # (selected_site_bee_ratio, elite_site_bee_ratio)
        "recruited_bee_ratio": [0.1],
        "dance_radius": [0.1, ],
        "dance_radius_damp": [0.99],
    }
    acor_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "sample_count": [50],
        "q": [0.5],
        "zeta": [1.0],
    }
    nmra_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "bp": [0.75],
    }

    ## Physics-based group
    mvo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "wep_minmax": [(0.2, 1.0)]
    }
    two_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    eo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }

    sa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "max_sub_iter": [10],
        "t0": [1000],
        "t1": [1],
        "move_count": [5],
        "mutation_rate": [0.1],
        "mutation_step_size": [0.1],
        "mutation_step_size_damp": [0.99]
    }
    hgso_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "n_clusters": [2]
    }

    ## Human-based group
    tlo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }
    qsa_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }

    lcbo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "r1": [2.35]
    }
    ica_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "empire_count": [5],
        "selection_pressure": [1],
        "assimilation_coeff": [1.5],
        "revolution_prob": [0.05],
        "revolution_rate": [0.1, ],
        "revolution_step_size": [0.1],
        "revolution_step_size_damp": [0.99],
        "zeta": [0.1]
    }
    ca_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "accepted_rate": [0.2],
    }

    ## Bio-based group
    iwo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "seeds": [(2, 10)],
        "exponent": [2],
        "sigma": [(0.5, 0.001)]
    }
    sma_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "z": [0.03],
    }

    ## System-based group
    aeo_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }

    wca_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "nsr": [4],
        "C": [2],
        "dmax": [1e-6]
    }

    ## Math-based group
    sca_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
    }

    hc_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "neighbour_size": [50]
    }

    ## Music-based group
    hs_paras = {
        "epoch": EPOCHS, "pop_size": POP_SIZES,
        "n_new": [20],
        "c_r": [0.95],
        "pa_r": [0.05]
    }

    MHA_SSNN_MODELS = [
        {"name": "GA-SSNN", "class": "GaSonia", "param_grid": ga_paras},
        {"name": "OCRO-SSNN", "class": "CroSonia", "param_grid": cro_paras},

        {"name": "PSO-SSNN", "class": "PsoSonia", "param_grid": pso_paras},
        {"name": "WOA-SSNN", "class": "WoaSonia", "param_grid": woa_paras},

        {"name": "OTWO-SSNN", "class": "TwoSonia", "param_grid": two_paras},
        {"name": "EO-SSNN", "class": "EoSonia", "param_grid": eo_paras},

        {"name": "TLO-SSNN", "class": "TloSonia", "param_grid": tlo_paras},

        {"name": "SMA-SSNN", "class": "SmaSonia", "param_grid": sma_paras},

        {"name": "SCA-SSNN", "class": "ScaSonia", "param_grid": sca_paras},
        {"name": "HS-SSNN", "class": "HsSonia", "param_grid": hs_paras},

        {"name": "AEO-SSNN", "class": "AeoSonia", "param_grid": aeo_paras},
        {"name": "IAEO-SSNN", "class": "ImprovedAeoSonia", "param_grid": aeo_paras},
    ]






