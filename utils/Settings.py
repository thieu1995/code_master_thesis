#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:37, 31/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%



SPF_3D_NETWORK = "3D"

SPF_PATH_SAVE_BASE = "history/results/"

SPF_LOAD_DATA_FROM = "dataset/ggcluster/"

SPF_DATA_FILENAME = ["gg_cpu", "gg_ram"] #["it_eu_5m", "it_uk_5m", "worldcup98_5m"]
SPF_DATA_COLS = [[1], [2]]
SPF_DATA_WINDOWS = [
	[
		[1, 2, 3],
		[1, 2, 3, 4, 5, 6],
	],
	[
		[1, 2, 3],
		[1, 2, 3, 4, 5, 6],
	],
	# [
	# 	[1, 2, 3],
	# 	[1, 2, 3, 4, 5, 6],
	# ],
	# [
	# 	[1, 2, 3],
	# 	[1, 2, 3, 4, 5, 6],
	# ],
	# [
	# 	[1, 2, 3],
	# 	[1, 2, 3, 4, 5, 6],
	# ],
]

## Default settings
SPF_HIDDEN_SIZES_HYBRID = [(5, False), ]             # (num_node, checker), default checker is True
SPF_DOMAIN_RANGE_HYBRID = (-1, 1)                   # For all hybrid models
SPF_ACTIVATIONS = [("elu", "elu")]
SPF_HIDDEN_SIZES_HYBRID_RNN = [([5, ], False), ]     # For hybrid LSTM
SPF_DROP_OUTS = [(0.2,)]
SPF_ELM_ACTIVATION = [("elu")]
SPF_FLNN_ACTIVATION = ["elu"]
SPF_EXPAND_FUNC = ["chebyshev", "legendre", "laguerre", "power", "trigonometric"]


SPF_HYBRID_SONIA_PARAS = [
	{
		"stimulation_level": 0.15, # [0.15, 0.25, 0.5],
		"positive_number": 0.25,
		"distance_level": 0.15,
		"max_cluster": 1000,
		"mutation_id": 0,
		"clustering_type": "immune_full",  # immune_full: cluster + mutation, else: cluster
	}
]
SPF_HYBRID_CNN_PARAS = [
	{
		"filters_size": 8,  # [0.15, 0.25, 0.5],
		"kernel_size": 2,
		"pool_size": 2,
	}
]

###### Setting for paper running on server ==============================
epochs = [1000]
hidden_sizes_traditional = [(20, False), ]  # (num_node, checker), default checker is True
learning_rates = [0.15]
optimizers = ['SGD']  ## SGD, Adam, Adagrad, Adadelta, RMSprop, Adamax, Nadam
losses = ["mse"]
batch_sizes = [64]
dropouts = [(0.2,)]
pop_sizes = [50]


###================= Settings models for paper ============================####

#### : DTR
dtr_final = {
	"criterion": ["mse"],               # mse, mae, friedman_mse
	"splitter": ['best'],               # best, random
	"max_depth": [list(range(3, 10)) ],   # Using gridsearch to find the optimal one
}

#### : KNN
knn_final = {
	"n_neighbors": [list(range(2, 10)), ],
	"weights": ['uniform'],      # uniform, distance
	"p": [2],                   # 1: manhattan_distance, 2: euclidean_distance
}

#### SVR
svr_final = {
	"kernel": ["rbf", "linear", "poly"],
	"C": [100],
	"gamma": [0.1]
}

#### NuSVR
nu_svr_final = {
	"nu": [0.5],                            # default
	"C": [1.0],                             # default
	"kernel": ["rbf", "linear", "poly"],
	"gamma": ["scale"]                      # default
}




#### : Immune-SONIA
sonia_final = {
	"clustering_type": ['immune_full'],  # immune_full: cluster + mutation, else: cluster
	"stimulation_level": [0.25], #[0.15, 0.25, 0.5],
	"positive_number": [0.25],
	"distance_level": [0.15],

	"max_cluster": [500],  # default
	"mutation_id": [0],  # default

	"epoch": epochs,
	"batch_size": batch_sizes,
	"learning_rate": learning_rates,
	"activations": SPF_ACTIVATIONS,
	"optimizer": optimizers,
	"loss": losses
}


#### ============== Hybrid MLP/RNN/LSTM/GRU/CNN ==============================######

#### : GA-MLP/RNN/LSTM/GRU/CNN
ga_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.8],  # 0.85 -> 0.97
	"pm": [0.2]  # 0.005 -> 0.10
}


#### : WOA-MLP/RNN/LSTM/GRU/CNN
woa_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : HHO-MLP/RNN/LSTM/GRU/CNN
hho_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : TLO-MLP/RNN/LSTM/GRU/CNN
tlo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
}

#### : WDO-MLP/RNN/LSTM/GRU/CNN
efo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"r_rate": [0.3],
	"ps_rate": [0.85],
	"p_field": [0.1],
	"n_field": [0.45],
}

#### : SCA-MLP/RNN/LSTM/GRU/CNN
sca_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : GCO-MLP/RNN/LSTM/GRU/CNN
gco_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"cr": [0.7],
	"f": [1.25],
}

#### : AEO-MLP/RNN/LSTM/GRU/CNN
aeo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}


########################################## Not using ==========================


#### : LCBO-MLP/RNN/LSTM/GRU/CNN
lcbo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"r1": [2.35]
}

#### : WDO-MLP/RNN/LSTM/GRU/CNN
wdo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"RT": [3],
	"g": [0.2],
	"alp": [0.4],
	"c": [0.4],
	"max_v": [0.3]
}

#### : WDO-MLP/RNN/LSTM/GRU/CNN
aso_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"alpha": [50],
	"beta": [0.2],
}

#### : NRO-MLP/RNN/LSTM/GRU/CNN
nro_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : FPA-MLP/RNN/LSTM/GRU/CNN
fpa_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"p": [0.8]
}

#### : PSO-MLP/RNN/LSTM/GRU/CNN
pso_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"c1": [2.0],
	"c2": [2.0],
	"w_min": [0.4],
	"w_max": [0.9]
}

#### : HGSO-MLP/RNN/LSTM/GRU/CNN
hgso_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"n_clusters": [2, ]
}

#### : MVO-MLP/RNN/LSTM/GRU/CNN
mvo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ]
}

#### : EO-MLP/RNN/LSTM/GRU/CNN
eo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : DE-MLP/RNN/LSTM/GRU/CNN
de_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"wf": [0.8],
	"cr": [0.9]
}
