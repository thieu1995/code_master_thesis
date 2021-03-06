#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:44, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, reshape, savetxt
from csv import DictWriter
from pathlib import Path
from pandas import read_csv, DataFrame


def save_results_to_csv(data: dict, filename=None, pathsave=None):
	## Check the parent directories
	Path(pathsave).mkdir(parents=True, exist_ok=True)
	with open(f"{pathsave}/{filename}.csv", 'a') as file:
		w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=data.keys())
		if file.tell() == 0:
			w.writeheader()
		w.writerow(data)
	return None


def save_to_csv_dict(data: dict, filename=None, pathsave=None):
	## Check the parent directories
	Path(pathsave).mkdir(parents=True, exist_ok=True)
	## Reshape data
	data_shaped = {}
	for key, value in data.items():
		data_shaped[key] = array(value).reshape(-1)
	df = DataFrame(data_shaped, columns=data_shaped.keys())
	df.to_csv(f"{pathsave}/{filename}.csv", index=False, header=True)
	return None


def save_to_csv(data: list, header: list, filename=None, pathsave=None):
	## Check the parent directories
	Path(pathsave).mkdir(parents=True, exist_ok=True)
	# Convert data and header to dictionary
	mydict = {}
	for idx, h in enumerate(header):
		mydict[h] = array(data[idx]).reshape(-1)
	df = DataFrame(mydict, columns=header)
	df.to_csv(f"{pathsave}/{filename}.csv", index=False, header=True)
	return None


def load_csv(path_to_data=None, cols=None):
	"""
	:param path_to_data:
	:type path_to_data:
	:param features_selected:  example -> ["bytes"], ["degree", "wind", ...]
	:type features_selected:
	:param features_index:  example -> ["date"], ["time"], ....
	:type features_index:
	:return:
	:rtype:
	"""
	df = read_csv(f"{path_to_data}.csv", usecols=cols)
	return df.values


def save_number_of_vms(data=None, pathfile=None):
	t0 = reshape(data, (-1, 1))
	savetxt(pathfile, t0, delimiter=",")


def save_scaling_results_to_csv(data=None, path_file=None):
	savetxt(path_file + ".csv", array(data), delimiter=",")
	return None


def load_number_of_vms(pathfile=None, delimiter=",", header=None):
	df = read_csv(pathfile, sep=delimiter, header=header)
	return df.values[:, 0:1]