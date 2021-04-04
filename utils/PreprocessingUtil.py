#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:46, 09/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import reshape, array, log, exp, sign, abs, power, floor
from numpy.random import seed, permutation
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pandas import Series
from matplotlib import pyplot as plt


class CheckDataset:
    """
        Checking whether data is stationary or non-stationary (trend, seasonality, ...)
        https://machinelearningmastery.com/time-series-data-stationary-python/
        https://machinelearningmastery.com/difference-time-series-dataset-python/
    """

    def check_by_plot_raw_data(self, pathfile=None):
        self.series = Series.from_csv(pathfile, header=0)
        self.series.plot()
        plt.show()

    def check_by_summary_statistic(self, pathfile=None, draw_history=True):
        """
        You can split your time series into two (or more) partitions and compare the mean and variance of each group.
        If they differ and the difference is statistically significant, the time series is likely non-stationary.

        Because we are looking at the mean and variance, we are assuming that the data conforms to a Gaussian
        (also called the bell curve or normal) distribution. ==> Stationary
        """
        self.series = Series.from_csv(pathfile, header=0)
        X = self.series.values
        split = int(len(X) / 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()

        self.series.hist()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))

        if draw_history:
            self.series.hist()
            plt.show()

    def _checking_consecutive__(self, df, time_name="timestamp", time_different=300):
        """
        :param df: Type of this must be dataframe
        :param time_name: the column name of date time
        :param time_different by seconds: 300 = 5 minutes
            https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html
        :return:
        """
        consecutive = True
        for i in range(df.shape[0] - 1):
            diff = (df[time_name].iloc[i + 1] - df[time_name].iloc[i]).seconds
            if time_different != diff:
                print("===========Not consecutive at: {}, different: {} ====================".format(i + 3, diff))
                consecutive = False
        return consecutive


class MiniBatch(object):
    def __init__(self, X_train, y_train, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size

    def random_mini_batches(self, seed_number=None):
        X, Y = self.X_train.T, self.y_train.T
        mini_batch_size = self.batch_size

        m = X.shape[1]  # number of training examples
        mini_batches = []
        seed(seed_number)

        # Step 1: Shuffle (X, Y)
        perm = list(permutation(m))
        shuffled_X = X[:, perm]
        shuffled_Y = Y[:, perm].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


class TimeSeries:
    def __init__(self, data=None, train_split=0.8):
        self.data_original = data
        if train_split < 1.0:
            self.train_split = int(train_split * self.data_original.shape[0])
        else:
            self.train_split = train_split
        self.multi_size = len(self.data_original[0])        # Remember if using multi-variate --> The last column is the predicted output.

    def _scaling__(self, scale_type="std", separate=True):
        """
        :param dataset: 2D numpy array
        :param scale_type: std / minmax
        :return:
        """
        self.data_new =[]
        self.data_mean=[]
        self.data_std = []
        self.data_min = []
        self.data_max = []

        self.data_kurtosis = []
        self.data_mean_kur = []
        self.data_std_kur = []
        self.lamda_boxcox = []
        self.data_boxcox = []

        for i in range(self.multi_size):
            col_data_original = self.data_original[:,i]

            if separate:
                col_data_mean, col_data_std = col_data_original[:self.train_split].mean(axis=0), col_data_original[:self.train_split].std(axis=0)
                col_data_min, col_data_max = col_data_original[:self.train_split].min(axis=0), col_data_original[:self.train_split].max(axis=0)
            else:
                col_data_mean, col_data_std = col_data_original.mean(axis=0), col_data_original.std(axis=0)
                col_data_min, col_data_max = col_data_original.min(axis=0), col_data_original.max(axis=0)

            if scale_type == "std":
                col_data_new = (col_data_original - col_data_mean) / col_data_std
            elif scale_type == "minmax":
                col_data_new = (col_data_original - col_data_min) / (col_data_max - col_data_min)
            elif scale_type == "loge":
                col_data_new = log(col_data_original)

            elif scale_type == "kurtosis":
                col_data_new = sign(col_data_original - col_data_mean) * power(abs(col_data_original - col_data_mean), 1.0 / 3)
            elif scale_type == "kurtosis_std":
                col_data_kurtosis = sign(col_data_original - col_data_mean) * power(abs(col_data_original - col_data_mean), 1.0 / 3)
                col_data_mean_kur, col_data_std_kur = col_data_original[:self.train_split].mean(axis=0), col_data_original[:self.train_split].std(axis=0)
                col_data_new = (col_data_kurtosis - col_data_mean_kur) / col_data_std_kur
                self.data_kurtosis.append(col_data_kurtosis)
                self.data_mean_kur.append(col_data_mean_kur)
                self.data_std_kur.append(col_data_std_kur)
            elif scale_type == "boxcox":
                col_data_new, col_lamda_boxcox = boxcox(col_data_original.flatten())
            elif scale_type == "boxcox_std":
                col_data_boxcox, col_lamda_boxcox = boxcox(col_data_original.flatten())
                col_data_boxcox = col_data_boxcox.reshape(-1, 1)
                col_data_mean,col_data_std = col_data_boxcox[:self.train_split].mean(axis=0), col_data_boxcox[:self.train_split].std(axis=0)
                col_data_new = (col_data_boxcox - col_data_mean) / col_data_std
                self.lamda_boxcox.append(col_lamda_boxcox)
                self.data_boxcox.append(col_data_boxcox)

            self.data_mean.append(col_data_mean)
            self.data_std.append(col_data_std)
            self.data_min.append(col_data_min)
            self.data_max.append(col_data_max)
            self.data_new.append(col_data_new)
        return array(self.data_new)

    def _inverse_scaling__(self, data=None, scale_type="std"):
        if scale_type == "std":
            return self.data_std[self.multi_size-1] * data + self.data_mean[self.multi_size-1]
        elif scale_type == "minmax":
            return data * (self.data_max[self.multi_size-1] - self.data_min[self.multi_size-1]) + self.data_min[self.multi_size-1]
        elif scale_type == "loge":
            return exp(data)

        elif scale_type == "kurtosis":
            return power(data, 3) + self.data_mean[self.multi_size-1]
        elif scale_type == "kurtosis_std":
            temp = self.data_std_kur[self.multi_size-1] * data + self.data_mean_kur[self.multi_size-1]
            return power(temp, 3) + self.data_mean[self.multi_size-1]

        elif scale_type == "boxcox":
            return inv_boxcox(data, self.lamda_boxcox[self.multi_size-1])
        elif scale_type == "boxcox_std":
            boxcox_invert = self.data_std[self.multi_size-1] * data + self.data_mean[self.multi_size-1]
            return inv_boxcox(boxcox_invert, self.lamda_boxcox[self.multi_size-1])

    def _make_train_test_data__(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param history_column: python list time in the past you want to use. (1, 2, 5) means (t-1, t-2, t-5) predict time t
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param pre_type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset[self.multi_size-1]) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append([])
            for j in range(self.multi_size):
                for vl in reshape(dataset[j][indices], (history_size, 1)):
                    data[i-start_index].append(vl)
            labels.append(dataset[self.multi_size-1][i + history_column[-1]])
        if pre_type == "3D":
            return array(data), array(labels)
        return reshape(array(data), (-1, history_size*self.multi_size)), reshape(array(labels), (-1, 1))


    def _make_train_test_with_expanded__(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D", expand_func=None):
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset[self.multi_size - 1]) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append([])
            for j in range(self.multi_size):
                for vl in reshape(dataset[j][indices], (history_size, 1)):
                    data[i - start_index].append(vl)
            labels.append(dataset[self.multi_size - 1][i + history_column[-1]])

        # Make the matrix X and Column Matrix y
        data = reshape(array(data), (-1, history_size * self.multi_size))
        labels = reshape(array(labels), (-1, 1))

        ## Expanded function applied here
        data = expand_func(data)

        if pre_type == "3D":
            return array(data), labels
        return data, labels
