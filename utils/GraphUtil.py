#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:43, 09/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import array, arange, amax
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas import DataFrame
from config import Config
from pathlib import Path
import platform

if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
if platform.system() == "Windows":
    import matplotlib
    matplotlib.use("TkAgg")


def draw_multiple_lines(list_lines: list, list_legends: list, list_colors: list, list_markers: list,
                        xy_labels: list, title: str, filename: str, pathsave: str, exts: list):
    plt.gcf().clear()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for idx, line in enumerate(list_lines):
        ax.plot(line[0], line[1], label=list_legends[idx])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])
    ax.set_title(title)
    ax.grid('on')

    for idx, ext in enumerate(exts):
        fig.savefig(pathsave + filename + ext, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if platform.system() != "Linux":
        fig.show()
    plt.close()


def visualize_cluster(X, labels):
    K = int(amax(labels) + 1)
    if X.shape[1] < 3:
        for i in range(K):
            X0 = X[labels == i, :]
            plt.scatter(X0[:, 0], X0[:, 1], Config.LIST_COLOURS[i], marker='o')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc='upper left')
            plt.title(f"Number of clusters = {K}")
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(K):
            X0 = X[labels == i, :]
            ax.plot(X0[:, 0], X0[:, 1], X0[:, 2], Config.LIST_COLOURS[i], marker='o', markersize=2, alpha=.8)
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
    plt.show()
    plt.close()


def draw_predict_line_with_error(data:list, error:list, filename:str, pathsave:str, exts: list):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # Import Data
    df = DataFrame({'y_true': data[0], 'y_pred': data[1]})
    list_data = [df.loc[:, "y_true"], df.loc[:, "y_pred"]]
    list_data[0].rename("Observed", inplace=True)
    list_data[1].rename("Predicted", inplace=True)

    # Draw Plot
    plt.rcParams['figure.figsize'] = 10, 6.5

    # sns.set(color_codes=True)
    my_fig = plt.figure(constrained_layout=True)
    gs = my_fig.add_gridspec(nrows=2, ncols=5)

    re_data = 100 * (data[0] - data[1]) / data[0]
    ax3 = my_fig.add_subplot(gs[0, :])
    sns.lineplot(data=re_data, ax=ax3)
    ax3.set(ylabel='RE(%)')

    ax1 = my_fig.add_subplot(gs[1, :3])
    sns.lineplot(data=list_data, ax=ax1)
    ax1.set(xlabel='Timestep (5 minutes)', ylabel=r'Value', title='Performance Prediction: RMSE=' + str(error[1]))
    # ax1.set(xlabel='Months', ylabel=r'Streamflow ($m^3/sec$)')

    ax2 = my_fig.add_subplot(gs[1:, 3:])
    sns.regplot(x="y_true", y="y_pred", data=df, ax=ax2)
    ax2.set(xlabel=r'Observed', ylabel=r'Predicted', title='Linear Relationship: MAE=' + str(error[0]))
    # ax2.set(xlabel=r'Observed ($m^3/s$)', ylabel=r'Predicted ($m^3/s$)')
    ax2.legend(['Fit'])

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    # if platform.system() != "Linux":
        # plt.show()
    plt.close()

    # # plt.tight_layout()
    # plt.savefig(pathsave + filename + ".png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    return None


def draw_predict(y_test=None, y_pred=None, filename=None, pathsave=None):
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('CPU')
    plt.xlabel('Timestamp')
    plt.legend(['Actual', 'Predict'], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None


def draw_true_predict(data: list, title:str, legends: list, coordinate_titles:list, filename: str, pathsave: str, exts: list):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    plt.plot(data[0], linestyle='-')
    plt.plot(data[1], linestyle='--')
    plt.ylabel(coordinate_titles[1])
    plt.xlabel(coordinate_titles[0])
    plt.title(title)
    plt.legend(legends, loc='upper right')
    plt.savefig(pathsave + filename + ".png")

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    # if platform.system() != "Linux":
    # plt.show()
    plt.close()

    # # plt.tight_layout()
    # plt.savefig(pathsave + filename + ".png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    return None


def __create_time_steps__(length):
    return list(range(-length, 0))

def _plot_history_true_future_prediciton__(plot_data, delta, title):
    """
    :param plot_data: 2D-numpy array
    :param delta:
    :param title:
    :return:
    """
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = __create_time_steps__(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    plt.show()
    return plt


def _plot_train_history__(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    return 0


def multi_step_plot(history, true_future, prediction, num_steps):
    plt.figure(figsize=(12, 6))
    num_in = __create_time_steps__(len(history))
    num_out = len(true_future)

    plt.plot(num_in, array(history[:, 1]), label='History')
    plt.plot(arange(num_out) / num_steps, array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(arange(num_out) / num_steps, array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
