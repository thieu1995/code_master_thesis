from utils.ClusterUtil.algorithm.kmeans import BaseKmeans, KMeansPlusPlus, KMeansSklearn
from utils.PreprocessingUtil import TimeSeries
from config import Config
from numpy import reshape
from pandas import read_csv
from utils.GraphUtil import visualize_cluster

df = read_csv(f'{Config.DATA_INPUT}/gg_cpu.csv', usecols=[1], header=0, index_col=False)
timeseries = TimeSeries(data=df.values, train_split=0.7)
data_new = timeseries._scaling__("minmax")
X_test, y_test = timeseries._make_train_test_data__(data_new, [1, 2, 3], timeseries.train_split, None, "2D")
X_train, y_train = timeseries._make_train_test_data__(data_new, [1, 2, 3], 0, timeseries.train_split, "2D")
print("Processing data done!!!")

#clustering = BaseKmeans(8)
clustering = KMeansPlusPlus(8)
#clustering = KMeansSklearn(8)

n_clusters, centers, list_clusters, labels, feature_label = clustering._cluster__(X_data=X_train)
print('Centers found by our algorithm:')
print(n_clusters)
print(centers)
s1, s2, s3 = clustering._evaluation__(X_train, labels ,0)
print("s1 = {}, s2 = {}, s3 = {}".format(s1, s2, s3))

visualize_cluster(X_train, reshape(labels, (len(labels))))



