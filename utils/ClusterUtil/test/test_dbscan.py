from utils.ClusterUtil.algorithm.dbscan import DbscanSklearn
from utils.PreprocessingUtil import TimeSeries
from config import Config
from utils.GraphUtil import visualize_cluster
from pandas import read_csv
from numpy import reshape

# parameters
sliding = 2

df = read_csv(f'{Config.DATA_INPUT}/gg_cpu.csv', usecols=[1], header=0, index_col=False)
timeseries = TimeSeries(data=df.values, train_split=0.7)
data_new = timeseries._scaling__("minmax")
X_test, y_test = timeseries._make_train_test_data__(data_new, [1, 2, 3], timeseries.train_split, None, "2D")
X_train, y_train = timeseries._make_train_test_data__(data_new, [1, 2, 3], 0, timeseries.train_split, "2D")
print("Processing data done!!!")

clustering = DbscanSklearn(eps=sliding*0.03, min_samples=2)
n_clusters, centers, list_clusters, labels, feature_label = clustering._cluster__(X_data=X_train)
print('Centers found by our algorithm:')
print(n_clusters)
print(centers)
s1, s2, s3 = clustering._evaluation__(X_train, labels ,0)
print(f"s1 = {s1}, s2 = {s2}, s3 = {s3}")

visualize_cluster(X_train, reshape(labels, (len(labels))))


