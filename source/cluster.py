import pandas as pd
import numpy as np
import utils
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from math import sin, cos, sqrt, atan2, radians
from estimation.kmeans_clustering import KMeansCluster

global num_clusters


def main():
    global num_clusters
    preprocessors = [
        preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    ]
    num_clusters = 4
    dir_path = "F:\SpaceApp\genex\geneflow_detector\data_sets"
    file_name = "falcon_sightings.csv"
    f = open(file_name, 'w')
    f.write("var falconOcPoints = [")
    geo_data = utils.load_full_dataset(dir_path, file_name)

    month_frame = geo_data.timestamp.dt.month
    current_month = month_frame.iloc[0]
    i = 0
    prv_i = 0
    cnt = 0
    while True:
        while True:
            i += 1
            if i >= len(geo_data):
                break
            next_month = month_frame.iloc[i]
            if next_month != current_month:
                current_month = next_month
                break
        if i >= len(geo_data):
            break
        chunk = geo_data.loc[prv_i:i]

        if len(chunk) < num_clusters:
            continue
        cluster(chunk, f)
        prv_i = i
    f.write("];")
    f.close()


def cluster(data_frame, file_wr):
    global num_clusters
    f = file_wr
    file_str = ""
    preprocessors = [
        preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1)
    ]

    data_columns = list(data_frame.columns.values)
    time_stmp = str(data_frame['timestamp'].iloc[0])

    data_columns.remove("timestamp")
    data_data_mapper = DataFrameMapper([(data_columns, preprocessors)])
    geo_data = data_data_mapper.fit_transform(data_frame)

    pred = KMeansCluster()

    pred.fit(geo_data)

    cluster_array = {i: geo_data[np.where(pred.model.labels_ == i)] for i in range(num_clusters)}
    radious_list = []

    for i in range(num_clusters):
        file_str += "['"+ time_stmp + "','"
        cluster_val = cluster_array[i]
        cluster_centre = pred.model.cluster_centers_[i]
        distance_arr = []
        for point in cluster_val:
            distance_arr.append(global_distance(cluster_centre, point))
        radious_list.append(get_percentile_threshold(0.8, distance_arr))
        file_str += str(cluster_centre[0]) + "','" + str(cluster_centre[1]) + "',"
        file_str += str(radious_list[i]) + ","
        file_str += str(i) + "],"

    file_wr.write(file_str)


def global_distance(centre, point):
    c_lat, c_long = centre
    p_lat, p_long = point

    R = 6373.0

    lat1 = radians(c_lat)
    lon1 = radians(c_long)
    lat2 = radians(p_lat)
    lon2 = radians(p_long)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    if distance is not None:
        if distance < 1:
            return 0.0
        else:
            return distance
    else:
        return 0.0


def get_percentile_threshold(quntile, data_list):
    var = np.array(data_list)  # input array
    if var.size:
        return np.percentile(var, quntile*100)

if __name__ == '__main__':
    main()
