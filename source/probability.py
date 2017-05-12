import numpy as np
from math import sin, cos, sqrt, atan2, radians


global num_clusters


def main():
    global num_clusters
    num_clusters = 4
    dir_path = "F:\SpaceApp\genex\geneflow_detector"

    cnt =0
    cluster_list = []
    file_str = ""
    with open('falcon_geneflow_final.csv') as f:
        for line in f:
            line = line.strip()
            data = line.split(",")
            cluster_list.append(list(map(float,data[1:])))
            time_stmp = data[0]
            if cnt ==num_clusters-1:
                for i in range(num_clusters):
                    file_str += "['" + time_stmp + "','"
                    distance_list = []
                    radious_list = []
                    r = cluster_list[i][2]
                    lat = cluster_list[i][0]
                    long = cluster_list[i][1]
                    file_str += str(lat) +"','" + str(long)+ "',"
                    for j in range(num_clusters):
                        if j == i:
                            continue
                        adj_r = cluster_list[j][2]
                        radious_list.append(adj_r)
                        adj_lat = cluster_list[j][0]
                        adj_long = cluster_list[j][1]
                        d = global_distance([lat,long], [adj_lat, adj_long])
                        distance_list.append(d)

                    prob = float(sum(radious_list))/sum(distance_list)
                    file_str += str(prob) + "," + str(int(cluster_list[i][3])) + "],"

                cluster_list = []
                cnt =-1

            cnt += 1

    f = open('falcn_geneflow_prob.js', 'w')
    f.write("var falconProbPoints = [")
    f.write(file_str)
    f.write("];")
    f.close()


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
