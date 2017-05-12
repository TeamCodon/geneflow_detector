from sklearn.cluster import KMeans
from utils import Logger
import time


class KMeansCluster():
    params = None
    model = None
    mae = 0

    default_params = {'n_clusters': 4, 'n_init': 12, 'n_jobs': -1}

    def __init__(self, params=default_params):
        self.params = params
        self.estimator = KMeans(**params)

    def fit(self, train_data):

        self.model = _fit(self.estimator, train_data)


def _fit(estimator, x_train):
    Logger.info("Training k-means clustering model started")
    start_time = time.time()

    model = estimator.fit(x_train)

    Logger.info("Training Gradient boosting model completed (Elapsed time: " + str(
        time.time() - start_time) + " seconds)")
    return model
