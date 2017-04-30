import pandas as pd

from enum import IntEnum
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing


def load_full_dataset(data_set_file):
    geo_df = pd.read_csv(data_set_file+"\\falcon-tracked.csv")
    climate_df = pd.read_csv(data_set_file+"\\falcon-tracked-temp.csv")

    preprocessors = [
        preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1)
    ]
    data_columns = list(geo_df.columns.values)
    data_data_mapper = DataFrameMapper([(data_columns, preprocessors)])
    geo_data = data_data_mapper.fit_transform(geo_df)

    data_columns = list(geo_df.columns.values)
    data_data_mapper = DataFrameMapper([(data_columns, preprocessors)])
    climate_data = data_data_mapper.fit_transform(climate_df)

    Logger.info("Loaded data sets: all data \n")
    return geo_data, climate_data


class EvaluationScoreCalculator:
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    def add_new_prediction(self, prediction, response_value):
        if prediction == 1:
            if prediction == response_value:
                self.true_positives += 1
            else:
                self.false_positives += 1
        else:
            if prediction == response_value:
                self.true_negatives += 1
            else:
                self.false_negatives += 1

    def print(self):
        Logger.info("Model evaluation stats calculated")
        Logger.info("True positives: " + str(self.true_positives))
        Logger.info("True negatives: " + str(self.true_negatives))
        Logger.info("False positives: " + str(self.false_positives))
        Logger.info("False negatives: " + str(self.false_negatives))
        Logger.info("Model evaluation scores calculated")
        Logger.info("Recall: " + str(self.get_recall()))
        Logger.info("Precision: " + str(self.get_precision()))
        Logger.info("F1: " + str(self.get_f1_score()))
        Logger.info("True negative rate: " + str(self.get_true_negative_rate()))
        Logger.info("Accuracy: " + str(self.get_accuracy()))

    def get_accuracy(self):
        denominator = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if denominator == 0:
            return -1
        else:
            return (self.true_positives + self.true_negatives) / denominator

    def get_recall(self):
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return -1
        else:
            return self.true_positives / denominator

    def get_precision(self):
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return -1
        else:
            return self.true_positives / denominator

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()

        denominator = precision + recall
        if denominator == 0:
            return -1
        else:
            return (precision * recall * 2) / denominator

    def get_true_negative_rate(self):
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return -1
        else:
            return self.true_negatives / denominator


class LogLevel(IntEnum):
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    OFF = 5


class Logger:
    current_log_level = LogLevel.INFO

    @staticmethod
    def set_log_level(log_level):
        Logger.current_log_level = log_level

    @staticmethod
    def info(msg):
        Logger.__print_log(LogLevel.INFO, msg)

    @staticmethod
    def debug(msg):
        Logger.__print_log(LogLevel.DEBUG, msg)

    @staticmethod
    def warn(msg):
        Logger.__print_log(LogLevel.WARN, msg)

    @staticmethod
    def error(msg):
        Logger.__print_log(LogLevel.ERROR, msg)

    @staticmethod
    def __print_log(msg_log_level, msg):
        if msg_log_level >= Logger.current_log_level:
            print("[" + msg_log_level.name + "] " + msg)
