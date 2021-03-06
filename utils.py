import pandas as pd

from enum import IntEnum


def load_full_dataset(data_set_file, file_name):
    geo_df = pd.read_csv(data_set_file+"\\"+file_name)

    geo_df['timestamp'] = pd.to_datetime(geo_df['timestamp'])

    # climate_df = pd.read_csv(data_set_file+"\\falcon-tracked-temp.csv")

    Logger.info("Loaded data sets: all data \n")
    return geo_df


def preprocess_full_dataset(data_set_file, file_name):
    geo_df = pd.read_csv(data_set_file+"\\falcon_geneflow_final.csv")

    geo_df.columns = ['timestamp','location-long', 'location-lat', 'radious', 'cluster_no']

    geo_df.to_csv(data_set_file+"\\"+file_name, index=False)


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
