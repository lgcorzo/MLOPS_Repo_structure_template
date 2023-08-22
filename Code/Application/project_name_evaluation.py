import os
import logging
import pickle
from statistics import mean
from typing import Dict, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from tqdm import tqdm

from Code.Application.project_name_algorithm import read_cnc_csv, read_file
from Code.Application.project_name_model import ProjectNameModel
from Code.Utils.env_variables import Env

RAW_PATH = '../../Data/Results'
NUM_GRAM: int = 3
N = 1  # position in ranking 0 if compared with other, 1 if compared with itself
PKL_FILE = 'project_name_model.pkl'
THRESHOLD_SCORE = 0.95
cnc_path = os.path.join(RAW_PATH, 'CNC')

cwd = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(cwd, PKL_FILE)


def load_cncs(path_files_in: str, df: pd.DataFrame) -> dict:
    cnc_list = df['CNC'] + df['Extension'].fillna('')
    knowledge_dict = {cnc_name: read_file(os.path.join(path_files_in, cnc_name))
                      for cnc_name in cnc_list if cnc_name in os.listdir(path_files_in)}

    return knowledge_dict


def model_fit(path_files_in: str, df: pd.DataFrame, cnc_path_in: str) -> None:
    logging.info('---------------------------------CREATE KNOWLEDGE DATABASE-------------------------------------------------')
    env = Env()
    mlflow.set_tracking_uri(env.remote_server_uri)
    mlflow.set_experiment(env.experiment_name)
    with mlflow.start_run(run_name=env.run_name):
        knowledge_dict = load_cncs(path_files_in, df)
        model = ProjectNameModel(cnc_path_in)
        model.fit(list(knowledge_dict.values()), list(knowledge_dict.keys()))
        mlflow.log_metric("accuracy", 1.0)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=env.registered_model_name)

    logging.info('---------------------------------------PKL GENERATED------------------------------------------------------')


def train_test_split_parts(df, part_name) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_df = df[df['PrdRefDst'] == part_name]
    train_df = df[df['PrdRefDst'] != part_name]
    return train_df, test_df


def create_train_test_dict(knowledge_dict: dict, df: pd.DataFrame, part_name: str) -> Tuple[Dict, Dict]:
    train_df, test_df = train_test_split_parts(df, part_name)
    cnc_list_train = train_df['CNC'] + train_df['Extension'].fillna('')
    cnc_list_test = test_df['CNC'] + test_df['Extension'].fillna('')
    knowledge_train_dict = {cnc_name: knowledge_dict[cnc_name] for cnc_name in cnc_list_train
                            if cnc_name in knowledge_dict}
    knowledge_test_dict = {cnc_name: knowledge_dict[cnc_name] for cnc_name in cnc_list_test
                           if cnc_name in knowledge_dict}

    return knowledge_train_dict, knowledge_test_dict


def model_train(path_files_in: str, part_name: str, df: pd.DataFrame, cnc_path_in=cnc_path) -> float:
    env = Env()
    mlflow.set_tracking_uri(env.remote_server_uri)
    mlflow.set_experiment(env.experiment_name)
    with mlflow.start_run(run_name=env.run_name) as mlops_run:
        knowledge_train_dict, knowledge_test_dict = create_train_test_dict(load_cncs(path_files_in, df), df, part_name)
        model = ProjectNameModel(cnc_path_in)
        model.fit(list(knowledge_train_dict.values()), list(knowledge_train_dict.keys()))
        score = model.score(list(knowledge_test_dict.values()), list(knowledge_test_dict.keys()))
        mlflow.log_metric("accuracy", score / 2)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=env.registered_model_name)
    return score /2


def get_evaluation_results(path_files_in: str, df: pd.DataFrame) -> bool:
    deploy = False
    parts_name = df['PrdRefDst'].unique()
    score = []
    for index, part in enumerate(tqdm(parts_name, ascii=True)):
        score.append(model_train(path_files_in, part, df))

    result_score = mean(score)

    logging.info('-----------------------------------------------------------------------------------------------------------')
    logging.info('                                           Evaluating Model                                                ')
    logging.info('-----------------------------------------------------------------------------------------------------------')
    logging.info("Score: {0:.2f} %".format(100 * result_score))

    if result_score >= THRESHOLD_SCORE:
        deploy = True

    return deploy


def load_ml_model() -> object:
    env = Env()
    mlflow.set_tracking_uri(env.remote_server_uri)
    mlflow.set_experiment(env.experiment_name)
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{env.registered_model_name}'"):
        mv = dict(mv)
        logged_model = mv["source"]

    model = mlflow.sklearn.load_model(logged_model)

    return model


def main() -> None:
    path_files = str(os.path.join(cwd, cnc_path))
    cnc_df = read_cnc_csv()
    get_evaluation_results(path_files, cnc_df)


if __name__ == '__main__':  # pragma: no cover
    main()
