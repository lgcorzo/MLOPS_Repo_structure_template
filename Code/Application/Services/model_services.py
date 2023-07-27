import os
import pandas as pd

from Code.Application.smart_machine_config_algorithm import read_cnc_csv, clean_data_cncs
from Code.Application.smart_machine_config_evaluation import model_fit, load_model
from Code.Domain.Models.machine_configuration import MachineConfiguration

cwd = os.path.dirname(os.path.abspath(__file__))
cnc_path = os.path.join(cwd, '../../../Data/Results/CNC/')
global pickle_model
pickle_model = ''


def fit_model_service() -> bool:
    csv_path = os.path.join(cwd, '../../../Data/Results/CNC', 'cnc_post_ext.csv')
    cnc_df = read_cnc_csv(csv_path)
    global pickle_model
    pickle_model = load_model()
    if (pickle_model is None):
        model_fit(cnc_path, cnc_df, csv_path)
        pickle_model = load_model()
    return True


def predict_model_service(file_data: str) -> dict:
    global pickle_model
    split_data = clean_data_cncs(file_data)
    data = {'cnc_db': split_data}
    series = pd.Series(data)
    sorted_df = pickle_model.predict_probea(series, 5)
    post = MachineConfiguration(sorted_df)
    return post.as_dict()
