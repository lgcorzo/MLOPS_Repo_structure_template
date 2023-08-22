import os
import pandas as pd

from Code.Application.project_name_algorithm import read_cnc_csv
from Code.Application.project_name_evaluation import model_fit, load_ml_model
from Code.Domain.Models.project_name import ProjectName


cwd = os.path.dirname(os.path.abspath(__file__))
cnc_path = os.path.join(cwd, '../../../Data/Results/CNC/')
global pickle_model
pickle_model = ''


def init_model_service() -> bool:
    global pickle_model
    csv_path = os.path.join(cwd, '../../../Data/Results/CNC', 'cnc_post_ext.csv')
    cnc_df = read_cnc_csv(csv_path)
    pickle_model = load_ml_model()
    if (pickle_model is None):
        model_fit(cnc_path, cnc_df, csv_path)
        pickle_model = load_ml_model()
    return True


def predict_model_service(file_data: str) -> dict:
    global pickle_model
    split_data = file_data
    data = {'cnc_db': split_data}
    series = pd.Series(data)
    sorted_df = pickle_model.predict_probea(series, 5)
    post = ProjectName(sorted_df)
    return post.as_dict()
