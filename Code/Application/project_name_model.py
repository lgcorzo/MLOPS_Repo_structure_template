import logging
from typing import Any, List
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from transformers import AutoModel, AutoTokenizer

from Code.Application.project_name_algorithm import (
    read_cnc_csv)


NUM_COMP = 10
names_transform = {
    'metric': 'metric',
    'DIS_PsfFile': 'post',
    'WrkRef': 'machine',
    'f_cluster': 'file',
}


def compare_documents(doc1, doc2) -> Any:
    logging.info('compare_documents start')
    similarity = cosine_similarity(doc1, doc2).detach().numpy()[0][0]
    logging.info('compare_documents finished')
    return similarity


def resume_document(doc1, model) -> Any:
    logging.info('compare_documents start')
    input_ids1 = model.pretrained_tokenizer(doc1, return_tensors="pt", max_length=512).input_ids
    output1 = model.pretrained_model(input_ids1).last_hidden_state[:, 0, :]
    logging.info('compare_documents finished')
    return output1.detach().numpy()


# Define a function to compute cosine similarity between two vectors
def cosine_similarity(x: Any, y: Any):
    x_torch = torch.tensor(x, dtype=torch.float64)
    y_torch = torch.tensor(y, dtype=torch.float64)
    logging.info('cosine_similarity start')
    x_norm = torch.norm(x_torch, dim=1, keepdim=True)
    y_norm = torch.norm(y_torch, dim=1, keepdim=True)
    similarity = torch.matmul(x_torch, y_torch.T) / (x_norm * y_norm)
    logging.info('cosine_similarity end')
    return similarity


def app_llm_metric_multiset(row: pd.DataFrame, cnc_comp: pd.Series) -> float:
    return compare_documents(row['cnc_db'], cnc_comp['cnc_db'])


def get_file_column_from_probea_results(row: pd.DataFrame) -> pd.Series:
    return row['file']


class ProjectNameModel(BaseEstimator):
    """Mixin class for ProjectNameModel classifiers in scikit-learn format."""
    knowledge_tokenized: List[Any]
    knowledge_cnc_name: List[Any]
    pretrained_model: Any
    pretrained_tokenizer: Any
    cnc_df: pd.DataFrame

    def __init__(self, cnc_path) -> None:
        self.knowledge_tokenized = []
        self.knowledge_cnc_name = []
        self.pretrained_model = None
        self.pretrained_tokenizer = None
        self.cnc_df = read_cnc_csv(cnc_path)

    def load_pretrained_llm(self, llm_type: str = "microsoft/codebert-base") -> None:
        logging.info(f'loading pretrained model {llm_type}')
        if (self.pretrained_model is None):
            self.pretrained_model = AutoModel.from_pretrained(llm_type)
        logging.info(f'loading pretrained tokenizer {llm_type}')
        if (self.pretrained_tokenizer is None):
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(llm_type)
        logging.info(f'{llm_type} models loaded')

    def fit(self, x_in: [list], y_in: list):
        logging.info('start fitting process')
        self.knowledge_tokenized = []
        self.load_pretrained_llm()
        for element in tqdm(x_in):
            self.knowledge_tokenized.append(resume_document(doc1=element, model=self))
        self.knowledge_cnc_name = y_in
        logging.info('fitting process finished')
        return self

    def predict_probea(self, x_in: pd.Series, num_results: int = NUM_COMP) -> pd.DataFrame:
        logging.info('start predict_probea')
        df_in = pd.DataFrame()
        df_in['cnc_db'] = self.knowledge_tokenized
        df_in['f_cluster'] = self.knowledge_cnc_name
        cnc_df = self.cnc_df.copy()
        cnc_df['f_cluster'] = cnc_df['CNC'] + cnc_df['Extension'].fillna('')
        x_in['cnc_db'] = resume_document(doc1=x_in['cnc_db'], model=self)
        df_in['metric'] = df_in.apply(app_llm_metric_multiset, cnc_comp=x_in, axis=1)
        table_dataframe_merges = pd.merge(
            df_in,
            cnc_df,
            how='left',
            on='f_cluster'
        )
        table_dataframe_merges = table_dataframe_merges.sort_values(by='metric',
                                                                    ascending=False,
                                                                    ignore_index=True).head(num_results)

        table_dataframe_merges_selected = table_dataframe_merges[names_transform.keys()]
        logging.info('finish predict_probea')
        return table_dataframe_merges_selected.rename(columns=names_transform)

    # Revisar loop duplicado
    def score(self, x_test: list, y_test: list) -> float:
        p = 0
        n = 0
        data = {'cnc_db': x_test, 'f_cluster': y_test}
        data_df = pd.DataFrame(data)
        result_df = data_df.apply(self.predict_probea, num_results=1, axis=1)
        cnc_df = self.cnc_df.copy()
        cnc_df['f_cluster'] = cnc_df['CNC'] + cnc_df['Extension'].fillna('')

        for index, cnc_name in enumerate(data_df['f_cluster']):
            post_model = result_df[index]['post']
            post_test = cnc_df['DIS_PsfFile'][cnc_df['f_cluster'] == cnc_name]
            if post_model.iloc[0] == post_test.iloc[0]:
                p += 1
            else:
                n += 1
        return p / (p + n)

    def predict(self, x_in: list) -> pd.Series:
        data = {'cnc_db': x_in}
        data_df = pd.DataFrame(data)
        result_df = data_df.apply(self.predict_probea, num_results=1, axis=1)
        out_df = result_df.apply(get_file_column_from_probea_results)
        result_out = out_df[0]
        result_out.name = 'file'
        return result_out
