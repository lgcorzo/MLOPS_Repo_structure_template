import pandas as pd

from sklearn.base import BaseEstimator
from typing import Any, List

from Code.Application.smart_machine_config_algorithm import jacc_metric_multiset, read_cnc_csv

NUM_COMP = 10
names_transform = {
    'metric': 'metric',
    'DIS_PsfFile': 'post',
    'WrkRef': 'machine',
    'f_cluster': 'file',
}


def app_jacc_metric_multiset(row: pd.DataFrame, cnc_comp: pd.Series) -> float:
    return jacc_metric_multiset(row['cnc_db'], cnc_comp['cnc_db'])


def get_file_column_from_probea_results(row: pd.DataFrame) -> pd.Series:
    return row['file']


class SmartMachineConfigModel(BaseEstimator):
    """Mixin class for SmartMachineConfigModel classifiers in scikit-learn format."""
    knowledge_kgrams: List[Any]
    knowledge_cnc_name: List[Any]
    cnc_df: pd.DataFrame

    def __init__(self, cnc_path) -> None:
        self.knowledge_kgrams = []
        self.knowledge_cnc_name = []
        self.cnc_df = read_cnc_csv(cnc_path)

    def fit(self, x_in: [list], y_in: list):
        """Perform fit on x_in and returns labels for X.

               Parameters
               ----------
               x_in : {array-like, sparse matrix} of shape (n_samples, n_features)
                   The input samples.

               y_in : array with the machine labels

               Returns
               -------
               self : ndarray of shape (n_samples,)

        """
        self.knowledge_kgrams = x_in
        self.knowledge_cnc_name = y_in
        return self

    def predict_probea(self, x_in: pd.Series, num_results: int = NUM_COMP) -> pd.DataFrame:
        """Perform prediction on x_in and returns labels for x_in returning metric, post file and machine

               Parameters
               ----------
               x_in : List of lists  of shape (n_samples, n_features)
                   The input samples.
               num_results : int Number of elements to show in the list of  similar results
               Returns
               -------
               self : dataframe of shape with the list of metrics
        """
        df_in = pd.DataFrame()
        df_in['cnc_db'] = self.knowledge_kgrams
        df_in['f_cluster'] = self.knowledge_cnc_name
        cnc_df = self.cnc_df.copy()
        cnc_df['f_cluster'] = cnc_df['CNC'] + cnc_df['Extension'].fillna('')

        df_in['metric'] = df_in.apply(app_jacc_metric_multiset, cnc_comp=x_in, axis=1)
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
        return table_dataframe_merges_selected.rename(columns=names_transform)

    # Revisar loop duplicado
    def score(self, x_test: list, y_test: list) -> float:
        """
        Return the percentage of the ratio between the matches and the total compared

        Parameters:
        -----------
        x_test : list of the kgrams from the test cnc files
        y_test : list of the cnc names from the test cnc files

        Returns
        -------
        score : float
            prevalence = positive/(positive+negative)
        """
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
        """Perform prediction on x_in and returns labels for x_in.

               Parameters
               ----------
               x_in : {array-like, dict}
                   The input samples.

               Returns
               -------
               result_out : result cluster series
        """

        data = {'cnc_db': x_in}
        data_df = pd.DataFrame(data)
        result_df = data_df.apply(self.predict_probea, num_results=1, axis=1)
        out_df = result_df.apply(get_file_column_from_probea_results)
        result_out = out_df[0]
        result_out.name = 'file'
        return result_out
