import numpy as np
import pandas as pd
import re
import os

from collections import Counter

RAW_PATH = '../../Data/Results'
CSV_FILE = 'cnc_post_ext.csv'

cnc_path = os.path.join(RAW_PATH, 'CNC')
csv_path = os.path.join(cnc_path, CSV_FILE)
NUM_GRAM: int = 3


def split_text(data: str, k_gram_length: int) -> list:
    """
    Split the text given in data into n-grams

    Parameters
    ----------
        data: str
            text that will be split
        k_gram_length: int
            length of the grams

    Returns
    -------
        list:
            list of all k-grams for the given text
    """
    return [data[index:index + k_gram_length] for index in range(len(data) - k_gram_length + 1)]


def read_file(path_file: str) -> str:
    return open(path_file, 'r', encoding='ISO-8859-1').read()


def remove_coordinate_numbers(data: str) -> str:
    pat1 = r'-[0-9]+[.][0-9]+'
    pat2 = r'[0-9]+[.][0-9]+'
    return re.sub(r'|'.join((pat1, pat2)), '', data)


def clean_data_cncs(cnc_data: str) -> list:
    return split_text(remove_coordinate_numbers(cnc_data).replace('\r\n', '').replace('\n', ''), NUM_GRAM)


def read_cnc_csv(csv_path_in=csv_path) -> pd.DataFrame:
    """
    Reads the csv file in CSV_PATH and return teh DataFrame with the columns
    WrkRef - Machine; CNC - CNC name; DIS_PsfFile - postprocessor name;
    DIS_CfgFile - seed file; PrdRefDst - part used; Extension - extension cnc file

    Returns
    -------
        pd.DataFrame:
            dataframe with
    """
    return pd.read_csv(csv_path_in, sep=';', dtype={'CNC': str}, encoding='ISO-8859-1')


def intersection_count(a: list, b: list) -> np.ndarray:
    """
    Intersection length function for a multiset

    Parameters
    ----------
        a: list
            list of k-grams
        b: list
            list of k-grams to compare

    Returns
    -------
        int:
            length of the intersection between the a and b lists
    """
    a = dict(Counter(a))
    b = dict(Counter(b))
    count = [min(a[item], b[item]) for item in a if item in b]
    return np.sum(count)


def jacc_metric_multiset(a: list, b: list) -> float:
    """
    Calculate the Jaccard coefficient for two given multisets

    Parameters
    ----------
        a: list
            list of all k-grams
        b: list
            list of all k-grams

    Returns
    -------
        float:
            value for jaccard coefficient
    """
    return 0 if not a or not b else 2 * intersection_count(a, b) / len(a + b)
