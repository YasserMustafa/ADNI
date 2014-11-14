"""Read and clean the CSF data"""

import pandas as pd
from read import read
from patient_info import clean_visits

BASE_DIR = '/phobos/alzheimers/adni/'

CSF_FILES = ['UPENNBIOMK.csv', 'UPENNBIOMK2.csv', 'UPENNBIOMK3.csv',
             'UPENNBIOMK4_09_06_12.csv', 'UPENNBIOMK5_10_31_13.csv',
             'UPENNBIOMK6_07_02_13.csv', 'UPENNBIOMK7.csv']

def read_csf():
    """
    Read in CSF results from each file and concatenate them into a
    single data frame
    """
    data = []
    for csf_file in CSF_FILES:
        data.append(read(BASE_DIR+csf_file))

    return pd.concat(data, ignore_index=True)

CSF = read_csf()

if 'VISCODE2' in CSF.columns:
    CSF = clean_visits(CSF)
else:
    CSF['VISCODE2'] = CSF['VISCODE']
