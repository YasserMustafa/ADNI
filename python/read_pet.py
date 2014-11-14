"""Read and clean the UCSF Free-surfer data"""

import pandas as pd
import StringIO
from patient_info import clean_visits

BASE_DIR = '/phobos/alzheimers/adni/'

FDG_FILE = BASE_DIR + 'UCBERKELEYFDG_03_13_14.csv'
AV_FILE = BASE_DIR + 'UCBERKELEYAV45_07_30_14.csv'

FDG = pd.read_csv(StringIO.StringIO(open(FDG_FILE)
                                    .read().replace('\x00', '')))
AV = pd.read_csv(StringIO.StringIO(open(AV_FILE)
                                   .read().replace('\x00', '')))

if 'VISCODE2' in FDG.columns:
    FDG = clean_visits(FDG)
else:
    FDG['VISCODE2'] = FDG['VISCODE']

if 'VISCODE2' in AV.columns:
    AV = clean_visits(AV)
else:
    AV['VISCODE2'] = AV['VISCODE']
