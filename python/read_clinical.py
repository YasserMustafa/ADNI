"""Read in data from clinical tests"""

from read import read

BASE_DIR = '/phobos/alzheimers/adni/'

MMSE_FILE = BASE_DIR + 'MMSE.csv'
CDR_FILE = BASE_DIR + 'CDR.csv'

MMSE = read(MMSE_FILE)
CDR = read(CDR_FILE)

MMSE.loc[MMSE['VISCODE2'] == 'sc', 'VISCODE2'] = 'bl'
CDR.loc[CDR['VISCODE2'] == 'sc', 'VISCODE2'] = 'bl'
