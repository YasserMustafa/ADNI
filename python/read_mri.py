"""Read and clean the UCSF Free-surfer data"""

import pandas as pd
from read import read
from patient_info import clean_visits

BASE_DIR = '/phobos/alzheimers/adni/'

# data from ADNIGO/ADNI2
DICTIONARY_51_FILE = BASE_DIR + 'UCSFFSX51_DICT_08_01_14.csv'
DATA_51_FILE = BASE_DIR + 'UCSFFSX51_08_01_14.csv'

# data from ADNI1
DICTIONARY_FILE = BASE_DIR + 'UCSFFSX_DICT_08_01_14.csv'
DATA_FILE = BASE_DIR + 'UCSFFSX_08_01_14.csv'

FSX_51 = read(DATA_51_FILE)
FSX = read(DATA_FILE)

if 'VISCODE2' in FSX.columns:
    FSX = clean_visits(FSX)
else:
    FSX['VISCODE2'] = FSX['VISCODE']

if 'VISCODE2' in FSX_51.columns:
    FSX_51 = clean_visits(FSX_51)
else:
    FSX_51['VISCODE2'] = FSX_51['VISCODE']

def find_unique(src, target):
    """
    Keyword Arguments:
    src    -- the original dataframe
    target -- the dataframe we are looking for matches in
    """
    uniq_idx = ~src.FLDNAME.isin(target.FLDNAME)
    uniq_elems = src[['FLDNAME', 'TEXT']][uniq_idx == True]
    return uniq_elems

def show_unique():
    """
    Show the unique data fields within each dataset ADNI1 has been
    processed with a different version of Freesurfer as compared to
    ADNIGO/2
    """
    fsx_51_dict = pd.read_csv(DICTIONARY_51_FILE)
    fsx_dict = pd.read_csv(DICTIONARY_FILE)

    print "Data fields present only in ADNIGO/2\n"
    print find_unique(fsx_51_dict, fsx_dict)

    print "Data fields present only in ADNI1\n"
    print find_unique(fsx_dict, fsx_51_dict)

def read_fsx():
    """
    Read in the Freesurfer 5.1 data (for ADNIGO/2)
    and the associated data dictionary
    """
    fsx_dict = pd.read_csv(DICTIONARY_51_FILE)
    fsx = pd.read_csv(DATA_51_FILE)

    return fsx, fsx_dict


def main():
    """
    Main entry point for module
    """
    #show_unique()

if __name__ == "__main__":
    main()
