"""Read data from a .csv file and return a pandas dataframe"""

import pandas as pd
import StringIO

def read(file_name):
    """
    Keyword Arguments:
    file_name -- read the contents of file_name into a dataframe
    """
    return pd.read_csv(StringIO.StringIO(open(file_name)
                                         .read().replace('\x00', '')))
