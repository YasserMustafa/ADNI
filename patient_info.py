"""
Read in the data dictionary as well as the diagnostic summary.
This module should be used to filter dataframes to get patients
belonging to a particular class (AD/CN/MCIc/MCInc)

"""

import pandas as pd
import StringIO

BASE_DIR = '/phobos/alzheimers/adni/'

# diagnostic summary data
DXSUM_FILE = BASE_DIR + 'DXSUM_PDXCONV_ADNIALL.csv'

# data dictionary for all ADNI data
DATADIC_FILE = BASE_DIR + 'DATADIC.csv'

# data dictionary for the ARM assignments
ARM_FILE = BASE_DIR + 'ARM.csv'

# data file for the Registries
REG_FILE = BASE_DIR + 'REGISTRY.csv'

DXSUM = pd.read_csv(StringIO.StringIO(open(DXSUM_FILE)
                                      .read().replace('\x00', '')))
DICT = pd.read_csv(StringIO.StringIO(open(DATADIC_FILE)
                                     .read().replace('\x00', '')))
ARM = pd.read_csv(StringIO.StringIO(open(ARM_FILE)
                                    .read().replace('\x00', '')))
REG = pd.read_csv(StringIO.StringIO(open(REG_FILE)
                                    .read().replace('\x00', '')))

# make the ADNI1 variables compatible with those in ADNIGO/2
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 1), 'DXCHANGE'] = 1
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 2), 'DXCHANGE'] = 2
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 3), 'DXCHANGE'] = 3
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 1), 'DXCHANGE'] = 4
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 3), 'DXCHANGE'] = 5
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 2), 'DXCHANGE'] = 6
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 1), 'DXCHANGE'] = 7
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 2), 'DXCHANGE'] = 8
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 3), 'DXCHANGE'] = 9

# merge ARM data with DXSUM. ADNI Training slides 2
DXARM = pd.merge(DXSUM[['RID', 'Phase', 'VISCODE', 'VISCODE2', 'DXCHANGE']],
                 ARM[['RID', 'Phase', 'ARM', 'ENROLLED']],
                 on=['RID', 'Phase'])
"""
1: Normal
2: Serious Memory Complaints (SMC)
3: EMCI
4: LMCI
5: AD
"""
NORMAL = 1
SMC = 2
EMCI = 3
LMCI = 4
AD = 5

BASE_DATA = DXARM[(DXARM['VISCODE2'] == 'bl') &
                  DXARM['ENROLLED'].isin([1, 2, 3])]
BASE_DATA.loc[(BASE_DATA['DXCHANGE'].isin([1, 7, 9])) &
              ~(BASE_DATA['ARM'] == 11), 'DXBASELINE'] = NORMAL
BASE_DATA.loc[(BASE_DATA['DXCHANGE'].isin([1, 7, 9])) &
              (BASE_DATA['ARM'] == 11), 'DXBASELINE'] = SMC
BASE_DATA.loc[(BASE_DATA['DXCHANGE'].isin([2, 4, 8])) &
              (BASE_DATA['ARM'] == 10), 'DXBASELINE'] = EMCI
BASE_DATA.loc[(BASE_DATA['DXCHANGE'].isin([2, 4, 8])) &
              ~(BASE_DATA['ARM'] == 10), 'DXBASELINE'] = LMCI
BASE_DATA.loc[(BASE_DATA['DXCHANGE'].isin([3, 5, 6])),
              'DXBASELINE'] = AD
DXARM = pd.merge(DXARM, BASE_DATA[['RID', 'DXBASELINE']], on='RID')

DXARM_REG = pd.merge(DXARM, REG[['RID', 'Phase', 'VISCODE', 'VISCODE2',
                                 'EXAMDATE', 'PTSTATUS', 'RGCONDCT',
                                 'RGSTATUS', 'VISTYPE']],
                     on=['RID', 'Phase', 'VISCODE', 'VISCODE2'])

class Data(object):
    """Class to hold the subsets of a dataset"""

    def __init__(self, data):
        """Initialize the sub-dict. for this set of features"""

        self.data = data
        try:
            self.sub_dx = DXSUM[DXSUM['RID'].isin(data.RID)]
        except AttributeError:
            print 'Data frame %s had no attribute RID!' % data.__name__
            raise

    def filter_bl_ad(self):
        """Find all baseline rows that were AD"""
        rid = DXARM[(DXARM['VISCODE2'] == 'bl') &
                    (DXARM['DXBASELINE'] == AD)].RID
        
