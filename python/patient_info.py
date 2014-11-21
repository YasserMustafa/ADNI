"""
Read in the data dictionary as well as the diagnostic summary.

This module should be used to filter dataframes to get patients
belonging to a particular class (AD/CN/MCIc/MCInc)

"""

import pandas as pd
import numpy as np
from read import read
import matplotlib.pyplot as plt

BASE_DIR = '/phobos/alzheimers/adni/'

# diagnostic summary data
DXSUM_FILE = BASE_DIR + 'DXSUM_PDXCONV_ADNIALL.csv'

# data dictionary for all ADNI data
DATADIC_FILE = BASE_DIR + 'DATADIC.csv'

# data dictionary for the ARM assignments
ARM_FILE = BASE_DIR + 'ARM.csv'

# data file for the Registries
REG_FILE = BASE_DIR + 'REGISTRY.csv'

DXSUM = read(DXSUM_FILE)
DICT = read(DATADIC_FILE)
ARM = read(ARM_FILE)
REG = read(REG_FILE)

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

"""
Key for DXCHANGE
"""

NL = 1
MCI = 2
AD = 3
NL_MCI = 4
MCI_AD = 5
NL_AD = 6
MCI_NL = 7
AD_MCI = 8
AD_NL = 9

# make the ADNI1 variables compatible with those in ADNIGO/2
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 1), 'DXCHANGE'] = NL
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 2), 'DXCHANGE'] = MCI
DXSUM.loc[(DXSUM['DXCONV'] == 0) &
          (DXSUM['DXCURREN'] == 3), 'DXCHANGE'] = AD
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 1), 'DXCHANGE'] = NL_MCI
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 3), 'DXCHANGE'] = MCI_AD
DXSUM.loc[(DXSUM['DXCONV'] == 1) &
          (DXSUM['DXCONTYP'] == 2), 'DXCHANGE'] = NL_AD
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 1), 'DXCHANGE'] = MCI_NL
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 2), 'DXCHANGE'] = AD_MCI
DXSUM.loc[(DXSUM['DXCONV'] == 2) &
          (DXSUM['DXREV'] == 3), 'DXCHANGE'] = AD_NL

# merge ARM data with DXSUM. ADNI Training slides 2
DXARM = pd.merge(DXSUM[['RID', 'Phase', 'VISCODE', 'VISCODE2', 'DXCHANGE']],
                 ARM[['RID', 'Phase', 'ARM', 'ENROLLED']],
                 on=['RID', 'Phase'])

BASE_DATA = DXARM.loc[(DXARM['VISCODE2'] == 'bl') &
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

def clean_visits(data):
    """
    Keyword Arguments:
    data -- The data-frame to clean
    """
    result = data['VISCODE2'].isnull()
    for i in xrange(len(result)):
        if result[i]:
            data.loc[i, 'VISCODE2'] = data.loc[i, 'VISCODE']

    return data

def get_dx(data):
    """
    Keyword Arguments:
    data -- The data we want Diagnoisis information for
    Note that this function just appends a new column to the df
    rather than return a new data structure

    Returns the new dataframe with DX info.

    """
    merged = pd.merge(data, DXARM_REG, on=['RID', 'VISCODE2'],
                      how='inner')

    cols = []
    for column in data.columns:
        if column in merged.columns:
            cols.append(column)

    cols.extend(['DXCHANGE', 'DXBASELINE'])
    merged = merged[cols]
    merged.loc[merged['DXCHANGE'].isin([1, 7, 9]), 'DX'] = 'NL'
    merged.loc[merged['DXCHANGE'].isin([2, 4, 8]), 'DX'] = 'MCI'
    merged.loc[merged['DXCHANGE'].isin([3, 5, 6]), 'DX'] = 'AD'

    return merged

def get_baseline_classes(data, phase=''):
    """
    Keyword Arguments:
    data -- The data to segment
    """
    # store patients in each group
    dx_base = {}

    # RIDs of patients we want to consider
    # first get all patients belong to the correct phase
    # and having baseline measurements
    # then only consider those that have measurements in the data matrix
    if phase == 'ADNI1':
        idx = get_adni1_idx(data)
        rid = DXARM_REG.loc[idx, 'RID']
    else:
        rid = data['RID'].unique()

    for patient in rid:
        try:
            info = DXARM_REG[DXARM_REG['RID'] == patient]
            dx_baseline = info.DXBASELINE.values[0]
            change = info.DXCHANGE
            if dx_baseline == 1 or dx_baseline == 2:
                dx_base[patient] = 'NL' # normal control
            elif dx_baseline == 3 or dx_baseline == 4:
                dx_base[patient] = 'MCI' # mild cognitive impairment
                if MCI_AD in change.values:
                    dx_base[patient] += '-C'
                elif MCI_NL in change.values:
                    dx_base[patient] += '-REV'
                elif MCI in change.values:
                    dx_base[patient] += '-NC'
            elif dx_baseline == 5:
                dx_base[patient] = 'AD' # alzheimer's disease
        except IndexError:
            print 'WARNING: No diagnostic info. for RID=%d'%patient

    return dx_base

def get_adni1_idx(data):
    """
    Extract from DXARM_REG the indices of the rows that belong only to
    ADNI1 patients, and also exist in 'data'
    Keyword Arguments: data -- The data-set we want to
    consider

    """
    return ((DXARM_REG['Phase'] == 'ADNI1') &
            (DXARM_REG['VISCODE'] == 'bl') & # belonging to ADNI1
            (DXARM_REG['RGCONDCT'] == 1) & # was the visit conducted?
            (DXARM_REG['RID'].isin(data['RID'])))

def count_visits(data, modality='', plot=True):
    """
    Keyword Arguments:
    data -- The subset of the data we want visit stats for
    """
    all_visits = np.sort(np.r_[data.VISCODE2.unique(), data.VISCODE.unique()])
    all_visits = np.array([visit for visit in all_visits
                           if str(visit) != 'nan' and\
                           str(visit) != 'f' and\
                           str(visit)[0] != 'v' and\
                           str(visit) != 'nv'])
    all_visits = np.unique(all_visits)
    print "Unique visit codes:", all_visits
    columns = np.r_[['Phase'], ['Count'], ['DXBASELINE'], all_visits]
    rid = data['RID'].unique()
    stats = pd.DataFrame(columns=columns)
    dx_base = get_baseline_classes(data)

    for patient in rid:
        reg_info = DXARM_REG[DXARM_REG['RID'] == patient]
        mode_info = data[data['RID'] == patient]
        values = []
        if mode_info.VISCODE2.isnull().any():
            visits = mode_info['VISCODE']
        else:
            visits = mode_info['VISCODE2']
        if pd.Series('bl').isin(reg_info.VISCODE2).any()\
           or pd.Series('sc').isin(reg_info.VISCODE2).any():
            phase = reg_info[reg_info.VISCODE2 == 'bl'].Phase.unique()[0]
            values.append(phase)
            values.append(len(visits.unique()))
            values.append(dx_base[patient])
            for visit in all_visits:
                if visit in visits.values:
                    values.append('True')
                else:
                    values.append('False')
            stats.loc[patient] = values

    print "Total patients = ", len(stats)
    if plot:
        phases = stats['Phase'].unique()
        counts = []
        for phase in phases:
            counts.append(stats[stats['Phase'] == phase].Count.values)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(counts, bins=np.arange(1, max(stats.Count+2)),
                label=phases.tolist(),
                color=['r', 'g', 'b'][:len(phases)])
        ax.set_xlabel('Number of visits', fontsize=28)
        ax.set_ylabel('Number of patients', fontsize=28)
        ax.set_title('Histogram of patient visits for '+modality+' data',
                     fontsize=30)
        ax.legend()
        ax.yaxis.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.xaxis.set_ticks(np.arange(1, max(stats.Count+2)))
        fig.show()
    return stats

def plot_dx(stats, modality=''):
    """
    Show the distribution of diagnoses against patient-visit counts
    Keyword Arguments:
    stats -- The stats to plot
    """
    dx_base = np.sort(stats['DXBASELINE'].unique())
    counts = []
    for dx in dx_base:
        counts.append(stats[stats['DXBASELINE'] == dx].Count.values)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=np.arange(1, max(stats.Count+2)),
            label=dx_base.tolist())
    ax.set_xlabel('Number of visits', fontsize=28)
    ax.set_ylabel('Number of patients', fontsize=28)
    ax.set_title('Histogram of patient visits for '+modality+' data',
                 fontsize=30)
    ax.legend()
    ax.yaxis.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.xaxis.set_ticks(np.arange(1, max(stats.Count+2)))
    fig.show()
