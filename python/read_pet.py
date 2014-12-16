"""Read and clean the UCSF Free-surfer data"""

import pandas as pd
import StringIO
from patient_info import clean_visits
import numpy as np
import matplotlib.pyplot as plt
from patient_info import get_dx, get_baseline_classes, DXARM_REG

BASE_DIR = '/phobos/alzheimers/adni/'

FDG_FILE = BASE_DIR + 'UCBERKELEYFDG_03_13_14.csv'
AV_FILE = BASE_DIR + 'UCBERKELEYAV45_07_30_14.csv'

FDG = pd.read_csv(StringIO.StringIO(open(FDG_FILE)
                                    .read().replace('\x00', '')))
AV = pd.read_csv(StringIO.StringIO(open(AV_FILE)
                                   .read().replace('\x00', '')))
FDG['ROI'] = FDG['ROINAME'] + '_' + FDG['ROILAT']

if 'VISCODE2' in FDG.columns:
    FDG = clean_visits(FDG)
else:
    FDG['VISCODE2'] = FDG['VISCODE']

if 'VISCODE2' in AV.columns:
    AV = clean_visits(AV)
else:
    AV['VISCODE2'] = AV['VISCODE']

MCI_AD = 5

def flatten_pet():
    """
    Reshape FDG data so that each row represents a visit rather than a
    region. Records are sorted by RID, with the VISCODE2 used to break
    ties within patients.

    """
    fdg = get_dx(FDG)
    data = []

    visit_features = ['RID', 'VISCODE2', 'DX']
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    regions = np.sort(fdg[:5]['ROI'].unique())

    grouped = fdg.groupby(visit_features, as_index=False)
    idx = sorted(grouped.indices.keys())
    base_dx = get_baseline_classes(fdg)

    for vis in idx:
        rid = vis[0]
        conv = base_dx[rid] == 'MCI-C'
        visit = grouped.get_group(vis).sort('ROI')[features]
        data.append(list(vis) +
                    [conv] +
                    visit.values.flatten().tolist())

    columns = visit_features + ['CONV']
    for roi in regions:
        columns.extend([roi+'_'+feature for feature in features])

    data = pd.DataFrame(data, columns=columns)
    data.loc[(data['DX'] == 'MCI') & (data['CONV']), 'DX'] = 'MCI-C'
    data.loc[(data['DX'] == 'MCI') & (~data['CONV']), 'DX'] = 'MCI-NC'
    data.pop('CONV')

    return data

def average_pet_features():
    """
    Return a df with each patient containing features that are the
    mean of every feature across all regions

    """
    fdg = get_dx(FDG)
    grouped = fdg.groupby(['RID', 'VISCODE2', 'DX'], as_index=False)
    agg = grouped.aggregate(np.mean)

    visit_features = ['RID', 'VISCODE2', 'DX']
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    base_dx = get_baseline_classes(agg)
    res = agg.RID.apply(lambda x: base_dx[x] == 'MCI-C')
    agg['CONV'] = res
    agg.loc[(agg['DX'] == 'MCI') & (agg['CONV']), 'DX'] = 'MCI-C'
    agg.loc[(agg['DX'] == 'MCI') & (~agg['CONV']), 'DX'] = 'MCI-NC'

    return agg[visit_features+features]

def plot_features():
    """
    Show the distribution of the features given the diagnoisis
    """
    features = ['MEAN', 'MEDIAN', 'MODE', 'MIN', 'MAX', 'STDEV']
    regions = FDG['ROI'].unique()
    rid = FDG['RID'].unique()
    dx_base = get_baseline_classes(FDG)
    groups = list(set(dx_base.values()))

    stats = {}
    for feature in features:
        stats[feature] = {}
        for group in groups:
            stats[feature][group] = []

    for patient in rid:
        if patient in dx_base:
            patient_info = FDG[FDG['RID'] == patient]
            visit = np.sort(patient_info['VISCODE2'].unique())[0]
            info = patient_info[patient_info['VISCODE2'] == visit]
            for feature in features:
                stats[feature][dx_base[patient]].append(info[feature].mean())

    for feature in features:
        if 'MCI-REV' in stats[feature]:
            stats[feature].pop('MCI-REV')

        idx = 0
        fig = plt.figure()
        for group in stats[feature].keys():
            idx += 1
            ax = fig.add_subplot(2, 2, idx)
            ax.hist(stats[feature][group], bins=50)
            ax.set_xlabel('Average value')
            ax.set_ylabel('Number of patients')
            ax.set_title('Dx = '+group)
            ax.yaxis.grid(True)

        fig.suptitle('Average of the '+feature+' of the PET value')
        fig.show()

    return stats
